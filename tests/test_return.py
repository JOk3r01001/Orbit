"""Offline deorbit, handoff, policy-contract and control ownership checks."""

from dataclasses import replace
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mission_control.commands import DeorbitTarget, MissionTask, OrbitTarget, TaskType
from mission_control.descent import DescentProfile, DescentSample, ReturnMission, braking_metrics, entry_rejection
from mission_control.orbital_controller import DeorbitController
from mission_control.orbits import OrbitElements, OrbitalTelemetry, plan_deorbit, plan_orbit_change
from mission_control.supervisor import ControlCommand, FlightState, MissionSupervisor
from mission_control.krpc_adapter import KRPCOrbitalAdapter
from scripts.return_to_ground import TelemetryClock, return_plan, run_return

try:
    import numpy as np
    from scripts.descent_train import DescentEnv, load_policy, policy_contract
except ImportError:
    DescentEnv = None

ROOT = Path(__file__).resolve().parents[1]


def sample(**changes):
    defaults = dict(ut_s=1000, height_m=20_000, vertical_mps=-200, horizontal_mps=100,
                    tilt_deg=30, fuel_fraction=.8, acceleration_mps2=30, gravity_mps2=9.81,
                    latitude_deg=0, longitude_deg=0, situation="flying", north_mps=100,
                    apoapsis_m=150_000, periapsis_m=20_000)
    return DescentSample(**{**defaults, **changes})


def mission(**changes):
    return ReturnMission(MissionTask("deorbit-test", TaskType.DEORBIT, DeorbitTarget(), timeout_s=7200), **changes)


def flight(**changes):
    values = dict(ut_s=100, vessel_id="ship", body="Kerbin", situation="orbiting", altitude_m=150_000,
                  altitude_agl_m=150_000, apoapsis_m=150_000, periapsis_m=120_000,
                  vertical_speed_mps=0, horizontal_speed_mps=2200, tilt_deg=90, fuel_fraction=1,
                  latitude_deg=0, longitude_deg=0,
                  orbital=OrbitalTelemetry(600_000, 3.5316e12, 100, 1000, 5000, 200_000, 320, -1))
    return FlightState(**{**values, **changes})


class DeorbitTests(unittest.TestCase):
    def test_deorbit_command_roundtrip_and_strict_limits(self):
        command = mission().deorbit
        self.assertEqual(command, MissionTask.from_dict(command.to_dict()))
        for pe in (-10, 0, 69_000, True, float("nan")):
            with self.subTest(pe=pe), self.assertRaises(ValueError):
                DeorbitTarget(pe)
        with self.assertRaises(ValueError):
            MissionTask.from_dict({**command.to_dict(), "target": {"periapsis_m": 20_000, "apoapsis_m": 150_000}})

    def test_retrograde_burn_reaches_low_periapsis_from_orbital_invariants(self):
        elements = OrbitElements(600_000, 3.5316e12, 150_000, 120_000)
        burn, = plan_deorbit(elements, DeorbitTarget())
        self.assertEqual(burn.at_apsis, "apoapsis")
        self.assertLess(burn.delta_v_mps, 0)
        radius = elements.body_radius_m + elements.apoapsis_m
        speed = elements.speed_at(elements.apoapsis_m) + burn.delta_v_mps
        energy = speed ** 2 / 2 - elements.gravitational_parameter / radius
        a = -elements.gravitational_parameter / (2 * energy)
        e = math.sqrt(1 + 2 * energy * (radius * speed) ** 2 / elements.gravitational_parameter ** 2)
        self.assertAlmostEqual(a * (1 - e) - elements.body_radius_m, 20_000, delta=.01)

    def test_regular_orbit_guard_is_preserved(self):
        with self.assertRaises(ValueError):
            plan_orbit_change(OrbitElements(600_000, 3.5316e12, 150_000, 120_000), OrbitTarget(150_000, 70_000))

    def test_deorbit_burn_continues_below_70km_periapsis_then_settles(self):
        controller = DeorbitController()
        actuator = Mock()
        supervisor = MissionSupervisor({TaskType.DEORBIT: controller}, actuator)
        initial = flight()
        supervisor.submit(mission().deorbit, initial)
        controller.update(replace(initial, ut_s=101))
        during = replace(initial, ut_s=controller.center_ut, periapsis_m=60_000, situation="sub_orbital")
        command = controller.update(during)
        self.assertEqual(command.attitude, "orbital_retrograde")
        self.assertGreater(command.throttle, 0)
        at_goal = replace(during, ut_s=during.ut_s + .2, periapsis_m=20_100)
        supervisor.tick(at_goal)
        self.assertEqual(supervisor.report()["status"], "running")
        supervisor.tick(replace(at_goal, ut_s=at_goal.ut_s + 3.1))
        self.assertEqual(supervisor.report()["status"], "succeeded")
        actuator.stop.assert_called()


class HandoffTests(unittest.TestCase):
    def test_profile_and_return_json_roundtrip(self):
        config = ReturnMission.from_dict(json.loads((ROOT / "examples/return/mission.json").read_text()))
        self.assertEqual(ReturnMission.from_dict(config.to_dict()), config)
        with self.assertRaises(ValueError):
            DescentProfile.from_dict({"max_heigth_m": 10_000})

    def test_lower_periapsis_alone_does_not_trigger_handoff(self):
        self.assertEqual(mission().handoff(sample(height_m=80_000, situation="sub_orbital")), "wait")
        self.assertEqual(mission().handoff(sample(vertical_mps=100)), "wait")
        self.assertEqual(mission().handoff(sample()), "ready")

    def test_fast_recoverable_descent_is_accepted_without_manual_slowdown(self):
        state = sample(vertical_mps=-400, horizontal_mps=500, acceleration_mps2=50)
        self.assertIsNone(entry_rejection(state, DescentProfile()))
        self.assertEqual(mission().handoff(state), "ready")

    def test_horizontal_velocity_is_included_in_braking_margin(self):
        slow = sample(horizontal_mps=0)
        fast = replace(slow, horizontal_mps=1500)
        self.assertGreater(braking_metrics(fast, DescentProfile())["required_height_m"],
                           braking_metrics(slow, DescentProfile())["required_height_m"])
        with self.assertRaisesRegex(ValueError, "braking height"):
            mission().handoff(fast)

    def test_unsafe_late_or_unpowered_handoff_rejected(self):
        for state in (sample(height_m=500), sample(acceleration_mps2=5),
                      sample(horizontal_mps=2500), sample(intact=False)):
            with self.subTest(state=state), self.assertRaises(ValueError):
                mission().handoff(state)

    def test_clock_detects_pause_rewind_and_jump(self):
        with patch("scripts.return_to_ground.time.monotonic", side_effect=[0, 4]):
            clock = TelemetryClock(100)
            with self.assertRaises(TimeoutError):
                clock.advance(100)
        for ut in (99, 103):
            with self.assertRaises(RuntimeError):
                TelemetryClock(100).advance(ut)

    def test_snapshot_preview_works_without_flight_packages(self):
        result = subprocess.run([sys.executable, "-S", "-m", "scripts.return_to_ground", "--snapshot",
                                 "examples/telemetry/orbit_snapshot.json"], cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["burns"][0]["attitude"], "orbital_retrograde")


class ReturnRunnerTests(unittest.TestCase):
    def adapters(self):
        events = []
        adapter = Mock()
        adapter.vessel = SimpleNamespace(parts=SimpleNamespace(all=[1, 2, 3], parachutes=[]))
        adapter.initial_fuel = 100
        adapter.stop.side_effect = lambda: events.append("stop")
        descent = Mock()
        descent.read.side_effect = [sample(), sample(), sample(ut_s=1000.2)]
        descent.begin_current.side_effect = lambda **_: (events.append("begin") or [0] * 26, {})
        descent.previous = sample(ut_s=1001, situation="landed", vertical_mps=0, horizontal_mps=0)
        descent.step.return_value = ([0] * 26, 100, True, False, {"is_success": True, "reason": "success"})
        return adapter, descent, events

    def run_return_fixture(self, adapter, descent, **kwargs):
        with tempfile.TemporaryDirectory() as directory, patch("scripts.return_to_ground.run_live", return_value={"result": {"status": "succeeded"}}):
            return run_return(mission(), adapter, descent, log_dir=Path(directory), **kwargs)

    def test_cutoff_precedes_single_ppo_handoff_and_no_reload(self):
        adapter, descent, events = self.adapters()
        model = Mock()
        model.predict.return_value = ([0, 0, 0], None)
        report = self.run_return_fixture(adapter, descent, model=model)
        self.assertEqual(report["status"], "succeeded")
        self.assertEqual(events[:2], ["stop", "begin"])
        descent.begin_current.assert_called_once_with(expected_vessel=adapter.vessel, fuel_reference=100)
        descent.sc.load.assert_not_called()
        descent.sc.save.assert_not_called()

    def test_capture_saves_handoff_without_policy_or_reload(self):
        adapter, descent, events = self.adapters()
        report = self.run_return_fixture(adapter, descent, capture_save="descent_training_start")
        self.assertEqual(report["status"], "handoff_saved")
        descent.sc.save.assert_called_once_with("descent_training_start")
        descent.begin_current.assert_not_called()
        descent.sc.load.assert_not_called()

    def test_handoff_failure_cuts_throttle_without_running_policy(self):
        adapter, descent, events = self.adapters()
        descent.read.side_effect = [sample(), sample(height_m=500)]
        model = Mock()
        report = self.run_return_fixture(adapter, descent, model=model)
        self.assertEqual(report["status"], "failed")
        model.predict.assert_not_called()
        adapter.stop.assert_called()

    def test_failed_deorbit_never_enters_landing(self):
        adapter, descent, events = self.adapters()
        with tempfile.TemporaryDirectory() as directory, patch("scripts.return_to_ground.run_live", return_value={"result": {"status": "failed"}}):
            report = run_return(mission(), adapter, descent, model=Mock(), log_dir=Path(directory))
        self.assertEqual(report["status"], "failed")
        descent.begin_current.assert_not_called()
        descent.read.assert_called_once()

    def test_insufficient_thrust_rejected_before_deorbit(self):
        adapter, descent, events = self.adapters()
        descent.read.side_effect = [sample(acceleration_mps2=5)]
        with tempfile.TemporaryDirectory() as directory, patch("scripts.return_to_ground.run_live") as deorbit:
            report = run_return(mission(), adapter, descent, model=Mock(), log_dir=Path(directory))
        self.assertEqual(report["status"], "failed")
        self.assertAlmostEqual(report["preflight"]["thrust_weight"], 5 / 9.81)
        self.assertIn("0.510", report["reason"])
        self.assertIn("minimum 1.300", report["reason"])
        deorbit.assert_not_called()
        adapter.stop.assert_called()

    def test_preview_reports_available_thrust_and_mass_without_actuation(self):
        state = flight()
        result = return_plan(mission(), state)
        engine = result["engine_preflight"]
        self.assertEqual(engine["mass_kg"], 5000)
        self.assertEqual(engine["available_thrust_n"], 200_000)
        self.assertAlmostEqual(engine["available_acceleration_mps2"], 40)
        self.assertAlmostEqual(engine["surface_thrust_weight_estimate"], 40 / (3.5316e12 / 600_000 ** 2))
        self.assertTrue(engine["current_thrust_weight_ok"])

    def test_surface_retrograde_uses_rotating_body_velocity(self):
        body = SimpleNamespace(reference_frame=object())
        craft = SimpleNamespace(orbit=SimpleNamespace(body=body), surface_reference_frame=object(),
                                velocity=Mock(return_value=(0, 0, 100)), auto_pilot=SimpleNamespace(),
                                control=SimpleNamespace(throttle=1))
        sc = SimpleNamespace(active_vessel=craft, rails_warp_factor=0, physics_warp_factor=0,
                             transform_direction=Mock(return_value=(0, 100, 0)))
        adapter = KRPCOrbitalAdapter.__new__(KRPCOrbitalAdapter)
        adapter.body, adapter.vessel, adapter.space_center = body, craft, sc
        adapter.apply(ControlCommand(attitude="surface_retrograde"))
        craft.velocity.assert_called_once_with(body.reference_frame)
        sc.transform_direction.assert_called_once_with((0, 0, 100), body.reference_frame, craft.surface_reference_frame)
        self.assertEqual(craft.auto_pilot.target_direction, (0, -1, 0))
        self.assertEqual(craft.control.throttle, 0)


@unittest.skipIf(DescentEnv is None, "Optional flight dependencies unavailable")
class DescentPolicyTests(unittest.TestCase):
    def env(self):
        env = DescentEnv(connection=SimpleNamespace(space_center=Mock(), close=Mock()))
        env.vessel = SimpleNamespace(auto_pilot=SimpleNamespace(), control=SimpleNamespace(throttle=0), surface_reference_frame=object())
        env.previous = sample()
        env.target = env.default_target(env.previous)
        env.throttle, env.lean, env.started_ut, env.timeout, env.contact = 0, (0, 0), 1000, 900, None
        return env

    def test_policy_observes_braking_attitude_and_orbital_context(self):
        env = self.env()
        observation = env.observation(env.previous)
        self.assertEqual(observation.shape, (26,))
        self.assertTrue(np.isfinite(observation).all())
        self.assertTrue(env.observation_space.contains(observation))
        self.assertNotEqual(observation[12], env.observation(replace(env.previous, periapsis_m=30_000))[12])

    def test_directed_thrust_normalized_and_bounded(self):
        env = self.env()
        env.apply_policy_action([1, 1, 1])
        vector = env.vessel.auto_pilot.target_direction
        self.assertAlmostEqual(sum(v * v for v in vector), 1)
        self.assertGreaterEqual(vector[0], math.cos(math.radians(85)) - 1e-10)
        self.assertEqual(env.vessel.control.throttle, 1)
        env.previous = replace(env.previous, nose_up=-.1)
        env.apply_policy_action([1, 0, 0])
        self.assertEqual(env.vessel.control.throttle, 0)

    def test_zero_thrust_terminal_observation_is_finite(self):
        env = self.env()
        self.assertTrue(np.isfinite(env.observation(replace(env.previous, acceleration_mps2=0))).all())

    def test_old_or_missing_policy_rejected_before_connection(self):
        from scripts.return_to_ground import main
        with tempfile.TemporaryDirectory() as directory, patch("mission_control.krpc_adapter.connect_orbital_client") as connect:
            with self.assertRaises(SystemExit):
                main(["--execute", "--model", str(Path(directory) / "absent.zip")])
            connect.assert_not_called()
            path = Path(directory) / "old.zip"
            path.touch()
            path.with_suffix(".json").write_text(json.dumps({"contract": {"observation_version": 1}}))
            with self.assertRaisesRegex(ValueError, "contract/profile"):
                load_policy(path, DescentProfile())

    def test_real_gym_ppo_training_and_policy_contract_reload(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        craft = SimpleNamespace(control=SimpleNamespace(throttle=0), auto_pilot=Mock(),
                                surface_reference_frame=object(), orbit=SimpleNamespace(body=SimpleNamespace(name="Kerbin")),
                                resources=SimpleNamespace(amount=lambda _: 100), parts=SimpleNamespace(all=[1, 2], parachutes=[]))
        sc = SimpleNamespace(active_vessel=craft, load=Mock())
        env = DescentEnv(connection=SimpleNamespace(space_center=sc, close=Mock()))
        ut = [1000]

        def read():
            ut[0] += .2
            return sample(ut_s=ut[0])

        env.read = read
        with tempfile.TemporaryDirectory() as directory, patch("scripts.landing_train.time.sleep"):
            check_env(env, warn=True, skip_render_check=True)
            model = PPO("MlpPolicy", env, n_steps=8, batch_size=4, n_epochs=1,
                        policy_kwargs={"net_arch": [8]}, device="cpu", seed=4)
            model.learn(total_timesteps=8)
            path = Path(directory) / "descent.zip"
            model.save(str(path))
            path.with_suffix(".json").write_text(json.dumps({"contract": policy_contract(env.profile)}))
            loaded = load_policy(path, env.profile)
            self.assertEqual(loaded.num_timesteps, 8)
            loaded.set_env(env)
            loaded.learn(total_timesteps=8, reset_num_timesteps=False)
            self.assertEqual(loaded.num_timesteps, 16)
            load_count = sc.load.call_count
            observation, _ = env.begin_current(expected_vessel=craft, fuel_reference=100)
            self.assertEqual(observation.shape, (26,))
            self.assertEqual(sc.load.call_count, load_count)
            with self.assertRaisesRegex(ValueError, "profile mismatch"):
                load_policy(path, replace(env.profile, max_height_m=25_000))
        env.close()


if __name__ == "__main__":
    unittest.main()
