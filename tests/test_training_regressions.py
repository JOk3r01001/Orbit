"""Exercise control/reward regressions with optional flight packages stubbed.

These tests do not emulate KSP physics or validate Gym/SB3 integration.
"""

import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

try:
    from krpc.services.spacecenter import AutoPilot
except ImportError:
    AutoPilot = None


ROOT = Path(__file__).resolve().parents[1]


def load_training_module(filename):
    stubs = {}
    for name in ("gymnasium", "krpc", "numpy", "stable_baselines3",
                 "stable_baselines3.common", "stable_baselines3.common.callbacks",
                 "stable_baselines3.common.env_checker", "stable_baselines3.common.monitor"):
        stubs[name] = ModuleType(name)
    class StubEnv:
        def reset(self, *, seed=None, options=None):
            pass

    stubs["gymnasium"].Env = StubEnv
    stubs["gymnasium"].spaces = SimpleNamespace()
    stubs["numpy"].clip = lambda value, low, high: min(high, max(low, value))
    stubs["numpy"].isfinite = math.isfinite
    stubs["stable_baselines3"].PPO = object
    stubs["stable_baselines3.common.callbacks"].BaseCallback = object
    stubs["stable_baselines3.common.callbacks"].CheckpointCallback = object
    stubs["stable_baselines3.common.env_checker"].check_env = Mock()
    stubs["stable_baselines3.common.monitor"].Monitor = object
    spec = importlib.util.spec_from_file_location(f"scripts._offline_{filename}", ROOT / "scripts" / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", stubs):
        spec.loader.exec_module(module)
    return module


class TrainingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launch = load_training_module("orbit_command")
        cls.apoapsis = load_training_module("apoapsis_brain")

    @unittest.skipIf(AutoPilot is None, "Optional kRPC client is not installed")
    def test_reset_engages_installed_krpc_autopilot(self):
        # Use the real generated client class with a recording RPC backend.
        # This catches API changes that an unrestricted Mock would conceal.
        client = Mock()
        autopilot = AutoPilot(client, 1)
        vessel = SimpleNamespace(
            auto_pilot=autopilot, mass=1000, available_thrust=100,
            control=SimpleNamespace(throttle=0, current_stage=1),
        )
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.vessel = None
        env.save_name = "launchpad_ready"
        env.conn = Mock()
        env._select_command = Mock()
        env._rebind_vessel_objects = Mock(side_effect=lambda: setattr(env, "vessel", vessel))
        env._get_liquid_fuel = lambda: 100
        env._get_speed_values = lambda: (70, 0, 0)
        env._safe_apoapsis = lambda: 70
        env._safe_periapsis = lambda: -500_000
        env._get_obs = lambda: [0] * 16
        env._command_info = lambda: {}
        with patch.object(self.launch.time, "sleep"), patch("builtins.print"):
            observation, _ = env.reset(seed=0)
        calls = [call for call in client._invoke.call_args_list if call.args[1] == "AutoPilot_set_Engaged"]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(call.args[2] == [autopilot, True] for call in calls))
        env.conn.space_center.load.assert_called_once_with("launchpad_ready")
        self.assertEqual(len(observation), 16)
        self.assertFalse(env._episode_finished)

    @unittest.skipIf(AutoPilot is None, "Optional kRPC client is not installed")
    def test_staging_reengages_installed_krpc_autopilot(self):
        client = Mock()
        autopilot = AutoPilot(client, 1)
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.vessel = SimpleNamespace(auto_pilot=autopilot, control=Mock())
        env._rebind_vessel_objects = Mock()
        with patch.object(self.launch.time, "sleep"):
            env._activate_next_stage(now=123)
        env.vessel.control.activate_next_stage.assert_called_once()
        env._rebind_vessel_objects.assert_called_once()
        self.assertEqual(env.last_stage_time, 123)
        self.assertEqual(client._invoke.call_args.args[:3], ("SpaceCenter", "AutoPilot_set_Engaged", [autopilot, True]))

    def test_live_goal_update_preserves_reserve_without_ksp_calls(self):
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.conn = Mock()
        env.set_command(dict(target_ap=180_000, target_pe=140_000, min_fuel_fraction=0.15))
        self.assertEqual(env.command_min_fuel_fraction, 0.15)
        self.assertEqual(env.conn.mock_calls, [])
        with self.assertRaises(ValueError):
            env.set_command(dict(target_ap=190_000))
        self.assertEqual(env.command_target_ap, 180_000)

    def test_step_exception_stops_engine(self):
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env._episode_finished = False
        env.vessel = SimpleNamespace(control=SimpleNamespace(throttle=1))
        env._step = Mock(side_effect=RuntimeError("Lost telemetry"))
        with self.assertRaisesRegex(RuntimeError, "Lost telemetry"):
            env.step([1, 1])
        self.assertEqual(env.vessel.control.throttle, 0)
        with self.assertRaisesRegex(RuntimeError, "reset"):
            env.step([1, 1])

    def test_interrupt_and_failed_reset_cut_throttle(self):
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.vessel = SimpleNamespace(control=SimpleNamespace(throttle=1))
        env._episode_finished = False
        env._step = Mock(side_effect=KeyboardInterrupt)
        with self.assertRaises(KeyboardInterrupt):
            env.step([1, 1])
        self.assertEqual(env.vessel.control.throttle, 0)
        env._reset = Mock(side_effect=RuntimeError("Save load failed"))
        env.vessel.control.throttle = 1
        with self.assertRaisesRegex(RuntimeError, "Save load failed"):
            env.reset()
        self.assertEqual(env.vessel.control.throttle, 0)
        self.assertTrue(env._episode_finished)

    def test_failed_checker_releases_launch_environment(self):
        env = Mock()
        module = load_training_module("check_launch")
        with patch.object(module, "KSPCommandedOrbitalEnv", return_value=env), \
                patch.object(module, "check_env", side_effect=RuntimeError("Invalid observation")):
            with self.assertRaisesRegex(RuntimeError, "Invalid observation"):
                module.main()
        env.close.assert_called_once()

    def test_success_and_timeout_stop_engine_before_return(self):
        for success in (True, False):
            with self.subTest(success=success):
                env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
                env.set_command(dict(target_ap=120_000, target_pe=100_000))
                env.current_step = 0
                env.max_steps = 2
                env._episode_return = 0.0
                env._episode_reward_terms = {}
                env.curriculum_stage = 3
                env.maximum_ap_overshoot = 40_000
                env.maximum_pe_overshoot = 40_000
                env._episode_finished = False
                env.dt = 0
                env.throttle_intervention_penalty_scale = self.launch.THROTTLE_INTERVENTION_PENALTY_SCALE
                env.pitch_intervention_penalty_scale = self.launch.PITCH_INTERVENTION_PENALTY_SCALE
                env._get_speed_values = lambda: (100_000, 0, 2000)
                env._safe_apoapsis = lambda: 120_000
                env._safe_periapsis = lambda: 100_000 if success else 80_000
                env._safe_time_to_apoapsis = lambda: 10
                env._compute_safe_throttle = lambda **kwargs: 1
                env._compute_safe_pitch = lambda **kwargs: 0
                env.vessel = SimpleNamespace(control=SimpleNamespace(throttle=1),
                                             auto_pilot=Mock(), available_thrust=100)
                env._handle_staging = Mock()
                env._get_obs = lambda: [0] * 16
                env.flight = SimpleNamespace(surface_altitude=100_000, vertical_speed=0, speed=2000)
                env.surface_flight = SimpleNamespace(pitch=0)
                env._horizontal_speed = lambda speed, vertical: speed
                env._current_fuel_fraction = lambda: 0.5
                env._compute_reward = lambda **kwargs: 5.0
                env._command_info = lambda: {}
                episode_rewards = []
                with patch.object(self.launch.time, "sleep"), patch("builtins.print"):
                    for _ in range(2):
                        _, reward, terminated, truncated, info = env.step([1, 0])
                        episode_rewards.append(reward)
                        self.assertAlmostEqual(sum(info["reward_terms"].values()), reward)
                        if terminated or truncated:
                            break
                self.assertEqual(terminated, success)
                self.assertEqual(truncated, not success)
                self.assertEqual(info["is_success"], success)
                self.assertEqual(env.vessel.control.throttle, 0)
                self.assertTrue(env._episode_finished)
                self.assertAlmostEqual(info["episode_return"], sum(episode_rewards))
                self.assertAlmostEqual(sum(info["episode_reward_terms"].values()), sum(episode_rewards))

    def test_low_orbit_tolerance_cannot_cut_throttle_below_atmosphere(self):
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.set_command(dict(target_ap=93_820, target_pe=70_463))
        env.coast_throttle_cap = 0.02
        for periapsis in (60_915, 68_000, 70_000):
            with self.subTest(periapsis=periapsis):
                throttle = env._compute_safe_throttle(
                    raw_throttle=0, alt=80_814, ap=101_556, pe=periapsis, time_to_ap=30,
                )
                pitch = env._compute_safe_pitch(
                    raw_pitch_fraction=1, alt=80_814, ap=101_556, pe=periapsis, time_to_ap=30,
                )
                self.assertEqual(throttle, 0.35)
                self.assertEqual(pitch, 10)
        self.assertEqual(env._compute_safe_throttle(
            raw_throttle=1, alt=80_814, ap=101_556, pe=71_000, time_to_ap=30,
        ), 0)
        # Before the next burn window, a request may be capped for coasting,
        # but is no longer unconditionally cut by the orbit-completion guard.
        self.assertEqual(env._compute_safe_throttle(
            raw_throttle=1, alt=80_814, ap=101_556, pe=60_915, time_to_ap=1401,
        ), 0.02)

    def test_reward_terms_account_for_complete_shaping_reward(self):
        cases = (
            dict(alt=500, vertical_speed=20, horizontal_speed=10, ap=10_000, pe=-500_000,
                 pitch=90, throttle=1, time_to_ap=100),
            dict(alt=30_000, vertical_speed=100, horizontal_speed=500, ap=90_000, pe=-300_000,
                 pitch=70, throttle=0.7, time_to_ap=150),
            dict(alt=80_000, vertical_speed=10, horizontal_speed=1500, ap=120_000, pe=40_000,
                 pitch=10, throttle=0, time_to_ap=200),
            dict(alt=100_000, vertical_speed=0, horizontal_speed=2100, ap=120_000, pe=95_000,
                 pitch=5, throttle=0.5, time_to_ap=30),
        )
        for telemetry in cases:
            with self.subTest(telemetry=telemetry):
                env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
                env.set_command(dict(target_ap=120_000, target_pe=100_000))
                env.prev_ap = 80_000
                env.prev_pe = -350_000
                env.prev_horizontal_speed = 400
                env.current_step = 100
                env._target_pitch_for_altitude = lambda altitude: 60
                reward = env._compute_reward(**telemetry)
                self.assertTrue(math.isfinite(reward))
                self.assertAlmostEqual(sum(env._last_reward_terms.values()), reward)
                self.assertAlmostEqual(env._last_reward_terms["time"], -0.006)

    def test_reward_guidance_requires_safe_periapsis_for_low_target(self):
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.set_command(dict(target_ap=93_820, target_pe=70_463))
        env.prev_ap = 101_556
        env.prev_pe = 67_000
        env.prev_horizontal_speed = 2000
        env.current_step = 100
        env._compute_reward(
            alt=80_814, vertical_speed=0, horizontal_speed=2000,
            ap=101_556, pe=68_000, pitch=5, throttle=0.35, time_to_ap=30,
        )
        self.assertAlmostEqual(env._last_reward_terms["coast_burn_guidance"], 0.008)

    def test_apoapsis_progress_is_rewarded_without_a_remaining_distance_charge(self):
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.set_command(dict(target_ap=180_000, target_pe=100_000))
        env.current_step = 100
        env.prev_ap = 79_000
        env.prev_pe = -350_000
        env.prev_horizontal_speed = 500
        env._target_pitch_for_altitude = lambda altitude: 60
        telemetry = dict(
            alt=30_000, vertical_speed=100, horizontal_speed=500, ap=80_000,
            pe=-350_000, pitch=60, throttle=1, time_to_ap=150,
        )
        env._compute_reward(**telemetry)
        self.assertEqual(env._last_reward_terms["apoapsis_deficit"], 0)
        self.assertGreater(env._last_reward_terms["orbit_progress"], 0)
        env._compute_reward(**{**telemetry, "ap": 78_000})
        self.assertEqual(env._last_reward_terms["apoapsis_deficit"], 0)
        self.assertLess(env._last_reward_terms["orbit_progress"], 0)

    def test_waiting_in_coast_or_requesting_a_stalled_burn_does_not_earn_reward(self):
        for time_to_ap, throttle in ((200, 0), (30, 0.5)):
            with self.subTest(time_to_ap=time_to_ap, throttle=throttle):
                env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
                env.set_command(dict(target_ap=120_000, target_pe=100_000))
                env.current_step = 100
                env.prev_ap = 120_000
                env.prev_pe = 40_000
                env.prev_horizontal_speed = 2000
                telemetry = dict(
                    alt=100_000, vertical_speed=0, horizontal_speed=2000, ap=120_000,
                    pe=40_000, pitch=5, throttle=throttle, time_to_ap=time_to_ap,
                )
                env._compute_reward(**telemetry)  # Consume any one-time milestones.
                reward = env._compute_reward(**telemetry)
                self.assertEqual(env._last_reward_terms["milestones"], 0)
                self.assertEqual(env._last_reward_terms["coast_burn_guidance"], 0)
                self.assertAlmostEqual(reward, -0.006)

    def test_burn_guidance_rewards_measured_periapsis_progress(self):
        env = self.launch.KSPCommandedOrbitalEnv.__new__(self.launch.KSPCommandedOrbitalEnv)
        env.set_command(dict(target_ap=120_000, target_pe=100_000))
        env.current_step = 100
        env.prev_ap = 120_000
        env.prev_pe = 40_000
        env.prev_horizontal_speed = 2000
        env._compute_reward(
            alt=100_000, vertical_speed=0, horizontal_speed=2000, ap=120_000,
            pe=40_100, pitch=5, throttle=0.5, time_to_ap=30,
        )
        guidance = env._last_reward_terms["coast_burn_guidance"]
        self.assertGreater(guidance, 0)
        self.assertAlmostEqual(guidance, 0.0008)

    def test_intervention_defaults_keep_cost_small_without_changing_guard_commands(self):
        with patch.object(self.launch.krpc, "connect", create=True), \
                patch.object(self.launch.np, "array", side_effect=lambda values, **kwargs: values, create=True), \
                patch.object(self.launch.np, "float32", float, create=True), \
                patch.object(self.launch.spaces, "Box", create=True), patch("builtins.print"):
            env = self.launch.KSPCommandedOrbitalEnv()
        self.assertEqual(env._compute_intervention_penalty(0, 0), 0)
        self.assertAlmostEqual(env._compute_intervention_penalty(1, 1), 0.02)
        self.assertEqual(env._compute_safe_throttle(
            raw_throttle=0, alt=100, ap=100, pe=-500_000, time_to_ap=100,
        ), 1)
        self.assertEqual(env._compute_safe_pitch(raw_pitch_fraction=0, alt=100), 80)

    def test_callback_logs_success_and_failure_returns_separately(self):
        for success in (True, False):
            with self.subTest(success=success):
                callback = self.launch.KSPTrainingCallback.__new__(self.launch.KSPTrainingCallback)
                callback.training_env = Mock()
                callback.num_timesteps = 123
                callback.logger = Mock()
                callback.locals = {
                    "dones": [True],
                    "infos": [{
                        "is_success": success, "episode_return": -12.5,
                        "reward_terms": {"time": -0.006},
                        "episode_reward_terms": {"time": -12.5},
                    }],
                }
                with patch("builtins.print"):
                    self.assertTrue(callback._on_step())
                outcome = "success" if success else "failure"
                callback.logger.record_mean.assert_any_call(f"episodes/{outcome}_return", -12.5)
                callback.logger.record_mean.assert_any_call("reward_step/time", -0.006, exclude="stdout")
                callback.logger.record_mean.assert_any_call("reward_episode/time", -12.5, exclude="stdout")

    def test_unsafe_periapsis_overrides_apoapsis_success(self):
        env = self.apoapsis.ApoapsisSpecialistEnv.__new__(self.apoapsis.ApoapsisSpecialistEnv)
        env.target_apoapsis = 120_000
        env.previous_absolute_apoapsis_error = 0
        for periapsis, expected_success in ((60_000, False), (100_000, True)):
            with self.subTest(periapsis=periapsis):
                reward, success, failure, info = env._reward(
                    dict(apoapsis=120_000, periapsis=periapsis, fuel_fraction=0.5, available_thrust=100),
                    dict(applied_throttle=0),
                )
                self.assertEqual(success, expected_success)
                self.assertEqual(failure, not expected_success)
                self.assertEqual(info["is_success"], int(expected_success))
                self.assertEqual(reward > 0, expected_success)

    def test_explicit_apoapsis_goal_is_preserved_and_unsupported_goal_rejected(self):
        env = self.apoapsis.ApoapsisSpecialistEnv.__new__(self.apoapsis.ApoapsisSpecialistEnv)
        env._choose_target(dict(apoapsis=100_000), dict(target_apoapsis=150_000))
        self.assertEqual(env.target_apoapsis, 150_000)
        with self.assertRaises(ValueError):
            env._choose_target(dict(apoapsis=100_000), dict(target_apoapsis=90_000))
        self.assertEqual(env.target_apoapsis, 150_000)

    def test_resumed_training_preserves_timestep_count_and_uses_chosen_model(self):
        fake_ppo = Mock()
        with patch.object(self.apoapsis, "PPO", fake_ppo), patch("builtins.print"):
            self.apoapsis._train_agent("env", "checkpoints", "metrics", "chosen.zip", 12345)
        fake_ppo.load.assert_called_once_with(
            "chosen.zip", env="env", device="cpu", tensorboard_log=self.apoapsis.TENSORBOARD_DIRECTORY,
        )
        fake_ppo.assert_not_called()
        fake_ppo.load.return_value.learn.assert_called_once_with(
            total_timesteps=12345, callback=["checkpoints", "metrics"], reset_num_timesteps=False,
        )

    def test_missing_resume_checkpoint_does_not_connect_to_ksp(self):
        with patch.object(self.apoapsis.os.path, "isfile", return_value=False), \
                patch.object(self.apoapsis, "ApoapsisSpecialistEnv") as constructor:
            with self.assertRaises(FileNotFoundError):
                self.apoapsis.train_agent("missing.zip")
        constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
