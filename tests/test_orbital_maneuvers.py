from dataclasses import replace
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from mission_control.commands import MissionTask, OrbitTarget, TaskType
from mission_control.krpc_adapter import KRPCOrbitalAdapter, _install_rpc_deadline
from mission_control.orbital_controller import OrbitalManeuverController, PID
from mission_control.orbits import OrbitElements, OrbitalTelemetry, burn_duration, plan_orbit_change
from mission_control.supervisor import ControlCommand, FlightState, MissionSupervisor, Status
from scripts.orbit_maneuver import main, plan_report, run_live

ROOT = Path(__file__).resolve().parents[1]
RADIUS = 600_000
MU = 3.5316e12

try:
    from krpc.encoder import Encoder
    from krpc.schema.KRPC_pb2 import ConnectionResponse
except ImportError:
    Encoder = ConnectionResponse = None


def task(**kwargs):
    return MissionTask("orbit-test", TaskType.CHANGE_ORBIT, OrbitTarget(150_000, 120_000, 1000, 1000), timeout_s=7200, **kwargs)


def telemetry(ut=100, **kwargs):
    orbital = OrbitalTelemetry(RADIUS, MU, 600, 100, 5000, 200_000, 320, 1)
    values = dict(
        ut_s=ut, vessel_id="test-vessel", body="Kerbin", situation="orbiting",
        altitude_m=100_000, altitude_agl_m=100_000, apoapsis_m=120_000, periapsis_m=90_000,
        vertical_speed_mps=0, horizontal_speed_mps=2200, tilt_deg=90,
        fuel_fraction=1, latitude_deg=0, longitude_deg=0, orbital=orbital,
    )
    return FlightState(**{**values, **kwargs})


def apply_impulse(elements, burn):
    """Recover apsides from independent energy/angular-momentum equations."""
    altitude = elements.periapsis_m if burn.at_apsis == "periapsis" else elements.apoapsis_m
    radius = elements.body_radius_m + altitude
    speed = elements.speed_at(altitude) + burn.delta_v_mps
    energy = speed * speed / 2 - elements.gravitational_parameter / radius
    momentum = radius * speed
    semi_major = -elements.gravitational_parameter / (2 * energy)
    eccentricity = math.sqrt(max(0, 1 + 2 * energy * momentum ** 2 / elements.gravitational_parameter ** 2))
    return OrbitElements(
        elements.body_radius_m, elements.gravitational_parameter,
        semi_major * (1 + eccentricity) - elements.body_radius_m,
        semi_major * (1 - eccentricity) - elements.body_radius_m,
    )


class PlannerTests(unittest.TestCase):
    def test_raise_and_lower_circular_orbits(self):
        for start, end in ((100_000, 150_000), (150_000, 100_000)):
            with self.subTest(start=start, end=end):
                elements = OrbitElements(RADIUS, MU, start, start)
                burns = plan_orbit_change(elements, OrbitTarget(end, end, 1, 1))
                self.assertEqual(len(burns), 2)
                self.assertTrue(all((burn.delta_v_mps > 0) == (end > start) for burn in burns))
                for burn in burns:
                    elements = apply_impulse(elements, burn)
                self.assertAlmostEqual(elements.apoapsis_m, end, delta=0.1)
                self.assertAlmostEqual(elements.periapsis_m, end, delta=0.1)

    def test_lowering_below_old_periapsis_changes_periapsis_first(self):
        burns = plan_orbit_change(OrbitElements(RADIUS, MU, 180_000, 160_000), OrbitTarget(120_000, 90_000))
        self.assertEqual([burn.changed_apsis for burn in burns], ["periapsis", "apoapsis"])

    def test_randomized_plans_reach_target_according_to_orbital_invariants(self):
        rng = random.Random(42)
        for _ in range(100):
            start_pe = rng.uniform(72_000, 175_000)
            start_ap = rng.uniform(start_pe, 180_000)
            end_pe = rng.uniform(72_000, 175_000)
            end_ap = rng.uniform(max(80_000, end_pe), 180_000)
            elements = OrbitElements(RADIUS, MU, start_ap, start_pe)
            target = OrbitTarget(end_ap, end_pe, 1, 1)
            for burn in plan_orbit_change(elements, target):
                elements = apply_impulse(elements, burn)
                self.assertGreater(elements.periapsis_m, 70_000)
            self.assertAlmostEqual(elements.apoapsis_m, end_ap, delta=1.1)
            self.assertAlmostEqual(elements.periapsis_m, end_pe, delta=1.1)

    def test_near_circular_tolerances_do_not_require_impossible_apsis_crossing(self):
        cases = (
            (150_000, 149_000, OrbitTarget(151_000, 150_500, 1000, 1000)),
            (150_000, 149_000, OrbitTarget(151_000, 150_999, 1000, 1)),
            (170_000, 151_000, OrbitTarget(150_500, 150_000, 1000, 1000)),
        )
        for ap, pe, target in cases:
            with self.subTest(ap=ap, pe=pe, target=target):
                elements = OrbitElements(RADIUS, MU, ap, pe)
                for burn in plan_orbit_change(elements, target):
                    self.assertGreater(burn.tolerance_m, 0)
                    elements = apply_impulse(elements, burn)
                self.assertLessEqual(abs(elements.apoapsis_m - target.apoapsis_m), target.apoapsis_tolerance_m + 0.1)
                self.assertLessEqual(abs(elements.periapsis_m - target.periapsis_m), target.periapsis_tolerance_m + 0.1)

    def test_rejects_atmospheric_targets_and_starting_orbits(self):
        for elements, target in (
            (OrbitElements(RADIUS, MU, 120_000, 60_000), OrbitTarget(150_000, 120_000)),
            (OrbitElements(RADIUS, MU, 120_000, 90_000), OrbitTarget(150_000, 70_000)),
        ):
            with self.assertRaises(ValueError):
                plan_orbit_change(elements, target)

    def test_already_satisfied_goal_has_no_burns(self):
        self.assertEqual(plan_orbit_change(OrbitElements(RADIUS, MU, 150_000, 120_000), task().target), [])

    def test_burn_time_accounts_for_mass_loss_and_rejects_no_thrust(self):
        duration = burn_duration(100, 5000, 100_000, 320)
        self.assertGreater(duration, 0)
        self.assertLess(duration, 100 / (100_000 / 5000))
        with self.assertRaises(ValueError):
            burn_duration(100, 5000, 0, 320)


class PIDAndControllerTests(unittest.TestCase):
    def test_pid_saturation_does_not_wind_up_integrator(self):
        pid = PID()
        for _ in range(100):
            self.assertEqual(pid.update(100, 0.2), 1)
        self.assertEqual(pid.integral, 0)
        self.assertEqual(pid.update(0, 0.2), 0)
        with self.assertRaises(ValueError):
            pid.update(1, 0)

    def test_pid_output_respects_dynamic_impulse_cap(self):
        self.assertLessEqual(PID().update(100, 0.2, output_max=0.05), 0.05)

    def test_controller_waits_for_window_and_alignment_then_stops_at_apsis_target(self):
        controller = OrbitalManeuverController()
        initial = telemetry()
        controller.start(task(), initial)
        command = controller.update(telemetry(100.2))
        self.assertEqual(command.throttle, 0)
        self.assertEqual(command.attitude, "orbital_prograde")
        ut = 100.2
        while ut + 0.2 < controller.center_ut:
            ut += 0.2
            command = controller.update(telemetry(ut, orbital=replace(initial.orbital, prograde_alignment=0)))
            self.assertEqual(command.throttle, 0)
        ut += 0.2
        command = controller.update(telemetry(ut))
        self.assertEqual(controller.phase, "burn")
        self.assertGreater(command.throttle, 0)
        command = controller.update(telemetry(ut + 0.2, apoapsis_m=150_000))
        self.assertEqual(command.throttle, 0)
        self.assertEqual(controller.phase, "planning")
        controller.update(telemetry(ut + 0.4, apoapsis_m=150_000))
        self.assertEqual(controller.burn.changed_apsis, "periapsis")

    def test_retrograde_burn_requires_retrograde_alignment(self):
        controller = OrbitalManeuverController()
        initial = telemetry(apoapsis_m=175_000, periapsis_m=130_000)
        controller.start(task(), initial)
        command = controller.update(replace(initial, ut_s=100.2))
        self.assertEqual(command.attitude, "orbital_retrograde")
        self.assertEqual(command.throttle, 0)

    def test_long_burn_is_rejected(self):
        initial = telemetry(orbital=replace(telemetry().orbital, available_thrust_n=1))
        with self.assertRaisesRegex(ValueError, "too long"):
            OrbitalManeuverController().start(task(), initial)

    def test_orbital_attitude_is_mutually_exclusive_with_surface_angles(self):
        with self.assertRaises(ValueError):
            ControlCommand(pitch_deg=0, heading_deg=90, attitude="orbital_prograde")


class FakeVessel:
    def __init__(self):
        self.body = SimpleNamespace(name="Kerbin", reference_frame="rotating", non_rotating_reference_frame="inertial",
                                    equatorial_radius=RADIUS, gravitational_parameter=MU)
        self.orbit = SimpleNamespace(body=self.body, apoapsis_altitude=120_000, periapsis_altitude=90_000,
                                     time_to_apoapsis=600, time_to_periapsis=100)
        self.resources = SimpleNamespace(amount=lambda name: 100)
        self.control = SimpleNamespace(throttle=0.7, activate_next_stage=Mock())
        self.auto_pilot = SimpleNamespace(engaged=False)
        self.surface_reference_frame = "surface"
        self.mass = 5000
        self.available_thrust = 200_000
        self.specific_impulse = 320
        self.situation = SimpleNamespace(name="orbiting")

    def velocity(self, frame):
        return (2200, 0, 0)

    def direction(self, frame):
        return (1, 0, 0)

    def flight(self, frame):
        return SimpleNamespace(mean_altitude=100_000, surface_altitude=100_000, vertical_speed=0,
                               horizontal_speed=2200, pitch=0, latitude=0, longitude=0)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.vessel = FakeVessel()
        self.connection = SimpleNamespace(
            space_center=SimpleNamespace(active_vessel=self.vessel, rails_warp_factor=0,
                                         physics_warp_factor=0, ut=100, load=Mock()), close=Mock(),
        )
        self.adapter = KRPCOrbitalAdapter(self.connection)

    def test_preview_does_not_change_controls_or_load_a_save(self):
        state = self.adapter.read()
        result = plan_report(task(), state)
        self.adapter.close()
        self.assertEqual(result["mode"], "plan_only")
        self.assertEqual(self.vessel.control.throttle, 0.7)
        self.assertFalse(self.vessel.auto_pilot.engaged)
        self.connection.space_center.load.assert_not_called()
        self.vessel.control.activate_next_stage.assert_not_called()

    def test_orbital_actuation_uses_inertial_vector_and_engaged_property(self):
        self.adapter.apply(ControlCommand(throttle=0.3, attitude="orbital_retrograde"))
        self.assertEqual(self.vessel.auto_pilot.target_direction, (-1, 0, 0))
        self.assertEqual(self.vessel.auto_pilot.reference_frame, "inertial")
        self.assertTrue(self.vessel.auto_pilot.engaged)
        self.assertEqual(self.vessel.control.throttle, 0.3)
        self.adapter.stop()
        self.assertEqual(self.vessel.control.throttle, 0)
        self.assertFalse(self.vessel.auto_pilot.engaged)

    def test_switching_active_vessel_blocks_commands_but_stops_original(self):
        other = FakeVessel()
        self.connection.space_center.active_vessel = other
        with self.assertRaisesRegex(RuntimeError, "Active vessel changed"):
            self.adapter.apply(ControlCommand(throttle=1))
        self.adapter.stop()
        self.assertEqual(self.vessel.control.throttle, 0)
        self.assertEqual(other.control.throttle, 0.7)

    def test_staging_and_warp_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "never stages"):
            self.adapter.apply(ControlCommand(stage=True))
        self.connection.space_center.physics_warp_factor = 1
        with self.assertRaisesRegex(RuntimeError, "time warp"):
            self.adapter.read()

    @unittest.skipIf(ConnectionResponse is None, "Optional kRPC client is not installed")
    def test_rpc_deadline_covers_missing_size_prefix_and_missing_body(self):
        for partial in (b"", b"\x05"):
            with self.subTest(partial=partial):
                local, remote = socket.socketpair()
                try:
                    connection = SimpleNamespace(_socket=local)
                    _install_rpc_deadline(connection, 0.05)
                    if partial:
                        remote.sendall(partial)
                    with self.assertRaises(TimeoutError):
                        connection.receive_message(ConnectionResponse)
                finally:
                    local.close()
                    remote.close()

    @unittest.skipIf(ConnectionResponse is None, "Optional kRPC client is not installed")
    def test_bounded_reader_preserves_krpc_message_framing(self):
        local, remote = socket.socketpair()
        try:
            connection = SimpleNamespace(_socket=local)
            _install_rpc_deadline(connection, 0.5)
            response = ConnectionResponse()
            response.client_identifier = b"offline-client"
            remote.sendall(Encoder.encode_message_with_size(response))
            decoded = connection.receive_message(ConnectionResponse)
            self.assertEqual(decoded.client_identifier, b"offline-client")
        finally:
            local.close()
            remote.close()


class RunnerTests(unittest.TestCase):
    def test_offline_cli_plans_without_optional_dependencies_or_ksp(self):
        result = subprocess.run([
            sys.executable, "-m", "scripts.orbit_maneuver", "--snapshot", "examples/telemetry/orbit_snapshot.json",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["mode"], "plan_only")
        self.assertEqual(len(report["burns"]), 2)
        self.assertTrue(all(burn["delta_v_mps"] < 0 for burn in report["burns"]))

    def test_cli_cannot_execute_an_offline_snapshot(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
            main(["--snapshot", "examples/telemetry/orbit_snapshot.json", "--execute"])
        self.assertEqual(error.exception.code, 2)

    def test_runner_holds_goal_then_records_success(self):
        adapter = Mock()
        adapter.read.side_effect = [telemetry(), telemetry(100.2, apoapsis_m=150_000, periapsis_m=120_000),
                                    telemetry(100.4, apoapsis_m=150_000, periapsis_m=120_000)]
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch("scripts.orbit_maneuver.time.sleep"), patch("builtins.print"):
            result = run_live(task(settle_time_s=0.2), adapter, log_dir=directory)
            self.assertEqual(result["result"]["status"], "succeeded")
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 1)
        self.assertGreaterEqual(adapter.stop.call_count, 2)

    def test_transport_failure_and_interrupt_release_controls(self):
        for exception, expected in ((ConnectionError("Disconnected"), "failed"), (KeyboardInterrupt(), "aborted")):
            with self.subTest(exception=type(exception).__name__):
                adapter = Mock()
                adapter.read.side_effect = [telemetry(), exception]
                with tempfile.TemporaryDirectory(dir=ROOT) as directory, patch("scripts.orbit_maneuver.time.sleep"), patch("builtins.print"):
                    result = run_live(task(), adapter, log_dir=directory)
                self.assertEqual(result["result"]["status"], expected)
                self.assertGreaterEqual(adapter.stop.call_count, 2)


class FiniteBurnSimulationTests(unittest.TestCase):
    def test_closed_loop_raise_and_lower_in_two_body_simulation(self):
        """Integrate finite thrust and propellant use, not the planner's impulses."""
        for start_ap, start_pe, target_ap, target_pe in (
            (120_000, 90_000, 150_000, 120_000),
            (175_000, 130_000, 150_000, 120_000),
        ):
            with self.subTest(start_ap=start_ap, target_ap=target_ap):
                initial = OrbitElements(RADIUS, MU, start_ap, start_pe)
                vector = [RADIUS + start_pe, 0, 0, initial.speed_at(start_pe), 5000.0]
                clock = 100.0
                dt = 0.5
                actuator = SimpleNamespace(command=ControlCommand())
                actuator.apply = lambda command: setattr(actuator, "command", command)
                actuator.stop = lambda: setattr(actuator, "command", ControlCommand())

                def observe():
                    x, y, vx, vy, mass = vector
                    radius = math.hypot(x, y)
                    speed2 = vx * vx + vy * vy
                    energy = speed2 / 2 - MU / radius
                    momentum = x * vy - y * vx
                    semi_major = -MU / (2 * energy)
                    eccentricity = math.sqrt(max(0, 1 + 2 * energy * momentum ** 2 / MU ** 2))
                    ap = semi_major * (1 + eccentricity) - RADIUS
                    pe = semi_major * (1 - eccentricity) - RADIUS
                    mean_motion = math.sqrt(MU / semi_major ** 3)
                    if eccentricity < 1e-7:
                        time_pe, time_ap = 0, math.pi / mean_motion
                    else:
                        cos_e = max(-1, min(1, (1 - radius / semi_major) / eccentricity))
                        sin_e = (x * vx + y * vy) / (eccentricity * math.sqrt(MU * semi_major))
                        anomaly = math.atan2(sin_e, cos_e)
                        mean_anomaly = (anomaly - eccentricity * math.sin(anomaly)) % (2 * math.pi)
                        time_pe = (-mean_anomaly) % (2 * math.pi) / mean_motion
                        time_ap = (math.pi - mean_anomaly) % (2 * math.pi) / mean_motion
                    sign = -1 if actuator.command.attitude == "orbital_retrograde" else 1
                    orbital = OrbitalTelemetry(RADIUS, MU, time_ap, time_pe, mass, 200_000, 320, sign)
                    return telemetry(
                        clock, altitude_m=radius - RADIUS, altitude_agl_m=radius - RADIUS,
                        apoapsis_m=ap, periapsis_m=pe, orbital=orbital,
                        fuel_fraction=(mass - 4000) / 1000,
                    )

                def derivative(values):
                    x, y, vx, vy, mass = values
                    radius = math.hypot(x, y)
                    speed = math.hypot(vx, vy)
                    sign = -1 if actuator.command.attitude == "orbital_retrograde" else 1
                    thrust = 200_000 * actuator.command.throttle
                    return [
                        vx, vy, -MU * x / radius ** 3 + sign * thrust * vx / (mass * speed),
                        -MU * y / radius ** 3 + sign * thrust * vy / (mass * speed),
                        -thrust / (320 * 9.80665),
                    ]

                mission = MissionTask(
                    "simulated-transfer", TaskType.CHANGE_ORBIT,
                    OrbitTarget(target_ap, target_pe, 1000, 1000), timeout_s=7200,
                )
                controller = OrbitalManeuverController()
                supervisor = MissionSupervisor({TaskType.CHANGE_ORBIT: controller}, actuator)
                supervisor.submit(mission, observe())
                for _ in range(15_000):
                    k1 = derivative(vector)
                    k2 = derivative([v + dt * d / 2 for v, d in zip(vector, k1)])
                    k3 = derivative([v + dt * d / 2 for v, d in zip(vector, k2)])
                    k4 = derivative([v + dt * d for v, d in zip(vector, k3)])
                    vector = [v + dt * (a + 2*b + 2*c + d) / 6 for v, a, b, c, d in zip(vector, k1, k2, k3, k4)]
                    clock += dt
                    supervisor.tick(observe())
                    if supervisor.status != Status.RUNNING:
                        break
                self.assertEqual(supervisor.status, Status.SUCCEEDED, supervisor.report())
                self.assertEqual(actuator.command.throttle, 0)
                self.assertLess(vector[-1], 5000)

    def test_paused_simulation_and_large_telemetry_gaps_fail(self):
        for next_state in (telemetry(), telemetry(103)):
            with self.subTest(ut=next_state.ut_s):
                adapter = Mock()
                adapter.read.side_effect = [telemetry(), next_state]
                with tempfile.TemporaryDirectory(dir=ROOT) as directory, \
                        patch("scripts.orbit_maneuver.time.sleep"), \
                        patch("scripts.orbit_maneuver.time.monotonic", side_effect=[0, 4]), patch("builtins.print"):
                    result = run_live(task(), adapter, log_dir=directory)
                self.assertEqual(result["result"]["status"], "failed")
                self.assertGreaterEqual(adapter.stop.call_count, 2)


if __name__ == "__main__":
    unittest.main()
