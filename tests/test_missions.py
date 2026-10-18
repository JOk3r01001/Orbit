import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from mission_control.commands import LandingTarget, MissionTask, OrbitTarget, TaskType
from mission_control.supervisor import ControlCommand, FlightState, MissionSupervisor, Status
from scripts.launch_command import validate_launch_command


ROOT = Path(__file__).resolve().parents[1]


def mission(kind=TaskType.REACH_ORBIT, **kwargs):
    target = LandingTarget(0, 0) if kind == TaskType.LAND else OrbitTarget(120_000, 100_000)
    return MissionTask("test-001", kind, target, **kwargs)


def state(**kwargs):
    defaults = dict(
        ut_s=100, vessel_id="rocket-1", body="Kerbin", situation="pre_launch",
        altitude_m=70, altitude_agl_m=10, apoapsis_m=70, periapsis_m=-500_000,
        vertical_speed_mps=0, horizontal_speed_mps=0, tilt_deg=0,
        fuel_fraction=1, latitude_deg=0, longitude_deg=0,
    )
    return FlightState(**{**defaults, **kwargs})


def orbit_state(**kwargs):
    return state(**{
        **dict(ut_s=101, situation="orbiting", altitude_m=100_000,
               altitude_agl_m=100_000, apoapsis_m=120_000, periapsis_m=100_000),
        **kwargs,
    })


class RecordingActuator:
    def __init__(self):
        self.commands = []
        self.stops = 0

    def apply(self, command):
        self.commands.append(command)

    def stop(self):
        self.stops += 1


class FixedController:
    def __init__(self):
        self.starts = []
        self.error = None

    def start(self, task, telemetry):
        self.starts.append((task, telemetry))

    def update(self, telemetry):
        if self.error:
            raise self.error
        return ControlCommand(0.5, 80, 90)


class MissionContractTests(unittest.TestCase):
    def test_example_tasks_round_trip_and_launch_translation(self):
        for filename in ("reach_orbit.json", "change_orbit.json", "land.json"):
            with self.subTest(filename=filename):
                task = MissionTask.from_dict(json.loads((ROOT / "examples" / filename).read_text()))
                self.assertEqual(task, MissionTask.from_dict(task.to_dict()))
                if task.task == TaskType.REACH_ORBIT:
                    options = task.launch_options()
                    self.assertEqual(validate_launch_command(options), options)

    def test_unknown_and_wrong_target_fields_rejected(self):
        payload = mission().to_dict()
        for modified in (
            {**payload, "urgncy": 1},
            {**payload, "task": "dock"},
            {**payload, "body": "Mun"},
            {**payload, "schema_version": True},
            {**payload, "target": {"latitude_deg": 0, "longitude_deg": 0}},
        ):
            with self.subTest(payload=modified), self.assertRaises(ValueError):
                MissionTask.from_dict(modified)

    def test_nonfinite_and_boolean_numeric_commands_rejected(self):
        for bad in (True, float("nan"), float("inf"), "120000", 10**400):
            payload = mission().to_dict()
            payload["target"]["apoapsis_m"] = bad
            with self.subTest(value=bad), self.assertRaises(ValueError):
                MissionTask.from_dict(payload)

    def test_inverted_or_atmospheric_orbits_rejected(self):
        for ap, pe in ((100_000, 110_000), (120_000, 60_000)):
            with self.subTest(ap=ap, pe=pe), self.assertRaises(ValueError):
                OrbitTarget(ap, pe)

    def test_launch_reserve_is_preserved_and_invalid_requests_not_mutated(self):
        command = dict(target_ap=180_000, target_pe=140_000, min_fuel_fraction=0.15)
        original = copy.deepcopy(command)
        self.assertEqual(validate_launch_command(command)["min_fuel_fraction"], 0.15)
        self.assertEqual(command, original)
        for invalid in (dict(target_ap=190_000), dict(target_ap=120_000, urgency=2),
                        dict(target_ap=120_000, min_fuel_fraction=0.2), dict(target_pe=90_000),
                        dict(target_ap=10**400)):
            with self.subTest(command=invalid), self.assertRaises(ValueError):
                validate_launch_command(invalid)


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.actuator = RecordingActuator()
        self.controller = FixedController()
        self.supervisor = MissionSupervisor({kind: self.controller for kind in TaskType}, self.actuator)

    def test_rejection_does_not_acquire_controls(self):
        supervisor = MissionSupervisor({}, self.actuator)
        with self.assertRaisesRegex(ValueError, "No controller"):
            supervisor.submit(mission(), state())
        self.assertEqual(supervisor.status, Status.IDLE)
        self.assertEqual(self.actuator.stops, 0)

    def test_only_one_controller_can_own_vessel(self):
        self.supervisor.submit(mission(), state())
        with self.assertRaisesRegex(ValueError, "mission is running"):
            self.supervisor.submit(mission(TaskType.LAND), state())
        self.assertEqual(len(self.controller.starts), 1)
        self.assertEqual(self.actuator.stops, 1)

    def test_orbit_requires_settle_period_with_engines_off(self):
        self.supervisor.submit(mission(), state())
        self.supervisor.tick(orbit_state())
        self.assertEqual(self.supervisor.status, Status.RUNNING)
        self.assertEqual(self.actuator.commands[-1].throttle, 0)
        self.supervisor.tick(orbit_state(ut_s=104))
        self.assertEqual(self.supervisor.status, Status.SUCCEEDED)
        self.assertEqual(self.actuator.stops, 2)

    def test_goal_excursion_restarts_settling(self):
        self.supervisor.submit(mission(), state())
        self.supervisor.tick(orbit_state())
        self.supervisor.tick(orbit_state(ut_s=103, apoapsis_m=150_000))
        self.assertEqual(self.actuator.commands[-1].throttle, 0.5)
        self.supervisor.tick(orbit_state(ut_s=104))
        self.assertEqual(self.supervisor.status, Status.RUNNING)
        self.supervisor.tick(orbit_state(ut_s=107))
        self.assertEqual(self.supervisor.status, Status.SUCCEEDED)

    def test_atmospheric_periapsis_cannot_succeed_with_loose_tolerance(self):
        task = replace(mission(), target=OrbitTarget(120_000, 75_000))
        self.supervisor.submit(task, state())
        self.supervisor.tick(orbit_state(periapsis_m=69_000))
        self.supervisor.tick(orbit_state(ut_s=105, periapsis_m=69_000))
        self.assertEqual(self.supervisor.status, Status.RUNNING)

    def test_timeout_and_fuel_violation_stop_engine_even_at_goal(self):
        for task, telemetry, reason in (
            (mission(timeout_s=4), orbit_state(ut_s=104), "timed out"),
            (mission(), orbit_state(fuel_fraction=0.01), "Fuel reserve"),
        ):
            with self.subTest(reason=reason):
                supervisor = MissionSupervisor({TaskType.REACH_ORBIT: self.controller}, self.actuator)
                supervisor.submit(task, state())
                report = supervisor.tick(telemetry)
                self.assertEqual(report["status"], "failed")
                self.assertIn(reason, report["reason"])

    def test_vessel_change_destruction_and_rewound_clock_fail(self):
        for telemetry in (state(ut_s=101, vessel_id="other"), state(ut_s=101, intact=False), state(ut_s=99)):
            with self.subTest(telemetry=telemetry):
                supervisor = MissionSupervisor({TaskType.REACH_ORBIT: self.controller}, self.actuator)
                supervisor.submit(mission(), state())
                self.assertEqual(supervisor.tick(telemetry)["status"], "failed")

    def test_controller_exception_cuts_throttle(self):
        self.controller.error = RuntimeError("policy unavailable")
        self.supervisor.submit(mission(), state())
        report = self.supervisor.tick(state(ut_s=101))
        self.assertEqual(report["status"], "failed")
        self.assertIn("policy unavailable", report["reason"])
        self.assertEqual(self.actuator.stops, 2)

    def test_abort_allows_next_mission(self):
        self.supervisor.submit(mission(), state())
        self.assertEqual(self.supervisor.abort()["status"], "aborted")
        self.supervisor.submit(mission(TaskType.CHANGE_ORBIT), orbit_state())
        self.assertEqual(self.supervisor.status, Status.RUNNING)
        self.assertEqual(len(self.controller.starts), 2)

    def test_landing_requires_prepared_descent(self):
        with self.assertRaisesRegex(ValueError, "descent"):
            self.supervisor.submit(mission(TaskType.LAND), orbit_state())

    def test_landing_requires_contact_speed_attitude_and_site(self):
        initial = state(situation="flying", altitude_agl_m=1000, vertical_speed_mps=-20)
        task = mission(TaskType.LAND)
        good = state(ut_s=101, situation="landed", altitude_agl_m=2,
                     touchdown_vertical_speed_mps=-1, touchdown_horizontal_speed_mps=0.5)
        for bad in (
            replace(good, situation="flying"), replace(good, vertical_speed_mps=-10),
            replace(good, horizontal_speed_mps=5), replace(good, tilt_deg=60),
            replace(good, longitude_deg=10),
            replace(good, touchdown_vertical_speed_mps=-20),
            replace(good, touchdown_horizontal_speed_mps=None),
        ):
            with self.subTest(telemetry=bad):
                supervisor = MissionSupervisor({TaskType.LAND: self.controller}, self.actuator)
                supervisor.submit(task, initial)
                supervisor.tick(bad)
                supervisor.tick(replace(bad, ut_s=105))
                self.assertEqual(supervisor.status, Status.RUNNING)
        self.supervisor.submit(task, initial)
        self.supervisor.tick(good)
        self.supervisor.tick(replace(good, ut_s=104))
        self.assertEqual(self.supervisor.status, Status.SUCCEEDED)


if __name__ == "__main__":
    unittest.main()
