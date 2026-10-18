"""Coordinate one controller per vessel; never load a training save.

Concrete controllers compute commands. Only the supervisor calls the actuator.
No KSP adapter or trained policy is bundled with this foundation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .commands import LandingTarget, MissionTask, OrbitTarget, TaskType, number


class Status(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class FlightState:
    ut_s: float
    vessel_id: str
    body: str
    situation: str
    altitude_m: float  # Above sea level, not terrain.
    altitude_agl_m: float
    apoapsis_m: float
    periapsis_m: float
    vertical_speed_mps: float
    horizontal_speed_mps: float
    tilt_deg: float  # Angle from local upright.
    fuel_fraction: float
    latitude_deg: float
    longitude_deg: float
    intact: bool = True
    touchdown_vertical_speed_mps: float | None = None
    touchdown_horizontal_speed_mps: float | None = None

    def __post_init__(self) -> None:
        number(self.ut_s, "ut_s", 0, 1e15)
        for name in ("altitude_m", "altitude_agl_m", "apoapsis_m", "periapsis_m", "vertical_speed_mps"):
            number(getattr(self, name), name, -1e12, 1e12)
        number(self.horizontal_speed_mps, "horizontal_speed_mps", 0, 1e9)
        number(self.tilt_deg, "tilt_deg", 0, 180)
        number(self.fuel_fraction, "fuel_fraction", 0, 1)
        number(self.latitude_deg, "latitude_deg", -90, 90)
        number(self.longitude_deg, "longitude_deg", -180, 180)
        if not isinstance(self.vessel_id, str) or not self.vessel_id:
            raise ValueError("vessel_id must be a nonempty string")
        if type(self.intact) is not bool:
            raise ValueError("intact must be boolean")
        for name in ("touchdown_vertical_speed_mps", "touchdown_horizontal_speed_mps"):
            value = getattr(self, name)
            if value is not None:
                number(value, name, -1e9 if name == "touchdown_vertical_speed_mps" else 0, 1e9)


@dataclass(frozen=True)
class ControlCommand:
    throttle: float = 0.0
    pitch_deg: float | None = None
    heading_deg: float | None = None
    stage: bool = False

    def __post_init__(self) -> None:
        number(self.throttle, "throttle", 0, 1)
        if (self.pitch_deg is None) != (self.heading_deg is None):
            raise ValueError("pitch_deg and heading_deg must be supplied together")
        if self.pitch_deg is not None:
            number(self.pitch_deg, "pitch_deg", -90, 90)
            number(self.heading_deg, "heading_deg", 0, 360)
        if type(self.stage) is not bool:
            raise ValueError("stage must be boolean")


class Controller(Protocol):
    """Controllers calculate commands and must not write to KSP directly."""

    def start(self, task: MissionTask, state: FlightState) -> None: ...
    def update(self, state: FlightState) -> ControlCommand: ...


class Actuator(Protocol):
    def apply(self, command: ControlCommand) -> None: ...
    def stop(self) -> None:
        """Cut throttle and release controller-owned control inputs."""
        ...


class MissionSupervisor:
    def __init__(self, controllers: dict[TaskType, Controller], actuator: Actuator):
        self.controllers = dict(controllers)
        self.actuator = actuator
        self.status = Status.IDLE
        self.task: MissionTask | None = None
        self.reason = "No mission submitted"
        self._controller: Controller | None = None
        self._vessel_id = ""
        self._started_ut = 0.0
        self._last_ut = 0.0
        self._settled_since: float | None = None

    def submit(self, task: MissionTask, state: FlightState) -> dict:
        """Reject unsupported requests without changing the vessel or an active task."""
        if self.status == Status.RUNNING:
            raise ValueError("A mission is running; abort it before submitting another")
        if task.task not in self.controllers:
            raise ValueError(f"No controller installed for {task.task.value}")
        if not state.intact or state.body != task.body:
            raise ValueError("The vessel must be intact and on the requested body")
        if state.fuel_fraction < task.minimum_fuel_fraction:
            raise ValueError("The vessel already has less than the requested fuel reserve")
        if task.task == TaskType.REACH_ORBIT and state.situation not in {"pre_launch", "landed"}:
            raise ValueError("reach_orbit starts on the launchpad; use change_orbit in orbit")
        if task.task == TaskType.CHANGE_ORBIT and not (
            state.situation == "orbiting" and state.periapsis_m > 70_000
        ):
            raise ValueError("change_orbit requires a stable Kerbin orbit above the atmosphere")
        if task.task == TaskType.LAND and not (
            state.situation in {"flying", "sub_orbital"}
            and 0 <= state.altitude_agl_m <= 10_000
            and state.vertical_speed_mps <= 0
        ):
            raise ValueError("land requires descent below 10 km AGL; deorbit is a separate future capability")
        self.task = task
        self._controller = self.controllers[task.task]
        self._vessel_id = state.vessel_id
        self._started_ut = self._last_ut = state.ut_s
        self._settled_since = None
        self.status = Status.RUNNING
        self.reason = "Mission accepted"
        try:
            # Release the previous task's controls before starting a new controller.
            self.actuator.stop()
            self._controller.start(task, state)
        except Exception as exc:
            self._finish(Status.FAILED, f"Controller startup failed: {exc}")
        return self.report()

    def tick(self, state: FlightState) -> dict:
        if self.status != Status.RUNNING:
            raise ValueError("No mission is running")
        assert self.task is not None and self._controller is not None
        if state.vessel_id != self._vessel_id or state.body != self.task.body:
            self._finish(Status.FAILED, "Vessel or body changed")
        elif not state.intact:
            self._finish(Status.FAILED, "Vessel destroyed")
        elif state.ut_s <= self._last_ut:
            self._finish(Status.FAILED, "Telemetry simulation time did not advance")
        else:
            self._last_ut = state.ut_s
            if state.ut_s - self._started_ut >= self.task.timeout_s:
                self._finish(Status.FAILED, "Mission timed out")
            elif state.fuel_fraction < self.task.minimum_fuel_fraction:
                self._finish(Status.FAILED, "Fuel reserve violated")
            else:
                try:
                    if self._goal_reached(state):
                        if self._settled_since is None:
                            self._settled_since = state.ut_s
                        # Verify the goal with engines off before handing off.
                        self.actuator.apply(ControlCommand())
                        if state.ut_s - self._settled_since >= self.task.settle_time_s:
                            self._finish(Status.SUCCEEDED, "Goal held with engines off")
                    else:
                        self._settled_since = None
                        command = self._controller.update(state)
                        if not isinstance(command, ControlCommand):
                            raise ValueError("Controller must return a ControlCommand")
                        if command.stage and self.task.task != TaskType.REACH_ORBIT:
                            raise ValueError("Staging is only supported during launch")
                        self.actuator.apply(command)
                except Exception as exc:
                    self._finish(Status.FAILED, f"Control update failed: {exc}")
        return self.report()

    def _goal_reached(self, state: FlightState) -> bool:
        assert self.task is not None
        target = self.task.target
        if isinstance(target, OrbitTarget):
            return (
                state.altitude_m > 70_000 and state.periapsis_m > 70_000
                and abs(state.apoapsis_m - target.apoapsis_m) <= target.apoapsis_tolerance_m
                and abs(state.periapsis_m - target.periapsis_m) <= target.periapsis_tolerance_m
            )
        assert isinstance(target, LandingTarget)
        # Great-circle distance on stock Kerbin (radius 600 km).
        import math
        lat1, lat2 = map(math.radians, (state.latitude_deg, target.latitude_deg))
        dlat = lat2 - lat1
        dlon = math.radians(target.longitude_deg - state.longitude_deg)
        hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        distance_m = 1_200_000 * math.asin(math.sqrt(min(1.0, max(0.0, hav))))
        return (
            state.situation == "landed" and state.altitude_agl_m <= 10
            and state.touchdown_vertical_speed_mps is not None
            and state.touchdown_horizontal_speed_mps is not None
            and abs(state.touchdown_vertical_speed_mps) <= target.max_vertical_speed_mps
            and state.touchdown_horizontal_speed_mps <= target.max_horizontal_speed_mps
            and abs(state.vertical_speed_mps) <= target.max_vertical_speed_mps
            and state.horizontal_speed_mps <= target.max_horizontal_speed_mps
            and state.tilt_deg <= target.max_tilt_deg and distance_m <= target.radius_m
        )

    def abort(self) -> dict:
        if self.status != Status.RUNNING:
            raise ValueError("No mission is running")
        self._finish(Status.ABORTED, "Aborted by commander")
        return self.report()

    def _finish(self, status: Status, reason: str) -> None:
        self.status, self.reason = status, reason
        self._controller = None
        try:
            self.actuator.stop()
        except Exception as exc:
            self.status = Status.FAILED
            self.reason += f"; actuator stop failed: {exc}"

    def report(self) -> dict:
        return {
            "task_id": self.task.task_id if self.task else None,
            "status": self.status.value,
            "reason": self.reason,
            "elapsed_s": self._last_ut - self._started_ut,
        }
