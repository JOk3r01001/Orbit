"""Strict, versioned commands shared by human and future AI commanders."""

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any


class TaskType(str, Enum):
    REACH_ORBIT = "reach_orbit"
    CHANGE_ORBIT = "change_orbit"
    LAND = "land"


def number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not minimum <= value <= maximum or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and between {minimum} and {maximum}")
    return float(value)


def fields(data: Any, allowed: set[str], required: set[str], name: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be an object")
    unknown = data.keys() - allowed
    missing = required - data.keys()
    if unknown or missing:
        raise ValueError(f"{name}: unknown fields {sorted(unknown)}; missing fields {sorted(missing)}")


@dataclass(frozen=True)
class OrbitTarget:
    apoapsis_m: float
    periapsis_m: float
    apoapsis_tolerance_m: float = 10_000.0
    periapsis_tolerance_m: float = 10_000.0

    def __post_init__(self) -> None:
        number(self.apoapsis_m, "apoapsis_m", 80_000, 180_000)
        number(self.periapsis_m, "periapsis_m", 70_000, self.apoapsis_m)
        number(self.apoapsis_tolerance_m, "apoapsis_tolerance_m", 1, 10_000)
        number(self.periapsis_tolerance_m, "periapsis_tolerance_m", 1, 10_000)


@dataclass(frozen=True)
class LandingTarget:
    latitude_deg: float
    longitude_deg: float
    radius_m: float = 500.0
    max_vertical_speed_mps: float = 2.0
    max_horizontal_speed_mps: float = 1.0
    max_tilt_deg: float = 10.0

    def __post_init__(self) -> None:
        number(self.latitude_deg, "latitude_deg", -90, 90)
        number(self.longitude_deg, "longitude_deg", -180, 180)
        number(self.radius_m, "radius_m", 1, 10_000)
        number(self.max_vertical_speed_mps, "max_vertical_speed_mps", 0.1, 5)
        number(self.max_horizontal_speed_mps, "max_horizontal_speed_mps", 0.1, 5)
        number(self.max_tilt_deg, "max_tilt_deg", 1, 30)


@dataclass(frozen=True)
class MissionTask:
    task_id: str
    task: TaskType
    target: OrbitTarget | LandingTarget
    schema_version: int = 1
    body: str = "Kerbin"
    minimum_fuel_fraction: float = 0.02
    urgency: float = 0.5
    timeout_s: float = 900.0
    settle_time_s: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a nonempty string")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Only schema_version 1 is supported")
        if self.body != "Kerbin":
            raise ValueError("The first mission profile supports Kerbin only")
        if not isinstance(self.task, TaskType):
            raise ValueError("task must be a TaskType")
        target_type = LandingTarget if self.task == TaskType.LAND else OrbitTarget
        if not isinstance(self.target, target_type):
            raise ValueError(f"{self.task.value} requires {target_type.__name__}")
        number(self.minimum_fuel_fraction, "minimum_fuel_fraction", 0, 0.15)
        number(self.urgency, "urgency", 0, 1)
        number(self.timeout_s, "timeout_s", 1, 86_400)
        number(self.settle_time_s, "settle_time_s", 0.1, self.timeout_s)

    @classmethod
    def from_dict(cls, data: Any) -> "MissionTask":
        fields(data, set(cls.__dataclass_fields__), {"task_id", "task", "target"}, "mission")
        try:
            kind = TaskType(data["task"])
        except (ValueError, TypeError) as exc:
            raise ValueError("task must be reach_orbit, change_orbit, or land") from exc
        target_type = LandingTarget if kind == TaskType.LAND else OrbitTarget
        required = ({"latitude_deg", "longitude_deg"} if kind == TaskType.LAND
                    else {"apoapsis_m", "periapsis_m"})
        fields(data["target"], set(target_type.__dataclass_fields__), required, "target")
        return cls(**{**data, "task": kind, "target": target_type(**data["target"])})

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["task"] = self.task.value
        return result

    def launch_options(self) -> dict[str, float]:
        """Translate a launch task to the existing environment's command API."""
        if self.task != TaskType.REACH_ORBIT:
            raise ValueError("Only reach_orbit tasks have launch options")
        assert isinstance(self.target, OrbitTarget)
        return {
            "target_ap": self.target.apoapsis_m,
            "target_pe": self.target.periapsis_m,
            "ap_tolerance": self.target.apoapsis_tolerance_m,
            "pe_tolerance": self.target.periapsis_tolerance_m,
            "min_fuel_fraction": self.minimum_fuel_fraction,
            "urgency": self.urgency,
        }
