"""Shared descent-policy contract and telemetry handoff checks; no flight I/O."""

from dataclasses import asdict, dataclass
import math

from .commands import MissionTask, TaskType, fields, number
from .landing import LandingSample


@dataclass(frozen=True)
class DescentSample(LandingSample):
    north_mps: float = 0
    east_mps: float = 0
    nose_up: float = 1
    nose_north: float = 0
    nose_east: float = 0
    apoapsis_m: float = 0
    periapsis_m: float = 0
    tank_fill_fraction: float = 1

    def __post_init__(self):
        super().__post_init__()
        for key in ("north_mps", "east_mps", "apoapsis_m", "periapsis_m"):
            number(getattr(self, key), key, -1e12, 1e12)
        for key in ("nose_up", "nose_north", "nose_east"):
            number(getattr(self, key), key, -1, 1)
        number(self.tank_fill_fraction, "tank_fill_fraction", 0, 1)


@dataclass(frozen=True)
class DescentProfile:
    min_height_m: float = 100
    max_height_m: float = 30_000
    max_downward_mps: float = 1_200
    max_horizontal_mps: float = 2_000
    min_thrust_weight: float = 1.3
    reaction_time_s: float = 3
    braking_safety_factor: float = 1.5
    clearance_m: float = 50
    timeout_s: float = 900

    def __post_init__(self):
        for key, low, high in (
            ("min_height_m", 50, 50_000), ("max_height_m", self.min_height_m, 60_000),
            ("max_downward_mps", 10, 3_000), ("max_horizontal_mps", 1, 3_000),
            ("min_thrust_weight", 1.1, 10), ("reaction_time_s", 0, 30),
            ("braking_safety_factor", 1.1, 5), ("clearance_m", 10, 500), ("timeout_s", 30, 3600),
        ):
            number(getattr(self, key), key, low, high)

    @classmethod
    def from_dict(cls, data):
        fields(data, set(cls.__dataclass_fields__), set(), "descent_profile")
        return cls(**data)


def braking_metrics(s, profile):
    """Conservative constant-acceleration screening, not an entry predictor.

    Charge all surface-relative speed against altitude, ignore helpful drag,
    allow reaction delay and reserve a safety factor. Thrust is fully available
    and attitude can recover within that delay; neither is guaranteed in KSP.
    """
    speed = math.hypot(s.vertical_mps, s.horizontal_mps)
    net = s.acceleration_mps2 - s.gravity_mps2
    distance = speed * speed / (2 * net) if net > 0 else math.inf
    required = (distance * profile.braking_safety_factor
                + max(0, -s.vertical_mps) * profile.reaction_time_s + profile.clearance_m)
    return {"estimated_stopping_distance_m": distance, "required_height_m": required,
            "height_margin_m": s.height_m - required,
            "ballistic_time_to_ground_s": s.height_m / max(.1, -s.vertical_mps)}


def entry_rejection(s, profile):
    if not s.intact or s.situation not in {"flying", "sub_orbital"}:
        return "Expected an intact vessel in flight"
    if not profile.min_height_m <= s.height_m <= profile.max_height_m:
        return "Height outside descent policy entry profile"
    if not -profile.max_downward_mps <= s.vertical_mps < 0:
        return "Vertical speed outside descent policy entry profile"
    if s.horizontal_mps > profile.max_horizontal_mps:
        return "Horizontal speed outside descent policy entry profile"
    if s.acceleration_mps2 < profile.min_thrust_weight * s.gravity_mps2:
        return "Insufficient thrust/weight for descent entry"
    if braking_metrics(s, profile)["height_margin_m"] < 0:
        return "Insufficient estimated braking height for descent entry"
    return None


@dataclass(frozen=True)
class ReturnMission:
    deorbit: MissionTask
    landing: MissionTask | None = None
    profile: DescentProfile = DescentProfile()
    handoff_height_m: float = 30_000
    minimum_handoff_height_m: float = 1_000
    coast_timeout_s: float = 7200

    def __post_init__(self):
        if self.deorbit.task != TaskType.DEORBIT:
            raise ValueError("return.deorbit must be a deorbit task")
        if self.landing is not None and self.landing.task != TaskType.LAND:
            raise ValueError("return.landing must be a land task")
        number(self.handoff_height_m, "handoff_height_m", self.profile.min_height_m, self.profile.max_height_m)
        number(self.minimum_handoff_height_m, "minimum_handoff_height_m", self.profile.min_height_m, self.handoff_height_m)
        number(self.coast_timeout_s, "coast_timeout_s", 1, 86_400)

    @classmethod
    def from_dict(cls, data):
        fields(data, set(cls.__dataclass_fields__), {"deorbit"}, "return_mission")
        values = dict(data)
        values["deorbit"] = MissionTask.from_dict(values["deorbit"])
        if values.get("landing") is not None:
            values["landing"] = MissionTask.from_dict(values["landing"])
        values["profile"] = DescentProfile.from_dict(values.get("profile", {}))
        return cls(**values)

    def to_dict(self):
        result = asdict(self)
        result["deorbit"] = self.deorbit.to_dict()
        result["landing"] = self.landing.to_dict() if self.landing else None
        return result

    def handoff(self, sample):
        """Return wait/ready, or reject a missed/unsafe window explicitly."""
        if not sample.intact or sample.situation in {"landed", "splashed"}:
            raise ValueError("Return ended before landing handoff")
        if sample.height_m < self.minimum_handoff_height_m:
            raise ValueError("Missed landing handoff: below minimum handoff height")
        if sample.vertical_mps >= 0 or sample.height_m > self.handoff_height_m:
            return "wait"
        reason = entry_rejection(sample, self.profile)
        if reason:
            raise ValueError(reason)
        return "ready"
