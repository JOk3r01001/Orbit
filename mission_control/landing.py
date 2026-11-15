"""Pure telemetry, reward and touchdown checks for the first landing curriculum."""

from dataclasses import dataclass
import math

from .commands import LandingTarget, number


@dataclass(frozen=True)
class LandingSample:
    ut_s: float
    height_m: float
    vertical_mps: float
    horizontal_mps: float
    tilt_deg: float
    fuel_fraction: float
    acceleration_mps2: float
    gravity_mps2: float
    latitude_deg: float
    longitude_deg: float
    situation: str
    intact: bool = True

    def __post_init__(self):
        for key, low, high in (
            ("ut_s", 0, 1e15), ("height_m", -100, 1e7),
            ("vertical_mps", -1e5, 1e5), ("horizontal_mps", 0, 1e5),
            ("tilt_deg", 0, 180), ("fuel_fraction", 0, 1),
            ("acceleration_mps2", 0, 1e5), ("gravity_mps2", .01, 1000),
            ("latitude_deg", -90, 90), ("longitude_deg", -180, 180),
        ):
            number(getattr(self, key), key, low, high)
        if type(self.intact) is not bool or not isinstance(self.situation, str):
            raise ValueError("Invalid vessel state")


def distance_to_target(sample, target):
    lat1, lat2 = map(math.radians, (sample.latitude_deg, target.latitude_deg))
    dlat = lat2 - lat1
    dlon = math.radians(target.longitude_deg - sample.longitude_deg)
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 1_200_000 * math.asin(math.sqrt(min(1, max(0, hav))))


def descent_reference(sample, target):
    """Reward guidance only: never replaces the policy's throttle action."""
    braking = max(.1, .4 * (sample.acceleration_mps2 - sample.gravity_mps2))
    return -min(30, math.sqrt(target.max_vertical_speed_mps ** 2 + 2 * braking * max(0, sample.height_m - 5)))


def potential(sample, target):
    return (max(0, sample.height_m) / 500
            + abs(sample.vertical_mps - descent_reference(sample, target)) / 30
            + sample.horizontal_mps / 5 + sample.tilt_deg / 90)


def transition_reward(previous, current, target):
    dt = current.ut_s - previous.ut_s
    number(dt, "telemetry interval", 1e-6, 2)
    progress = 5 * (potential(previous, target) - potential(current, target))
    fuel = 5 * max(0, previous.fuel_fraction - current.fuel_fraction)
    time = .02 * dt
    return progress - fuel - time, {"progress_reward": progress, "fuel_penalty": fuel, "time_penalty": time}


def touchdown_failure(previous, current, target):
    """Use pre-contact speed too: landed telemetry alone hides impact velocity.

    This sampled bound is deliberately conservative; it cannot measure a peak
    impact between RPC samples or certify structural loads.
    """
    if not current.intact:
        return "vessel_damaged"
    if current.situation != "landed":
        return "not_landed"
    if max(abs(previous.vertical_mps), abs(current.vertical_mps)) > target.max_vertical_speed_mps:
        return "hard_touchdown"
    if max(previous.horizontal_mps, current.horizontal_mps) > target.max_horizontal_speed_mps:
        return "horizontal_touchdown_speed"
    if max(previous.tilt_deg, current.tilt_deg) > target.max_tilt_deg:
        return "touchdown_tilt"
    if distance_to_target(current, target) > target.radius_m:
        return "outside_landing_site"
    return None
