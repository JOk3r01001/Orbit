"""Mission and telemetry types for stock KSP parachute recovery."""

from dataclasses import asdict, dataclass

from .commands import MissionTask, TaskType, fields, number


@dataclass(frozen=True)
class ParachuteRecovery:
    deploy_above_m: float = 100
    deploy_below_m: float = 10_000
    max_touchdown_vertical_mps: float = 8
    max_touchdown_horizontal_mps: float = 5
    settle_time_s: float = 3
    timeout_s: float = 7_200
    allow_splashdown: bool = True
    entry_attitude: str = "surface_retrograde"

    def __post_init__(self):
        number(self.deploy_above_m, "deploy_above_m", 10, 10_000)
        number(self.deploy_below_m, "deploy_below_m", self.deploy_above_m + 1, 60_000)
        number(self.max_touchdown_vertical_mps, "max_touchdown_vertical_mps", .1, 20)
        number(self.max_touchdown_horizontal_mps, "max_touchdown_horizontal_mps", .1, 20)
        number(self.timeout_s, "timeout_s", 1, 86_400)
        number(self.settle_time_s, "settle_time_s", .1, self.timeout_s)
        if type(self.allow_splashdown) is not bool:
            raise ValueError("allow_splashdown must be a boolean")
        if self.entry_attitude not in ("surface_retrograde", "passive"):
            raise ValueError("entry_attitude must be surface_retrograde or passive")


@dataclass(frozen=True)
class ParachuteReturnMission:
    deorbit: MissionTask
    recovery: ParachuteRecovery = ParachuteRecovery()

    def __post_init__(self):
        if self.deorbit.task != TaskType.DEORBIT:
            raise ValueError("return.deorbit must be a deorbit task")

    @classmethod
    def from_dict(cls, payload):
        fields(payload, {"deorbit", "recovery"}, {"deorbit"}, "parachute_return")
        config = payload.get("recovery", {})
        fields(config, set(ParachuteRecovery.__dataclass_fields__), set(), "recovery")
        return cls(MissionTask.from_dict(payload["deorbit"]), ParachuteRecovery(**config))

    def to_dict(self):
        return {"deorbit": self.deorbit.to_dict(), "recovery": asdict(self.recovery)}


@dataclass(frozen=True)
class RecoverySample:
    ut_s: float
    situation: str
    height_m: float
    vertical_mps: float
    horizontal_mps: float
    latitude_deg: float
    longitude_deg: float
    intact: bool
    chute_states: tuple[str, ...]

    def __post_init__(self):
        number(self.ut_s, "ut_s", 0, 1e15)
        number(self.height_m, "height_m", -1e7, 1e9)
        number(self.vertical_mps, "vertical_mps", -1e5, 1e5)
        number(self.horizontal_mps, "horizontal_mps", 0, 1e5)
        number(self.latitude_deg, "latitude_deg", -90, 90)
        number(self.longitude_deg, "longitude_deg", -180, 180)

    def in_deploy_window(self, config):
        return (self.situation in {"flying", "sub_orbital"} and self.vertical_mps < 0
                and config.deploy_above_m <= self.height_m <= config.deploy_below_m)

    def speeds_within_limits(self, config):
        return (abs(self.vertical_mps) <= config.max_touchdown_vertical_mps
                and self.horizontal_mps <= config.max_touchdown_horizontal_mps)
