"""Two-body, tangential apsis-change planning. No simulator dependencies."""

from dataclasses import dataclass
import math

from .commands import OrbitTarget, number

STANDARD_GRAVITY = 9.80665
MINIMUM_MANEUVER_PERIAPSIS_M = 71_000.0


@dataclass(frozen=True)
class OrbitElements:
    body_radius_m: float
    gravitational_parameter: float
    apoapsis_m: float
    periapsis_m: float

    def __post_init__(self):
        number(self.body_radius_m, "body_radius_m", 1, 1e9)
        number(self.gravitational_parameter, "gravitational_parameter", 1, 1e25)
        number(self.apoapsis_m, "apoapsis_m", 0, 1e12)
        number(self.periapsis_m, "periapsis_m", 0, self.apoapsis_m)

    @property
    def semi_major_axis_m(self):
        return self.body_radius_m + (self.apoapsis_m + self.periapsis_m) / 2

    @property
    def period_s(self):
        return 2 * math.pi * math.sqrt(self.semi_major_axis_m ** 3 / self.gravitational_parameter)

    def speed_at(self, altitude_m):
        number(altitude_m, "altitude_m", 0, 1e12)
        radius = self.body_radius_m + altitude_m
        squared_speed = self.gravitational_parameter * (2 / radius - 1 / self.semi_major_axis_m)
        if squared_speed <= 0:
            raise ValueError("Radius is outside the bound trajectory")
        return math.sqrt(squared_speed)


@dataclass(frozen=True)
class OrbitalTelemetry:
    """Extra telemetry required by the orbital controller, in SI units."""

    body_radius_m: float
    gravitational_parameter: float
    time_to_apoapsis_s: float
    time_to_periapsis_s: float
    mass_kg: float
    available_thrust_n: float
    specific_impulse_s: float
    prograde_alignment: float

    def __post_init__(self):
        for name, low, high in (
            ("body_radius_m", 1, 1e9), ("gravitational_parameter", 1, 1e25),
            ("time_to_apoapsis_s", 0, 1e9), ("time_to_periapsis_s", 0, 1e9),
            ("mass_kg", 0.001, 1e12), ("available_thrust_n", 0, 1e15),
            ("specific_impulse_s", 0, 1e6), ("prograde_alignment", -1, 1),
        ):
            number(getattr(self, name), name, low, high)


@dataclass(frozen=True)
class ApsisBurn:
    at_apsis: str
    changed_apsis: str
    target_altitude_m: float
    delta_v_mps: float  # Signed: positive prograde, negative retrograde.
    tolerance_m: float

    @property
    def attitude(self):
        return "orbital_prograde" if self.delta_v_mps >= 0 else "orbital_retrograde"


def apsis_delta_v(elements: OrbitElements, changed_apsis: str, target_altitude_m: float):
    """Tangential impulse needed at the opposite apsis, without swapping axes."""
    number(target_altitude_m, "target_altitude_m", 0, 1e12)
    if changed_apsis == "apoapsis":
        fixed = elements.periapsis_m
        if target_altitude_m < fixed:
            raise ValueError("Lower periapsis first before lowering apoapsis below it")
    elif changed_apsis == "periapsis":
        fixed = elements.apoapsis_m
        if target_altitude_m > fixed:
            raise ValueError("Raise apoapsis first before raising periapsis above it")
    else:
        raise ValueError("changed_apsis must be apoapsis or periapsis")
    target = OrbitElements(
        elements.body_radius_m, elements.gravitational_parameter,
        max(fixed, target_altitude_m), min(fixed, target_altitude_m),
    )
    return target.speed_at(fixed) - elements.speed_at(fixed)


def plan_orbit_change(elements: OrbitElements, target: OrbitTarget) -> list[ApsisBurn]:
    """Plan up to two impulses for in-plane orbital size/shape changes.

    Lowering both apsides below the old periapsis requires changing periapsis
    first; otherwise apoapsis is changed first. No inclination/transfer planning.
    """
    if elements.periapsis_m <= 70_000:
        raise ValueError("Start in a stable Kerbin orbit above the atmosphere")
    if target.periapsis_m < MINIMUM_MANEUVER_PERIAPSIS_M:
        raise ValueError("Orbital maneuvers require target periapsis of at least 71 km")
    if (
        abs(elements.apoapsis_m - target.apoapsis_m) <= target.apoapsis_tolerance_m
        and abs(elements.periapsis_m - target.periapsis_m) <= target.periapsis_tolerance_m
    ):
        return []
    order = ("periapsis", "apoapsis") if target.apoapsis_m < elements.periapsis_m else ("apoapsis", "periapsis")
    current = elements
    burns = []
    for changed in order:
        nominal = getattr(target, f"{changed}_m")
        desired = nominal
        actual = getattr(current, f"{changed}_m")
        tolerance = getattr(target, f"{changed}_tolerance_m")
        # Near-circular targets may be met within tolerance without crossing
        # the fixed apsis. Spend only the remaining tolerance on that burn.
        if changed == "periapsis" and desired > current.apoapsis_m:
            desired = current.apoapsis_m
        elif changed == "apoapsis" and desired < current.periapsis_m:
            desired = current.periapsis_m
        tolerance -= abs(desired - nominal)
        if tolerance <= 0:
            raise ValueError("A preliminary apsis adjustment is required for the requested tolerances")
        # A preliminary geometry change can be necessary even inside tolerance.
        geometry_requires_change = (
            changed == "apoapsis" and target.periapsis_m >= current.apoapsis_m + target.periapsis_tolerance_m
        ) or (changed == "periapsis" and target.apoapsis_m <= current.periapsis_m - target.apoapsis_tolerance_m)
        if abs(desired - actual) <= tolerance and not geometry_requires_change:
            continue
        if changed == order[0]:
            other_tolerance = target.periapsis_tolerance_m if changed == "apoapsis" else target.apoapsis_tolerance_m
            tolerance = min(tolerance, (target.apoapsis_m - target.periapsis_m + other_tolerance) / 2)
        delta_v = apsis_delta_v(current, changed, desired)
        burns.append(ApsisBurn(
            "periapsis" if changed == "apoapsis" else "apoapsis", changed, desired, delta_v, tolerance,
        ))
        current = OrbitElements(
            current.body_radius_m, current.gravitational_parameter,
            desired if changed == "apoapsis" else current.apoapsis_m,
            desired if changed == "periapsis" else current.periapsis_m,
        )
    return burns


def burn_duration(delta_v_mps, mass_kg, thrust_n, specific_impulse_s):
    """Ideal constant-thrust burn time accounting for propellant mass loss."""
    number(delta_v_mps, "delta_v_mps", -1e6, 1e6)
    number(mass_kg, "mass_kg", 0.001, 1e12)
    number(thrust_n, "thrust_n", 0.001, 1e15)
    number(specific_impulse_s, "specific_impulse_s", 0.001, 1e6)
    exhaust_velocity = specific_impulse_s * STANDARD_GRAVITY
    burned_mass = mass_kg * -math.expm1(-abs(delta_v_mps) / exhaust_velocity)
    return burned_mass * exhaust_velocity / thrust_n
