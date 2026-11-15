"""Apsis maneuver state machine with bounded PID throttle control."""

from .commands import MissionTask, OrbitTarget, TaskType, number
from .orbits import OrbitElements, apsis_delta_v, burn_duration, plan_orbit_change
from .supervisor import ControlCommand, FlightState


class PID:
    def __init__(self, kp=0.15, ki=0.01, kd=0.02, integral_limit=10.0):
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd), ("integral_limit", integral_limit)):
            number(value, name, 0, 1e6)
        self.kp, self.ki, self.kd, self.integral_limit = kp, ki, kd, integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, error, dt, output_max=1.0):
        number(error, "PID error", -1e9, 1e9)
        number(dt, "PID dt", 1e-6, 60)
        number(output_max, "output_max", 0, 1)
        integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error * dt))
        derivative = 0 if self.previous_error is None else (error - self.previous_error) / dt
        raw = self.kp * error + self.ki * integral + self.kd * derivative
        output = max(0.0, min(output_max, raw))
        if raw == output or (raw > output_max and error < 0) or (raw < 0 and error > 0):
            self.integral = integral
        self.previous_error = error
        return output


def elements_from_state(state: FlightState):
    if state.orbital is None:
        raise ValueError("Orbital telemetry is required")
    return OrbitElements(
        state.orbital.body_radius_m, state.orbital.gravitational_parameter,
        state.apoapsis_m, state.periapsis_m,
    )


class OrbitalManeuverController:
    """Compute commands only; the supervisor/actuator owns the vessel."""

    def __init__(self, minimum_alignment=0.98, max_burn_fraction=0.05, max_burns=6):
        number(minimum_alignment, "minimum_alignment", 0.9, 1)
        number(max_burn_fraction, "max_burn_fraction", 0.001, 0.1)
        if type(max_burns) is not int or max_burns < 1:
            raise ValueError("max_burns must be positive")
        self.minimum_alignment = minimum_alignment
        self.max_burn_fraction = max_burn_fraction
        self.max_burns = max_burns
        self.pid = PID()
        self.phase = "idle"
        self.burn = None
        self.burns_started = 0
        self.center_ut = None

    def start(self, task: MissionTask, state: FlightState):
        if task.task != TaskType.CHANGE_ORBIT or not isinstance(task.target, OrbitTarget):
            raise ValueError("This controller only accepts change_orbit tasks")
        if state.body != "Kerbin" or state.situation != "orbiting":
            raise ValueError("Start in a Kerbin orbit")
        elements = elements_from_state(state)
        burns = plan_orbit_change(elements, task.target)
        self.task = task
        self.phase = "planning"
        self.burn = None
        self.burns_started = 0
        self.center_ut = None
        self._last_ut = state.ut_s
        self._mission_started_ut = state.ut_s
        self.pid.reset()
        if burns:
            self._validate_duration(state, burns[0])

    def _validate_duration(self, state, burn):
        orbital = state.orbital
        duration = burn_duration(
            burn.delta_v_mps, orbital.mass_kg, orbital.available_thrust_n, orbital.specific_impulse_s,
        )
        if duration > elements_from_state(state).period_s * self.max_burn_fraction:
            raise ValueError("Burn is too long for this apsis-burn controller; use more thrust or a smaller maneuver")
        return duration

    def update(self, state: FlightState) -> ControlCommand:
        elements = elements_from_state(state)
        if state.periapsis_m <= 70_000 or state.altitude_m <= 70_000:
            raise RuntimeError("Maneuver left the safe orbital envelope")
        dt = state.ut_s - self._last_ut
        if dt <= 0:
            raise RuntimeError("Orbital telemetry time did not advance")
        # The supervisor can temporarily hold engines off while settling. If
        # it returns control later, discard derivative/integral history.
        if dt > 2:
            self.pid.reset()
        dt = min(dt, 2.0)
        self._last_ut = state.ut_s
        orbital = state.orbital
        if self.burn is None:
            plan = plan_orbit_change(elements, self.task.target)
            if not plan:
                self.phase = "settling"
                return ControlCommand()
            if self.burns_started >= self.max_burns:
                raise RuntimeError("Maneuver correction limit reached")
            self.burn = plan[0]
            self.duration_s = self._validate_duration(state, self.burn)
            wait = getattr(orbital, f"time_to_{self.burn.at_apsis}_s")
            # Defer a too-close event rather than beginning a late burn.
            if wait < self.duration_s / 2 + 10:
                wait += elements.period_s
            self.center_ut = state.ut_s + wait
            if self.center_ut + self.duration_s / 2 >= self.task.timeout_s + self._mission_started_ut:
                raise RuntimeError("Next burn window exceeds the mission timeout")
            self.phase = "coast"
            self.pid.reset()

        attitude = self.burn.attitude
        direction = 1 if self.burn.delta_v_mps >= 0 else -1
        aligned = orbital.prograde_alignment * direction >= self.minimum_alignment
        if self.phase == "coast":
            start_ut = self.center_ut - self.duration_s / 2
            if state.ut_s > self.center_ut + max(5.0, self.duration_s / 2):
                raise RuntimeError("Missed aligned burn window; engines remain off")
            if state.ut_s < start_ut or not aligned:
                return ControlCommand(attitude=attitude)
            self.phase = "burn"
            self.burns_started += 1
            self._burn_started_ut = state.ut_s
            self.pid.reset()

        error = self.burn.target_altitude_m - getattr(state, f"{self.burn.changed_apsis}_m")
        tolerance = self.burn.tolerance_m
        if abs(error) <= tolerance:
            self.burn = None
            self.phase = "planning"
            self.pid.reset()
            return ControlCommand(attitude=attitude)
        allowed_duration = min(max(30.0, self.duration_s * 2 + 10), elements.period_s * self.max_burn_fraction)
        if state.ut_s - self._burn_started_ut > allowed_duration:
            raise RuntimeError("Burn exceeded its estimated duration; engines stopped")
        remaining = apsis_delta_v(elements, self.burn.changed_apsis, self.burn.target_altitude_m)
        if remaining * direction <= 0:
            self.burn = None
            self.phase = "planning"
            return ControlCommand(attitude=attitude)
        if not aligned:
            return ControlCommand(attitude=attitude)
        if orbital.available_thrust_n <= 0 or orbital.specific_impulse_s <= 0:
            raise RuntimeError("No usable active engine remains")
        acceleration = orbital.available_thrust_n / orbital.mass_kg
        # Never request enough impulse to spend more than half the estimated
        # remaining delta-v in one observed control interval.
        cap = min(1.0, abs(remaining) / (2 * acceleration * max(dt, 0.1)))
        throttle = self.pid.update(abs(remaining), dt, cap)
        return ControlCommand(throttle=throttle, attitude=attitude)

    def report(self):
        return {
            "phase": self.phase, "burns_started": self.burns_started,
            "changed_apsis": self.burn.changed_apsis if self.burn else None,
            "burn_center_ut": getattr(self, "center_ut", None),
        }
