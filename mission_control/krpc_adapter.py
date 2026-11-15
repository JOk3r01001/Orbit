"""kRPC 0.6 orbital telemetry and actuator. No connection on module import."""

import math
import socket
import time

from .commands import number
from .orbits import OrbitalTelemetry
from .supervisor import ControlCommand, FlightState


def _install_rpc_deadline(connection, timeout_s):
    """Bound reads, including the 0.6 client's polling protobuf-size loop.

    This small compatibility boundary uses the pinned client's RPC connection.
    kRPC 0.6 has no public connect/response timeout argument. Its receive method
    has no read-ahead buffer, so exact socket reads preserve message framing.
    """
    from krpc.decoder import Decoder

    rpc_socket = connection._socket

    def read_exact(count, deadline):
        chunks = []
        remaining = count
        while remaining:
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise TimeoutError("kRPC response deadline exceeded")
            rpc_socket.settimeout(budget)
            block = rpc_socket.recv(min(4096, remaining))
            if not block:
                raise ConnectionError("kRPC connection closed")
            chunks.append(block)
            remaining -= len(block)
        return b"".join(chunks)

    def receive_message(message_type):
        deadline = time.monotonic() + timeout_s
        prefix = bytearray()
        try:
            for _ in range(10):
                prefix.extend(read_exact(1, deadline))
                try:
                    length = Decoder.decode_message_size(bytes(prefix))
                except IndexError:
                    continue
                if length > 8 * 1024 * 1024:
                    raise ValueError("Unexpectedly large kRPC response")
                return Decoder.decode_message(read_exact(length, deadline), message_type)
            raise ValueError("Invalid kRPC response size prefix")
        finally:
            try:
                rpc_socket.settimeout(timeout_s)
            except OSError:
                pass

    rpc_socket.settimeout(timeout_s)
    connection.receive_message = receive_message


def connect_orbital_client(timeout_s=5.0):
    """Open a read-only initial connection; controls are only written by apply/stop."""
    import krpc

    number(timeout_s, "timeout_s", 0.1, 30)
    previous_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout_s)
    try:
        client = krpc.connect(name="Orbit_maneuver_controller", stream_port=None)
    finally:
        socket.setdefaulttimeout(previous_timeout)
    try:
        _install_rpc_deadline(client._rpc_connection, timeout_s)
    except BaseException:
        client.close()
        raise
    return client


class KRPCOrbitalAdapter:
    """Bound to one remote vessel object for the lifetime of a mission.

    Constructing/reading/closing is read-only. apply() and stop() write controls.
    No save loading, engine activation, staging, or time warp is performed.
    """

    def __init__(self, connection):
        self.connection = connection
        self.space_center = connection.space_center
        self.vessel = self.space_center.active_vessel
        self.body = self.vessel.orbit.body
        if self.body.name != "Kerbin":
            raise ValueError("The orbital adapter currently supports Kerbin only")
        # kRPC 0.6 Vessel has no public GUID property. Its Python hash/equality
        # identify the remote object within this connection; never use its name.
        self.vessel_id = f"krpc:{hash(self.vessel)}"
        self.initial_fuel = float(self.vessel.resources.amount("LiquidFuel"))
        if self.initial_fuel <= 0:
            raise ValueError("The first orbital vehicle profile requires LiquidFuel")

    def _assert_bound(self):
        if self.space_center.active_vessel != self.vessel:
            raise RuntimeError("Active vessel changed; refusing commands to another vessel")
        if self.vessel.orbit.body != self.body:
            raise RuntimeError("Vessel changed sphere of influence")
        if self.space_center.rails_warp_factor or self.space_center.physics_warp_factor:
            raise RuntimeError("Set time warp to normal speed before orbital control")

    def read(self):
        self._assert_bound()
        orbit = self.vessel.orbit
        frame = self.body.non_rotating_reference_frame
        orbital_flight = self.vessel.flight(frame)
        surface_flight = self.vessel.flight(self.body.reference_frame)
        attitude_flight = self.vessel.flight(self.vessel.surface_reference_frame)
        velocity = self.vessel.velocity(frame)
        direction = self.vessel.direction(frame)
        speed = math.sqrt(sum(value * value for value in velocity))
        norm = math.sqrt(sum(value * value for value in direction))
        if speed <= 0 or norm <= 0:
            raise ValueError("Cannot determine orbital direction")
        alignment = sum(a * b for a, b in zip(direction, velocity)) / (norm * speed)
        orbital = OrbitalTelemetry(
            body_radius_m=float(self.body.equatorial_radius),
            gravitational_parameter=float(self.body.gravitational_parameter),
            time_to_apoapsis_s=float(orbit.time_to_apoapsis),
            time_to_periapsis_s=float(orbit.time_to_periapsis),
            mass_kg=float(self.vessel.mass),
            available_thrust_n=float(self.vessel.available_thrust),
            specific_impulse_s=float(self.vessel.specific_impulse),
            prograde_alignment=max(-1, min(1, alignment)),
        )
        values = dict(
            vessel_id=self.vessel_id, body=self.body.name, situation=self.vessel.situation.name,
            altitude_m=float(orbital_flight.mean_altitude),
            altitude_agl_m=float(surface_flight.surface_altitude),
            apoapsis_m=float(orbit.apoapsis_altitude), periapsis_m=float(orbit.periapsis_altitude),
            vertical_speed_mps=float(surface_flight.vertical_speed),
            horizontal_speed_mps=float(surface_flight.horizontal_speed),
            tilt_deg=90 - float(attitude_flight.pitch),
            fuel_fraction=min(1, max(0, float(self.vessel.resources.amount("LiquidFuel")) / self.initial_fuel)),
            latitude_deg=float(surface_flight.latitude), longitude_deg=float(surface_flight.longitude),
            orbital=orbital,
        )
        self._assert_bound()
        return FlightState(ut_s=float(self.space_center.ut), **values)

    def apply(self, command: ControlCommand):
        self._assert_bound()
        if command.stage:
            raise ValueError("The orbital actuator never stages")
        if command.attitude is not None:
            frame = self.body.non_rotating_reference_frame
            velocity = self.vessel.velocity(frame)
            norm = math.sqrt(sum(value * value for value in velocity))
            if norm <= 0:
                raise ValueError("Cannot orient without orbital velocity")
            sign = 1 if command.attitude == "orbital_prograde" else -1
            autopilot = self.vessel.auto_pilot
            autopilot.reference_frame = frame
            autopilot.target_direction = tuple(sign * value / norm for value in velocity)
            autopilot.engaged = True
        elif command.pitch_deg is not None:
            self.vessel.auto_pilot.reference_frame = self.vessel.surface_reference_frame
            self.vessel.auto_pilot.target_pitch_and_heading(command.pitch_deg, command.heading_deg)
            self.vessel.auto_pilot.engaged = True
        self.vessel.control.throttle = command.throttle

    def stop(self):
        # Stop the bound vessel even if the user switched the active vessel.
        errors = []
        for action in (
            lambda: setattr(self.vessel.control, "throttle", 0.0),
            lambda: setattr(self.vessel.auto_pilot, "engaged", False),
        ):
            try:
                action()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    def close(self):
        self.connection.close()
