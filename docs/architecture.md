# Mission and controller contract

## First profile

Use stock Kerbin and one controllable rocket. The launch environment targets an
eastward orbit with apoapsis between 80 and 180 km and periapsis at least 70 km.
The mission supervisor requires both the vessel altitude and periapsis to be
strictly above 70 km before declaring an orbital goal reached. A target at the
boundary does not relax that physical check.

The v1 return coordinator runs an explicit deorbit task from stable orbit,
then uses surface telemetry and the stock parachute safety state for recovery.
It accepts a dry landing or splashdown under the configured limits. Powered
landing tasks and target coordinates remain part of the experimental interfaces.

## Mission lifecycle

The commander submits an immutable `MissionTask`. The supervisor rejects
unsupported controller types and inappropriate starting states before touching
vessel controls. An active task must be aborted before another is accepted.

```text
idle -> running -> succeeded
                -> failed
                -> aborted
```

After a terminal result, a new task can be submitted using the current vessel's
telemetry. The supervisor never reloads a save. Each accepted task binds to the
reported vessel identity and body; a change causes failure and a stop request.

The actuator owns flight commands and direct stock chute deployment. A controller's
`start(task, state)` initializes its internal state; `update(state)` returns a
validated `ControlCommand`. Controllers must not call kRPC directly. Staging is
permitted only for the launch task. The orbital actuator checks its bound remote
vessel object before commands and refuses commands after an active-vessel or
sphere-of-influence change. The first orbital profile never stages.

Controller exceptions, actuator errors, reserve violations, mission timeout,
vessel destruction, and non-advancing simulation timestamps end the task. Every
terminal path requests a stop. If stopping fails, the report records failure;
the core cannot guarantee hardware state after a lost kRPC connection.

## Telemetry and timing

`FlightState` carries:

- KSP universal time and stable vessel identity.
- Body and situation (`pre_launch`, `landed`, `flying`, `sub_orbital`, `orbiting`).
- Altitude above sea level and altitude above terrain, separately.
- Apoapsis/periapsis altitudes above sea level.
- Surface-relative vertical/horizontal speeds and tilt from local upright.
- Fuel fraction relative to the task's defined starting load.
- Latitude, longitude, and vessel integrity.
- Recorded vertical/horizontal speeds at touchdown, when known.
- Optional `OrbitalTelemetry`: body radius/gravity parameter, times to apsides,
  current mass, available thrust, specific impulse, and prograde alignment.

The orbital kRPC adapter samples the bound vessel using the correct reference
frames. Orbital velocity and alignment use a body-centered non-rotating frame;
surface velocities use the rotating body frame, and tilt uses a local surface
frame. Samples are sequential RPC reads, not atomic physics-tick snapshots.
The landing environments retain pre-contact sampled speeds and contact data
during settling; zero velocity after a hard impact is not evidence of a safe
landing. RPC samples cannot measure impact peaks between reads.

Mission timeouts use simulation time. The supervisor consumes samples supplied
by a caller. The orbital runner rejects gaps greater than two simulation seconds
and fails after three wall-clock seconds without advancing simulation time.
After connection setup, RPC response reads have deadlines, including the kRPC
0.6 size-prefix polling path. Initial service discovery still uses the SDK's
handshake. If the server is unreachable, a cutoff request cannot be guaranteed.
Stable-goal checks assume regular telemetry. The legacy launch training script
still uses wall-clock sleeps; its physics-time stepping remains future work.

## Completion and handoffs

For an orbit task, both orbital errors must be within the requested tolerances,
the orbit must clear the atmosphere, and the fuel reserve must be satisfied.

For v1 return, require an intact craft, a fully open parachute observed before
contact, and settled `landed` or permitted `splashed` status within the speed
limits. The separately defined powered landing goal uses dry-ground contact,
target radius, speed and tilt limits.

When a candidate goal is observed, command zero throttle and no staging. The goal
must remain satisfied for `settle_time_s` of simulation time before success. If
it leaves tolerance, restart settling and return control to the current
controller. A successful handoff releases controller-owned inputs via `stop()`.

`report()` returns task identity, status, reason, and elapsed simulation time.
The orbital runner persists JSONL telemetry and a final JSON report under
`training_output/maneuvers/`. Its standalone fuel fraction is measured against
the load at connection time, not the initial launch load.

## Integration boundary

The existing launch Gym environment still performs training resets, calculates
rewards, stages, and writes to kRPC itself. Its `set_command()` method is useful
for updating a goal without a save reload, but it is not a supervisor controller
implementation. Extract policy observation/action conversion and scripted guards
into a live controller, then add a single kRPC actuator and telemetry adapter.
Never give the legacy environment and supervisor concurrent control. The new
orbital controller already uses the supervisor and a single kRPC actuator;
automatic launch-to-orbit-controller handoff remains to be implemented.

`scripts.return_to_ground` uses the orbital supervisor for the deorbit burn.
Once that task releases the controls, the runner monitors entry, requests safe
chute deployment on the same vessel, and records touchdown. The engine stays
at zero throttle after the deorbit burn. See [parachute return](parachute-return.md).
The optional powered-descent path uses the same deorbit phase and hands control
to `DescentEnv.begin_current()` without a training reset. See the
[experimental model contract](return-and-descent.md).

The launch policy's current observation has 16 values. Changes to its shape,
ordering, normalization, action meaning, or scripted guard behavior require a
new version and fresh evaluation; matching tensor dimensions alone do not prove
checkpoint compatibility.
