# Mission and controller contract

## First profile

Use stock Kerbin and one controllable rocket. The launch environment targets an
eastward orbit with apoapsis between 80 and 180 km and periapsis at least 70 km.
The mission supervisor requires both the vessel altitude and periapsis to be
strictly above 70 km before declaring an orbital goal reached. A target at the
boundary does not relax that physical check.

The generic supervisor's initial landing task begins below 10 km above terrain.
The separate return coordinator now runs an explicit deorbit task and a
telemetry-gated handoff to descent PPO v2, with a configurable entry profile up
to 30 km in the example. It does not use the generic 10 km landing adapter.
Landing coordinates in examples are illustrative, not surveyed safe sites.

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

Only `Actuator.apply()` and `Actuator.stop()` own physical control. A controller's
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

For landing, require an intact vessel in `landed` state, within the target
radius, within touchdown and current speed limits, and within the tilt limit.
The initial distance calculation uses a 600 km spherical Kerbin radius. Splashdown
is not a successful landing in this first profile.

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

`scripts.return_to_ground` uses the orbital actuator/supervisor for deorbit,
then owns passive coast attitude at zero throttle. It explicitly calls `stop()`
before `DescentEnv.begin_current()` takes control of the same vessel. It never
calls training reset during a live handoff. The descent Gym environment currently
acts as the PPO inference bridge and owns its own kRPC writes; it is not a generic
`Controller` implementation. These phases run sequentially, never concurrently.
See [return and descent](return-and-descent.md) for the model contract and limits.

The launch policy's current observation has 16 values. Changes to its shape,
ordering, normalization, action meaning, or scripted guard behavior require a
new version and fresh evaluation; matching tensor dimensions alone do not prove
checkpoint compatibility.
