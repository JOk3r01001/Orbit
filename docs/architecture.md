# Mission and controller contract

## First profile

Use stock Kerbin and one controllable rocket. The launch environment targets an
eastward orbit with apoapsis between 80 and 180 km and periapsis at least 70 km.
The mission supervisor requires both the vessel altitude and periapsis to be
strictly above 70 km before declaring an orbital goal reached. A target at the
boundary does not relax that physical check.

The initial landing task begins in a prepared descending state below 10 km above
terrain. It does not initiate a return from orbit. Deorbit and atmospheric-return
preparation must be added before a complete orbital landing mission can run.
Landing coordinates in the example are illustrative, not a surveyed safe site.

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
permitted only for the launch task. The eventual actuator must verify that its
bound vessel is still the intended vessel, rather than applying commands to
whatever vessel happens to be active.

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

The future kRPC adapter must provide a consistent sample and correct reference
frames. Orbital velocity is measured in a body-centered non-rotating frame;
surface landing velocity is measured relative to the rotating body. The adapter
must record touchdown speeds at contact and retain them during settling; zero
velocity after a hard impact is not evidence of a safe landing. Missing contact
speed data prevents landing success.

Mission timeouts use simulation time. The current supervisor consumes samples
supplied by a caller; it has no polling loop, maximum sample-gap check, or
wall-clock connection watchdog. Those belong in the live runner. Stable-goal
checks assume regular telemetry; offline tests do not establish stability
between observations. The legacy training scripts still use wall-clock sleeps,
so physics-time stepping remains an integration milestone.

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
Persisted telemetry, metrics, and mission event logs will be added with the live
runner.

## Integration boundary

The existing launch Gym environment still performs training resets, calculates
rewards, stages, and writes to kRPC itself. Its `set_command()` method is useful
for updating a goal without a save reload, but it is not a supervisor controller
implementation. Extract policy observation/action conversion and scripted guards
into a live controller, then add a single kRPC actuator and telemetry adapter.
Never give the legacy environment and supervisor concurrent control.

The launch policy's current observation has 16 values. Changes to its shape,
ordering, normalization, action meaning, or scripted guard behavior require a
new version and fresh evaluation; matching tensor dimensions alone do not prove
checkpoint compatibility.
