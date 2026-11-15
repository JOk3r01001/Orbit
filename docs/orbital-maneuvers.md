# Phase 2: in-plane orbital maneuvers

The initial implementation changes orbital size and shape around stock Kerbin.
It uses deterministic two-body planning and PID throttle control, with KSP
autopilot controlling attitude. It requires no PPO training and does not modify
the released launch checkpoint.

## Components

- `mission_control/orbits.py`: vis-viva calculations, apsis-burn ordering, and
  ideal burn-duration estimates including propellant mass loss.
- `mission_control/orbital_controller.py`: PID with anti-windup and a
  plan/coast/burn state machine. This computes commands without calling kRPC.
- `mission_control/krpc_adapter.py`: telemetry and bound-vessel actuation for
  the pinned kRPC 0.6 client.
- `scripts/orbit_maneuver.py`: plan preview, explicit execution, telemetry
  watchdogs, reporting, and cleanup through `MissionSupervisor`.

The equations and reference-frame conventions follow the
[kRPC orbit API](https://krpc.github.io/krpc/0.6.0/python/api/space-center/orbit.html),
[autopilot API](https://krpc.github.io/krpc/0.6.0/python/api/space-center/auto-pilot.html),
and [circularization tutorial](https://krpc.github.io/krpc/0.6.0/tutorials/launch-into-orbit.html).
Orbit altitudes are converted to radii before applying vis-viva.

## Planning

An apoapsis change is planned at periapsis. A periapsis change is planned at
apoapsis. Signed delta-v chooses prograde or retrograde. When the requested
apoapsis is below the current periapsis, lower periapsis first. Otherwise change
apoapsis first. This avoids invalid intermediate geometries and keeps the
intermediate periapsis outside the atmosphere.

The planner usually returns zero, one, or two burns. Near-circular goals can
require stricter intermediate tolerances or a final apsis within the requested
acceptance band. It never changes the mission's requested bounds. Live execution
replans from the actual orbit after each burn and permits up to six corrective
burns before failure.

The first task profile retains 80–180 km target apoapsis and requires target
periapsis of at least 71 km. Inclination, ascending node, rendezvous, body
transfers, atmospheric deorbit, and orientation of the apsidal line are outside
this controller's scope.

## Control

The ideal duration centers a burn around its apsis. If there is insufficient
time to align before a window, the controller waits until the next orbit.
The orbital direction is refreshed in a non-rotating body frame. Engines remain
off until directional alignment exceeds 0.98. Retrograde alignment is measured
against the negative of the orbital velocity direction.

The PID controls remaining delta-v. Its output is limited to [0, 1], and its
integrator does not accumulate into saturation. A second output cap limits the
commanded impulse to half the estimated remaining delta-v per observed control
interval. The controller stops a burn within its apsis tolerance, stops/replans
after overshoot, and aborts excessive-duration burns. Estimated burns exceeding
5% of an orbital period are rejected; this is a short-burn approximation, not a
low-thrust trajectory optimizer.

The supervisor checks both requested apsides and the reserve, commands zero
throttle, and requires the goal to hold for the task's settling interval. On
completion or failure it requests cutoff and disengages the owned autopilot.
The adapter never activates an engine, stages, reloads a save, or changes warp.

## First live test

1. Leave the reference rocket in a stable orbit and stop launch training or
   evaluation so only one process will control the vessel.
2. Manually make a recoverable orbital save. Keep the orbital engine active,
   throttle at zero, kRPC running, and time warp at normal speed.
3. Preview the current-orbit plan:

   ```powershell
   python -m scripts.orbit_maneuver --mission examples/change_orbit.json
   ```

4. Review target, direction, delta-v, and available duration estimates. The
   example requests 150 x 120 km within 1 km. No controls are changed by preview.
5. Execute:

   ```powershell
   python -m scripts.orbit_maneuver --mission examples/change_orbit.json --execute
   ```

6. Inspect the final JSON and telemetry in `training_output/maneuvers/`. Check
   success, errors on both apsides, fuel consumption, alignment, and abort reason.

Waiting for an apsis can take many minutes. Do not manually time warp while the
runner is active; warp and stale/jumping telemetry are rejected. Ctrl+C aborts
and requests engine cutoff. A lost connection can prevent confirmation of that
cutoff, so check the vessel if transport fails.

The standalone fuel reserve is relative to the LiquidFuel load when the adapter
connects. It is not a whole-mission reserve or a delta-v estimate. Propellant
mixture/flow problems are detected through usable thrust, not a complete engine
and tank model. The current profile expects a usable engine throughout a task.

## Validation status

Offline tests verify burn plans using independent energy/angular-momentum
equations across randomized orbits, near-circular geometry, PID anti-windup,
control timing, alignment, transport deadlines, and cleanup. Finite-thrust
two-body simulations exercise closed-loop raising and lowering with propellant
mass loss. These simulations assume immediate attitude alignment and ideal
thrust, and do not reproduce KSP attitude dynamics, engine response, or RPC delay.

A read-only live preview succeeded from approximately 174.9 x 130.3 km to
150 x 120 km, proposing two retrograde burns totaling about 25.5 m/s. No engine
commands were sent for that preview. The first user-reported execution then
raised a roughly 102.1 x 81.2 km orbit to the 150 x 120 km goal, reporting
`succeeded` after 1,280.44 simulation seconds with engines-off settling. Both
apsides were inside the requested 1 km acceptance bands. This is one reference
vessel/test case; lowering burns, other targets and broad reliability still
need live evaluation.
