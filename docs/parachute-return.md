# V1 deorbit and parachute recovery

The return runner starts on the active vessel in a stable orbit around stock
Kerbin. It runs a retrograde burn at apoapsis through the orbital PID, waits
for the requested periapsis to settle with the engine off, and monitors entry
through parachute deployment and touchdown.

## Prepare and fly

Equip the returning craft with stock parachutes sized for its return mass and
thermal protection for the intended entry orientation. Verify the chute setup
with the actual recovery mass. Keep all chutes stowed in orbit and configure
their opening altitude and pressure in KSP. Have an active, throttleable engine
and propellant for the deorbit burn. Save a recoverable orbital backup, leave the
game at normal time speed with kRPC running, and stop other flight controllers.

From the project root with the virtual environment active:

```powershell
python -m scripts.return_to_ground
```

This preview shows the estimated burn and the live `parachute_states` list.
Confirm the expected chutes are all `stowed`. A positive engine estimate is
based on the current flight configuration and does not establish enough delta-v.
The return uses no landing PPO and has no powered-landing thrust-to-weight gate.

```powershell
python -m scripts.return_to_ground --execute
```

The example in [parachute.json](../examples/return/parachute.json) targets a
20 km periapsis with 500 m tolerance. After engine cutoff, the runner uses
surface-retrograde attitude through entry by default. Set `entry_attitude` to
`passive` in a custom `--mission` file for a craft designed to orient itself.
The burn planner uses orbital telemetry. Atmospheric survival and touchdown
location depend on the craft and entry conditions. The runner does not stage,
reload a save, or steer to a selected landing site.

## Deployment and touchdown

The example requests deployment between 100 m and 10 km above terrain, during
descent, only when the stock chute reports `safe`. Each chute receives at most
one deployment request. KSP still controls when the canopy opens according to
its part settings. The runner cuts throttle and releases entry attitude control
after the first deployment request so the vehicle can hang under the canopy.
See [the kRPC parachute API](https://krpc.github.io/krpc/0.6.0/python/api/space-center/parts.html#parachute).

The mission requires all original parts to survive, an observed fully open
canopy before contact, and a stable touchdown or splashdown for three seconds.
The example accepts up to 8 m/s vertical and 5 m/s horizontal speed. Set
`allow_splashdown` to `false` to require dry ground. The runner checks the last
airborne speed as well as contact telemetry because KSP can report near-zero
speed after an impact. These are sampled speeds rather than an exact impact
measurement. The limits are acceptance criteria for the example mission, and
the craft needs to be tested against them.

JSONL telemetry and the final JSON report are saved in
`training_output/returns/`. The report includes chute commands, the last
airborne sample, touchdown location and any failure reason. Ctrl+C requests
cutoff and releases flight controls. A deployed parachute stays deployed.

The default return path has offline coverage. A complete live recovery run
with the parachute-equipped craft is the remaining v1 flight validation.
