# Deorbit, telemetry handoff and descent PPO v2

This is the experimental powered-descent path. The v1 default is
[parachute recovery](parachute-return.md). Use `--powered` for a preview of the
powered experiment. Passing `--capture-handoff` or `--model` selects it as well.

The return prototype implements this sequence on the currently loaded Kerbin
vessel:

```text
stable orbit -> retrograde deorbit burn at apoapsis
             -> periapsis target held with engines off
             -> coast/atmospheric entry at zero throttle, surface retrograde
             -> telemetry handoff -> descent PPO -> touchdown/settling
```

Periapsis sets the osculating return trajectory. Handoff uses actual descending
altitude above terrain, velocity, thrust and estimated braking margin. It does
not wait until the craft reaches periapsis. The runtime never reloads a save,
stages, activates engines, deploys parachutes or starts learning online.

## Implementation and limits

- `DeorbitTarget` / `TaskType.DEORBIT` explicitly permit a periapsis target
  between 1 and 65 km. Ordinary `change_orbit` keeps its atmospheric safeguards.
- `DeorbitController` reuses the orbital PID/timing/alignment logic. It lowers
  periapsis at apoapsis and finishes while the craft remains above the atmosphere.
- `ReturnMission` combines the deorbit task, optional landing task, descent
  profile, handoff height window and coast timeout.
- `scripts.return_to_ground` runs the deorbit through `MissionSupervisor`,
  coasts, cuts/relinquishes orbital control, and initializes the descent
  environment on the same bound vessel without calling reset or loading a save.
- `scripts.descent_train` provides the new training/evaluation environment and
  checkpoint contract. It shares touchdown/settling checks with landing v1.

The example in `examples/return/mission.json` requests periapsis **20 km**, with
500 m tolerance. This is an experimental entry target, not a demonstrated safe
trajectory. The controller does not predict atmospheric drag, thermal loads,
terrain, impact location, or propellant sufficiency. Lowering periapsis alone
does not ensure a safe dry-ground landing. The craft needs suitable thermal and
attitude capability, a usable throttleable engine, landing legs, and enough fuel.

## Handoff contract

The example begins considering handoff at **30 km above terrain**, during descent,
and refuses a handoff below **1 km**. These are editable experiment settings, not
validated performance claims. The policy entry profile permits 100 m-30 km,
downward speed up to 1,200 m/s, horizontal speed up to 2,000 m/s and thrust/weight
of at least 1.3. A model must actually be trained/evaluated on the states it will
encounter; declaring these limits does not teach it to fly them.

An additional screening estimate uses all surface-relative speed:

```text
net acceleration = available thrust / mass - local gravity
stopping distance = (vertical speed² + horizontal speed²) / (2 * net acceleration)
required height = 1.5 * stopping distance + downward speed * 3 seconds + 50 m
```

The runtime accepts only a positive height margin and a matching policy profile.
This intentionally conservative check ignores useful drag but also assumes full
thrust and timely attitude recovery. It is not a trajectory solver, fuel check,
thermal check or safety guarantee. In the handoff window, an out-of-profile or
insufficient-margin descent fails explicitly; it does not silently fall back to
the old vertical policy or wait below the minimum height. High entries may need
a different entry plan, handoff altitude, or vehicle after live investigation.

The default `landing: null` chooses the ground footprint at handoff with a broad
10 km acceptance radius. A structured `land` mission can instead specify a
location and touchdown limits. There is no orbital targeting/site predictor;
choosing a distant coordinate does not make the deorbit burn target that site.

## First step: preview and capture a real training state

Keep a recoverable orbital backup. Load the intended return craft in stable
orbit, stop other flight-control scripts, keep the landing engine active,
throttle zero, landing legs already deployed and time warp normal. Verify its
thermal protection and propellant arrangement manually.

Read-only preview (no engine/autopilot writes):

```powershell
python -m scripts.return_to_ground --powered
```

The preview's `engine_preflight` section shows active-engine available thrust,
mass, current thrust/weight and a surface estimate. The current preflight minimum
is 1.3. Zero throttle is allowed: available thrust describes full-throttle
capability, not the thrust being fired now. If this check fails, verify the
intended engine is active, fueled and not thrust-limited; an underpowered craft
needs less mass or more thrust. Live failures retain the sampled TWR and limit
in the return report. The surface estimate ignores pressure-dependent thrust
changes and fuel consumption, so it is only diagnostic.

Offline preview without KSP or flight dependencies:

```powershell
python -m scripts.return_to_ground --powered --snapshot examples/telemetry/orbit_snapshot.json
```

There are no trained descent-v2 weights yet. Explicit capture mode performs the
deorbit and coast, then **saves** the first valid handoff state at zero throttle:

```powershell
python -m scripts.return_to_ground --execute --capture-handoff descent_training_start
```

The save name is written in KSP and an existing save with that name may be
overwritten. Capture does not land the rocket or pause the game: it requests
cutoff, saves, releases controls and exits. The craft continues descending.
Training will reload the saved state for each episode. If the entry is too fast,
too low or otherwise outside the profile, the command aborts and records why.
No live entry/capture has been validated by the automated tests.

## Train the descent brain

Use the existing training virtual environment:

```powershell
python -m scripts.descent_train check
python -m scripts.descent_train train
```

The check reloads the descent save and applies random actions; crashes are
expected possibilities. The policy now learns braking and thrust direction
from the captured state. It does **not** require a manually prepared 10-20 m/s
descent. Initial states still need to pass the braking-margin screen.

Train across several real prepared/captured saves by repeating `--save`:

```powershell
python -m scripts.descent_train train --save descent_training_start --save descent_start_02 --model training_output/descent_v2/multi_start.zip
```

Each episode selects one save using Gym's seeded RNG. There is no hidden physics
teleportation or fabricated observation randomization. Keep held-out saves for
evaluation, and keep the named saves unchanged when resuming a run. To train a
new experiment, select a new `--model` path. Training resumes an existing matching
checkpoint automatically and requests 250,000 additional steps by default.

Weights live under `training_output/descent_v2/`, separately from launch and
landing-v1 weights. Checkpoints are saved every 5,000 steps. Ctrl+C or an exception
during learning saves current parameters before cleanup. Back up both the ZIP
and JSON sidecar off-machine; power loss cannot save in-memory updates. A periodic
checkpoint can be resumed by passing its ZIP with `--model`, keeping its sidecar
and the original `--save`/mission settings.

```powershell
tensorboard --logdir training_output/descent_v2/tensorboard
python -m scripts.descent_train evaluate --episodes 6
```

## Policy interface

The v2 interface has **26 observations**: terrain altitude, vertical/north/east
surface velocity, tilt, LiquidFuel tank fill, available thrust/mass, local
gravity, engine-axis up/north/east components, apoapsis, periapsis, estimated
stopping distance, required height, ballistic time to ground, previous throttle
and two lean commands, elapsed time, local target north/east offsets, and target
vertical/horizontal speed, tilt and radius limits.

Surface velocity is measured in the rotating body frame and rotated into local
up/north/east axes. It is not measured relative to the local frame's moving
vessel-centered origin. Policy fuel input is tank fill, keeping that observation
consistent between a captured save and live return. Reserve checks use fuel
relative to the initial training load or, during return, the initial orbital load.

Three actions in [-1, 1] select throttle (mapped to [0, 1]), north lean and east
lean. The commanded thrust direction is normalized and capped at 85 degrees from
upright. KSP autopilot tracks that direction; PPO chooses it. Firing is blocked
while the actual engine axis points below the horizon. There is no scripted
braking throttle. This assumes the vessel's forward axis represents thrust and
the craft has sufficient attitude authority.

Reward uses changes in descent-speed/altitude/horizontal-speed/site-error
potential, minus fuel and time costs. The speed reference only shapes reward.
Success/failure adds +100/-100 and the shared touchdown check uses pre-contact
speed, integrity, dry-ground contact and engine-off settling. Sampling cannot
measure exact impact peaks between RPC reads. Reports retain sampled touchdown
data and failure reasons; no landing weights are published until flight goals
are demonstrated.

## Execute the trained sequence

After deterministic evaluation on representative and held-out descent states:

```powershell
python -m scripts.return_to_ground --execute --model training_output/descent_v2/descent_latest.zip
```

The runner loads weights and checks the JSON policy contract, dimensions,
training-step count and descent profile **before connecting/committing to a
deorbit**. Landing-v1 weights are incompatible. A successful training run or a
positive step count alone does not establish landing competence.

Return telemetry/reports go to `training_output/returns/`; training episode CSVs,
evaluation JSON and TensorBoard logs go to `training_output/descent_v2/`.
Ctrl+C requests cutoff. Vessel/body switches, warp, paused/rewound/gapped
telemetry, missed handoff windows and control errors stop the sequence. A broken
transport can prevent confirming cutoff. Ordinary orbital saves are never
automatically loaded during execution.

The prototype has offline coverage for deorbit calculations, handoff conditions,
policy compatibility, save capture and exclusive control ordering. Live deorbit,
atmospheric survival, capture, descent training and complete return success are
still pending. Launch-to-return integration remains separate work.
