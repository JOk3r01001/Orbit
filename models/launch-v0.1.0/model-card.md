# First experimental launch PPO weights

Release: [launch-v0.1.0](https://github.com/JOk3r01001/Orbit/releases/tag/launch-v0.1.0)

This release snapshots the latest saved launch policy at **1,025,855 training
steps**, using reward revision **2**. It is the first public weights milestone
for the Orbit project. It controls launch and orbital insertion; orbital
maneuvers, landing, and live supervisor integration are separate future work.

## Artifact and compatibility

- Model: `ppo_ksp_commanded_pilot_v4.zip` (Stable-Baselines3 PPO archive).
- SHA-256: `7cb832917fec31ece264bd733d7fbfe724888effd7dd5d3df24e871969f5ce85`.
- Source implementation: commit `b164bd5014f6a4e2682b80025ed9ca9c8f652646`.
- Recorded training environment: Python 3.12.10, Stable-Baselines3 2.9.2,
  PyTorch 2.15.0+cpu, NumPy 2.5.4, Gymnasium 1.4.0, and kRPC 0.6.0.
- The archive contains policy and value-network parameters, optimizer state,
  training counters, and reward-configuration metadata. It can resume training.
- The published model is a frozen snapshot. Later local training does not
  automatically update this release.

The release also provides `manifest.json`, `evaluation-summary.json`,
`requirements-training.lock.txt`, and `SHA256SUMS.txt`. The requirements file
captures installed distribution versions without local filesystem paths; it
does not include the editable Orbit project itself.

## Flight environment

Use stock Kerbin, an eastward-launching rocket, and a kRPC **0.6.0** server.
The corresponding KSP mod version file targets KSP 1.12.3 through 1.12.5. Craft
and KSP saves are not included in this release, so the original evaluation is
not yet fully reproducible from these artifacts alone.

The environment reloads `launchpad_ready` for each episode. Stage activation,
throttle/pitch guards, and attitude stabilization are provided by scripted
logic and KSP autopilot. Results describe this combined system, not an
unassisted policy. Transfer to different vessels is unverified.

The 16 observations, in order, are:

| Index | Value | Normalization |
| --- | --- | --- |
| 0 | Altitude above terrain | /100,000 m |
| 1 | Vertical speed | /1,000 m/s |
| 2 | Horizontal speed | /2,500 m/s |
| 3 | Apoapsis altitude | /200,000 m |
| 4 | Periapsis altitude | /200,000 m |
| 5 | Time to apoapsis | /300 s |
| 6 | LiquidFuel remaining | /initial LiquidFuel |
| 7 | Current mass | /initial mass, clipped to [0, 1.5] |
| 8 | Pitch | /90 degrees |
| 9 | Heading error from east | /180 degrees |
| 10 | Applied throttle | [0, 1] |
| 11 | Current stage | /max(1, initial stage) |
| 12 | Target apoapsis altitude | /200,000 m |
| 13 | Target periapsis altitude | /200,000 m |
| 14 | Requested minimum fuel fraction | fraction |
| 15 | Urgency | [0, 1] |

The final observation is float32 and clipped to [-10, 10]. Actions are two
values in [0, 1]: throttle request and pitch fraction (multiplied by 90 degrees).
Heading is fixed at 90 degrees. No external VecNormalize statistics are used.

The training target envelope is 80–180 km apoapsis, periapsis from 70 km up to
the selected apoapsis, and fuel-reserve requests up to 15% subject to the training
sampler's heuristic. This envelope is not a guarantee that a vessel can achieve
every target. The policy reached curriculum stage 3; the training seed was not
explicitly fixed. Episode duration is limited to 9,000 environment steps.

## Reported evaluation

The user reported the following `scripts/test_orbit.py` console summary before
release packaging. Evaluation uses deterministic actions and the same launchpad
save for six target-orbit commands:

| Target apoapsis (km) | Target periapsis (km) | Required fuel fraction |
| --- | --- | --- |
| 120 | 90 | 0.020 |
| 130 | 100 | 0.020 |
| 150 | 110 | 0.015 |
| 160 | 120 | 0.015 |
| 170 | 130 | 0.010 |
| 180 | 140 | 0.010 |

Aggregate results: **6/6 successes**, mean absolute apoapsis error **4.0 km**,
mean absolute periapsis error **9.8 km**, and mean fuel-fraction margin **+0.102**.
The 180 x 140 km case finished at 174.9 x 130.1 km, with 0.114 fuel fraction
remaining, after 6,172 steps; total reward was 397.28.

The original console report did not record the model hash. Therefore its exact
checkpoint identity is not independently verified. This release packages the
latest saved model and does not claim a separate six-flight re-evaluation during
publication. The six-case result is a same-vessel smoke test, not an estimate of
general reliability or precision across different craft, saves, or disturbances.

Success uses 10 km apoapsis/periapsis tolerances, altitude and periapsis above
70 km, and the requested fuel reserve. It ends as soon as those conditions pass;
there is no additional engine-off settling interval in the launch environment.
Fuel fraction measures LiquidFuel and is not a delta-v or landing-reserve budget.

## Use

From the matching source checkout:

```powershell
python -m pip install -r models/launch-v0.1.0/requirements-training.lock.txt
python -m pip install -e . --no-deps
python -m scripts.test_orbit --model path\to\ppo_ksp_commanded_pilot_v4.zip
```

Evaluation connects to KSP, reloads the launch save, and controls the vessel.
Use an isolated training save and one active flight-control process.

For continued learning, back up any existing model and place the downloaded file
at the repository root as `ppo_ksp_commanded_pilot_v4.zip`, then run:

```powershell
python -m scripts.orbit_command
```

The saved counter and curriculum continue, while the vessel starts again from
the launchpad save. Reward revision 2 removes the standing apoapsis-deficit
charge, reduces intervention coefficients to 0.01 each, and ties positive burn
guidance to measured periapsis progress. PPO uses learning rate 0.0003, rollout
length 1,024, batch size 256, gamma 0.995, and entropy coefficient 0.01.
