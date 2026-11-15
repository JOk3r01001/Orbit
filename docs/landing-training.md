# Phase 3: powered landing, first curriculum

This prototype trains a new PPO policy to control throttle during a near-vertical
descent on stock Kerbin. KSP autopilot holds the vehicle upright. No parachute,
staging, deorbit, horizontal navigation or automatic handoff from orbit is
implemented. Launch weights are separate and are not reused for this policy.

## Prepare the training save

Use a controllable rocket with landing legs, a throttleable LiquidFuel engine,
reaction wheels/SAS capability, and sufficient propellant. Deploy landing legs
before saving. An engine whose minimum thrust exceeds the weight of the vehicle
will not be suitable even if its maximum thrust is sufficient.

Prepare a short hop over flat, dry ground and save during descent:

- Recommended height: **300-500 m above terrain**; accepted range 100-1000 m.
- Recommended vertical speed: **10-20 m/s downward**; accepted range greater than
  zero and at most 40 m/s downward, subject to a stopping-distance check.
- Horizontal speed: **at most 0.5 m/s**. This policy cannot steer sideways.
- Tilt: at most 10 degrees from upright.
- Engine active, throttle zero, landing legs deployed, parachutes removed or
  stowed, normal time speed, kRPC running.
- Available thrust/weight at least 1.3; estimated stopping distance must be less
  than 60% of initial terrain altitude.

Name the KSP save **`landing_training_start`**. This is a separate descent save,
not `orbit_ready` or `launchpad_ready`. Every episode reloads it. Keep an original
copy and do not change its craft or state while resuming an existing experiment.
Ground altitude is terrain clearance of the vessel reference point; it is not
landing-leg clearance and does not reach zero when the legs touch down.

## Run

Stop all other flight-control scripts. From the repository root, with the
training virtual environment activated:

```powershell
python -m scripts.landing_train check
```

This actively reloads the save and tries random throttle actions. It checks the
Gym interface, not landing performance, and the rocket may crash during it.

Once the environment check works:

```powershell
python -m scripts.landing_train train
```

The default run requests 250,000 additional steps; PPO completes full rollouts,
so the actual count can be slightly higher. It automatically resumes
`training_output/landing_v1/landing_latest.zip` if present. Otherwise it starts a
new policy. Rerun the same command tomorrow to continue. Ctrl+C requests engine
cutoff and saves the current policy. Periodic checkpoints are written every
5,000 steps. Copy the `.zip` **and matching `.json` metadata** off the machine to
protect against disk loss; these files are ignored by Git. A forced process
termination or power loss cannot save the latest in-memory progress.

To resume a periodic checkpoint explicitly:

```powershell
python -m scripts.landing_train train --model training_output/landing_v1/checkpoints/landing_latest_5000_steps.zip
```

This resumes and then updates that specified path. To start a separate experiment
without overwriting the existing checkpoint:

```powershell
python -m scripts.landing_train train --fresh --model training_output/landing_v1/experiment_02.zip
```

Live reward graphs:

```powershell
tensorboard --logdir training_output/landing_v1/tensorboard
```

Evaluate the saved policy with deterministic actions:

```powershell
python -m scripts.landing_train evaluate --episodes 6
```

Repeated evaluations reload the same descent save. They measure repeatability
on that state, not generalization to other heights, craft, terrain or weather.

## Mission input, learning and termination

Without `--mission`, the landing target is the ground location beneath the
initial save. Defaults are a 500 m radius, vertical touchdown speed at most
2 m/s, horizontal speed at most 1 m/s, and tilt at most 10 degrees.

An optional `--mission path/to/land.json` accepts the existing structured land
task. Use the same argument when checking, training, resuming and evaluating.
The save must already be over that target's acceptance area. The coordinate
input does not add steering capability. Reserves are relative to the initial
LiquidFuel load, not whole-mission fuel. Stage-one episodes have a maximum
180-second simulation timeout (or a shorter mission timeout).

The 14 observations include terrain altitude, descent and horizontal speeds,
tilt, fuel, thrust/mass, gravity, a reference descent speed, previous throttle,
elapsed time and target acceptance limits/distance. The action is one throttle
fraction in [0, 1]. The descent-speed reference shapes reward only; it never
commands throttle. Throttle is blocked above 15 degrees tilt, and the
intervention is logged.

Reward is the change in a bounded-on-the-training-envelope descent potential,
minus fuel consumption and simulation-time costs. Standing still earns no
positive progress reward. Success adds 100; terminal failure or timeout subtracts
100. Reward components and intervention flags are returned by the environment;
episode return, length and success are logged through SB3 Monitor/TensorBoard.

Success requires dry-ground `landed` status, all original parts retained,
acceptable touchdown speed/tilt/location, and three seconds of engine-off
settling (or the mission's settling time). The check also uses the last
pre-contact speed: a hard impact followed by zero reported landed velocity is
not a success. This is a conservative sampled estimate, not an exact impact
sensor; short peaks between telemetry reads can be missed.

Damage, water contact, bouncing after contact, reserve/thrust failure, excessive
tilt/ascent/height, timeouts and telemetry problems stop the episode or run and
request engine cutoff. Loss of the connection can prevent that cutoff from
being confirmed. No engines are automatically activated and no stages are fired.

## Validation and next milestones

Offline tests cover touchdown rejection, settling, bounce, reserve enforcement,
reward costs, telemetry/action failures, and alignment intervention. KSP training
and landing performance remain unverified until a suitable descent save is
prepared and tested. There are no trained landing weights yet.

After reliable vertical landings, add multiple validated descent saves and
evaluate unseen starting conditions. Then extend the action/observation space
for lateral control and target navigation; that changes the policy interface
and needs a new policy version. Full deorbit/reentry and the supervisor's live
PPO inference adapter are later integration work.
