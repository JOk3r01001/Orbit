# KSP Launch Pilot — Training Guide

This README covers only the training of the first reinforcement-learning pilot: the **launch and lift-off pilot**.

The pilot is responsible for taking a rocket from the launchpad through atmospheric ascent and placing it into a safe preliminary orbit or handoff state. Orbital corrections, transfers, landing, and LLM mission-command logic are outside the scope of this document.

---

## 1. Training objective

The launch pilot learns to:

- launch safely from the pad,
- control throttle during ascent,
- follow a gravity-turn trajectory,
- maintain a useful heading,
- stage when required,
- raise apoapsis above the atmosphere,
- avoid crashes, flips, and fuel waste,
- finish in a state suitable for the next pilot.

A successful episode should end when the vessel reaches the launch pilot's handoff condition, for example:

- apoapsis above the atmosphere,
- vessel still controllable,
- sufficient fuel remaining,
- no collision or atmospheric re-entry,
- optionally, periapsis above the atmosphere if the launch pilot also performs circularization.

---

## 2. System architecture

```text
KSP telemetry
      ↓
Gymnasium environment
      ↓
PPO launch policy
      ↓
Throttle / pitch / staging commands
      ↓
KSP physics
      ↓
Reward and next observation
```

The launch pilot should make fast, low-level flight-control decisions. Higher-level mission planning should remain outside this model.

---

## 3. Requirements

Install the required Python packages:

```bash
pip install numpy gymnasium stable-baselines3 krpc tensorboard
```

You also need:

- Kerbal Space Program,
- the kRPC mod installed,
- the kRPC server running,
- a launch vehicle prepared for repeated training,
- a named KSP save that can be restored at the beginning of every episode.

---

## 4. KSP training save

Create a dedicated save, for example:

```text
launch_training_start
```

The save should contain:

- the rocket on the launchpad,
- engines and staging configured correctly,
- sufficient fuel,
- SAS or reaction-control capability if your environment uses it,
- no active time warp,
- a consistent starting state for every episode.

The environment should reload this save during every `reset()` call.

Do not train from an important career save. Training repeatedly reloads the same state and may issue unexpected control commands while the model is exploring.

---

## 5. Action space

A simple continuous action space can use:

```text
action[0] = throttle command
action[1] = pitch command
```

For Stable-Baselines3 PPO, both outputs can be represented in the range `[-1, 1]`.

Example conversion:

```python
throttle = (action[0] + 1.0) / 2.0
pitch_input = float(action[1])
```

This maps:

```text
throttle action -1.0 → 0% throttle
throttle action  0.0 → 50% throttle
throttle action  1.0 → 100% throttle
```

Staging can be handled in one of two ways:

1. **Rule-based staging** — activate the next stage when current-stage fuel is nearly empty.
2. **Learned staging** — add another action output and let the policy decide when to stage.

For the first working version, rule-based staging is usually easier and safer.

---

## 6. Observation space

The observation vector should contain enough information for the policy to understand the rocket's motion, trajectory, remaining resources, and current controls.

A practical launch observation vector may include:

```text
altitude
vertical speed
horizontal speed
apoapsis altitude
periapsis altitude
time to apoapsis
pitch
heading error
roll or angular velocity
fuel fraction
current mass
available thrust
throttle
current stage
```

Normalize large values before passing them to PPO.

Example:

```python
observation = np.array(
    [
        altitude / 150_000.0,
        vertical_speed / 2_000.0,
        horizontal_speed / 3_000.0,
        apoapsis / 150_000.0,
        periapsis / 150_000.0,
        time_to_apoapsis / 300.0,
        pitch / 90.0,
        heading_error / 180.0,
        fuel_fraction,
        mass / initial_mass,
        available_thrust / thrust_scale,
        current_throttle,
        stage_fraction,
    ],
    dtype=np.float32,
)

observation = np.clip(observation, -1.0, 1.0)
```

The normalization constants should match the expected range of your rocket and mission.

---

## 7. Reward design

The reward should encourage useful progress rather than only rewarding elapsed time or altitude.

A good starting reward can combine:

```text
positive:
- altitude progress
- apoapsis progress
- horizontal-speed progress after the gravity turn
- pointing near the desired trajectory
- reaching the handoff condition

negative:
- fuel consumption
- excessive angular motion
- large heading error
- descending during ascent
- unsafe periapsis
- crashing
- running out of fuel
- exceeding the episode limit
```

Example structure:

```python
reward = 0.0

reward += altitude_progress / 1_000.0
reward += apoapsis_progress / 1_000.0
reward += horizontal_speed_progress / 500.0

reward -= throttle * fuel_penalty_scale
reward -= abs(heading_error) * heading_penalty_scale
reward -= angular_velocity * instability_penalty_scale
reward -= time_penalty

if success:
    reward += 100.0

if crashed:
    reward -= 100.0
```

Use progress rewards based on the difference between the previous and current state:

```python
altitude_progress = current_altitude - previous_altitude
apoapsis_progress = current_apoapsis - previous_apoapsis
```

This is usually safer than rewarding raw altitude or apoapsis every step, which can cause rewards to grow simply because an episode lasts longer.

---

## 8. Episode termination

Terminate the episode when one of these conditions occurs.

### Success

Examples:

```text
apoapsis above target altitude
periapsis above minimum altitude
vessel remains controllable
fuel remains above a minimum reserve
```

### Failure

Examples:

```text
vessel crashes
vessel begins uncontrolled descent
rocket flips beyond a safe angle
fuel is depleted before success
periapsis becomes unrecoverably low
vehicle is destroyed
```

### Truncation

Use truncation when the maximum episode length is reached:

```python
truncated = episode_steps >= max_episode_steps
```

Gymnasium expects:

```python
observation, reward, terminated, truncated, info
```

---

## 9. Recommended curriculum

Do not begin with the full launch problem if the policy cannot yet produce stable behavior.

### Stage 1 — Vertical lift-off

Train the pilot to:

- use throttle,
- remain approximately vertical,
- climb without crashing.

### Stage 2 — Controlled pitch-over

Introduce a target pitch profile and reward:

- gradual pitch reduction,
- low heading error,
- controlled angular velocity.

### Stage 3 — Apoapsis targeting

Train the policy to:

- raise apoapsis toward a target,
- reduce throttle near the target,
- avoid overshooting excessively.

### Stage 4 — Full atmospheric ascent

Combine:

- throttle control,
- gravity turn,
- heading control,
- staging,
- apoapsis targeting.

### Stage 5 — Randomization

Randomize selected conditions:

- vehicle mass,
- fuel load,
- engine thrust,
- target orbit,
- small initial heading errors,
- aerodynamic disturbances.

Increase difficulty gradually. Keep evaluation conditions separate from training conditions.

---

## 10. PPO training configuration

A reasonable starting configuration is:

```python
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    n_steps=1024,
    batch_size=256,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs={
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128],
        }
    },
    tensorboard_log="training_output/tensorboard",
    verbose=1,
    seed=42,
)
```

These values are only a starting point. The best configuration depends on:

- simulation step duration,
- episode length,
- reward scale,
- observation normalization,
- rocket dynamics.

---

## 11. Environment validation

Before long training runs, validate the Gymnasium environment:

```python
from stable_baselines3.common.env_checker import check_env

env = KSPLaunchEnv()
check_env(env, warn=True, skip_render_check=True)
```

The checker may execute random actions. Make sure every reset safely restores the launch save.

---

## 12. Starting training

Example:

```bash
python train_launch_pilot.py
```

Or, if the training script supports command-line arguments:

```bash
python train_launch_pilot.py train \
    --save-name launch_training_start \
    --total-timesteps 500000 \
    --output-dir launch_training_output
```

The exact script name and arguments should match your implementation.

---

## 13. Checkpoints

Save checkpoints regularly:

```python
from stable_baselines3.common.callbacks import CheckpointCallback

checkpoint_callback = CheckpointCallback(
    save_freq=25_000,
    save_path="launch_training_output/checkpoints",
    name_prefix="launch_ppo",
)
```

Train with:

```python
model.learn(
    total_timesteps=500_000,
    callback=checkpoint_callback,
)
```

Do not assume the final checkpoint is the best checkpoint. Evaluate several saved models independently.

---

## 14. TensorBoard

Start TensorBoard with:

```bash
tensorboard --logdir launch_training_output/tensorboard
```

Useful PPO metrics include:

```text
rollout/ep_rew_mean
rollout/ep_len_mean
train/approx_kl
train/clip_fraction
train/entropy_loss
train/explained_variance
train/policy_gradient_loss
train/value_loss
```

Project-specific metrics should also be logged:

```text
launch/success_rate
launch/final_apoapsis_error
launch/final_periapsis
launch/fuel_fraction
launch/max_heading_error
launch/max_angular_velocity
launch/time_to_handoff
launch/staging_failures
```

---

## 15. Interpreting unstable training

Watch for this pattern:

```text
episode reward rises
episode length rises
value loss becomes extremely large
explained variance remains near zero
```

This can mean the critic is failing to predict returns, even though the total reward appears to improve.

Possible causes include:

- rewards that grow with episode length,
- very large terminal rewards,
- unnormalized observations,
- reward terms with very different scales,
- unstable or discontinuous episode resets,
- sparse success events,
- excessive learning rate,
- overly long rollouts.

Useful responses include:

- normalize observations,
- reduce reward magnitudes,
- use progress-based rewards,
- shorten the first curriculum,
- reduce the learning rate,
- compare earlier checkpoints,
- evaluate with deterministic actions.

---

## 16. Evaluation

Training reward is not enough. Evaluate the pilot separately with learning disabled.

Use:

```python
action, _ = model.predict(
    observation,
    deterministic=True,
)
```

Run multiple episodes and report:

```text
success rate
mean final apoapsis error
mean final periapsis
mean fuel remaining
mean time to handoff
crash rate
staging failure rate
```

A useful minimum evaluation is 100 independent flights.

Example result format:

```text
Launch success rate: 82 / 100
Mean apoapsis error: 3.4 km
Mean remaining fuel: 21%
Crash rate: 7%
Mean time to handoff: 164 seconds
```

---

## 17. Suggested handoff contract

The launch pilot should produce a clear state for the next pilot.

Example handoff condition:

```python
handoff_ready = (
    apoapsis >= target_apoapsis - tolerance
    and vessel_is_stable
    and fuel_fraction >= minimum_fuel_reserve
)
```

The launch pilot can then stop controlling the vessel and return a status object:

```python
handoff = {
    "pilot": "launch",
    "status": "complete",
    "apoapsis": current_apoapsis,
    "periapsis": current_periapsis,
    "fuel_fraction": fuel_fraction,
    "next_pilot": "orbit_correction",
}
```

This keeps the first pilot separate from later orbital maneuver models.

---

## 18. Training checklist

Before training:

- [ ] kRPC server is running.
- [ ] The correct vessel is active.
- [ ] The launch save exists.
- [ ] Engines and staging are configured.
- [ ] `reset()` reloads the vessel correctly.
- [ ] Observations contain finite values.
- [ ] Observations match `observation_space`.
- [ ] Actions match `action_space`.
- [ ] Throttle is set to zero on termination.
- [ ] Crashes and success are detected correctly.
- [ ] Checkpoints and TensorBoard directories exist.
- [ ] The environment passes `check_env`.

Before presenting results:

- [ ] Evaluate deterministic checkpoints.
- [ ] Report success rate over many flights.
- [ ] Include fuel and trajectory metrics.
- [ ] Compare the best checkpoint with a simple scripted baseline.
- [ ] Record at least one complete successful flight.

---

## Scope

This README covers only the **first launch/lift-off pilot**.

It does not cover:

- orbital apoapsis/periapsis correction,
- Mun or interplanetary transfers,
- landing,
- PID attitude-control baselines,
- LLM commander integration,
- automatic controller switching.
