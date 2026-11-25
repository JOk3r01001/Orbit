"""Version-two PPO: learn braking, thrust direction and touchdown from descent saves."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from mission_control.commands import LandingTarget
from mission_control.descent import DescentProfile, DescentSample, ReturnMission, braking_metrics, entry_rejection
from mission_control.landing import distance_to_target
from scripts.landing_train import LandingEnv


OUTPUT = Path("training_output/descent_v2")
LATEST = OUTPUT / "descent_latest.zip"
OBSERVATION_VERSION = 2
OBSERVATION_SIZE = 26


def policy_contract(profile):
    return {"observation_version": OBSERVATION_VERSION, "observation_size": OBSERVATION_SIZE,
            "action_size": 3, "curriculum": "directed_descent_v2", "profile": asdict(profile)}


def load_policy(path, profile):
    """Validate the contract before a return mission can commit to a deorbit."""
    path = Path(path)
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise ValueError("A trained descent_v2 .zip checkpoint and matching .json sidecar are required")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("contract") != policy_contract(profile):
        raise ValueError("Descent model contract/profile mismatch; v1 landing weights cannot be used")
    model = PPO.load(str(path), device="cpu")
    if (model.observation_space.shape != (OBSERVATION_SIZE,) or model.action_space.shape != (3,)
            or not np.all(model.action_space.low == -1) or not np.all(model.action_space.high == 1)):
        raise ValueError("Incompatible descent model observation/action spaces")
    if model.num_timesteps <= 0:
        raise ValueError("Descent policy has no training steps")
    return model


def target_offsets(s, target):
    north = math.radians(target.latitude_deg - s.latitude_deg) * 600_000
    dlon = (target.longitude_deg - s.longitude_deg + 180) % 360 - 180
    east = math.radians(dlon) * 600_000 * math.cos(math.radians(s.latitude_deg))
    return north, east


class DescentEnv(LandingEnv):
    """Three normalized actions: throttle, north lean, east lean.

    PPO chooses a thrust direction (up/north/east); autopilot tracks it. No
    scripted braking, staging or parachutes. Shares touchdown checks with v1.
    """

    maximum_steps = 15000

    def __init__(self, save_names=None, task=None, profile=None, connection=None):
        self.profile = profile or DescentProfile()
        self.save_names = save_names or ["descent_training_start"]
        super().__init__(self.save_names[0], task, connection)
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.episode_timeout_s = self.profile.timeout_s
        self.lean = (0, 0)

    def read(self):
        base = super().read()
        frame = self.vessel.surface_reference_frame
        surface_velocity = self.vessel.velocity(self.body.reference_frame)
        _, north, east = self.sc.transform_direction(surface_velocity, self.body.reference_frame, frame)
        nose = self.vessel.direction(frame)
        length = math.sqrt(sum(v * v for v in nose))
        if length <= 0:
            raise ValueError("Cannot determine vessel attitude")
        capacity = float(self.vessel.resources.max("LiquidFuel"))
        if capacity <= 0:
            raise ValueError("No LiquidFuel tank capacity")
        fill = float(self.vessel.resources.amount("LiquidFuel")) / capacity
        # Samples are sequential RPC reads; use the final simulation timestamp.
        return DescentSample(**{**asdict(base), "ut_s": float(self.sc.ut)},
                             north_mps=float(north), east_mps=float(east),
                             nose_up=max(-1, min(1, nose[0] / length)),
                             nose_north=max(-1, min(1, nose[1] / length)),
                             nose_east=max(-1, min(1, nose[2] / length)),
                             apoapsis_m=float(self.vessel.orbit.apoapsis_altitude),
                             periapsis_m=float(self.vessel.orbit.periapsis_altitude),
                             tank_fill_fraction=max(0, min(1, fill)))

    def reset(self, *, seed=None, options=None):
        # Initialize RNG before choosing a prepared save; no fabricated telemetry.
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        self.save_name = str(self.np_random.choice(self.save_names))
        return super().reset(seed=seed, options=options)

    def validate_initial(self, sample):
        reason = entry_rejection(sample, self.profile)
        if reason:
            raise ValueError(reason)

    def default_target(self, sample):
        # First return experiments accept a broad footprint; no site prediction.
        return LandingTarget(sample.latitude_deg, sample.longitude_deg, radius_m=10_000)

    def validate_target(self, sample):
        if distance_to_target(sample, self.target) > 100_000:
            raise ValueError("Landing target is more than 100 km from descent entry")

    def configure_attitude(self):
        self.lean = (0, 0)
        velocity = (self.previous.vertical_mps, self.previous.north_mps, self.previous.east_mps)
        norm = math.sqrt(sum(v * v for v in velocity))
        direction = tuple(-v / norm for v in velocity) if norm > .1 else (1, 0, 0)
        ap = self.vessel.auto_pilot
        ap.reference_frame = self.vessel.surface_reference_frame
        ap.target_direction = direction
        ap.engaged = True

    def apply_policy_action(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Expected three finite descent actions")
        action = np.clip(action, -1, 1)
        self.lean = (float(action[1]), float(action[2]))
        # Upward hemisphere, up to 85 degrees from vertical, with a circular
        # cone cap (diagonal inputs must not exceed the same tilt limit).
        horizontal = np.asarray(self.lean) * math.tan(math.radians(85))
        norm = float(np.linalg.norm(horizontal))
        if norm > math.tan(math.radians(85)):
            horizontal *= math.tan(math.radians(85)) / norm
        direction = np.array([1, horizontal[0], horizontal[1]])
        direction /= np.linalg.norm(direction)
        ap = self.vessel.auto_pilot
        ap.reference_frame = self.vessel.surface_reference_frame
        ap.target_direction = tuple(float(v) for v in direction)
        ap.engaged = True
        self.throttle = 0 if self.contact is not None else float((action[0] + 1) / 2)
        # Block firing while the actual engine axis still points below horizon.
        if self.previous.nose_up <= 0:
            self.throttle = 0
        self.vessel.control.throttle = self.throttle

    def alignment_intervention(self):
        return self.previous.nose_up <= 0

    def outside_envelope(self, sample):
        return (sample.height_m > self.profile.max_height_m * 1.2
                or sample.vertical_mps > 100 or sample.vertical_mps < -3500
                or sample.horizontal_mps > 3500)

    def observation(self, s):
        metrics = braking_metrics(s, self.profile)
        north, east = target_offsets(s, self.target)
        return np.asarray([
            s.height_m / 30_000, s.vertical_mps / 1000, s.north_mps / 2000, s.east_mps / 2000,
            s.tilt_deg / 180, s.tank_fill_fraction, s.acceleration_mps2 / 50, s.gravity_mps2 / 10,
            s.nose_up, s.nose_north, s.nose_east, s.apoapsis_m / 200_000, s.periapsis_m / 100_000,
            min(metrics["estimated_stopping_distance_m"], 1e8) / 30_000,
            min(metrics["required_height_m"], 1e8) / 30_000,
            metrics["ballistic_time_to_ground_s"] / 1000,
            self.throttle, *self.lean, (s.ut_s - self.started_ut) / self.timeout,
            north / 100_000, east / 100_000, self.target.max_vertical_speed_mps / 5,
            self.target.max_horizontal_speed_mps / 5, self.target.max_tilt_deg / 30,
            self.target.radius_m / 10_000,
        ], dtype=np.float32)

    def reward_transition(self, previous, sample):
        def potential(s):
            net = max(.1, .3 * (s.acceleration_mps2 - s.gravity_mps2))
            reference = -min(250, math.sqrt(self.target.max_vertical_speed_mps ** 2 + 2 * net * max(0, s.height_m - 5)))
            return (max(0, s.height_m) / 30_000 + abs(s.vertical_mps - reference) / 200
                    + s.horizontal_mps / 400 + distance_to_target(s, self.target) / 100_000
                    + s.tilt_deg / 180 * max(0, 1 - s.height_m / 1000))
        dt = sample.ut_s - previous.ut_s
        from mission_control.commands import number
        number(dt, "descent dt", 1e-6, 2)
        terms = {"progress_reward": 5 * (potential(previous) - potential(sample)),
                 "fuel_penalty": 5 * max(0, previous.fuel_fraction - sample.fuel_fraction), "time_penalty": .02 * dt}
        return terms["progress_reward"] - terms["fuel_penalty"] - terms["time_penalty"], terms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "evaluate"))
    parser.add_argument("--save", action="append", help="Repeat for multiple prepared descent saves")
    parser.add_argument("--return-mission", type=Path, default=Path("examples/return/mission.json"))
    parser.add_argument("--model", type=Path, default=LATEST)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--episodes", type=int, default=6)
    args = parser.parse_args(argv)
    if args.timesteps <= 0 or args.episodes <= 0 or args.model.suffix != ".zip":
        parser.error("Positive steps/episodes and a .zip model path are required")
    mission = ReturnMission.from_dict(json.loads(args.return_mission.read_text(encoding="utf-8")))
    saves = args.save or ["descent_training_start"]
    metadata = {"contract": policy_contract(mission.profile), "saves": saves,
                "landing_task": mission.landing.to_dict() if mission.landing else None}
    resume = args.model.is_file()
    model = None
    if args.mode != "check" and resume:
        saved = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
        if saved != metadata:
            parser.error("Checkpoint training configuration changed; use its original settings or a new --model path")
        model = load_policy(args.model, mission.profile)
    if args.mode == "evaluate" and model is None:
        parser.error("Train a descent_v2 policy before evaluation")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = DescentEnv(saves, mission.landing, mission.profile)
    try:
        if args.mode == "check":
            check_env(env, warn=True, skip_render_check=True)
            print("Descent interface check passed; flight performance remains unverified.")
            return
        monitored = Monitor(env, str(OUTPUT / f"episodes_{time.time_ns()}"), info_keywords=("is_success", "reason"))
        if model is None:
            model = PPO("MlpPolicy", monitored, n_steps=1024, batch_size=128, learning_rate=3e-4,
                        gamma=.995, ent_coef=.005, policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                        verbose=1, device="cpu", seed=42)
        else:
            model.set_env(monitored)
        model.tensorboard_log = str(OUTPUT / "tensorboard")
        if args.mode == "train":
            args.model.parent.mkdir(parents=True, exist_ok=True)
            args.model.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

            class Checkpoints(CheckpointCallback):
                def _on_step(self):
                    result = super()._on_step()
                    if self.n_calls % self.save_freq == 0:
                        sidecar = Path(self.save_path) / f"{self.name_prefix}_{self.num_timesteps}_steps.json"
                        sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                    return result

            try:
                model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not resume, tb_log_name="descent_v2",
                            callback=Checkpoints(save_freq=5000, save_path=str(OUTPUT / "checkpoints"), name_prefix=args.model.stem))
            finally:
                temporary = args.model.with_name(args.model.stem + "_saving.zip")
                model.save(str(temporary))
                temporary.replace(args.model)
                print(f"Descent model saved: {args.model}")
        else:
            results = []
            for i in range(args.episodes):
                observation, _ = env.reset(seed=1000 + i)
                done = False
                total = 0
                while not done:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminal, truncated, info = env.step(action)
                    total += reward
                    done = terminal or truncated
                results.append({**info, "reward": total, "save": env.save_name})
            path = OUTPUT / f"evaluation_{time.time_ns()}.json"
            path.write_text(json.dumps({"model": str(args.model), "episodes": results}, indent=2) + "\n", encoding="utf-8")
            print(f"Successes: {sum(r['is_success'] for r in results)}/{len(results)} | {path}")
    except KeyboardInterrupt:
        print("Descent run interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
