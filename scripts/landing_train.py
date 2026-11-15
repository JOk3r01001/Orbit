"""Stage-one vertical landing PPO: check, train or evaluate a prepared descent save."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from mission_control.commands import LandingTarget, MissionTask, TaskType
from mission_control.krpc_adapter import connect_orbital_client
from mission_control.landing import LandingSample, descent_reference, distance_to_target, touchdown_failure, transition_reward


OUTPUT = Path("training_output/landing_v1")
LATEST = OUTPUT / "landing_latest.zip"
OBSERVATION_VERSION = 1


class LandingEnv(gym.Env):
    """Throttle-only curriculum. Upright autopilot, no staging or parachutes.

    Observation: height, vertical/horizontal speed, tilt, fuel, thrust/mass,
    gravity, reference descent speed, last throttle, elapsed time, target
    vertical/horizontal limits, target distance and target tilt limit (14).
    """

    def __init__(self, save_name="landing_training_start", task=None, connection=None):
        super().__init__()
        if task is not None and task.task != TaskType.LAND:
            raise ValueError("Expected a land mission")
        self.action_space = spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.connection = connection or connect_orbital_client()
        self.sc = self.connection.space_center
        self.save_name, self.task = save_name, task
        self.vessel = None
        self.done = True

    def stop(self):
        if self.vessel is not None:
            errors = []
            for obj, name, value in ((self.vessel.control, "throttle", 0), (self.vessel.auto_pilot, "engaged", False)):
                try:
                    setattr(obj, name, value)
                except Exception as exc:
                    errors.append(str(exc))
            if errors:
                raise RuntimeError("Cutoff could not be confirmed: " + "; ".join(errors))

    def read(self):
        if self.sc.active_vessel != self.vessel or self.vessel.orbit.body != self.body:
            raise RuntimeError("Active vessel or body changed")
        if self.sc.rails_warp_factor or self.sc.physics_warp_factor:
            raise RuntimeError("Landing training requires normal time speed")
        flight = self.vessel.flight(self.body.reference_frame)
        attitude = self.vessel.flight(self.vessel.surface_reference_frame)
        mass = float(self.vessel.mass)
        if mass <= 0:
            raise RuntimeError("Invalid vessel mass")
        return LandingSample(
            ut_s=float(self.sc.ut), height_m=float(flight.surface_altitude),
            vertical_mps=float(flight.vertical_speed), horizontal_mps=float(flight.horizontal_speed),
            tilt_deg=90 - float(attitude.pitch),
            fuel_fraction=min(1, max(0, float(self.vessel.resources.amount("LiquidFuel")) / self.initial_fuel)),
            acceleration_mps2=float(self.vessel.available_thrust) / mass,
            gravity_mps2=float(self.body.gravitational_parameter) / (float(self.body.equatorial_radius) + float(flight.mean_altitude)) ** 2,
            latitude_deg=float(flight.latitude), longitude_deg=float(flight.longitude),
            situation=self.vessel.situation.name,
            intact=len(self.vessel.parts.all) >= self.initial_parts,
        )

    def observation(self, s):
        t = self.target
        return np.asarray([
            s.height_m / 500, s.vertical_mps / 50, s.horizontal_mps / 5,
            s.tilt_deg / 90, s.fuel_fraction, s.acceleration_mps2 / 30,
            s.gravity_mps2 / 10, descent_reference(s, t) / 50, self.throttle,
            (s.ut_s - self.started_ut) / self.timeout,
            t.max_vertical_speed_mps / 5, t.max_horizontal_speed_mps / 5,
            distance_to_target(s, t) / t.radius_m, t.max_tilt_deg / 30,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("Configure the mission with --mission, not reset options")
        try:
            self.stop()
            self.vessel = None  # A loaded save invalidates remote vessel references.
            self.sc.load(self.save_name)
            time.sleep(2)
            self.vessel = self.sc.active_vessel
            self.vessel.control.throttle = 0
            self.body = self.vessel.orbit.body
            if self.body.name != "Kerbin":
                raise ValueError("Landing stage one supports Kerbin only")
            self.initial_fuel = float(self.vessel.resources.amount("LiquidFuel"))
            self.initial_parts = len(self.vessel.parts.all)
            if self.initial_fuel <= 0:
                raise ValueError("A LiquidFuel engine is required")
            sample = self.read()
            if not (100 <= sample.height_m <= 1000 and -40 <= sample.vertical_mps < 0
                    and sample.horizontal_mps <= .5 and sample.tilt_deg <= 10
                    and sample.situation in {"flying", "sub_orbital"}):
                raise ValueError("Save must be descending at 100-1000 m AGL, -40..0 m/s vertical, <=0.5 m/s horizontal, <=10 degrees tilt")
            if sample.acceleration_mps2 < 1.3 * sample.gravity_mps2:
                raise ValueError("Keep a throttleable landing engine active with thrust/weight >=1.3")
            stopping_distance = sample.vertical_mps ** 2 / (2 * (sample.acceleration_mps2 - sample.gravity_mps2))
            if stopping_distance > .6 * sample.height_m:
                raise ValueError("Descent save is too fast/low for the initial curriculum")
            if any(chute.deployed for chute in self.vessel.parts.parachutes):
                raise ValueError("Retract/remove parachutes for powered-landing training")
            self.target = self.task.target if self.task else LandingTarget(sample.latitude_deg, sample.longitude_deg)
            if distance_to_target(sample, self.target) > self.target.radius_m:
                raise ValueError("Throttle-only stage cannot steer to a distant landing site")
            self.timeout = min(180, self.task.timeout_s) if self.task else 180
            self.settle_time = self.task.settle_time_s if self.task else 3
            if self.settle_time >= self.timeout:
                raise ValueError("Settling time must be shorter than the stage-one timeout (maximum 180 seconds)")
            self.reserve = self.task.minimum_fuel_fraction if self.task else 0
            self.started_ut = sample.ut_s
            self.previous = sample
            self.steps, self.throttle = 0, 0
            self.contact = None
            self.done = False
            ap = self.vessel.auto_pilot
            ap.reference_frame = self.vessel.surface_reference_frame
            ap.target_pitch_and_heading(90, 90)
            ap.engaged = True
            return self.observation(sample), {"target": asdict(self.target)}
        except BaseException:
            self.done = True
            self.stop()
            raise

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset the environment before stepping")
        try:
            action = np.asarray(action, dtype=float)
            if action.shape != (1,) or not np.isfinite(action).all():
                raise ValueError("Expected one finite throttle action")
            self.steps += 1
            # Alignment is an explicit safety intervention, never an autopilot burn.
            self.throttle = 0 if self.contact is not None or self.previous.tilt_deg > 15 else float(np.clip(action[0], 0, 1))
            self.vessel.control.throttle = self.throttle
            deadline = time.monotonic() + 3
            while True:
                time.sleep(.05)
                sample = self.read()
                dt = sample.ut_s - self.previous.ut_s
                if dt < 0 or dt > 2:
                    raise RuntimeError("Landing telemetry time rewound or jumped")
                if dt >= .1:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Landing simulation stopped advancing")
            reward, terms = transition_reward(self.previous, sample, self.target)
            reason, success = None, False
            if not sample.intact:
                reason = "vessel_damaged"
            elif sample.situation == "splashed":
                reason = "water_landing"
            elif sample.situation == "landed":
                if self.contact is None:
                    reason = touchdown_failure(self.previous, sample, self.target)
                    if reason is None:
                        self.contact = (sample.ut_s, abs(self.previous.vertical_mps), self.previous.horizontal_mps)
                if reason is None:
                    # Keep verifying upright, low-speed contact during settling.
                    reason = touchdown_failure(sample, sample, self.target)
                    success = reason is None and sample.ut_s - self.contact[0] >= self.settle_time
                    if success:
                        reason = "success"
                self.vessel.control.throttle = 0
                self.throttle = 0
            elif self.contact is not None:
                reason = "bounced_after_touchdown"
            elif sample.height_m > 1500 or sample.vertical_mps > 20 or sample.tilt_deg > 45:
                reason = "outside_training_envelope"
            elif sample.fuel_fraction < max(.001, self.reserve) or sample.acceleration_mps2 <= sample.gravity_mps2:
                reason = "insufficient_fuel_or_thrust"
            if sample.fuel_fraction < self.reserve:
                reason = "fuel_reserve_violated"
                success = False
            timed_out = sample.ut_s - self.started_ut >= self.timeout or self.steps >= 3000
            terminated = reason is not None
            truncated = timed_out and not terminated
            if truncated:
                reason = "timeout"
            if success:
                reward += 100
            elif terminated or truncated:
                reward -= 100
            info = {**terms, "is_success": success, "reason": reason or "running",
                    "height_m": sample.height_m, "vertical_mps": sample.vertical_mps,
                    "fuel_fraction": sample.fuel_fraction, "applied_throttle": self.throttle,
                    "alignment_intervention": self.previous.tilt_deg > 15,
                    "touchdown": self.contact}
            self.previous = sample
            self.done = terminated or truncated
            if self.done:
                self.stop()
                print(f"LANDING RESULT | {info['reason']} | v={sample.vertical_mps:.2f} m/s | fuel={sample.fuel_fraction:.3f}", flush=True)
            return self.observation(sample), float(reward), terminated, truncated, info
        except Exception:
            # A completely destroyed craft can invalidate every vessel RPC.
            # Continue only if a working server confirms it is gone.
            try:
                destroyed = self.vessel is not None and self.vessel not in self.sc.vessels
            except Exception:
                destroyed = False
            self.done = True
            if destroyed:
                self.vessel = None
                print("LANDING RESULT | vessel_destroyed", flush=True)
                return self.observation(self.previous), -100.0, True, False, {
                    "is_success": False, "reason": "vessel_destroyed",
                    "progress_reward": 0.0, "fuel_penalty": 0.0, "time_penalty": 0.0,
                    "height_m": self.previous.height_m, "vertical_mps": self.previous.vertical_mps,
                    "fuel_fraction": self.previous.fuel_fraction, "applied_throttle": self.throttle,
                    "alignment_intervention": False, "touchdown": self.contact,
                }
            self.stop()
            raise
        except BaseException:
            self.done = True
            self.stop()
            raise

    def close(self):
        try:
            self.stop()
        finally:
            self.connection.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "train", "evaluate"))
    parser.add_argument("--save", default="landing_training_start")
    parser.add_argument("--mission", type=Path, help="Optional land task; defaults to save's ground location")
    parser.add_argument("--model", type=Path, default=LATEST)
    parser.add_argument("--fresh", action="store_true", help="Start a new policy; use a new --model path to retain prior weights")
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--episodes", type=int, default=6)
    args = parser.parse_args(argv)
    if args.timesteps <= 0 or args.episodes <= 0:
        parser.error("Timesteps and episodes must be positive")
    if args.fresh and (args.mode != "train" or args.model.exists()):
        parser.error("--fresh requires train and a model path that does not already exist")
    if args.model.suffix != ".zip":
        parser.error("--model must end in .zip")
    task = MissionTask.from_dict(json.loads(args.mission.read_text(encoding="utf-8"))) if args.mission else None
    if task and task.task != TaskType.LAND:
        parser.error("--mission must contain a land task")
    if args.mode == "evaluate" and not args.model.is_file():
        parser.error("No landing checkpoint found; train first")
    resume = args.model.is_file() and not args.fresh
    config_path = args.model.with_suffix(".json")
    metadata = {"observation_version": OBSERVATION_VERSION, "save": args.save,
                "task": task.to_dict() if task else "save_ground_location",
                "curriculum": "vertical_throttle_only_v1"}
    if args.mode != "check" and resume and (not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != metadata):
        parser.error("Checkpoint metadata does not match this landing configuration; retain its .json sidecar")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = LandingEnv(args.save, task)
    model = None
    try:
        if args.mode == "check":
            check_env(env, warn=True, skip_render_check=True)
            print("Landing environment interface check passed; this is not a performance evaluation.")
            return
        monitored = Monitor(env, str(OUTPUT / f"episodes_{time.time_ns()}"), info_keywords=("is_success", "reason"))
        model = PPO.load(str(args.model), env=monitored, device="cpu") if resume else PPO(
            "MlpPolicy", monitored, n_steps=1024, batch_size=128, learning_rate=3e-4,
            gamma=.995, ent_coef=.005, policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
            tensorboard_log=str(OUTPUT / "tensorboard"), verbose=1, seed=42, device="cpu")
        if args.mode == "train":
            model.tensorboard_log = str(OUTPUT / "tensorboard")
            args.model.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            class LandingCheckpoint(CheckpointCallback):
                def _on_step(self):
                    keep_training = super()._on_step()
                    if self.n_calls % self.save_freq == 0:
                        path = Path(self.save_path) / f"{self.name_prefix}_{self.num_timesteps}_steps.json"
                        path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
                    return keep_training
            try:
                model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not resume,
                            tb_log_name="landing_v1", callback=LandingCheckpoint(
                                save_freq=5000, save_path=str(OUTPUT / "checkpoints"), name_prefix=args.model.stem))
            finally:
                # Runs on Ctrl+C and errors too; an OS crash still needs periodic checkpoints.
                temporary = args.model.with_name(args.model.stem + "_saving.zip")
                model.save(str(temporary))
                temporary.replace(args.model)
                print(f"Landing policy saved: {args.model}", flush=True)
        else:
            results = []
            for episode in range(args.episodes):
                obs, _ = monitored.reset(seed=1000 + episode)
                done = False
                total = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = monitored.step(action)
                    total += reward
                    done = terminated or truncated
                results.append({**info, "reward": total})
            report = OUTPUT / f"evaluation_{time.time_ns()}.json"
            report.write_text(json.dumps({"model": str(args.model), "episodes": results}, indent=2) + "\n", encoding="utf-8")
            print(f"Successes: {sum(r['is_success'] for r in results)}/{len(results)} | {report}")
    except KeyboardInterrupt:
        print("Landing run interrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
