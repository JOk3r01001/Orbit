
from __future__ import annotations

import os
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import krpc
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


# =============================================================================
# SETTINGS — CHANGE THESE VALUES FIRST
# =============================================================================

# Choose: "check", "train", or "evaluate".
RUN_MODE = "train"

# Named KSP save loaded at the beginning of every episode.
KSP_SAVE_NAME = "orbit_training_start"

# Separate output location for the apoapsis-only specialist.
OUTPUT_DIRECTORY = "apoapsis_specialist_output"
CHECKPOINT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "checkpoints")
TENSORBOARD_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "tensorboard")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIRECTORY, "apoapsis_specialist_final")

# Used only when RUN_MODE = "evaluate".
MODEL_TO_EVALUATE = FINAL_MODEL_PATH + ".zip"
EVALUATION_EPISODES = 20

# PPO training length.
TOTAL_TIMESTEPS = 250_000

# One environment step waits this many real seconds for KSP physics.
STEP_DURATION = 0.20

# Maximum number of steps in one episode.
MAX_EPISODE_STEPS = 2500

# Wait after loading a KSP save.
SAVE_LOAD_WAIT = 3.0

# Random apoapsis targets used by this first curriculum.
TARGET_APOAPSIS_MIN = 110_000.0
TARGET_APOAPSIS_MAX = 180_000.0
MINIMUM_APOAPSIS_INCREASE = 10_000.0

# Episode succeeds when apoapsis is within this distance of the target.
APOAPSIS_SUCCESS_TOLERANCE = 1_000.0

# Episode fails when apoapsis exceeds the target by more than this amount.
MAX_APOAPSIS_OVERSHOOT = 5_000.0

# Hard safety boundary only. Periapsis is not part of the learned goal/reward.
MINIMUM_SAFE_PERIAPSIS = 70_000.0

# Engines remain blocked until SAS points sufficiently close to prograde.
MINIMUM_ALIGNMENT = 0.97

# Throttle is limited near the target to reduce overshooting.
THROTTLE_TAPER_START = 20_000.0
THROTTLE_FINE_START = 5_000.0
MAX_THROTTLE_TAPER = 0.30
MAX_THROTTLE_FINE = 0.10

# Reward parameters.
PROGRESS_REWARD_METERS = 1_000.0
PROGRESS_REWARD_CLIP = 2.0
FUEL_PENALTY_SCALE = 0.003
TIME_PENALTY = 0.001
SUCCESS_REWARD = 100.0
FAILURE_PENALTY = 150.0

# Resource names used to calculate fuel fraction.
FUEL_RESOURCES = ("LiquidFuel", "Oxidizer")

# Random seed.
SEED = 42


# =============================================================================
# GYMNASIUM ENVIRONMENT
# =============================================================================

class ApoapsisSpecialistEnv(gym.Env):
    """
    Goal-conditioned PPO environment for raising apoapsis only.

    Action:
        action[0] = throttle command in [0, 1]
                    0 -> coast
                    1 -> full requested throttle

        KSP SAS remains in prograde mode. The environment may reduce or block
        the requested throttle for alignment and overshoot safety.

    Observations (12):
        0   normalized current apoapsis
        1   normalized current periapsis (safety/context only)
        2   normalized target apoapsis
        3   normalized signed apoapsis error
        4   normalized time to apoapsis
        5   normalized time to periapsis
        6   fuel fraction
        7   normalized orbital speed
        8   previous applied throttle
        9   current mass fraction
        10  prograde alignment
        11  normalized acceleration capability (available thrust / mass)

    Periapsis is deliberately excluded from the goal, success condition, and
    progress reward. It is retained only as a physical safety observation and
    termination boundary. A separate future model should specialize in
    periapsis changes.
    """

    def __init__(self) -> None:
        super().__init__()

        # PPO controls one continuous value: throttle.
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32,
        )

        print("Connecting to KSP through kRPC...")
        self.connection = krpc.connect(name="PPO Apoapsis Specialist")
        self.space_center = self.connection.space_center

        self.vessel: Any = None
        self.control: Any = None
        self.orbit: Any = None
        self.flight: Any = None

        self.target_apoapsis = 0.0
        self.starting_periapsis = 0.0

        self.initial_fuel = 1.0
        self.initial_mass = 1.0
        self.previous_absolute_apoapsis_error = 0.0
        self.previous_throttle = 0.0
        self.episode_steps = 0

        # Observation normalization scales.
        self.altitude_scale = 250_000.0
        self.error_scale = 150_000.0
        self.speed_scale = 4_000.0
        self.acceleration_scale = 30.0

    # -------------------------------------------------------------------------
    # KSP SETUP
    # -------------------------------------------------------------------------

    def _load_training_save(self) -> None:
        """Load the same starting state at the beginning of every episode."""

        self.space_center.load(KSP_SAVE_NAME)
        time.sleep(SAVE_LOAD_WAIT)

        # Loading a save invalidates old kRPC object references.
        self.vessel = self.space_center.active_vessel
        self.control = self.vessel.control
        self.orbit = self.vessel.orbit

        self._activate_available_engines()

        # Body-centered non-rotating frame gives meaningful orbital velocity.
        reference_frame = self.orbit.body.non_rotating_reference_frame
        self.flight = self.vessel.flight(reference_frame)

        self.space_center.rails_warp_factor = 0
        self.space_center.physics_warp_factor = 0

        self.control.throttle = 0.0
        self.control.sas = True

        try:
            self.control.speed_mode = self.space_center.SpeedMode.orbit
        except Exception:
            # This only changes the navball speed mode.
            pass

        self._set_prograde_sas()

    def _activate_available_engines(self) -> None:
        """Activate fueled engines so available_thrust is non-zero."""

        engines = self.vessel.parts.engines

        if not engines:
            raise RuntimeError("No engines were found on the vessel.")

        for engine in engines:
            if engine.has_fuel and engine.max_thrust > 0.0:
                try:
                    engine.active = True
                except Exception:
                    pass

        time.sleep(0.25)

        if self.vessel.available_thrust <= 0.0:
            raise RuntimeError(
                "No usable engine could be activated. Check engine fuel, "
                "staging, fuel flow, and engine status."
            )

    def _set_prograde_sas(self) -> None:
        """Enable SAS and keep the vessel pointing prograde."""

        self.control.sas = True
        time.sleep(0.05)

        try:
            self.control.sas_mode = self.space_center.SASMode.prograde
        except Exception as exc:
            raise RuntimeError(
                "Could not enable prograde SAS. Ensure the vessel has SAS "
                "capability and prograde mode is available."
            ) from exc

    # -------------------------------------------------------------------------
    # TELEMETRY
    # -------------------------------------------------------------------------

    @staticmethod
    def _safe_number(value: float, fallback: float = 0.0) -> float:
        """Replace NaN or infinity with a finite fallback."""

        value = float(value)
        return value if np.isfinite(value) else fallback

    def _fuel_amount(self) -> float:
        """Return total configured propellant currently on the vessel."""

        total = 0.0

        for resource_name in FUEL_RESOURCES:
            try:
                total += float(self.vessel.resources.amount(resource_name))
            except Exception:
                pass

        return total

    def _fuel_fraction(self) -> float:
        """Return remaining fuel as a value from 0 to 1."""

        if self.initial_fuel <= 0.0:
            return 0.0

        return float(
            np.clip(
                self._fuel_amount() / self.initial_fuel,
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _alignment(
        first_vector: tuple[float, float, float],
        second_vector: tuple[float, float, float],
    ) -> float:
        """Return directional alignment from -1 to 1."""

        first = np.asarray(first_vector, dtype=np.float64)
        second = np.asarray(second_vector, dtype=np.float64)

        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))

        if denominator < 1e-10:
            return 0.0

        result = np.dot(first, second) / denominator
        return float(np.clip(result, -1.0, 1.0))

    def _get_prograde_alignment(self) -> float:
        """Measure how closely the vessel points prograde."""

        return self._alignment(
            self.flight.direction,
            self.flight.prograde,
        )

    def _telemetry(self) -> dict[str, float]:
        """Read all KSP values needed by this environment."""

        period = max(
            self._safe_number(self.orbit.period, fallback=1.0),
            1.0,
        )

        mass = max(
            self._safe_number(self.vessel.mass, fallback=1.0),
            1.0,
        )

        available_thrust = max(
            self._safe_number(self.vessel.available_thrust),
            0.0,
        )

        return {
            "apoapsis": self._safe_number(self.orbit.apoapsis_altitude),
            "periapsis": self._safe_number(self.orbit.periapsis_altitude),
            "period": period,
            "time_to_apoapsis": self._safe_number(
                self.orbit.time_to_apoapsis,
                fallback=period,
            ),
            "time_to_periapsis": self._safe_number(
                self.orbit.time_to_periapsis,
                fallback=period,
            ),
            "speed": self._safe_number(self.flight.speed),
            "fuel_fraction": self._fuel_fraction(),
            "mass": mass,
            "available_thrust": available_thrust,
            "acceleration_capability": available_thrust / mass,
        }

    # -------------------------------------------------------------------------
    # TARGET AND OBSERVATION
    # -------------------------------------------------------------------------

    def _choose_target(self, telemetry: dict[str, float]) -> None:
        """Choose a higher random apoapsis target for this episode."""

        starting_apoapsis = telemetry["apoapsis"]

        lowest_target = max(
            TARGET_APOAPSIS_MIN,
            starting_apoapsis + MINIMUM_APOAPSIS_INCREASE,
        )

        highest_target = max(
            TARGET_APOAPSIS_MAX,
            lowest_target + 1_000.0,
        )

        self.target_apoapsis = float(
            self.np_random.uniform(lowest_target, highest_target)
        )

    def _apoapsis_error(self, telemetry: dict[str, float]) -> float:
        """Return signed target AP minus current AP, in meters."""

        return self.target_apoapsis - telemetry["apoapsis"]

    def _observation(
        self,
        telemetry: dict[str, float],
    ) -> np.ndarray:
        """Convert raw KSP values into the normalized PPO observation."""

        apoapsis_error = self._apoapsis_error(telemetry)
        period = telemetry["period"]

        time_to_apoapsis = np.clip(
            telemetry["time_to_apoapsis"] / period,
            0.0,
            1.0,
        )

        time_to_periapsis = np.clip(
            telemetry["time_to_periapsis"] / period,
            0.0,
            1.0,
        )

        prograde_alignment = self._get_prograde_alignment()

        mass_fraction = np.clip(
            telemetry["mass"] / max(self.initial_mass, 1.0),
            0.0,
            1.0,
        )

        observation = np.array(
            [
                telemetry["apoapsis"] / self.altitude_scale,       # 0
                telemetry["periapsis"] / self.altitude_scale,     # 1 safety
                self.target_apoapsis / self.altitude_scale,        # 2
                apoapsis_error / self.error_scale,                 # 3
                time_to_apoapsis,                                 # 4
                time_to_periapsis,                                # 5
                telemetry["fuel_fraction"],                       # 6
                telemetry["speed"] / self.speed_scale,            # 7
                self.previous_throttle,                            # 8
                mass_fraction,                                    # 9
                prograde_alignment,                               # 10
                telemetry["acceleration_capability"]
                / self.acceleration_scale,                        # 11
            ],
            dtype=np.float32,
        )

        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )

        return np.clip(observation, -1.0, 1.0).astype(np.float32)

    # -------------------------------------------------------------------------
    # ACTION AND SAFETY GUARDS
    # -------------------------------------------------------------------------

    def _apply_action(
        self,
        action: np.ndarray,
    ) -> dict[str, float | str | bool]:
        """Convert PPO's throttle output into a guarded KSP command."""

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (1,):
            raise ValueError(
                f"Expected one throttle action, received shape {action.shape}."
            )

        action = np.clip(action, 0.0, 1.0)
        requested_throttle = float(action[0])

        current_apoapsis = self._safe_number(
            self.orbit.apoapsis_altitude
        )
        remaining_apoapsis = self.target_apoapsis - current_apoapsis

        prograde_alignment = self._get_prograde_alignment()
        aligned = prograde_alignment >= MINIMUM_ALIGNMENT

        # Hard cutoff: once target AP is reached, never continue burning.
        if remaining_apoapsis <= 0.0:
            applied_throttle = 0.0
            throttle_cap = 0.0
            throttle_guard = "target_reached"

        # Alignment guard: never burn while SAS is rotating to prograde.
        elif not aligned:
            applied_throttle = 0.0
            throttle_cap = 0.0
            throttle_guard = "waiting_for_alignment"

        # Fine-control zone immediately before the target.
        elif remaining_apoapsis <= THROTTLE_FINE_START:
            throttle_cap = MAX_THROTTLE_FINE
            applied_throttle = min(requested_throttle, throttle_cap)
            throttle_guard = "fine_taper"

        # Wider approach zone before the target.
        elif remaining_apoapsis <= THROTTLE_TAPER_START:
            throttle_cap = MAX_THROTTLE_TAPER
            applied_throttle = min(requested_throttle, throttle_cap)
            throttle_guard = "approach_taper"

        else:
            throttle_cap = 1.0
            applied_throttle = requested_throttle
            throttle_guard = "none"

        self.control.throttle = applied_throttle
        self.previous_throttle = applied_throttle

        return {
            "mode": "prograde",
            "alignment": prograde_alignment,
            "aligned": aligned,
            "requested_throttle": requested_throttle,
            "applied_throttle": applied_throttle,
            "throttle_cap": throttle_cap,
            "throttle_guard": throttle_guard,
            "remaining_apoapsis": remaining_apoapsis,
        }

    # -------------------------------------------------------------------------
    # APOAPSIS-ONLY REWARD
    # -------------------------------------------------------------------------

    def _reward(
        self,
        telemetry: dict[str, float],
        action_info: dict[str, float | str | bool],
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        """
        Reward only progress toward target apoapsis.

        Periapsis does not affect progress, success, or precision reward. It is
        checked only as a hard physical safety condition.
        """

        apoapsis_error = self._apoapsis_error(telemetry)
        absolute_apoapsis_error = abs(apoapsis_error)

        improvement = (
            self.previous_absolute_apoapsis_error
            - absolute_apoapsis_error
        )

        progress_reward = float(
            np.clip(
                improvement / PROGRESS_REWARD_METERS,
                -PROGRESS_REWARD_CLIP,
                PROGRESS_REWARD_CLIP,
            )
        )

        applied_throttle = float(action_info["applied_throttle"])
        fuel_penalty = applied_throttle * FUEL_PENALTY_SCALE
        time_penalty = TIME_PENALTY

        # Success depends only on apoapsis precision.
        success = absolute_apoapsis_error <= APOAPSIS_SUCCESS_TOLERANCE

        # Apoapsis-specific failure guard.
        apoapsis_overshoot = (
            telemetry["apoapsis"]
            > self.target_apoapsis + MAX_APOAPSIS_OVERSHOOT
        )

        # Periapsis remains safety-only, not a learned target.
        unsafe_periapsis = (
            telemetry["periapsis"] < MINIMUM_SAFE_PERIAPSIS
        )

        out_of_fuel = (
            telemetry["fuel_fraction"] <= 0.001
            and not success
        )

        no_thrust = (
            telemetry["available_thrust"] <= 0.0
            and not success
        )

        failure = (
            apoapsis_overshoot
            or unsafe_periapsis
            or out_of_fuel
            or no_thrust
        )

        reward = progress_reward - fuel_penalty - time_penalty

        if success:
            reward += SUCCESS_REWARD
        elif failure:
            reward -= FAILURE_PENALTY

        self.previous_absolute_apoapsis_error = absolute_apoapsis_error

        info = {
            "success": success,
            "is_success": int(success),
            "apoapsis_overshoot": apoapsis_overshoot,
            "unsafe_periapsis": unsafe_periapsis,
            "out_of_fuel": out_of_fuel,
            "no_thrust": no_thrust,
            "apoapsis_error": apoapsis_error,
            "absolute_apoapsis_error": absolute_apoapsis_error,
            "progress_reward": progress_reward,
            "fuel_penalty": fuel_penalty,
            "time_penalty": time_penalty,
            "fuel_fraction": telemetry["fuel_fraction"],
        }

        return reward, success, failure, info

    # -------------------------------------------------------------------------
    # REQUIRED GYMNASIUM METHODS
    # -------------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode from the named orbital save."""

        super().reset(seed=seed)
        self._load_training_save()

        self.episode_steps = 0
        self.previous_throttle = 0.0

        self.initial_fuel = self._fuel_amount()
        self.initial_mass = max(float(self.vessel.mass), 1.0)

        if self.initial_fuel <= 0.0:
            raise RuntimeError(
                "No LiquidFuel or Oxidizer was found. "
                "Check FUEL_RESOURCES at the top of the script."
            )

        telemetry = self._telemetry()

        if telemetry["periapsis"] < MINIMUM_SAFE_PERIAPSIS:
            raise RuntimeError(
                "The training save is not in a safe stable orbit."
            )

        if telemetry["available_thrust"] <= 0.0:
            raise RuntimeError(
                "The vessel has no available thrust after engine activation."
            )

        self.starting_periapsis = telemetry["periapsis"]
        self._choose_target(telemetry)

        self.previous_absolute_apoapsis_error = abs(
            self._apoapsis_error(telemetry)
        )

        observation = self._observation(telemetry)

        info = {
            "starting_apoapsis": telemetry["apoapsis"],
            "starting_periapsis": telemetry["periapsis"],
            "target_apoapsis": self.target_apoapsis,
        }

        print(
            "\nNew apoapsis-specialist episode:"
            f"\n  Start PE:  {telemetry['periapsis'] / 1000:.1f} km"
            f"\n  Start AP:  {telemetry['apoapsis'] / 1000:.1f} km"
            f"\n  Target AP: {self.target_apoapsis / 1000:.1f} km"
            "\n  PE role: safety only"
        )

        return observation, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply one PPO action and advance the KSP simulation."""

        self.episode_steps += 1
        action_info = self._apply_action(action)

        time.sleep(STEP_DURATION)

        telemetry = self._telemetry()
        observation = self._observation(telemetry)

        reward, success, failure, reward_info = self._reward(
            telemetry,
            action_info,
        )

        terminated = success or failure
        truncated = self.episode_steps >= MAX_EPISODE_STEPS

        if terminated or truncated:
            self.control.throttle = 0.0

            print(
                "\nEpisode ended:"
                f"\n  Success: {success}"
                f"\n  Overshoot: {reward_info['apoapsis_overshoot']}"
                f"\n  Unsafe PE: {reward_info['unsafe_periapsis']}"
                f"\n  Out of fuel: {reward_info['out_of_fuel']}"
                f"\n  No thrust: {reward_info['no_thrust']}"
                f"\n  Timed out: {truncated}"
                f"\n  Final AP: {telemetry['apoapsis'] / 1000:.1f} km"
                f"\n  Target AP: {self.target_apoapsis / 1000:.1f} km"
                f"\n  AP error: "
                f"{reward_info['absolute_apoapsis_error']:.0f} m"
                f"\n  Final PE: {telemetry['periapsis'] / 1000:.1f} km"
            )

        info = {
            **action_info,
            **reward_info,
            "current_apoapsis": telemetry["apoapsis"],
            "current_periapsis": telemetry["periapsis"],
            "target_apoapsis": self.target_apoapsis,
            "final_apoapsis_error": reward_info[
                "absolute_apoapsis_error"
            ],
            "episode_steps": self.episode_steps,
        }

        return (
            observation,
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        """Safely stop the engine and close kRPC."""

        try:
            if self.control is not None:
                self.control.throttle = 0.0
        except Exception:
            pass

        try:
            self.connection.close()
        except Exception:
            pass


# =============================================================================
# TENSORBOARD METRICS
# =============================================================================

class ApoapsisMetricsCallback(BaseCallback):
    """Write apoapsis-specialist metrics into the SB3 logger."""

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "absolute_apoapsis_error" not in info:
                continue

            self.logger.record(
                "apoapsis/error_m",
                float(info["absolute_apoapsis_error"]),
            )
            self.logger.record(
                "apoapsis/current_m",
                float(info["current_apoapsis"]),
            )
            self.logger.record(
                "apoapsis/target_m",
                float(info["target_apoapsis"]),
            )
            self.logger.record(
                "apoapsis/applied_throttle",
                float(info["applied_throttle"]),
            )
            self.logger.record(
                "apoapsis/alignment",
                float(info["alignment"]),
            )
            self.logger.record(
                "apoapsis/fuel_fraction",
                float(info["fuel_fraction"]),
            )

            if bool(info.get("success", False)):
                self.logger.record("apoapsis/success_event", 1.0)

            if bool(info.get("apoapsis_overshoot", False)):
                self.logger.record("apoapsis/overshoot_event", 1.0)

        return True


# =============================================================================
# TRAINING
# =============================================================================

def train_agent() -> None:
    """Create a new apoapsis-specialist PPO model and train it."""

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    os.makedirs(TENSORBOARD_DIRECTORY, exist_ok=True)

    raw_env = ApoapsisSpecialistEnv()

    env = Monitor(
        raw_env,
        info_keywords=(
            "is_success",
            "final_apoapsis_error",
            "fuel_fraction",
            "apoapsis_overshoot",
        ),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=CHECKPOINT_DIRECTORY,
        name_prefix="apoapsis_specialist",
    )

    metrics_callback = ApoapsisMetricsCallback()

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        device="cpu",
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
        tensorboard_log=TENSORBOARD_DIRECTORY,
        verbose=1,
        seed=SEED,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, metrics_callback],
        )

        model.save(FINAL_MODEL_PATH)

        print(
            f"\nTraining finished. Model saved to: "
            f"{FINAL_MODEL_PATH}.zip"
        )
    finally:
        env.close()


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_agent() -> None:
    """Evaluate the apoapsis specialist with deterministic actions."""

    env = ApoapsisSpecialistEnv()
    model = PPO.load(MODEL_TO_EVALUATE, device="cpu")

    successes = 0
    final_errors: list[float] = []
    remaining_fuel: list[float] = []

    try:
        for episode in range(EVALUATION_EPISODES):
            observation, reset_info = env.reset(seed=SEED + episode)

            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}

            while not (terminated or truncated):
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

                (
                    observation,
                    _reward,
                    terminated,
                    truncated,
                    final_info,
                ) = env.step(action)

            success = bool(final_info.get("success", False))
            successes += int(success)

            final_error = float(
                final_info.get("absolute_apoapsis_error", np.nan)
            )
            final_errors.append(final_error)

            fuel_fraction = float(
                final_info.get("fuel_fraction", np.nan)
            )
            remaining_fuel.append(fuel_fraction)

            print(
                f"\nEvaluation episode {episode + 1}:"
                f"\n  Success: {success}"
                f"\n  Final AP error: {final_error:.0f} m"
                f"\n  Fuel remaining: {fuel_fraction:.3f}"
                f"\n  Starting AP: "
                f"{reset_info['starting_apoapsis'] / 1000:.1f} km"
                f"\n  Target AP: "
                f"{reset_info['target_apoapsis'] / 1000:.1f} km"
            )

        success_rate = 100.0 * successes / EVALUATION_EPISODES

        print(
            "\nEvaluation summary:"
            f"\n  Success rate: {successes}/{EVALUATION_EPISODES} "
            f"({success_rate:.1f}%)"
            f"\n  Mean final AP error: "
            f"{np.nanmean(final_errors):.0f} m"
            f"\n  Mean remaining fuel: "
            f"{np.nanmean(remaining_fuel):.3f}"
        )
    finally:
        env.close()


# =============================================================================
# ENVIRONMENT CHECK
# =============================================================================

def check_environment() -> None:
    """Verify the Gymnasium interface using SB3's checker."""

    env = ApoapsisSpecialistEnv()

    try:
        check_env(env, warn=True, skip_render_check=True)
        print("\nThe environment passed the SB3 environment checker.")
    finally:
        env.close()


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================

def main() -> None:
    if RUN_MODE == "check":
        check_environment()
    elif RUN_MODE == "train":
        train_agent()
    elif RUN_MODE == "evaluate":
        evaluate_agent()
    else:
        raise ValueError(
            'RUN_MODE must be "check", "train", or "evaluate".'
        )


if __name__ == "__main__":
    main()
