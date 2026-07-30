
from __future__ import annotations

import os
import time
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import krpc
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


# =============================================================================
# SETTINGS — CHANGE THESE VALUES FIRST
# =============================================================================

# Choose: "check", "train", or "evaluate"
RUN_MODE = "train"

# Name of the KSP save loaded at the beginning of every episode.
KSP_SAVE_NAME = "orbit_training_start"

# Training output folders.
OUTPUT_DIRECTORY = "orbit_correction_throttle_only_output"
CHECKPOINT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "checkpoints")
TENSORBOARD_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "tensorboard")
FINAL_MODEL_PATH = os.path.join(OUTPUT_DIRECTORY, "orbit_correction_throttle_only_final")

# Used only when RUN_MODE = "evaluate".
MODEL_TO_EVALUATE = FINAL_MODEL_PATH + ".zip"
EVALUATION_EPISODES = 20

# PPO training length.
TOTAL_TIMESTEPS = 500_000

# One environment step waits this many real seconds for KSP physics.
STEP_DURATION = 0.20

# Maximum number of steps in one episode.
MAX_EPISODE_STEPS = 2500

# Wait after loading a KSP save.
SAVE_LOAD_WAIT = 3.0

# Target apoapsis range for the first curriculum.
TARGET_APOAPSIS_MIN = 90_000.0
TARGET_APOAPSIS_MAX = 180_000.0

# The target must be at least this far above the starting apoapsis.
MINIMUM_APOAPSIS_INCREASE = 10_000.0

# Success means both orbital values are within this tolerance.
SUCCESS_TOLERANCE = 2_000.0

# Kerbin atmosphere ends at 70 km.
MINIMUM_SAFE_PERIAPSIS = 70_000.0

# Engines are blocked until the vessel is pointing close enough to prograde.
MINIMUM_ALIGNMENT = 0.97

# Resource names used to calculate fuel fraction.
FUEL_RESOURCES = ("LiquidFuel", "Oxidizer")

# Random seed.
SEED = 42


# =============================================================================
# GYMNASIUM ENVIRONMENT
# =============================================================================

class OrbitCorrectionEnv(gym.Env):
    """
    PPO environment for raising apoapsis from an existing orbit.

    Action:
        action[0] = throttle command
             0 -> zero throttle / coast
             1 -> full throttle

        The action space intentionally has only one value. KSP SAS is kept
        permanently in prograde mode for this first curriculum. PPO learns
        when to burn and how much throttle to use.

    Observation:
        0  normalized current apoapsis
        1  normalized current periapsis
        2  normalized target apoapsis
        3  normalized target periapsis
        4  normalized apoapsis error
        5  normalized periapsis error
        6  time to apoapsis as fraction of orbital period
        7  time to periapsis as fraction of orbital period
        8  fuel fraction
        9  normalized orbital speed
        10 previous applied throttle
        11 current mass fraction
        12 prograde alignment
        13 normalized acceleration capability (available thrust / mass)
    """

    def __init__(self) -> None:
        super().__init__()

        # PPO controls throttle only. SAS handles prograde attitude.
        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(14,),
            dtype=np.float32,
        )

        print("Connecting to KSP through kRPC...")
        self.connection = krpc.connect(name="PPO Orbit Correction")
        self.space_center = self.connection.space_center

        self.vessel: Any = None
        self.control: Any = None
        self.orbit: Any = None
        self.flight: Any = None

        self.target_apoapsis = 0.0
        self.target_periapsis = 0.0

        self.initial_fuel = 1.0
        self.initial_mass = 1.0
        self.previous_total_error = 0.0
        self.previous_throttle = 0.0
        self.episode_steps = 0

        self.altitude_scale = 250_000.0
        self.error_scale = 150_000.0
        self.speed_scale = 4_000.0
        self.acceleration_scale = 30.0

    # -------------------------------------------------------------------------
    # KSP SETUP
    # -------------------------------------------------------------------------

    def _load_training_save(self) -> None:
        """Load the same starting orbit at the beginning of every episode."""

        self.space_center.load(KSP_SAVE_NAME)
        time.sleep(SAVE_LOAD_WAIT)

        self.vessel = self.space_center.active_vessel
        self.control = self.vessel.control
        self.orbit = self.vessel.orbit

        reference_frame = self.orbit.body.non_rotating_reference_frame
        self.flight = self.vessel.flight(reference_frame)

        self.space_center.rails_warp_factor = 0
        self.space_center.physics_warp_factor = 0

        self.control.throttle = 0.0
        self.control.sas = True

        try:
            self.control.speed_mode = self.space_center.SpeedMode.orbit
        except Exception:
            pass

        self._set_prograde_sas()

    # -------------------------------------------------------------------------
    # TELEMETRY
    # -------------------------------------------------------------------------

    @staticmethod
    def _safe_number(value: float, fallback: float = 0.0) -> float:
        """Replace NaN or infinity with a safe finite value."""

        value = float(value)
        return value if np.isfinite(value) else fallback

    def _fuel_amount(self) -> float:
        """Return total LiquidFuel + Oxidizer currently on the vessel."""

        total = 0.0

        for resource_name in FUEL_RESOURCES:
            try:
                total += float(self.vessel.resources.amount(resource_name))
            except Exception:
                pass

        return total

    def _fuel_fraction(self) -> float:
        """Return remaining fuel between 0 and 1."""

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

        denominator = np.linalg.norm(first) * np.linalg.norm(second)

        if denominator < 1e-10:
            return 0.0

        result = np.dot(first, second) / denominator
        return float(np.clip(result, -1.0, 1.0))

    def _get_prograde_alignment(self) -> float:
        """Measure how closely the vessel is pointing prograde."""

        return self._alignment(
            self.flight.direction,
            self.flight.prograde,
        )

    def _telemetry(self) -> dict[str, float]:
        """Read all KSP values needed by the environment."""

        period = max(
            self._safe_number(self.orbit.period, fallback=1.0),
            1.0,
        )

        mass = max(self._safe_number(self.vessel.mass, fallback=1.0), 1.0)
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
        """Raise apoapsis while keeping periapsis near its starting value."""

        starting_apoapsis = telemetry["apoapsis"]
        starting_periapsis = telemetry["periapsis"]

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
        self.target_periapsis = float(starting_periapsis)

    def _errors(
        self,
        telemetry: dict[str, float],
    ) -> tuple[float, float, float]:
        """Return apoapsis error, periapsis error, and combined error."""

        apoapsis_error = self.target_apoapsis - telemetry["apoapsis"]
        periapsis_error = self.target_periapsis - telemetry["periapsis"]
        total_error = abs(apoapsis_error) + abs(periapsis_error)

        return apoapsis_error, periapsis_error, total_error

    def _observation(
        self,
        telemetry: dict[str, float],
    ) -> np.ndarray:
        """Convert raw KSP values into the normalized PPO observation."""

        apoapsis_error, periapsis_error, _ = self._errors(telemetry)
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
                telemetry["periapsis"] / self.altitude_scale,     # 1
                self.target_apoapsis / self.altitude_scale,        # 2
                self.target_periapsis / self.altitude_scale,       # 3
                apoapsis_error / self.error_scale,                 # 4
                periapsis_error / self.error_scale,               # 5
                time_to_apoapsis,                                 # 6
                time_to_periapsis,                                # 7
                telemetry["fuel_fraction"],                       # 8
                telemetry["speed"] / self.speed_scale,            # 9
                self.previous_throttle,                            # 10
                mass_fraction,                                    # 11
                prograde_alignment,                               # 12
                telemetry["acceleration_capability"]
                / self.acceleration_scale,                        # 13
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
    # ACTIONS
    # -------------------------------------------------------------------------

    def _set_prograde_sas(self) -> None:
        """Enable SAS and keep the vessel pointing prograde."""

        self.control.sas = True
        time.sleep(0.05)

        try:
            self.control.sas_mode = self.space_center.SASMode.prograde
        except Exception as exc:
            raise RuntimeError(
                "Could not enable prograde SAS. Ensure the vessel has "
                "SAS capability and the prograde mode is available."
            ) from exc

    def _apply_action(
        self,
        action: np.ndarray,
    ) -> dict[str, float | str | bool]:
        """Convert PPO's single output into a safe throttle command."""

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (1,):
            raise ValueError(
                f"Expected one throttle action, received shape {action.shape}."
            )

        action = np.clip(action, 0.0, 1.0)

        # The PPO action already uses KSP's 0..1 throttle range.
        requested_throttle = float(action[0])

        prograde_alignment = self._get_prograde_alignment()
        aligned = prograde_alignment >= MINIMUM_ALIGNMENT

        # Safety guard: never burn while SAS is still rotating to prograde.
        applied_throttle = requested_throttle if aligned else 0.0

        self.control.throttle = applied_throttle
        self.previous_throttle = applied_throttle

        return {
            "mode": "prograde",
            "alignment": prograde_alignment,
            "aligned": aligned,
            "requested_throttle": requested_throttle,
            "applied_throttle": applied_throttle,
        }

    # -------------------------------------------------------------------------
    # REWARD
    # -------------------------------------------------------------------------

    def _reward(
        self,
        telemetry: dict[str, float],
        action_info: dict[str, float | str | bool],
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        """Reward reductions in orbital error and penalize waste/failure."""

        apoapsis_error, periapsis_error, total_error = self._errors(telemetry)
        improvement = self.previous_total_error - total_error

        progress_reward = float(
            np.clip(improvement / 1_000.0, -5.0, 5.0)
        )

        applied_throttle = float(action_info["applied_throttle"])
        fuel_penalty = applied_throttle * 0.003
        time_penalty = 0.001

        success = (
            abs(apoapsis_error) <= SUCCESS_TOLERANCE
            and abs(periapsis_error) <= SUCCESS_TOLERANCE
        )

        unsafe_orbit = telemetry["periapsis"] < MINIMUM_SAFE_PERIAPSIS
        out_of_fuel = telemetry["fuel_fraction"] <= 0.001 and not success
        no_thrust = telemetry["available_thrust"] <= 0.0 and not success
        failure = unsafe_orbit or out_of_fuel or no_thrust

        reward = progress_reward - fuel_penalty - time_penalty

        if success:
            reward += 100.0
        if failure:
            reward -= 100.0

        self.previous_total_error = total_error

        info = {
            "success": success,
            "unsafe_orbit": unsafe_orbit,
            "out_of_fuel": out_of_fuel,
            "no_thrust": no_thrust,
            "apoapsis_error": apoapsis_error,
            "periapsis_error": periapsis_error,
            "total_error": total_error,
            "progress_reward": progress_reward,
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
        """Start a new episode from the saved orbit."""

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
                "The vessel has no available thrust. Activate the orbital "
                "engine before creating the training save."
            )

        self._choose_target(telemetry)
        _, _, self.previous_total_error = self._errors(telemetry)
        observation = self._observation(telemetry)

        info = {
            "starting_apoapsis": telemetry["apoapsis"],
            "starting_periapsis": telemetry["periapsis"],
            "target_apoapsis": self.target_apoapsis,
            "target_periapsis": self.target_periapsis,
        }

        print(
            "\nNew episode:"
            f"\n  Start:  {telemetry['periapsis'] / 1000:.1f} x "
            f"{telemetry['apoapsis'] / 1000:.1f} km"
            f"\n  Target: {self.target_periapsis / 1000:.1f} x "
            f"{self.target_apoapsis / 1000:.1f} km"
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

        info = {
            **action_info,
            **reward_info,
            "current_apoapsis": telemetry["apoapsis"],
            "current_periapsis": telemetry["periapsis"],
            "target_apoapsis": self.target_apoapsis,
            "target_periapsis": self.target_periapsis,
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
# TRAINING
# =============================================================================

def train_agent() -> None:
    """Create a new PPO model and train it."""

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(CHECKPOINT_DIRECTORY, exist_ok=True)
    os.makedirs(TENSORBOARD_DIRECTORY, exist_ok=True)

    env = Monitor(OrbitCorrectionEnv())

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=CHECKPOINT_DIRECTORY,
        name_prefix="orbit_correction_throttle_only",
    )

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
        tensorboard_log=TENSORBOARD_DIRECTORY,
        verbose=1,
        seed=SEED,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
        )
        model.save(FINAL_MODEL_PATH)
        print(f"\nTraining finished. Model saved to: {FINAL_MODEL_PATH}.zip")
    finally:
        env.close()


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_agent() -> None:
    """Run the trained model without exploration."""

    env = OrbitCorrectionEnv()
    model = PPO.load(MODEL_TO_EVALUATE)

    successes = 0
    final_errors: list[float] = []

    try:
        for episode in range(EVALUATION_EPISODES):
            observation, reset_info = env.reset(seed=SEED + episode)

            terminated = False
            truncated = False
            final_info: dict[str, Any] = {}

            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                (
                    observation,
                    _reward,
                    terminated,
                    truncated,
                    final_info,
                ) = env.step(action)

            success = bool(final_info.get("success", False))
            successes += int(success)

            final_error = float(final_info.get("total_error", np.nan))
            final_errors.append(final_error)

            print(
                f"\nEvaluation episode {episode + 1}:"
                f"\n  Success: {success}"
                f"\n  Final combined error: {final_error:.0f} m"
                f"\n  Fuel remaining: "
                f"{float(final_info.get('fuel_fraction', 0.0)):.3f}"
                f"\n  Target: {reset_info['target_periapsis'] / 1000:.1f} x "
                f"{reset_info['target_apoapsis'] / 1000:.1f} km"
            )

        success_rate = 100.0 * successes / EVALUATION_EPISODES

        print(
            "\nEvaluation summary:"
            f"\n  Success rate: {successes}/{EVALUATION_EPISODES} "
            f"({success_rate:.1f}%)"
            f"\n  Mean final combined error: "
            f"{np.nanmean(final_errors):.0f} m"
        )
    finally:
        env.close()


# =============================================================================
# ENVIRONMENT CHECK
# =============================================================================

def check_environment() -> None:
    """Verify that the environment follows the Gymnasium interface."""

    env = OrbitCorrectionEnv()

    try:
        check_env(env, warn=True, skip_render_check=True)
        print("\nThe environment passed the Stable-Baselines3 checker.")
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