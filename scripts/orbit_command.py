import os
import time

import gymnasium as gym
from gymnasium import spaces
import krpc
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


class KSPCommandedOrbitalEnv(gym.Env):
    """
    Goal-conditioned PPO environment for launching a multi-stage rocket
    into a commanded Kerbin orbit.

    Actions:
        action[0] = throttle, 0.0 to 1.0
        action[1] = pitch fraction, 0.0 to 1.0
                    converted to 0 to 90 degrees

    Observations:
        0  altitude
        1  vertical speed
        2  horizontal speed
        3  apoapsis
        4  periapsis
        5  time to apoapsis
        6  fuel fraction
        7  mass fraction
        8  pitch
        9  heading error
        10 throttle
        11 current stage
        12 commanded apoapsis
        13 commanded periapsis
        14 commanded minimum remaining fuel
        15 commanded urgency
    """

    metadata = {"render_modes": []}

    def __init__(self, randomize_commands=True):
        super().__init__()

        print("Connecting to KSP for commanded reinforcement learning...")
        self.conn = krpc.connect(name="PPO_Commanded_Trainer")

        # action[0] = throttle
        # action[1] = pitch fraction
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 12 telemetry values + 4 commander values
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(16,),
            dtype=np.float32,
        )

        # Environment timing
        self.dt = 0.1
        self.max_steps = 8000

        # Allowed commander target ranges
        self.randomize_commands = randomize_commands

        self.minimum_target_ap = 80_000.0
        self.maximum_target_ap = 180_000.0

        self.minimum_target_pe = 70_000.0

        self.maximum_minimum_fuel_fraction = 0.15

        # Default commander instruction
        self.command_target_ap = 100_000.0
        self.command_target_pe = 90_000.0
        self.command_min_fuel_fraction = 0.10
        self.command_urgency = 0.50

        # The pilot succeeds when it gets within these tolerances
        self.ap_tolerance = 10_000.0
        self.pe_tolerance = 10_000.0

        # Safety-intervention penalty strengths
        self.throttle_intervention_penalty_scale = 2.0
        self.pitch_intervention_penalty_scale = 2.0

        # KSP runtime objects
        self.vessel = None
        self.flight = None
        self.surface_flight = None
        self.body = None
        self.ref_frame = None

        # Episode state
        self.current_step = 0
        self.last_stage_time = 0.0
        self.stage_count_start = 1

        self.initial_fuel = 1.0
        self.initial_mass = 1.0

        self.prev_altitude = 0.0
        self.prev_ap = 0.0
        self.prev_pe = -600_000.0
        self.prev_horizontal_speed = 0.0

        # One-time reward milestones
        self.ap_above_atmosphere_awarded = False
        self.ap_near_target_awarded = False
        self.pe_positive_awarded = False
        self.pe_50k_awarded = False
        self.pe_near_target_awarded = False

    # --------------------------------------------------
    # RESET AND COMMAND SELECTION
    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Select a random training command or use an external command
        self._select_command(options)

        print("\nLoading launchpad save...")
        self.conn.space_center.load("launchpad_ready")
        time.sleep(2.0)

        self._rebind_vessel_objects()

        # Initial autopilot direction
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, 90)

        # Reset episode counters
        self.current_step = 0
        self.last_stage_time = time.time()

        self._reset_reward_milestones()

        # Store initial resources
        self.initial_fuel = max(
            1.0,
            self._get_liquid_fuel(),
        )

        self.initial_mass = max(
            1.0,
            float(self.vessel.mass),
        )

        self.stage_count_start = max(
            1,
            int(self.vessel.control.current_stage),
        )

        # Initial values used for delta rewards
        alt, _, horizontal_speed = self._get_speed_values()

        self.prev_altitude = alt
        self.prev_horizontal_speed = horizontal_speed
        self.prev_ap = self._safe_apoapsis()
        self.prev_pe = self._safe_periapsis()

        # Start launch
        self.vessel.control.throttle = 1.0
        time.sleep(0.5)

        # Try several stages until engine thrust appears
        for attempt in range(4):
            self._rebind_vessel_objects()

            if self.vessel.available_thrust > 0.1:
                print("Launch engine thrust detected.")
                break

            print(
                f"Initial launch staging attempt "
                f"{attempt + 1}..."
            )

            self.vessel.control.activate_next_stage()
            self.last_stage_time = time.time()

            time.sleep(1.0)

        self._rebind_vessel_objects()

        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, 90)
        self.vessel.control.throttle = 1.0

        observation = self._get_obs()
        info = self._command_info()

        return observation, info

    def _select_command(self, options):
        """
        During training, random commands are generated.

        During deployment, an external command can be supplied through:

        env.reset(
            options={
                "target_ap": 100000,
                "target_pe": 90000,
                "min_fuel_fraction": 0.05,
                "urgency": 0.5,
            }
        )
        """

        # External command supplied by an LLM or mission manager
        if options is not None and "target_ap" in options:
            self.command_target_ap = float(
                np.clip(
                    options["target_ap"],
                    self.minimum_target_ap,
                    self.maximum_target_ap,
                )
            )

            self.command_target_pe = float(
                np.clip(
                    options.get(
                        "target_pe",
                        self.minimum_target_pe,
                    ),
                    self.minimum_target_pe,
                    self.command_target_ap,
                )
            )

            self.command_min_fuel_fraction = float(
                np.clip(
                    options.get(
                        "min_fuel_fraction",
                        0.0,
                    ),
                    0.0,
                    self.maximum_minimum_fuel_fraction,
                )
            )

            self.command_urgency = float(
                np.clip(
                    options.get(
                        "urgency",
                        0.5,
                    ),
                    0.0,
                    1.0,
                )
            )

        # Random command used during PPO training
        elif self.randomize_commands:
            self.command_target_ap = float(
                self.np_random.uniform(
                    self.minimum_target_ap,
                    self.maximum_target_ap,
                )
            )

            self.command_target_pe = float(
                self.np_random.uniform(
                    self.minimum_target_pe,
                    self.command_target_ap,
                )
            )

            self.command_min_fuel_fraction = float(
                self.np_random.uniform(
                    0.0,
                    self.maximum_minimum_fuel_fraction,
                )
            )

            self.command_urgency = float(
                self.np_random.uniform(
                    0.0,
                    1.0,
                )
            )

        print(
            "MISSION COMMAND | "
            f"Target Ap: "
            f"{self.command_target_ap / 1000:.1f} km | "
            f"Target Pe: "
            f"{self.command_target_pe / 1000:.1f} km | "
            f"Minimum fuel: "
            f"{self.command_min_fuel_fraction:.2f} | "
            f"Urgency: "
            f"{self.command_urgency:.2f}"
        )

    def _reset_reward_milestones(self):
        self.ap_above_atmosphere_awarded = False
        self.ap_near_target_awarded = False
        self.pe_positive_awarded = False
        self.pe_50k_awarded = False
        self.pe_near_target_awarded = False

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, action):
        self.current_step += 1

        terminated = False
        truncated = False
        success = False

        # Read state before applying the next action
        alt, _, _ = self._get_speed_values()

        ap = self._safe_apoapsis()
        pe = self._safe_periapsis()
        time_to_ap = self._safe_time_to_apoapsis()

        # Read raw PPO commands
        raw_throttle = float(
            np.clip(
                action[0],
                0.0,
                1.0,
            )
        )

        raw_pitch_fraction = float(
            np.clip(
                action[1],
                0.0,
                1.0,
            )
        )
        
        # Pitch requested directly by PPO, before safety correction
        requested_pitch = raw_pitch_fraction * 90.0

        # Apply safety/control helpers
        throttle_cmd = self._compute_safe_throttle(
            raw_throttle=raw_throttle,
            alt=alt,
            ap=ap,
            pe=pe,
            time_to_ap=time_to_ap,
        )

        pitch_cmd = self._compute_safe_pitch(
            raw_pitch_fraction=raw_pitch_fraction,
            alt=alt,
            ap=ap,
            pe=pe,
            time_to_ap=time_to_ap,
        )

        # Measure how strongly the safety layer corrected PPO's action
        throttle_intervention = abs(
            throttle_cmd - raw_throttle
        )

        # Normalize pitch difference to approximately 0.0–1.0
        pitch_intervention = abs(
            pitch_cmd - requested_pitch
        ) / 90.0

        safety_intervention_penalty = (
            self.throttle_intervention_penalty_scale
            * throttle_intervention
            + self.pitch_intervention_penalty_scale
            * pitch_intervention
        )

        # Send commands to KSP
        self.vessel.control.throttle = throttle_cmd

        self.vessel.auto_pilot.target_pitch_and_heading(
            pitch_cmd,
            90,
        )

        # Automated staging
        self._handle_staging(throttle_cmd)

        time.sleep(self.dt)

        # Observation after the physics step
        observation = self._get_obs()

        alt = float(self.flight.surface_altitude)
        pitch = float(self.surface_flight.pitch)
        vertical_speed = float(self.flight.vertical_speed)
        speed = float(self.flight.speed)

        horizontal_speed = self._horizontal_speed(
            speed,
            vertical_speed,
        )

        ap = self._safe_apoapsis()
        pe = self._safe_periapsis()
        time_to_ap = self._safe_time_to_apoapsis()
        fuel_fraction = self._current_fuel_fraction()

        reward = self._compute_reward(
            alt=alt,
            vertical_speed=vertical_speed,
            horizontal_speed=horizontal_speed,
            ap=ap,
            pe=pe,
            pitch=pitch,
            throttle=throttle_cmd,
            time_to_ap=time_to_ap,
        )
        # PPO actions that need less safety correction are rewarded
        reward -= safety_intervention_penalty

        ap_error = abs(
            ap - self.command_target_ap
        )

        pe_error = abs(
            pe - self.command_target_pe
        )

        orbit_within_tolerance = (
            ap_error <= self.ap_tolerance
            and pe_error <= self.pe_tolerance
            and pe >= 70_000.0
        )

        # --------------------------------------------------
        # Successful commanded orbit
        # --------------------------------------------------

        if orbit_within_tolerance:
            accuracy_bonus = 5000.0 / (
                1.0
                + (ap_error + pe_error) / 10_000.0
            )

            # Orbit and fuel requirement were both satisfied
            if (
                fuel_fraction
                >= self.command_min_fuel_fraction
            ):
                reward += 15_000.0
                reward += accuracy_bonus
                reward += 2_000.0

                success = True

                print(
                    f"[{self.current_step}] "
                    f"COMMANDED ORBIT ACHIEVED | "
                    f"Target Ap: "
                    f"{self.command_target_ap / 1000:.1f} km | "
                    f"Actual Ap: "
                    f"{ap / 1000:.1f} km | "
                    f"Target Pe: "
                    f"{self.command_target_pe / 1000:.1f} km | "
                    f"Actual Pe: "
                    f"{pe / 1000:.1f} km | "
                    f"Fuel: "
                    f"{fuel_fraction:.2f} | "
                    f"Accuracy bonus: "
                    f"{accuracy_bonus:.0f}"
                )

            # Orbit was reached, but the fuel command was missed
            else:
                fuel_shortfall = (
                    self.command_min_fuel_fraction
                    - fuel_fraction
                )

                reward -= 3_000.0
                reward -= 10_000.0 * fuel_shortfall

                print(
                    f"[{self.current_step}] "
                    f"ORBIT REACHED, "
                    f"BUT FUEL COMMAND MISSED | "
                    f"Required fuel: "
                    f"{self.command_min_fuel_fraction:.2f} | "
                    f"Actual fuel: "
                    f"{fuel_fraction:.2f}"
                )

            terminated = True

        # --------------------------------------------------
        # Failure: did not leave the launch area
        # --------------------------------------------------

        elif self.current_step > 200 and alt < 500:
            reward -= 3000.0

            print(
                f"[{self.current_step}] "
                f"FAILURE: Did not launch properly | "
                f"Alt: {alt:.1f} m | "
                f"Thrust: "
                f"{self.vessel.available_thrust:.1f} | "
                f"Throttle: "
                f"{throttle_cmd:.2f}"
            )

            terminated = True

        # --------------------------------------------------
        # Failure: crash or flip near launchpad
        # --------------------------------------------------

        elif (
            self.current_step > 50
            and alt < 100
            and pitch < 45
        ):
            reward -= 3000.0

            print(
                f"[{self.current_step}] "
                f"FAILURE: Crashed or flipped near pad."
            )

            terminated = True

        # --------------------------------------------------
        # Failure: falling without useful trajectory
        # --------------------------------------------------

        elif (
            self.current_step > 300
            and alt < self.prev_altitude - 500
            and vertical_speed < -100
            and ap < 50_000
        ):
            reward -= 2500.0

            print(
                f"[{self.current_step}] "
                f"FAILURE: Falling without useful trajectory."
            )

            terminated = True

        # --------------------------------------------------
        # Timeout
        # --------------------------------------------------

        elif self.current_step >= self.max_steps:
            reward -= 1500.0

            print(
                f"[{self.current_step}] "
                f"TIMEOUT | "
                f"Target Ap: "
                f"{self.command_target_ap / 1000:.1f} km | "
                f"Actual Ap: "
                f"{ap / 1000:.1f} km | "
                f"Target Pe: "
                f"{self.command_target_pe / 1000:.1f} km | "
                f"Actual Pe: "
                f"{pe / 1000:.1f} km"
            )

            truncated = True

        # Store current state for the next reward calculation
        self.prev_altitude = alt
        self.prev_ap = ap
        self.prev_pe = pe
        self.prev_horizontal_speed = horizontal_speed

        info = {
            # Success metrics
            "is_success": success,
            "altitude": alt,
            "apoapsis": ap,
            "periapsis": pe,
            "apoapsis_error": ap_error,
            "periapsis_error": pe_error,
            "horizontal_speed": horizontal_speed,
            "vertical_speed": vertical_speed,
            "pitch": pitch,
            "throttle": throttle_cmd,
            "fuel_fraction": fuel_fraction,

            # Safety-intervention metrics
            "raw_throttle": raw_throttle,
            "safe_throttle": throttle_cmd,
            "requested_pitch": requested_pitch,
            "safe_pitch": pitch_cmd,
            "throttle_intervention": throttle_intervention,
            "pitch_intervention": pitch_intervention,
            "safety_intervention_penalty": safety_intervention_penalty,

            "available_thrust": float(
                self.vessel.available_thrust
            ),
            **self._command_info(),
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # --------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------

    def _get_obs(self):
        alt = float(
            self.flight.surface_altitude
        )

        vertical_speed = float(
            self.flight.vertical_speed
        )

        speed = float(
            self.flight.speed
        )

        horizontal_speed = self._horizontal_speed(
            speed,
            vertical_speed,
        )

        ap = self._safe_apoapsis()
        pe = self._safe_periapsis()
        time_to_ap = self._safe_time_to_apoapsis()

        fuel_fraction = self._current_fuel_fraction()

        mass_fraction = float(
            np.clip(
                float(self.vessel.mass)
                / self.initial_mass,
                0.0,
                1.5,
            )
        )

        pitch = float(
            self.surface_flight.pitch
        )

        heading = float(
            self.surface_flight.heading
        )

        throttle = float(
            self.vessel.control.throttle
        )

        current_stage = int(
            self.vessel.control.current_stage
        )

        # East is heading 90 degrees
        heading_error = self._angle_error_degrees(
            heading,
            90.0,
        )

        observation = np.array(
            [
                # Current flight state: indices 0–11
                alt / 100_000.0,
                vertical_speed / 1000.0,
                horizontal_speed / 2500.0,
                ap / 200_000.0,
                pe / 200_000.0,
                time_to_ap / 300.0,
                fuel_fraction,
                mass_fraction,
                pitch / 90.0,
                heading_error / 180.0,
                throttle,
                current_stage
                / max(
                    1.0,
                    float(self.stage_count_start),
                ),

                # Commander instruction: indices 12–15
                self.command_target_ap
                / 200_000.0,

                self.command_target_pe
                / 200_000.0,

                self.command_min_fuel_fraction,

                self.command_urgency,
            ],
            dtype=np.float32,
        )

        observation = np.nan_to_num(
            observation,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        observation = np.clip(
            observation,
            -10.0,
            10.0,
        )

        return observation

    # --------------------------------------------------
    # REWARD
    # --------------------------------------------------

    def _compute_reward(
        self,
        alt,
        vertical_speed,
        horizontal_speed,
        ap,
        pe,
        pitch,
        throttle,
        time_to_ap,
    ):
        reward = 0.0

        # Current errors relative to the commander target
        ap_error = abs(
            ap - self.command_target_ap
        )

        pe_error = abs(
            pe - self.command_target_pe
        )

        # Errors during previous environment step
        previous_ap_error = abs(
            self.prev_ap - self.command_target_ap
        )

        previous_pe_error = abs(
            self.prev_pe - self.command_target_pe
        )

        # Positive if the rocket moved closer to the target
        ap_error_improvement = (
            previous_ap_error - ap_error
        )

        pe_error_improvement = (
            previous_pe_error - pe_error
        )

        delta_horizontal_speed = (
            horizontal_speed
            - self.prev_horizontal_speed
        )

        # --------------------------------------------------
        # 1. Liftoff and early ascent
        # --------------------------------------------------

        if alt > 100:
            reward += 1.0

        if alt < 1000:
            if pitch > 75:
                reward += 1.0
            else:
                reward -= 3.0

            reward += 0.01 * np.clip(
                vertical_speed,
                -50.0,
                100.0,
            )

        # --------------------------------------------------
        # 2. Move toward commanded apoapsis
        # --------------------------------------------------

        reward += 0.004 * np.clip(
            ap_error_improvement,
            -1000.0,
            1000.0,
        )

        # Bounded proximity reward
        reward += 8.0 * np.exp(
            -ap_error / 20_000.0
        )

        # Penalize significant apoapsis overshoot
        if (
            ap
            > self.command_target_ap + 20_000.0
        ):
            ap_overshoot = (
                ap - self.command_target_ap
            )

            reward -= min(
                50.0,
                0.001 * ap_overshoot,
            )

        # --------------------------------------------------
        # 3. Horizontal-speed progress
        # --------------------------------------------------

        reward += 0.05 * np.clip(
            delta_horizontal_speed,
            -20.0,
            20.0,
        )

        if alt > 10_000:
            reward += (
                0.001 * horizontal_speed
            )

        # --------------------------------------------------
        # 4. Gravity-turn shaping
        # --------------------------------------------------

        target_pitch = (
            self._target_pitch_for_altitude(
                alt
            )
        )

        pitch_error = abs(
            pitch - target_pitch
        )

        if 1000 < alt < 70_000:
            reward -= (
                0.03 * pitch_error
            )

        # --------------------------------------------------
        # 5. One-time apoapsis milestones
        # --------------------------------------------------

        if (
            not self.ap_above_atmosphere_awarded
            and ap >= 70_000.0
        ):
            reward += 300.0
            self.ap_above_atmosphere_awarded = True

        if (
            not self.ap_near_target_awarded
            and ap_error <= 20_000.0
        ):
            reward += 500.0
            self.ap_near_target_awarded = True

        # --------------------------------------------------
        # 6. Move periapsis toward commanded target
        # --------------------------------------------------

        if ap > 60_000.0:
            reward += 0.01 * np.clip(
                pe_error_improvement,
                -1000.0,
                1000.0,
            )

            reward += 12.0 * np.exp(
                -pe_error / 20_000.0
            )

        # Penalize significant periapsis overshoot
        if (
            pe
            > self.command_target_pe + 20_000.0
        ):
            pe_overshoot = (
                pe - self.command_target_pe
            )

            reward -= min(
                50.0,
                0.001 * pe_overshoot,
            )

        # One-time periapsis milestones
        if (
            not self.pe_positive_awarded
            and pe > 0.0
        ):
            reward += 300.0
            self.pe_positive_awarded = True

        if (
            not self.pe_50k_awarded
            and pe > 50_000.0
        ):
            reward += 700.0
            self.pe_50k_awarded = True

        if (
            not self.pe_near_target_awarded
            and pe_error <= 20_000.0
            and pe > 60_000.0
        ):
            reward += 1200.0
            self.pe_near_target_awarded = True

        # --------------------------------------------------
        # 7. Coast and circularization behavior
        # --------------------------------------------------

        ap_ready_for_coast = (
            ap
            >= self.command_target_ap - 10_000.0
        )

        ap_ready_for_burn = (
            ap
            >= self.command_target_ap - 20_000.0
        )

        pe_still_low = (
            pe
            < self.command_target_pe - 5_000.0
        )

        # Coast while far from apoapsis
        if (
            ap_ready_for_coast
            and time_to_ap > 90.0
        ):
            if throttle < 0.2:
                reward += 3.0

            elif throttle > 0.5:
                reward -= 8.0

        # Circularization burn near apoapsis
        if (
            ap_ready_for_burn
            and time_to_ap < 60.0
            and pe_still_low
        ):
            if pitch < 15.0:
                reward += 8.0
            else:
                reward -= (
                    0.3
                    * abs(pitch - 5.0)
                )

            reward += (
                0.002
                * horizontal_speed
            )

            if throttle > 0.3:
                reward += 5.0
            else:
                reward -= 3.0

        # --------------------------------------------------
        # 8. Urgency and bad-behavior penalties
        # --------------------------------------------------

        # Higher urgency means a larger penalty for every step
        reward -= (
            0.05
            + 0.15 * self.command_urgency
        )

        if (
            self.current_step > 30
            and alt < 100
            and throttle < 0.5
        ):
            reward -= 10.0

        if (
            alt < 3000
            and pitch < 60
        ):
            reward -= 10.0

        if (
            alt > 20_000
            and pitch > 80
            and horizontal_speed < 500
        ):
            reward -= 5.0

        if (
            alt > 1000
            and vertical_speed < -50
        ):
            reward -= 10.0

        return float(reward)

    # --------------------------------------------------
    # CONTROL HELPERS
    # --------------------------------------------------

    def _compute_safe_throttle(
        self,
        raw_throttle,
        alt,
        ap,
        pe,
        time_to_ap=None,
    ):
        # Full throttle during liftoff
        if alt < 1000:
            return 1.0

        # Keep burning until close to commanded apoapsis
        ascent_threshold = max(
            65_000.0,
            self.command_target_ap - 15_000.0,
        )

        if ap < ascent_threshold:
            return max(
                0.65,
                raw_throttle,
            )

        if time_to_ap is not None:
            # Coast when target Ap is nearly reached
            if (
                ap
                >= self.command_target_ap - 10_000.0
                and time_to_ap > 90.0
            ):
                return min(
                    raw_throttle,
                    0.10,
                )

            # Burn near apoapsis while Pe is still too low
            if (
                ap
                >= self.command_target_ap - 20_000.0
                and time_to_ap < 60.0
                and pe
                < self.command_target_pe - 5_000.0
            ):
                return max(
                    0.35,
                    raw_throttle,
                )

        return raw_throttle

    def _compute_safe_pitch(
        self,
        raw_pitch_fraction,
        alt,
        ap=None,
        pe=None,
        time_to_ap=None,
    ):
        requested_pitch = (
            raw_pitch_fraction * 90.0
        )

        # Prevent immediate sideways launch
        if alt < 500:
            return max(
                80.0,
                requested_pitch,
            )

        if alt < 1500:
            return max(
                70.0,
                requested_pitch,
            )

        # Keep circularization burn almost horizontal
        if (
            ap is not None
            and pe is not None
            and time_to_ap is not None
        ):
            if (
                ap
                >= self.command_target_ap - 20_000.0
                and time_to_ap < 60.0
                and pe
                < self.command_target_pe - 5_000.0
            ):
                return min(
                    requested_pitch,
                    10.0,
                )

        return requested_pitch

    @staticmethod
    def _target_pitch_for_altitude(alt):
        if alt < 1000:
            return 90.0

        if alt < 10_000:
            return float(
                np.interp(
                    alt,
                    [1000, 10_000],
                    [85.0, 65.0],
                )
            )

        if alt < 30_000:
            return float(
                np.interp(
                    alt,
                    [10_000, 30_000],
                    [65.0, 30.0],
                )
            )

        if alt < 60_000:
            return float(
                np.interp(
                    alt,
                    [30_000, 60_000],
                    [30.0, 5.0],
                )
            )

        return 0.0

    # --------------------------------------------------
    # STAGING
    # --------------------------------------------------

    def _handle_staging(self, throttle_cmd):
        """
        Handles:

        - full-stage burnout
        - side-booster burnout
        - no-active-engine situations
        """

        now = time.time()

        # Prevent multiple rapid staging commands
        if now - self.last_stage_time < 1.5:
            return

        # Do not stage while intentionally coasting
        if throttle_cmd < 0.1:
            return

        try:
            active_engines = [
                engine
                for engine
                in self.vessel.parts.engines
                if engine.active
            ]

            vessel_no_thrust = (
                self.vessel.available_thrust
                <= 0.1
            )

            # No engine is active
            if (
                not active_engines
                and vessel_no_thrust
            ):
                print(
                    f"[{self.current_step}] "
                    f"No active engines. "
                    f"Staging again..."
                )

                self._activate_next_stage(now)
                return

            if not active_engines:
                return

            alive_engines = [
                engine
                for engine in active_engines
                if engine.available_thrust > 0.1
            ]

            dead_engines = [
                engine
                for engine in active_engines
                if engine.available_thrust <= 0.1
            ]

            all_engines_dead = (
                len(dead_engines)
                == len(active_engines)
            )

            partial_flameout = bool(
                dead_engines
                and alive_engines
            )

            alt = float(
                self.flight.surface_altitude
            )

            # Full-stage burnout
            if (
                all_engines_dead
                or vessel_no_thrust
            ):
                print(
                    f"[{self.current_step}] "
                    f"Full burnout/no thrust. "
                    f"Active engines: "
                    f"{len(active_engines)}. "
                    f"Staging..."
                )

                self._activate_next_stage(now)
                return

            # Side boosters burned out while core engine still runs
            if (
                partial_flameout
                and alt > 100
            ):
                print(
                    f"[{self.current_step}] "
                    f"Partial flameout. "
                    f"Dead engines: "
                    f"{len(dead_engines)}, "
                    f"alive engines: "
                    f"{len(alive_engines)}. "
                    f"Staging side boosters..."
                )

                self._activate_next_stage(now)

        except Exception as exc:
            print(
                f"Staging check failed: {exc}"
            )

    def _activate_next_stage(self, now):
        self.vessel.control.activate_next_stage()

        self.last_stage_time = now

        time.sleep(0.5)

        self._rebind_vessel_objects()

        self.vessel.auto_pilot.engage()

    # --------------------------------------------------
    # KSP VALUE HELPERS
    # --------------------------------------------------

    def _rebind_vessel_objects(self):
        self.vessel = (
            self.conn.space_center.active_vessel
        )

        self.body = self.vessel.orbit.body

        self.ref_frame = (
            self.body.reference_frame
        )

        self.flight = self.vessel.flight(
            self.ref_frame
        )

        self.surface_flight = self.vessel.flight(
            self.vessel.surface_reference_frame
        )

    def _get_liquid_fuel(self):
        try:
            return float(
                self.vessel.resources.amount(
                    "LiquidFuel"
                )
            )

        except Exception:
            return 0.0

    def _current_fuel_fraction(self):
        return float(
            np.clip(
                self._get_liquid_fuel()
                / self.initial_fuel,
                0.0,
                1.5,
            )
        )

    def _safe_apoapsis(self):
        try:
            value = float(
                self.vessel.orbit.apoapsis_altitude
            )

            return float(
                np.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=500_000.0,
                    neginf=-100_000.0,
                )
            )

        except Exception:
            return 0.0

    def _safe_periapsis(self):
        try:
            value = float(
                self.vessel.orbit.periapsis_altitude
            )

            return float(
                np.nan_to_num(
                    value,
                    nan=-600_000.0,
                    posinf=500_000.0,
                    neginf=-600_000.0,
                )
            )

        except Exception:
            return -600_000.0

    def _safe_time_to_apoapsis(self):
        try:
            value = float(
                self.vessel.orbit.time_to_apoapsis
            )

            return float(
                np.nan_to_num(
                    value,
                    nan=0.0,
                    posinf=999.0,
                    neginf=0.0,
                )
            )

        except Exception:
            return 0.0

    def _get_speed_values(self):
        alt = float(
            self.flight.surface_altitude
        )

        vertical_speed = float(
            self.flight.vertical_speed
        )

        speed = float(
            self.flight.speed
        )

        horizontal_speed = self._horizontal_speed(
            speed,
            vertical_speed,
        )

        return (
            alt,
            vertical_speed,
            horizontal_speed,
        )

    def _command_info(self):
        return {
            "command_target_ap":
                self.command_target_ap,

            "command_target_pe":
                self.command_target_pe,

            "command_min_fuel_fraction":
                self.command_min_fuel_fraction,

            "command_urgency":
                self.command_urgency,
        }

    @staticmethod
    def _horizontal_speed(
        speed,
        vertical_speed,
    ):
        return float(
            max(
                0.0,
                np.sqrt(
                    max(
                        0.0,
                        speed**2
                        - vertical_speed**2,
                    )
                ),
            )
        )

    @staticmethod
    def _angle_error_degrees(
        angle,
        target,
    ):
        return float(
            (
                angle
                - target
                + 180.0
            )
            % 360.0
            - 180.0
        )

    def close(self):
        try:
            if self.vessel is not None:
                self.vessel.control.throttle = 0.0

        except Exception:
            pass

        try:
            if self.conn is not None:
                self.conn.close()

        except Exception:
            pass


# --------------------------------------------------
# TRAINING SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    env = KSPCommandedOrbitalEnv(
        randomize_commands=True
    )

    print(
        "Initializing commanded PPO "
        "reinforcement learning..."
    )

    # Separate v3 paths.
    model_path = (
        "./ppo_ksp_commanded_pilot_v3"
    )

    checkpoint_dir = (
        "./models_v3/"
    )

    tensorboard_dir = (
        "./ksp_tensorboard_v3/"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )

    os.makedirs(
        tensorboard_dir,
        exist_ok=True,
    )

    # Continue existing v3 training if the model exists
    if os.path.exists(
        model_path + ".zip"
    ):
        print(
            "Loading existing commanded "
            "PPO v3 brain..."
        )

        model = PPO.load(
            model_path,
            env=env,
            device="cpu",
            tensorboard_log=tensorboard_dir,
        )

    # Otherwise create a new 16-observation model
    else:
        print(
            "Creating a fresh commanded "
            "PPO v3 brain..."
        )

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cpu",
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=tensorboard_dir,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=checkpoint_dir,
        name_prefix="ksp_commanded_v3",
    )

    print(
        ">>> COMMENCING COMMANDED RL TRAINING. "
        "Press Ctrl+C to save and exit. <<<"
    )

    try:
        # Goal-conditioned training is harder than
        # the previous fixed-target task.
        model.learn(
            total_timesteps=1_000_000,
            callback=checkpoint_callback,
            reset_num_timesteps=False,
        )

        model.save(model_path)

        print(
            f"Training complete. "
            f"Brain saved to "
            f"{model_path}.zip"
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

        interrupt_path = (
            model_path + "_interrupt"
        )

        latest_path = (
            model_path + "_latest"
        )

        print(
            f"Saving interrupt model to "
            f"{interrupt_path}.zip ..."
        )

        model.save(interrupt_path)

        print(
            f"Saving latest model to "
            f"{latest_path}.zip ..."
        )

        model.save(latest_path)

        print(
            f"Also updating main model at "
            f"{model_path}.zip ..."
        )

        model.save(model_path)

        print(
            "Commanded brain saved safely."
        )

    finally:
        env.close()