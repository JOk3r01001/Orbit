import gymnasium as gym
from gymnasium import spaces
import numpy as np
import krpc
import time
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback


class KSPOrbitalEnv(gym.Env):
    """
    PPO environment for learning a multi-stage Kerbin launch to orbit.

    Action:
        action[0] = throttle command, 0.0 to 1.0
        action[1] = pitch command fraction, 0.0 to 1.0
                    converted to 0 to 90 degrees

    Observation:
        Normalized flight/orbit state.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        print("Connecting to KSP for Reinforcement Learning...")
        self.conn = krpc.connect(name="PPO_Trainer")

        # -----------------------------
        # ACTION SPACE
        # -----------------------------
        # [Throttle, Pitch Fraction]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # -----------------------------
        # OBSERVATION SPACE
        # -----------------------------
        # 12 normalized values:
        #
        # 0  altitude_norm
        # 1  vertical_speed_norm
        # 2  horizontal_speed_norm
        # 3  apoapsis_norm
        # 4  periapsis_norm
        # 5  time_to_apoapsis_norm
        # 6  fuel_fraction
        # 7  mass_fraction
        # 8  pitch_norm
        # 9  heading_error_norm
        # 10 throttle
        # 11 current_stage_norm
        #
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(12,),
            dtype=np.float32,
        )

        # Episode config
        self.dt = 0.1
        self.max_steps = 5000  # about 500 seconds

        # Orbit target
        self.target_pe = 70_000.0
        self.target_ap_low = 75_000.0
        self.target_ap_high = 120_000.0

        # Runtime variables
        self.vessel = None
        self.flight = None
        self.surface_flight = None
        self.body = None
        self.ref_frame = None

        self.current_step = 0
        self.last_stage_time = 0.0
        self.stage_count_start = 1

        self.initial_fuel = 1.0
        self.initial_mass = 1.0

        self.prev_ap = 0.0
        self.prev_pe = -600_000.0
        self.prev_horizontal_speed = 0.0
        self.prev_altitude = 0.0

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print("\nLoading launchpad save...")
        self.conn.space_center.load("launchpad_ready")
        time.sleep(2.0)

        self._rebind_vessel_objects()

        # Autopilot setup
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, 90)

        # Reset counters
        self.current_step = 0
        self.last_stage_time = time.time()

        # Initial resources
        self.initial_fuel = max(1.0, self._get_liquid_fuel())
        self.initial_mass = max(1.0, self.vessel.mass)
        self.stage_count_start = max(1, self.vessel.control.current_stage)

        # Initial previous values for delta rewards
        alt, vertical_speed, horizontal_speed = self._get_speed_values()

        self.prev_altitude = alt
        self.prev_horizontal_speed = horizontal_speed
        self.prev_ap = self._safe_apoapsis()
        self.prev_pe = self._safe_periapsis()

        # Start launch
        self.vessel.control.throttle = 1.0
        time.sleep(0.2)
        self.vessel.control.activate_next_stage()

        obs = self._get_obs()
        return obs, {}

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, action):
        self.current_step += 1

        terminated = False
        truncated = False

        # Read current values before applying action
        alt, vertical_speed, horizontal_speed = self._get_speed_values()
        ap = self._safe_apoapsis()

        # -----------------------------
        # APPLY ACTION
        # -----------------------------
        raw_throttle = float(np.clip(action[0], 0.0, 1.0))
        raw_pitch_fraction = float(np.clip(action[1], 0.0, 1.0))

        throttle_cmd = self._compute_safe_throttle(raw_throttle, alt, ap)
        pitch_cmd = self._compute_safe_pitch(raw_pitch_fraction, alt)

        self.vessel.control.throttle = throttle_cmd
        self.vessel.auto_pilot.target_pitch_and_heading(pitch_cmd, 90)

        # Automated staging
        self._handle_staging(throttle_cmd)

        time.sleep(self.dt)

        # Re-read after physics step
        obs = self._get_obs()

        alt = self.flight.surface_altitude
        pitch = self.surface_flight.pitch
        vertical_speed = self.flight.vertical_speed
        speed = self.flight.speed
        horizontal_speed = self._horizontal_speed(speed, vertical_speed)

        ap = self._safe_apoapsis()
        pe = self._safe_periapsis()
        time_to_ap = self._safe_time_to_apoapsis()

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

        # -----------------------------
        # TERMINATION CONDITIONS
        # -----------------------------

        # Success: stable orbit above atmosphere
        if pe > self.target_pe and ap > self.target_pe:
            circularity_error = abs(ap - pe)
            circularity_bonus = 5000.0 / (1.0 + circularity_error / 10_000.0)

            reward += 15_000.0
            reward += circularity_bonus

            print(
                f"[{self.current_step}] ORBIT ACHIEVED | "
                f"Ap: {ap / 1000:.1f} km | "
                f"Pe: {pe / 1000:.1f} km | "
                f"Bonus: {circularity_bonus:.0f}"
            )

            terminated = True

        # Failure: no launch
        elif self.current_step > 150 and alt < 500:
            reward -= 3000.0
            print(f"[{self.current_step}] FAILURE: Did not launch properly.")
            terminated = True

        # Failure: crash / flip near ground
        elif self.current_step > 50 and alt < 100 and pitch < 45:
            reward -= 3000.0
            print(f"[{self.current_step}] FAILURE: Crashed or flipped near pad.")
            terminated = True

        # Failure: falling back into atmosphere after reaching high altitude
        elif self.current_step > 300 and alt < self.prev_altitude - 500 and vertical_speed < -100 and ap < 50_000:
            reward -= 2500.0
            print(f"[{self.current_step}] FAILURE: Falling without useful trajectory.")
            terminated = True

        # Timeout
        elif self.current_step >= self.max_steps:
            reward -= 1500.0
            print(
                f"[{self.current_step}] TIMEOUT | "
                f"Ap: {ap / 1000:.1f} km | "
                f"Pe: {pe / 1000:.1f} km"
            )
            truncated = True

        # Store previous values
        self.prev_altitude = alt
        self.prev_ap = ap
        self.prev_pe = pe
        self.prev_horizontal_speed = horizontal_speed

        info = {
            "altitude": alt,
            "apoapsis": ap,
            "periapsis": pe,
            "horizontal_speed": horizontal_speed,
            "vertical_speed": vertical_speed,
            "pitch": pitch,
            "throttle": throttle_cmd,
        }

        return obs, reward, terminated, truncated, info

    # --------------------------------------------------
    # OBSERVATION
    # --------------------------------------------------

    def _get_obs(self):
        alt = self.flight.surface_altitude
        vertical_speed = self.flight.vertical_speed
        speed = self.flight.speed
        horizontal_speed = self._horizontal_speed(speed, vertical_speed)

        ap = self._safe_apoapsis()
        pe = self._safe_periapsis()
        time_to_ap = self._safe_time_to_apoapsis()

        fuel = self._get_liquid_fuel()
        mass = self.vessel.mass

        pitch = self.surface_flight.pitch
        heading = self.surface_flight.heading
        throttle = self.vessel.control.throttle

        current_stage = self.vessel.control.current_stage

        fuel_fraction = np.clip(fuel / self.initial_fuel, 0.0, 1.5)
        mass_fraction = np.clip(mass / self.initial_mass, 0.0, 1.5)

        # Heading target is east, 90 degrees
        heading_error = self._angle_error_degrees(heading, 90.0)

        obs = np.array(
            [
                alt / 100_000.0,
                vertical_speed / 1000.0,
                horizontal_speed / 2500.0,
                ap / 150_000.0,
                pe / 150_000.0,
                time_to_ap / 300.0,
                fuel_fraction,
                mass_fraction,
                pitch / 90.0,
                heading_error / 180.0,
                throttle,
                current_stage / max(1.0, self.stage_count_start),
            ],
            dtype=np.float32,
        )

        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, -10.0, 10.0)

        return obs

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

        delta_ap = ap - self.prev_ap
        delta_pe = pe - self.prev_pe
        delta_horizontal_speed = horizontal_speed - self.prev_horizontal_speed

        # --------------------------------------------------
        # 1. Liftoff and early ascent
        # --------------------------------------------------

        if alt > 100:
            reward += 1.0

        if alt < 1000:
            # Must point mostly upward near the ground
            if pitch > 75:
                reward += 1.0
            else:
                reward -= 3.0

            # Encourage positive vertical speed at launch
            reward += 0.01 * np.clip(vertical_speed, -50, 100)

        # --------------------------------------------------
        # 2. Apoapsis progress
        # --------------------------------------------------

        # Reward raising apoapsis, but clip to avoid huge spikes.
        reward += 0.003 * np.clip(delta_ap, -1000.0, 1000.0)

        # --------------------------------------------------
        # 3. Horizontal speed progress
        # --------------------------------------------------

        # Orbit needs horizontal velocity.
        reward += 0.05 * np.clip(delta_horizontal_speed, -20.0, 20.0)

        # Small absolute horizontal-speed reward after leaving lower atmosphere.
        if alt > 10_000:
            reward += 0.002 * horizontal_speed

        # --------------------------------------------------
        # 4. Gravity turn guidance
        # --------------------------------------------------

        target_pitch = self._target_pitch_for_altitude(alt)
        pitch_error = abs(pitch - target_pitch)

        # Do not force this too hard, but guide the model.
        if 1000 < alt < 70_000:
            reward -= 0.03 * pitch_error

        # --------------------------------------------------
        # 5. Apoapsis zone rewards
        # --------------------------------------------------

        if 70_000 < ap < 140_000:
            reward += 15.0

        if self.target_ap_low < ap < self.target_ap_high:
            reward += 30.0

        # --------------------------------------------------
        # 6. Periapsis progress
        # --------------------------------------------------

        # Reward periapsis improvement only after apoapsis is near space.
        if ap > 60_000:
            reward += 0.002 * np.clip(delta_pe, -1000.0, 1000.0)

        # Strong reward once periapsis gets closer to space.
        if pe > 0:
            reward += 100.0

        if pe > 30_000:
            reward += 300.0

        if pe > 50_000:
            reward += 700.0

        # --------------------------------------------------
        # 7. Coast/circularization behavior
        # --------------------------------------------------

        # If apoapsis is already high, it is okay to reduce throttle before circularization.
        if ap > 80_000 and time_to_ap > 30 and throttle < 0.2:
            reward += 2.0

        # Penalize burning too long after apoapsis is very high.
        if ap > 200_000 and throttle > 0.5:
            reward -= 20.0

        # --------------------------------------------------
        # 8. Bad behavior penalties
        # --------------------------------------------------

        # Wasting time
        reward -= 0.1

        # Sitting on the pad with low throttle
        if self.current_step > 30 and alt < 100 and throttle < 0.5:
            reward -= 10.0

        # Going horizontal too early
        if alt < 3000 and pitch < 60:
            reward -= 10.0

        # Straight up for too long
        if alt > 20_000 and pitch > 80 and horizontal_speed < 500:
            reward -= 5.0

        # Falling
        if alt > 1000 and vertical_speed < -50:
            reward -= 10.0

        # Apoapsis way too high without periapsis
        if ap > 250_000 and pe < 0:
            reward -= 50.0

        return float(reward)

    # --------------------------------------------------
    # CONTROL HELPERS
    # --------------------------------------------------

    def _compute_safe_throttle(self, raw_throttle, alt, ap):
        """
        Keeps throttle safe during early ascent, but still allows throttle
        control later for multi-stage ascent and circularization.
        """

        # Mandatory liftoff power
        if alt < 1000:
            return 1.0

        # Main ascent: do not let PPO completely shut off too early
        if ap < 65_000:
            return max(0.65, raw_throttle)

        # Near-space coast / circularization: full freedom
        return raw_throttle

    def _compute_safe_pitch(self, raw_pitch_fraction, alt):
        """
        Converts action to pitch while preventing obviously impossible
        early-launch behavior.
        """

        requested_pitch = raw_pitch_fraction * 90.0

        # Early launch: do not allow immediate sideways pitch
        if alt < 500:
            return max(80.0, requested_pitch)

        if alt < 1500:
            return max(70.0, requested_pitch)

        return requested_pitch

    def _target_pitch_for_altitude(self, alt):
        """
        Rough gravity-turn guide.
        This is only used in reward shaping, not as direct control.
        """

        if alt < 1000:
            return 90.0
        elif alt < 10_000:
            return np.interp(alt, [1000, 10_000], [85.0, 65.0])
        elif alt < 30_000:
            return np.interp(alt, [10_000, 30_000], [65.0, 30.0])
        elif alt < 60_000:
            return np.interp(alt, [30_000, 60_000], [30.0, 5.0])
        else:
            return 0.0

    # --------------------------------------------------
    # STAGING
    # --------------------------------------------------

    def _handle_staging(self, throttle_cmd):
        """
        Automated staging for multi-stage rockets.

        This avoids the problem in the original baseline where ANY engine with
        zero thrust could trigger staging too early.
        """

        now = time.time()

        if now - self.last_stage_time < 1.5:
            return

        if throttle_cmd < 0.1:
            return

        try:
            active_engines = [engine for engine in self.vessel.parts.engines if engine.active]

            if len(active_engines) == 0:
                return

            # Stage only if all active engines are producing basically no thrust.
            all_engines_dead = all(engine.available_thrust <= 0.1 for engine in active_engines)

            # Also stage if vessel available thrust is zero while throttle is commanded.
            vessel_no_thrust = self.vessel.available_thrust <= 0.1

            if all_engines_dead or vessel_no_thrust:
                print(f"[{self.current_step}] Staging...")
                self.vessel.control.activate_next_stage()
                self.last_stage_time = now

                time.sleep(0.5)
                self._rebind_vessel_objects()

                self.vessel.auto_pilot.engage()

        except Exception as e:
            print(f"Staging check failed: {e}")

    # --------------------------------------------------
    # KSP VALUE HELPERS
    # --------------------------------------------------

    def _rebind_vessel_objects(self):
        self.vessel = self.conn.space_center.active_vessel
        self.body = self.vessel.orbit.body
        self.ref_frame = self.body.reference_frame

        self.flight = self.vessel.flight(self.ref_frame)
        self.surface_flight = self.vessel.flight(self.vessel.surface_reference_frame)

    def _get_liquid_fuel(self):
        try:
            return float(self.vessel.resources.amount("LiquidFuel"))
        except Exception:
            return 0.0

    def _safe_apoapsis(self):
        try:
            ap = float(self.vessel.orbit.apoapsis_altitude)
            return np.nan_to_num(ap, nan=0.0, posinf=500_000.0, neginf=-100_000.0)
        except Exception:
            return 0.0

    def _safe_periapsis(self):
        try:
            pe = float(self.vessel.orbit.periapsis_altitude)
            return np.nan_to_num(pe, nan=-600_000.0, posinf=500_000.0, neginf=-600_000.0)
        except Exception:
            return -600_000.0

    def _safe_time_to_apoapsis(self):
        try:
            t = float(self.vessel.orbit.time_to_apoapsis)
            return np.nan_to_num(t, nan=0.0, posinf=999.0, neginf=0.0)
        except Exception:
            return 0.0

    def _get_speed_values(self):
        alt = self.flight.surface_altitude
        vertical_speed = self.flight.vertical_speed
        speed = self.flight.speed
        horizontal_speed = self._horizontal_speed(speed, vertical_speed)

        return alt, vertical_speed, horizontal_speed

    @staticmethod
    def _horizontal_speed(speed, vertical_speed):
        return max(0.0, np.sqrt(max(0.0, speed ** 2 - vertical_speed ** 2)))

    @staticmethod
    def _angle_error_degrees(angle, target):
        """
        Returns smallest signed difference between two headings.
        """
        return (angle - target + 180.0) % 360.0 - 180.0


# --------------------------------------------------
# TRAINING SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    env = KSPOrbitalEnv()

    print("Initializing PPO Reinforcement Learning...")

    model_path = "./ppo_ksp_brain"

    if os.path.exists(model_path + ".zip"):
        print("Loading existing PPO brain...")
        model = PPO.load(
            model_path,
            env=env,
            device="cpu",
        )
    else:
        print("Creating new PPO brain...")

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
            tensorboard_log="./ksp_tensorboard/",
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/",
        name_prefix="ksp_ppo",
    )

    print(">>> COMMENCING RL TRAINING. Press Ctrl+C to save and exit. <<<")

    try:
        model.learn(
            total_timesteps=500_000,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        model.save(model_path)
        print("Training complete. Brain saved.")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current brain state...")
        model.save(model_path)
        print("Brain saved.")