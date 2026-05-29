import gymnasium as gym
from gymnasium import spaces
import numpy as np
import krpc
import time
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class KSPOrbitalEnv(gym.Env):
    def __init__(self):
        super(KSPOrbitalEnv, self).__init__()
        print("Connecting to KSP for Reinforcement Learning...")
        self.conn = krpc.connect(name='PPO_Trainer')
        
        # --- THE ACTION SPACE (The Pilot) ---
        # 2 Dimensions: [Throttle (0 to 1), Pitch Fraction (0 to 1)]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # --- THE OBSERVATION SPACE (The Sensors) ---
        # 13 Dimensions matching your expert data
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # LOAD YOUR LAUNCHPAD SAVE STATE HERE!
        # Make sure you create a quicksave named 'launchpad_ready'
        self.conn.space_center.load('launchpad_ready') 
        time.sleep(2.0)
        
        self.vessel = self.conn.space_center.active_vessel
        self.ref_frame = self.vessel.orbit.body.reference_frame
        self.flight = self.vessel.flight(self.ref_frame)
        self.surface_flight = self.vessel.flight(self.vessel.surface_reference_frame)
        
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, 90)
        
        self.vessel.control.throttle = 1.0
        self.vessel.control.activate_next_stage() # Ignition
        
        self.last_stage_time = time.time()
        self.current_step = 0
        self.max_steps = 5000 # 500 seconds max flight time
        
        return self._get_obs(), {}

    def _get_obs(self):
        alt = self.flight.surface_altitude
        vel = self.flight.vertical_speed
        ap = self.vessel.orbit.apoapsis_altitude
        pe = self.vessel.orbit.periapsis_altitude
        time_to_ap = self.vessel.orbit.time_to_apoapsis
        
        try: fuel = self.vessel.resources.amount('LiquidFuel')
        except: fuel = 0.0
        
        mass = self.vessel.mass
        pitch = self.surface_flight.pitch
        heading = self.surface_flight.heading
        roll = self.surface_flight.roll
        pos_x, pos_y, pos_z = self.vessel.position(self.ref_frame)

        return np.array([
            alt, vel, fuel, mass,
            pitch, heading, roll, 
            pos_x, pos_y, pos_z, 
            ap, pe, time_to_ap
        ], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        terminated = False
        truncated = False
        reward = 0.0
        
        # 1. APPLY AI COMMANDS (Throttle and Pitch)
        ai_throttle = float(action[0])
        ai_pitch_fraction = float(action[1])
        
        self.vessel.control.throttle = ai_throttle
        ai_pitch_degrees = ai_pitch_fraction * 90.0
        self.vessel.auto_pilot.target_pitch_and_heading(ai_pitch_degrees, 90)

        # 2. THE FLIGHT ENGINEER (Automated Staging)
        active_engines = [e for e in self.vessel.parts.engines if e.active]
        booster_flamed_out = any(e.available_thrust == 0 for e in active_engines)
        
        if (booster_flamed_out or self.vessel.available_thrust == 0) and ai_throttle > 0 and (time.time() - self.last_stage_time) > 1.5:
            self.conn.space_center.active_vessel.control.activate_next_stage()
            self.last_stage_time = time.time()
            time.sleep(0.5)
            # Re-acquire hooks
            self.vessel = self.conn.space_center.active_vessel
            self.ref_frame = self.vessel.orbit.body.reference_frame
            self.flight = self.vessel.flight(self.ref_frame)
            self.surface_flight = self.vessel.flight(self.vessel.surface_reference_frame)
            self.vessel.auto_pilot.engage()

        time.sleep(0.1)
        obs = self._get_obs()
        
        # Extract variables for the Reward Function
        alt = obs[0]
        ap = obs[10]
        pe = obs[11]
        pitch = obs[4]

        # --- THE REWARD FUNCTION (The Circularization Upgrade) ---
        
        # 1. Continuous Rewards (Climbing and Coasting)
        reward += (alt / 100000.0) * 2.0  
        
        # Big breadcrumbs for getting the Apoapsis into the safe orbital zone (80km - 100km)
        if 80000 < ap < 100000:
            reward += 10.0  
            
        # 2. The Circularization Multiplier (The Magic Math)
        # We calculate how "round" the orbit is by finding the difference between Ap and Pe.
        orbit_diff = abs(ap - pe)
        
        # 3. Terminal States (Win / Loss)
        if pe > 75000: # We are officially in space!
            print(f"[{self.current_step}] >>> ORBIT ACHIEVED! <<<")
            
            # The Base Jackpot
            reward += 10000.0
            
            # THE CIRCULARIZATION BONUS:
            # Maximum bonus of 5000 if Ap and Pe perfectly match. 
            # Scales down smoothly the farther apart they are.
            circular_bonus = 5000.0 * (1.0 / (1.0 + (orbit_diff / 5000.0)))
            reward += circular_bonus
            
            print(f"Ap: {ap/1000:.1f}km | Pe: {pe/1000:.1f}km | Bonus: +{circular_bonus:.0f}")
            terminated = True
            
        elif alt < 100 and pitch < 45 and self.current_step > 50: 
            reward -= 5000.0
            print(f"[{self.current_step}] CRITICAL FAILURE: Crashed or Flipped.")
            terminated = True
            
        elif self.current_step >= self.max_steps: 
            reward -= 1000.0
            print(f"[{self.current_step}] TIMEOUT: Failed to circularize.")
            truncated = True

        return obs, reward, terminated, truncated, {}

# --- PPO TRAINING SCRIPT ---
if __name__ == "__main__":
    # Create the environment
    env = KSPOrbitalEnv()
    
    # We are starting a BRAND NEW PPO brain, but we are going to use the 
    # architecture of our Behavioral Cloning tests.
    print("Initializing PPO Reinforcement Learning...")
    
    model_path = "./ppo_ksp_brain"
    
    # Check if a previous PPO brain exists to continue training
    if os.path.exists(model_path + ".zip"):
        print("Loading existing PPO brain...")
        model = PPO.load(model_path, env=env, device="cpu")
    else:
        print("Creating new PPO brain...")
        model = PPO("MlpPolicy", env, verbose=1, device="cpu", 
                    tensorboard_log="./ksp_tensorboard/")

    # Save a checkpoint every 2000 steps
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path='./models/', name_prefix='ksp_ppo')

    print(">>> COMMENCING RL TRAINING. Press Ctrl+C to save and exit. <<<")
    try:
        # Let it play the game!
        model.learn(total_timesteps=500000, callback=checkpoint_callback)
        model.save(model_path)
        print("Training complete! Brain saved.")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current brain state...")
        model.save(model_path)