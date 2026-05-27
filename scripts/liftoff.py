import gymnasium as gym
from gymnasium import spaces
import numpy as np
import krpc
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

class KSPLiftoffEnv(gym.Env):
    def __init__(self):
        super().__init__()
        print("Connecting to KSP...")
        self.conn = krpc.connect(name='Liftoff_Trainer')
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.conn.space_center.load('hover_start')
        time.sleep(1.0)
        self.vessel = self.conn.space_center.active_vessel
        self.ref_frame = self.vessel.orbit.body.reference_frame
        self.flight = self.vessel.flight(self.ref_frame)
        self.surface_flight = self.vessel.flight(self.vessel.surface_reference_frame)
        self.start_parts = len(self.vessel.parts.all)
        
        # --- FIX: TURN ON SAS AUTOPILOT ---
        # This prevents the rocket from immediately tipping over at launch
        self.vessel.auto_pilot.engage()
        self.vessel.auto_pilot.target_pitch_and_heading(90, 90)
        
        self.vessel.control.throttle = 0.0
        self.current_step = 0
        self.last_stage_time = time.time()
        self.last_alt = self.flight.surface_altitude 
        self.has_launched = False
        return self._get_obs(), {}

    def _get_obs(self):
        alt = self.flight.surface_altitude
        vel = self.flight.vertical_speed
        try:
            fuel = self.vessel.resources.amount('LiquidFuel')
        except:
            fuel = 0.0
        return np.array([alt, vel, fuel], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        try:
            self.vessel.control.throttle = float(action[0])
            if float(action[1]) > 0.8:
                if (time.time() - self.last_stage_time) > 2.0:
                    self.vessel.control.activate_next_stage()
                    self.last_stage_time = time.time()
                    time.sleep(0.2) 
                    self.start_parts = len(self.vessel.parts.all)

            time.sleep(0.1)
            obs = self._get_obs()
            alt, vel, fuel = obs[0], obs[1], obs[2]
            current_parts = len(self.vessel.parts.all)
            pitch = self.surface_flight.pitch 
            mean_alt = self.flight.mean_altitude 
        except Exception:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32), -1000, True, False, {}

        # --- THE FIX: ANTI-HOVER REWARD LOGIC ---
        
        # 1. Base rewards (Climbing is good)
        reward = (alt / 10000) * 5.0
        alt_gained = alt - self.last_alt
        
        # Multiply alt_gained more aggressively so it craves speed
        reward += alt_gained * 5.0 
        self.last_alt = alt
        
        # 2. THE TIME PENALTY (Existential Angst)
        # We punish the AI slightly for every step it stays alive without finishing the mission.
        # This forces it to hurry up and reach 10km to stop the pain!
        reward -= 1.0 
            
        terminated = False
        
        # --- SAFETY AND CRASH LOGIC ---
        if alt >= 10000:
            reward += 5000  # Increased jackpot
            terminated = True
        elif current_parts < self.start_parts:
            reward -= 1000
            terminated = True
        elif pitch < 60 and alt < 100:
            reward -= 1000
            terminated = True
        elif mean_alt < 10 and self.has_launched and self.current_step > 50:
            reward -= 1000
            terminated = True
        elif alt < 15 and self.has_launched and self.current_step > 50:
            reward -= 1000
            terminated = True
        elif not self.has_launched:
            reward -= 50
        
        if alt > 30: self.has_launched = True

        # STAGING LOGIC (FUEL SANITY)
        if float(action[1]) > 0.8 and (time.time() - self.last_stage_time) > 2.0:
            if fuel < 5.0: reward += 300 
            else: reward -= 1000
            
        # Notice: Vertical Incentive and Thrust Bonus have been completely removed!

        return obs, reward, terminated, False, {}

# --- THE MAIN LOOP (LOADING A SAVED BRAIN) ---
if __name__ == "__main__":
    env = KSPLiftoffEnv()
    
    # 1. LOAD THE SAVED MODEL INSTEAD OF CREATING A NEW ONE
    print("Loading previously saved brain...")
    
    # Change the filename here if you want to load a specific checkpoint 
    # e.g., "./saved_brains/liftoff_dense_10000_steps"
    model = PPO.load("liftoff_dense_brain_final_v2", env=env, device="cpu")
    
    # 2. Periodic Auto-Save (Continuing from where it left off)
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path='./saved_brains/', 
        # You can change the prefix so you don't overwrite old checkpoints
        name_prefix='liftoff_dense_continued' 
    )
    
    print("Resuming Training! Press Ctrl+C to stop and save early.")
    
    # 3. The Early Interrupt Catch
    try:
        # You can adjust total_timesteps to however many MORE steps you want it to run
        model.learn(total_timesteps=80000, callback=checkpoint_callback, reset_num_timesteps=False) 
        
        print("Training complete! Saving final model.")
        model.save("liftoff_dense_brain_final_v3")
        
    except KeyboardInterrupt:
        print("\n>>> Training Interrupted by User! Saving current brain state... <<<")
        model.save("liftoff_dense_brain_interrupted")
        print("Model saved safely as 'liftoff_dense_brain_interrupted'. Safe to close.")