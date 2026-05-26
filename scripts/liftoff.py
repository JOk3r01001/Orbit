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
        
        # Action: [Throttle, Stage_Trigger]
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation: [Surface Altitude, Vertical Velocity]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Load the rocket sitting on the launchpad
        self.conn.space_center.load('hover_start')
        time.sleep(1.0)
        
        self.vessel = self.conn.space_center.active_vessel
        self.ref_frame = self.vessel.orbit.body.reference_frame
        self.flight = self.vessel.flight(self.ref_frame)
        
        # Engine OFF.
        self.vessel.control.throttle = 0.0
        
        self.current_step = 0
        self.last_stage_time = time.time()
        
        # Tracker for our "Breadcrumb" rewards
        self.last_alt = self.flight.surface_altitude 
        self.has_launched = False
        
        return self._get_obs(), {}

    def _get_obs(self):
        alt = self.flight.surface_altitude
        vel = self.flight.vertical_speed
        return np.array([alt, vel], dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        self.vessel.control.throttle = float(action[0])
        
        if float(action[1]) > 0.8:
            if (time.time() - self.last_stage_time) > 2.0:
                self.vessel.control.activate_next_stage()
                self.last_stage_time = time.time()

        time.sleep(0.1)

        obs = self._get_obs()
        alt = obs[0]
        vel = obs[1]

        reward = 0
        terminated = False
        truncated = False

        if alt > 30:
            self.has_launched = True

        # --- THE REWARD MATH ---
        
        # 1. The Breadcrumb Trail (Points for going UP, minus points for falling DOWN)
        alt_gained = alt - self.last_alt
        reward += alt_gained * 0.5  # 0.5 points for every meter climbed!
        self.last_alt = alt
        
        # 2. The Ultimate Goal
        if alt >= 10000:
            print(f"TARGET REACHED! 10km crossed in {self.current_step} steps.")
            reward += 2000 # Jackpot for finishing the ascent mission!
            terminated = True
            
        # 3. Crash / Explosion check (Only checks if it actually left the pad first)
        elif alt < 15 and self.has_launched and self.current_step > 50:
            reward -= 1000
            terminated = True
            
        # 4. Coward Penalty
        elif not self.has_launched:
            reward -= 2 # Stop sitting on the pad!

        # Timeout (It shouldn't take more than ~3 minutes to reach 10km)
        if self.current_step > 1500: 
            truncated = True 

        return obs, reward, terminated, truncated, {}

# --- The Main Loop ---
if __name__ == "__main__":
    env = KSPLiftoffEnv()
    
    print("Initializing Liftoff Brain...")
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path='./saved_brains/', 
        name_prefix='liftoff_model'
    )
    
    print("Starting Ascent Training! 3... 2... 1...")
    model.learn(total_timesteps=80000, callback=checkpoint_callback) 
    
    print("Training Complete. Saving brain to disk...")
    model.save("liftoff_bot_brain_final")