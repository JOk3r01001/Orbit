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
        
        # Crash detection trackers
        self.surface_flight = self.vessel.flight(self.vessel.surface_reference_frame)
        self.start_parts = len(self.vessel.parts.all)
        
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
        
        # --- NEW: THE EXPLOSION CATCHER ---
        # We wrap all physical interactions in a try/except block.
        try:
            self.vessel.control.throttle = float(action[0])
            
            if float(action[1]) > 0.8:
                if (time.time() - self.last_stage_time) > 2.0:
                    self.vessel.control.activate_next_stage()
                    self.last_stage_time = time.time()
                    print(">>> AI TRIGGERED STAGING! <<<")
                    
                    time.sleep(0.2) 
                    self.start_parts = len(self.vessel.parts.all)

            # Physics Tick
            time.sleep(0.1)

            obs = self._get_obs()
            alt = obs[0]
            vel = obs[1]

            # Grab telemetry for the crash checks
            current_parts = len(self.vessel.parts.all)
            pitch = self.surface_flight.pitch 
            mean_alt = self.flight.mean_altitude 
            
        except Exception as e:
            # If ANY of the above code fails, the ship no longer exists!
            print(">>> CATASTROPHIC EXPLOSION DETECTED! <<<")
            # Return a dummy observation, massive penalty, and force a reset
            return np.array([0.0, 0.0], dtype=np.float32), -1000, True, False, {}

        # -----------------------------------

        reward = 0
        terminated = False
        truncated = False

        if alt > 30:
            self.has_launched = True

        # --- THE REWARD MATH ---
        
        # 1. The Breadcrumb Trail 
        alt_gained = alt - self.last_alt
        reward += alt_gained * 2.0 
        self.last_alt = alt
        
        # 2. The Ultimate Goal
        if alt >= 10000:
            print(f"TARGET REACHED! 10km crossed in {self.current_step} steps.")
            reward += 2000 
            terminated = True
            
        # 3a. Broken Part Check
        elif current_parts < self.start_parts:
            print(">>> CRAFT BROKE APART! <<<")
            reward -= 1000
            terminated = True
            
        # 3b. Tip-Over Check 
        elif pitch < 60 and alt < 100:
            print(">>> ROCKET TIPPED OVER! <<<")
            reward -= 1000
            terminated = True
            
        # 3c. The Ocean Trap Check
        elif mean_alt < 10 and self.has_launched and self.current_step > 50:
            print(">>> SPLASHDOWN IN THE OCEAN! <<<")
            reward -= 1000
            terminated = True
            
        # 4. Normal crash check (on solid ground)
        elif alt < 15 and self.has_launched and self.current_step > 50:
            reward -= 1000
            terminated = True
            
        # 5. Coward Penalty
        elif not self.has_launched:
            reward -= 50

        # Timeout 
        if self.current_step > 1500: 
            truncated = True 

        return obs, reward, terminated, truncated, {}

# --- The Main Loop ---
if __name__ == "__main__":
    env = KSPLiftoffEnv()
    
    print("Loading existing brain...")
    model = PPO.load("./saved_brains/liftoff_model_5000_steps", env=env)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path='./saved_brains/', 
        name_prefix='liftoff_model'
    )
    
    print("Starting Ascent Training! 3... 2... 1...")
    model.learn(total_timesteps=80000, callback=checkpoint_callback) 
    
    print("Training Complete. Saving brain to disk...")
    model.save("liftoff_bot_brain_final")