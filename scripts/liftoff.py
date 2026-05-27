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
            resources = self.vessel.resources_in_decouple_stage(self.vessel.control.current_stage - 1, cumulative=True)
            fuel = resources.amount('LiquidFuel')
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

        # REWARD MATH
        reward = (alt / 10000) * 5.0
        alt_gained = alt - self.last_alt
        reward += alt_gained * 2.0
        self.last_alt = alt
        
        terminated = False
        
        # --- SAFETY AND CRASH LOGIC ---
        if alt >= 10000:
            print("TARGET REACHED!")
            reward += 2000 
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

        # STAGING LOGIC
        if float(action[1]) > 0.8 and (time.time() - self.last_stage_time) > 2.0:
            if fuel < 5.0: reward += 200 
            else: reward -= 500

        return obs, reward, terminated, False, {}

if __name__ == "__main__":
    env = KSPLiftoffEnv()
    
    # IMPLEMENTING DENSER NETWORK
    # We increase the hidden layers to 128 neurons each to handle the complex state space
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
    model = PPO("MlpPolicy", env, 
                policy_kwargs=policy_kwargs, 
                verbose=1, 
                device="cpu")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, 
        save_path='./saved_brains/', 
        name_prefix='liftoff_dense_model'
    )
    
    print("Starting Training with Denser Brain...")
    model.learn(total_timesteps=80000, callback=checkpoint_callback) 
    model.save("liftoff_dense_brain_final")