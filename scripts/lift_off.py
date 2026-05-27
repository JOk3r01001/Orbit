import time
from stable_baselines3 import PPO
from liftoff import KSPLiftoffEnv # Replace with the actual name of your python file

print("Booting up the AI Pilot...")

# 1. Load the environment
env = KSPLiftoffEnv()
obs, info = env.reset()

# 2. Load the fully trained brain! (Make sure the filename matches your saved zip)
# If you used the v2 name from earlier, it will be "liftoff_dense_brain_final_v2"
model = PPO.load("liftoff_dense_brain_final_v2", env=env, device="cpu")

print("Brain loaded. Commencing launch sequence in 3 seconds...")
time.sleep(3.0)

# --- THE FIX: GROUND CONTROL IGNITION ---
# We manually trigger the first stage so the AI doesn't get stuck doing math
env.vessel.control.activate_next_stage()

# 3. The Flight Loop (No training, just flying)
while True:
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Slow the brain down to match training speed
    time.sleep(0.05) 
    
    if terminated or truncated:
        print(f"Flight ended. Resetting to launchpad...")
        obs, info = env.reset()
        
        # Make sure to re-light the engine after every reset!
        time.sleep(1.5)
        env.vessel.control.activate_next_stage()