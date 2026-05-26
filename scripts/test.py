import krpc
import time

print("Attempting to connect to KSP...")

try:
    # 1. Connect to the server
    conn = krpc.connect(name='Sanity Check')
    vessel = conn.space_center.active_vessel
    print("SUCCESS: Connected to the game!")
    print(f"Active Vessel: {vessel.name}")

    # 2. Send Commands
    print("Initiating test launch in 3 seconds...")
    time.sleep(3)
    
    print("Ignition!")
    vessel.control.throttle = 1.0       # Throttle to 100%
    time.sleep(1)
    vessel.control.activate_next_stage() # Press the spacebar

except Exception as e:
    print(f"An error occurred: {e}")