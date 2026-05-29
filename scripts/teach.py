import krpc
import time
import csv

print("Connecting to KSP for Automated Expert Flight...")
conn = krpc.connect(name='Robot_Teacher')
vessel = conn.space_center.active_vessel
ref_frame = vessel.orbit.body.reference_frame
flight = vessel.flight(ref_frame)
surface_flight = vessel.flight(vessel.surface_reference_frame)

filename = "flight.csv"

# Ensure Autopilot is engaged before launch
vessel.auto_pilot.engage()
vessel.auto_pilot.target_pitch_and_heading(90, 90)

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'obs_alt', 'obs_vel', 'obs_fuel', 
        'obs_pitch', 'obs_heading', 'obs_roll', 
        'obs_pos_x', 'obs_pos_y', 'obs_pos_z',
        'obs_ap', 'obs_pe', 'obs_time_to_ap',  # <-- THE CRITICAL ADDITIONS
        'act_throttle', 'act_stage'
    ])

    print("\n>>> ROBOT TEACHER ARMED <<<")
    print("Initiating automated launch sequence in 3 seconds...")
    time.sleep(3.0)
    
    # Ignite first stage!
    vessel.control.throttle = 1.0
    vessel.control.activate_next_stage()
    last_stage_time = time.time()

    # --- NEW: State Machine Variables ---
    mission_phase = 0
    recording_paused = False

    try:
        while True:
            # --- 1. SENSOR READINGS ---
            alt = flight.surface_altitude
            vel = flight.vertical_speed
            ap = vessel.orbit.apoapsis_altitude
            pe = vessel.orbit.periapsis_altitude         # Tracks the bottom of your orbit
            time_to_ap = vessel.orbit.time_to_apoapsis   # Counts down to the highest point
            
            try: fuel = vessel.resources.amount('LiquidFuel')
            except: fuel = 0.0
            
            pitch = surface_flight.pitch
            heading = surface_flight.heading
            roll = surface_flight.roll
            pos_x, pos_y, pos_z = vessel.position(ref_frame)

            # --- 2. EXPERT LOGIC (State Machine) ---
            target_throttle = 0.0
            target_pitch = 90.0
            stage_action = 0.0
            recording_paused = False # Default to recording data

            # PHASE 0: Vertical Ascent (Clear the pad)
            if mission_phase == 0:
                target_pitch = 90.0
                target_throttle = 1.0
                if alt > 2000:
                    mission_phase = 1

            # PHASE 1: Smooth Gravity Turn
            elif mission_phase == 1:
                fraction = (alt - 2000) / (45000 - 2000)
                # Pitch smoothly from 90 down to 15 degrees
                target_pitch = max(15.0, 90.0 - (fraction * 75.0))
                target_throttle = 1.0
                
                if ap >= 100000:
                    mission_phase = 3
                elif alt >= 45000:
                    mission_phase = 2

            # PHASE 2: Push Apoapsis to 100k
            elif mission_phase == 2:
                # Keep nose slightly up (15 deg) to push Ap out of the atmosphere efficiently
                target_pitch = 15.0 
                target_throttle = 0.7
                if ap >= 100000:
                    print(f"\n>>> 100km APOAPSIS REACHED. COASTING... <<<")
                    mission_phase = 3

            # PHASE 3: Coast to Apoapsis
            elif mission_phase == 3:
                target_pitch = 0.0 # Lay perfectly horizontal (0 degrees) for the upcoming burn
                target_throttle = 0.0
                
                # CRITICAL AI FIX: If we are coasting for minutes, pause recording.
                if time_to_ap > 20.0:
                    recording_paused = True

                # Start the circularization burn 15 seconds before reaching the exact Apoapsis
                if time_to_ap <= 15.0 or time_to_ap > (vessel.orbit.period / 2):
                    print("\n>>> ORBITAL INSERTION BURN INITIATED <<<")
                    mission_phase = 4

            # PHASE 4: Orbital Insertion Burn
            elif mission_phase == 4:
                target_pitch = 0.0 # Stay flat
                target_throttle = 1.0
                
                # Stop burning when Periapsis clears the atmosphere (75km makes a safe loop)
                if pe >= 80000:
                    mission_phase = 5

            # Apply the calculated maneuvers to the ship
            vessel.auto_pilot.target_pitch_and_heading(target_pitch, 90)
            vessel.control.throttle = target_throttle

            # --- EXPERT STAGING LOGIC (Upgraded for Vessel Tracking) ---
            active_engines = [e for e in vessel.parts.engines if e.active]
            booster_flamed_out = any(e.available_thrust == 0 for e in active_engines)
            
            if (booster_flamed_out or vessel.available_thrust == 0) and target_throttle > 0 and (time.time() - last_stage_time) > 1.5:
                # 1. Force the globally active vessel to stage (prevents the crash)
                conn.space_center.active_vessel.control.activate_next_stage()
                last_stage_time = time.time()
                stage_action = 1.0
                print(">>> AUTO-STAGING TRIGGERED <<<")
                
                # 2. Give KSP a fraction of a second to compute the physics split
                time.sleep(0.5)
                
                # 3. RE-ACQUIRE THE ROCKET! (Prevents recording the falling debris)
                vessel = conn.space_center.active_vessel
                ref_frame = vessel.orbit.body.reference_frame
                flight = vessel.flight(ref_frame)
                surface_flight = vessel.flight(vessel.surface_reference_frame)

            
            # --- 3. RECORD DATA FOR THE AI ---
            if not recording_paused:
                writer.writerow([
                    alt, vel, fuel, 
                    pitch, heading, roll, 
                    pos_x, pos_y, pos_z, 
                    ap, pe, time_to_ap,         # <-- MATCHING DATA POINTS
                    target_throttle, stage_action
                ])
            
            # PHASE 5: End Script Successfully
            if mission_phase == 5:
                vessel.control.throttle = 0.0
                print(f"\n>>> ORBIT ACHIEVED! Ap: {ap/1000:.1f}km | Pe: {pe/1000:.1f}km <<<")
                print(f"Flawless expert data saved to {filename}.")
                break

            # Run at 10Hz to match the AI environment
            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\nRecording interrupted. Partial data saved to {filename}.")