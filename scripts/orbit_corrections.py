import os
import time

import gymnasium as gym
from gymnasium import spaces
import krpc
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class KSPOrbitCorrectionsEnv(gym.Env):
    """
    Goal-conditioned PPO environment for changing the orbit of a vessel.
    
    Actions:
        action[0] = throttle command, -1 to 1
                    mapped to 0 to 1

        action[1] = burn direction, -1 to 1
                    negative = retrograde
                    near zero = coast
                    positive = prograde

    Observations:
        0   normalized current apoapsis
        1   normalized current periapsis
        2   normalized target apoapsis
        3   normalized target periapsis
        4   normalized apoapsis error
        5   normalized periapsis error
        6   normalized time to apoapsis
        7   normalized time to periapsis
        8   fuel fraction
        9   normalized orbital speed
        10  previous throttle
        11  previous burn direction
        12  prograde alignment
        13  retrograde alignment
    """
    def __init__(self):
        super().__init__()

        print("Connecting to KSP for Reinforcement Learning...")
        self.conn = krpc.connect(name="PPO_Trainer")
        self.vessel = self.conn.space_center.active_vessel

