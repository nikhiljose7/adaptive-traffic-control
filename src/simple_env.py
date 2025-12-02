import os
import sys

# Check and set SUMO_HOME if missing
if 'SUMO_HOME' not in os.environ:
    common_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
    ]
    for path in common_paths:
        if os.path.exists(path):
            os.environ['SUMO_HOME'] = path
            print(f"Set SUMO_HOME to {path}")
            break

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sumolib
import traci

class TrafficSignalEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, net_file, route_file, use_gui=False, num_seconds=3600):
        super(TrafficSignalEnv, self).__init__()
        
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.sumo_binary = sumolib.checkBinary('sumo-gui') if self.use_gui else sumolib.checkBinary('sumo')
        
        # Define action and observation space
        # Action: 0 (NS Green), 1 (EW Green) - simplified for 2 phases
        # In our net.xml, we have 4 phases usually (G, y, r, r...) but let's assume we switch between major Green phases.
        # Let's say: 
        # Phase 0: NS Green (Index 0 in SUMO logic usually)
        # Phase 2: EW Green (Index 2 in SUMO logic usually)
        # We will control the duration or switch. 
        # For simplicity, let's say action is "keep current" or "switch". 
        # OR, let's say action is "choose phase 0" or "choose phase 2".
        self.action_space = spaces.Discrete(2) 
        
        # Observation: Queue length for each of the 4 incoming lanes
        # E0, E1, E2, E3. 
        # We can also include waiting time.
        # Let's say 4 lanes * 2 metrics = 8 dims
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.ts_id = "J0" # The junction ID
        self.lanes = ["E0_0", "E1_0", "E2_0", "E3_0"]
        
        self.step_counter = 0
        self.run_id = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.step_counter > 0:
            traci.close()
            
        self.run_id += 1
        sumo_cmd = [self.sumo_binary, "-n", self.net_file, "-r", self.route_file, "--no-step-log", "true", "--waiting-time-memory", "1000"]
        
        if self.use_gui:
            # Add view settings
            view_file = os.path.join(os.path.dirname(self.net_file), "view.xml")
            if os.path.exists(view_file):
                sumo_cmd.extend(["-g", view_file])
                
        # Start TRACI
        traci.start(sumo_cmd, label=str(self.run_id))
        self.conn = traci.getConnection(str(self.run_id))
        
        self.step_counter = 0
        
        return self._get_observation(), {}

    def step(self, action):
        # Apply action
        # Action 0: Phase 0 (NS Green)
        # Action 1: Phase 2 (EW Green)
        # Note: In SUMO, phases are 0, 1 (yellow), 2, 3 (yellow).
        # We need to handle yellow phases if we switch.
        
        current_phase = self.conn.trafficlight.getPhase(self.ts_id)
        
        target_phase = action * 2 # 0 -> 0, 1 -> 2
        
        if current_phase == target_phase:
            # Keep current phase
            pass
        else:
            # We need to switch. 
            # Ideally we should go through yellow. 
            # For this simple env, let's just set the phase directly for now to avoid complexity of timing yellow.
            # Or better, if we are in 0 and want 2, set to 1 (yellow) for a few steps then 2.
            # But standard RL envs often simplify this: "Set Phase".
            self.conn.trafficlight.setPhase(self.ts_id, target_phase)
            
        # Run simulation for 5 seconds (decision interval)
        # This is important because we don't want to make decisions every 1s (too jittery)
        reward = 0
        for _ in range(5):
            self.conn.simulationStep()
            self.step_counter += 1
            reward += self._compute_reward()
            if self.step_counter >= self.num_seconds:
                break
                
        observation = self._get_observation()
        terminated = self.step_counter >= self.num_seconds
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Get queue and waiting time for each lane
        obs = []
        for lane in self.lanes:
            queue = self.conn.lane.getLastStepHaltingNumber(lane)
            wait = self.conn.lane.getWaitingTime(lane)
            obs.extend([queue, wait])
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self):
        # Reward is negative total waiting time + queue
        total_queue = 0
        total_wait = 0
        for lane in self.lanes:
            total_queue += self.conn.lane.getLastStepHaltingNumber(lane)
            total_wait += self.conn.lane.getWaitingTime(lane)
        return -(total_queue + total_wait)

    def close(self):
        traci.close()
