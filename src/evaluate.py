import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from simple_env import TrafficSignalEnv
import numpy as np

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(project_root, "config", "network.net.xml")
    route_file = os.path.join(project_root, "config", "routes.rou.xml")
    model_path = os.path.join(project_root, "ppo_traffic_agent")
    
    # Create environment with GUI for visualization
    env = TrafficSignalEnv(net_file=net_file, route_file=route_file, use_gui=True)
    
    # Load model
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}. Please train first.")
        return

    model = PPO.load(model_path)
    
    # Evaluate
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    print("Starting evaluation...")
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        
    print(f"Evaluation finished! Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()
