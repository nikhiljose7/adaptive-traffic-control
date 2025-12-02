import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from simple_env import TrafficSignalEnv

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(project_root, "config", "network.net.xml")
    route_file = os.path.join(project_root, "config", "routes.rou.xml")
    
    # Create environment
    env = TrafficSignalEnv(net_file=net_file, route_file=route_file, use_gui=False)
    
    # Check environment
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")
    
    # Initialize Agent
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10)
    
    # Train
    print("Starting training...")
    model.learn(total_timesteps=10000)
    print("Training finished!")
    
    # Save model
    model_path = os.path.join(project_root, "ppo_traffic_agent")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    env.close()

if __name__ == "__main__":
    main()
