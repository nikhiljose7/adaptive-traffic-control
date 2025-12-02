import streamlit as st
import os
import sys
import time
import pandas as pd
import altair as alt
from stable_baselines3 import PPO
from simple_env import TrafficSignalEnv

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Traffic Control AI", layout="wide")

st.title("🚦 Adaptive Traffic Signal Control System")
st.markdown("This dashboard visualizes the performance of the Reinforcement Learning agent in controlling traffic signals.")

# Sidebar for controls
st.sidebar.header("Simulation Controls")
delay = st.sidebar.slider("Step Delay (ms)", 0, 1000, 100)
num_steps = st.sidebar.number_input("Number of Steps", min_value=100, max_value=3600, value=500)
show_gui = st.sidebar.checkbox("Show Simulation GUI", value=True)
start_btn = st.sidebar.button("Start Simulation")

# Placeholders for metrics
col1, col2, col3 = st.columns(3)
with col1:
    metric_reward = st.empty()
with col2:
    metric_wait = st.empty()
with col3:
    metric_queue = st.empty()

# Placeholder for charts
chart_placeholder = st.empty()

def run_simulation():
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    net_file = os.path.join(project_root, "config", "network.net.xml")
    route_file = os.path.join(project_root, "config", "routes.rou.xml")
    model_path = os.path.join(project_root, "ppo_traffic_agent")
    
    # Check model
    if not os.path.exists(model_path + ".zip"):
        st.error(f"Model not found at {model_path}. Please train first.")
        return

    # Initialize Env
    env = TrafficSignalEnv(net_file=net_file, route_file=route_file, use_gui=show_gui)
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    total_reward = 0
    data = []
    
    progress_bar = st.progress(0)
    
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Extract metrics from observation
        # Obs structure: [q0, w0, q1, w1, q2, w2, q3, w3]
        total_queue = sum(obs[0::2])
        total_wait = sum(obs[1::2])
        
        data.append({
            "Step": step,
            "Total Queue": total_queue,
            "Total Waiting Time": total_wait,
            "Reward": reward
        })
        
        # Update Metrics
        metric_reward.metric("Cumulative Reward", f"{total_reward:.2f}")
        metric_wait.metric("Current Waiting Time", f"{total_wait:.2f}")
        metric_queue.metric("Current Queue Length", f"{total_queue:.0f}")
        
        # Update Charts every 10 steps to save performance
        if step % 10 == 0:
            df = pd.DataFrame(data)
            
            c = alt.Chart(df).mark_line().encode(
                x='Step',
                y='Total Waiting Time',
                tooltip=['Step', 'Total Waiting Time']
            ).properties(title="Total Waiting Time over Time")
            
            chart_placeholder.altair_chart(c, use_container_width=True)
        
        progress_bar.progress((step + 1) / num_steps)
        
        if delay > 0:
            time.sleep(delay / 1000.0)
            
        if terminated or truncated:
            break
            
    env.close()
    st.success("Simulation Finished!")

if start_btn:
    run_simulation()
