import streamlit as st
import os
import sys
import time
import pandas as pd
import altair as alt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from simple_env import TrafficSignalEnv
import numpy as np
from datetime import datetime

# Ensure we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Traffic Control AI", 
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🚦 Adaptive Traffic Signal Control System</h1><p>Train and evaluate RL agents for intelligent traffic management</p></div>', unsafe_allow_html=True)

# Define paths globally
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
net_file = os.path.join(project_root, "config", "network.net.xml")
route_file = os.path.join(project_root, "config", "routes.rou.xml")

# Custom callback for Streamlit training visualization
class StreamlitCallback(BaseCallback):
    def __init__(self, progress_bar, metrics_placeholder, chart_placeholder, verbose=0):
        super().__init__(verbose)
        self.progress_bar = progress_bar
        self.metrics_placeholder = metrics_placeholder
        self.chart_placeholder = chart_placeholder
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # Check if episode is done
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
            
            # Update metrics
            if len(self.episode_rewards) > 0:
                col1, col2, col3, col4 = self.metrics_placeholder.columns(4)
                with col1:
                    st.metric("Episodes Completed", len(self.episode_rewards))
                with col2:
                    st.metric("Last Episode Reward", f"{self.episode_rewards[-1]:.2f}")
                with col3:
                    st.metric("Average Reward (Last 10)", f"{np.mean(self.episode_rewards[-10:]):.2f}")
                with col4:
                    st.metric("Training Progress", f"{self.num_timesteps}/{self.model.n_steps}")
                
                # Update chart every 5 episodes
                if len(self.episode_rewards) % 5 == 0:
                    df = pd.DataFrame({
                        'Episode': range(1, len(self.episode_rewards) + 1),
                        'Reward': self.episode_rewards
                    })
                    
                    chart = alt.Chart(df).mark_line(point=True, strokeWidth=3).encode(
                        x=alt.X('Episode:Q', title='Episode'),
                        y=alt.Y('Reward:Q', title='Episode Reward'),
                        tooltip=['Episode', 'Reward']
                    ).properties(
                        title='Training Progress - Episode Rewards',
                        height=300
                    )
                    
                    self.chart_placeholder.altair_chart(chart, use_container_width=True)
        
        # Update progress bar
        progress = self.num_timesteps / self.model._total_timesteps
        self.progress_bar.progress(min(progress, 1.0))
        
        return True

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Train Agent", "📊 Evaluate Agent", "📁 Model Manager"])

# ==================== TAB 1: TRAIN AGENT ====================
with tab1:
    st.header("Train a New RL Agent")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚙️ Training Configuration")
        
        model_name = st.text_input("Model Name", value=f"ppo_traffic_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        col_a, col_b = st.columns(2)
        with col_a:
            total_timesteps = st.number_input("Total Timesteps", min_value=1000, max_value=1000000, value=50000, step=1000,
                                              help="Total number of timesteps to train")
            learning_rate = st.number_input("Learning Rate", min_value=0.00001, max_value=0.01, value=0.0003, format="%.5f",
                                           help="Learning rate for the optimizer")
            n_steps = st.number_input("Steps per Update", min_value=128, max_value=4096, value=2048, step=128,
                                     help="Number of steps to collect before updating the policy")
        
        with col_b:
            batch_size = st.number_input("Batch Size", min_value=16, max_value=512, value=64, step=16,
                                        help="Minibatch size for optimization")
            n_epochs = st.number_input("Epochs per Update", min_value=1, max_value=20, value=10,
                                      help="Number of epochs when optimizing the surrogate loss")
            gamma = st.number_input("Discount Factor (γ)", min_value=0.9, max_value=0.999, value=0.99, format="%.3f",
                                   help="Discount factor for future rewards")
        
        use_gui_train = st.checkbox("Show SUMO GUI during training", value=False,
                                   help="⚠️ Warning: GUI will slow down training significantly")
    
    with col2:
        st.subheader("📋 Training Info")
        st.info("""
        **PPO Algorithm**
        
        Proximal Policy Optimization is a policy gradient method that:
        - Learns a policy to control traffic lights
        - Balances exploration and exploitation
        - Provides stable training
        
        **Tips:**
        - Start with 50,000 timesteps
        - Disable GUI for faster training
        - Monitor episode rewards
        """)
    
    st.markdown("---")
    
    # Training button
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        try:
            st.subheader("📈 Training Progress")
            
            # Create placeholders
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            
            status_text.info("🔄 Initializing environment...")
            
            # Create environment
            env = TrafficSignalEnv(net_file=net_file, route_file=route_file, use_gui=use_gui_train)
            
            status_text.info("🤖 Creating PPO model...")
            
            # Create model
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma
            )
            
            status_text.info("🏋️ Training in progress...")
            
            # Create callback
            callback = StreamlitCallback(progress_bar, metrics_placeholder, chart_placeholder)
            
            # Train
            start_time = time.time()
            model.learn(total_timesteps=total_timesteps, callback=callback)
            end_time = time.time()
            
            # Save model
            model_path = os.path.join(project_root, model_name)
            model.save(model_path)
            
            env.close()
            
            # Show success
            training_time = end_time - start_time
            st.success(f"""
            ✅ **Training Complete!**
            
            - Training Time: {training_time/60:.2f} minutes
            - Total Episodes: {len(callback.episode_rewards)}
            - Final Average Reward: {np.mean(callback.episode_rewards[-10:]):.2f}
            - Model saved as: `{model_name}.zip`
            """)
            
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            env.close()

# ==================== TAB 2: EVALUATE AGENT ====================
with tab2:
    st.header("Evaluate Trained Agent")
    
    # Sidebar-like configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎮 Simulation Controls")
        
        # List available models
        model_files = [f.replace('.zip', '') for f in os.listdir(project_root) if f.endswith('.zip') and 'ppo' in f.lower()]
        
        if model_files:
            selected_model = st.selectbox("Select Model", model_files, help="Choose a trained model to evaluate")
        else:
            st.warning("⚠️ No trained models found. Please train a model first.")
            selected_model = None
        
        col_a, col_b = st.columns(2)
        with col_a:
            num_steps = st.number_input("Number of Steps", min_value=100, max_value=3600, value=500,
                                       help="Duration of simulation in steps")
            delay = st.slider("Step Delay (ms)", 0, 1000, 100,
                            help="Delay between steps for visualization")
        with col_b:
            show_gui = st.checkbox("Show SUMO GUI", value=True,
                                  help="Display the traffic simulation window")
    
    with col2:
        st.subheader("ℹ️ Evaluation Info")
        st.info("""
        **Evaluation Mode**
        
        Test your trained agent:
        - Uses deterministic policy
        - Real-time visualization
        - Performance metrics
        
        **Metrics:**
        - Queue Length: Stopped vehicles
        - Waiting Time: Time vehicles wait
        - Cumulative Reward: Overall performance
        """)
    
    st.markdown("---")
    
    # Evaluation button
    if st.button("▶️ Start Evaluation", type="primary", use_container_width=True, disabled=(selected_model is None)):
        try:
            model_path = os.path.join(project_root, selected_model)
            
            # Create placeholders
            st.subheader("📊 Live Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                metric_reward = st.empty()
            with col2:
                metric_wait = st.empty()
            with col3:
                metric_queue = st.empty()
            
            chart_placeholder = st.empty()
            progress_bar = st.progress(0)
            
            # Initialize Env
            env = TrafficSignalEnv(net_file=net_file, route_file=route_file, use_gui=show_gui)
            model = PPO.load(model_path)
            
            obs, _ = env.reset()
            total_reward = 0
            data = []
            
            for step in range(num_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                # Extract metrics from observation
                total_queue = sum(obs[0::2])
                total_wait = sum(obs[1::2])
                
                data.append({
                    "Step": step,
                    "Total Queue": total_queue,
                    "Total Waiting Time": total_wait,
                    "Reward": reward
                })
                
                # Update Metrics
                metric_reward.metric("🎯 Cumulative Reward", f"{total_reward:.2f}", 
                                    delta=f"{reward:.2f}" if step > 0 else None)
                metric_wait.metric("⏱️ Waiting Time", f"{total_wait:.2f} s")
                metric_queue.metric("🚗 Queue Length", f"{int(total_queue)} vehicles")
                
                # Update Charts every 10 steps
                if step % 10 == 0 and len(data) > 1:
                    df = pd.DataFrame(data)
                    
                    chart = alt.Chart(df).mark_area(
                        line={'color':'#667eea'},
                        color=alt.Gradient(
                            gradient='linear',
                            stops=[alt.GradientStop(color='white', offset=0),
                                   alt.GradientStop(color='#667eea', offset=1)],
                            x1=1, x2=1, y1=1, y2=0
                        )
                    ).encode(
                        x=alt.X('Step:Q', title='Simulation Step'),
                        y=alt.Y('Total Waiting Time:Q', title='Waiting Time (s)'),
                        tooltip=['Step', 'Total Waiting Time', 'Total Queue']
                    ).properties(
                        title='Traffic Metrics Over Time',
                        height=300
                    )
                    
                    chart_placeholder.altair_chart(chart, use_container_width=True)
                
                progress_bar.progress((step + 1) / num_steps)
                
                if delay > 0:
                    time.sleep(delay / 1000.0)
                    
                if terminated or truncated:
                    break
            
            env.close()
            
            # Final statistics
            st.success("✅ Evaluation Complete!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Steps", len(data))
            with col2:
                st.metric("Average Queue", f"{np.mean([d['Total Queue'] for d in data]):.2f}")
            with col3:
                st.metric("Average Wait Time", f"{np.mean([d['Total Waiting Time'] for d in data]):.2f}")
                
        except Exception as e:
            st.error(f"❌ Evaluation failed: {str(e)}")
            try:
                env.close()
            except:
                pass

# ==================== TAB 3: MODEL MANAGER ====================
with tab3:
    st.header("Model Management")
    
    # List all models
    model_files = [(f, os.path.getsize(os.path.join(project_root, f)), 
                    os.path.getmtime(os.path.join(project_root, f))) 
                   for f in os.listdir(project_root) if f.endswith('.zip') and 'ppo' in f.lower()]
    
    if model_files:
        st.subheader(f"📦 Available Models ({len(model_files)})")
        
        models_data = []
        for filename, size, mtime in model_files:
            models_data.append({
                "Model Name": filename.replace('.zip', ''),
                "Size (KB)": f"{size / 1024:.2f}",
                "Last Modified": datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        df = pd.DataFrame(models_data)
        st.dataframe(df, use_container_width=True)
        
        st.markdown("---")
        
        # Delete model
        st.subheader("🗑️ Delete Model")
        model_to_delete = st.selectbox("Select model to delete", [m['Model Name'] for m in models_data])
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("Delete Selected Model", type="secondary"):
                try:
                    os.remove(os.path.join(project_root, model_to_delete + '.zip'))
                    st.success(f"✅ Deleted {model_to_delete}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to delete: {str(e)}")
        
    else:
        st.info("📭 No models found. Train a model in the 'Train Agent' tab to get started!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🚦 Adaptive Traffic Control System | Powered by Reinforcement Learning (PPO) & SUMO</p>
</div>
""", unsafe_allow_html=True)
