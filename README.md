# Adaptive Traffic Signal Control System 🚦

An intelligent traffic signal control system powered by Reinforcement Learning (PPO) and SUMO (Simulation of Urban MObility). This system optimizes traffic light timings to reduce congestion and waiting times.

## 🌟 Features
- **Reinforcement Learning**: Uses Proximal Policy Optimization (PPO) to learn optimal signal phases.
- **SUMO Simulation**: Realistic traffic simulation using Eclipse SUMO.
- **Custom Environment**: Gymnasium-compatible environment for traffic control.
- **Interactive Dashboard**: Streamlit-based UI to visualize real-time metrics and run simulations.
- **Automatic Configuration**: Automatically detects SUMO installation path.

## 🛠️ Prerequisites
- **Python 3.8+**
- **Eclipse SUMO**: Download and install from [eclipse.dev/sumo](https://eclipse.dev/sumo/).

## 📦 Installation

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/yourusername/adaptive-traffic-control.git
    cd adaptive-traffic-control
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Train the Agent
Train the RL agent from scratch. This will save the model to `ppo_traffic_agent.zip`.
```bash
python src/train.py
```

### 2. Run the Streamlit Dashboard (Recommended)
The easiest way to run the simulation and see results.
```bash
streamlit run src/app.py
```
- Check **"Show Simulation GUI"** in the sidebar to see the traffic animation.
- Click **"Start Simulation"**.

### 3. Run Evaluation Script (CLI)
Run the evaluation directly with SUMO-GUI.
```bash
python src/evaluate.py
```

## 📂 Project Structure
```
project/
├── config/                 # SUMO configuration files
│   ├── network.net.xml     # Road network
│   ├── routes.rou.xml      # Traffic demand
│   └── view.xml            # GUI view settings
├── src/                    # Source code
│   ├── simple_env.py       # Custom RL Environment
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation script
│   └── app.py              # Streamlit Dashboard
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🧠 How it Works
The system observes the **queue length** and **waiting time** of vehicles at the intersection. Based on this state, the RL agent (PPO) decides whether to:
1.  Keep the current green phase.
2.  Switch to the next phase.

The goal is to minimize the total waiting time of all vehicles.
