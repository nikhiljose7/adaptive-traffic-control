# Adaptive Traffic Signal Control System 🚦

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![SUMO](https://img.shields.io/badge/SUMO-1.15%2B-green.svg)](https://eclipse.dev/sumo/)

An intelligent traffic signal control system powered by Reinforcement Learning (PPO) and SUMO (Simulation of Urban MObility). This system optimizes traffic light timings to reduce congestion and waiting times through adaptive learning.

## 🌟 Features

- **Reinforcement Learning**: Uses Proximal Policy Optimization (PPO) algorithm to learn optimal signal phases
- **SUMO Simulation**: Realistic traffic simulation using Eclipse SUMO
- **Custom Environment**: Gymnasium-compatible environment for traffic control
- **Interactive Dashboard**: Streamlit-based UI to visualize real-time metrics and run simulations
- **Automatic Configuration**: Automatically detects SUMO installation path
- **Real-time Metrics**: Monitor queue length, waiting time, and reward during training and evaluation

## 🛠️ Prerequisites

- **Python 3.8+**
- **Eclipse SUMO**: Download and install from [eclipse.dev/sumo](https://eclipse.dev/sumo/)

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nikhiljose7/adaptive-traffic-control.git
    cd adaptive-traffic-control
    ```

2.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify SUMO installation**:
    ```bash
    sumo --version
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

The system uses a **custom Gymnasium environment** that interfaces with SUMO. The RL agent observes:
- **Queue Length**: Number of stopped vehicles at each lane
- **Waiting Time**: Cumulative waiting time for vehicles

Based on these observations, the PPO agent learns to decide:
1. Keep the current green phase
2. Switch to the next phase

**Reward Function**: The agent is rewarded for minimizing total waiting time and queue length.

## 📊 Algorithm Details

### Proximal Policy Optimization (PPO)
- **Policy**: Multi-layer perceptron (MLP) neural network
- **Objective**: Maximize cumulative reward while minimizing vehicle waiting time
- **Training**: On-policy gradient algorithm with clipped surrogate objective
- **Advantages**: 
  - More stable than vanilla policy gradient
  - Better sample efficiency than DQN
  - Robust to hyperparameter choices

### Environment Specifications
- **Observation Space**: 8-dimensional continuous space (4 lanes × 2 metrics)
- **Action Space**: Discrete(2) - NS Green or EW Green
- **Episode Length**: 3600 simulation seconds (1 hour)
- **Decision Interval**: 5 seconds per action

## 🏗️ Project Architecture

```
adaptive-traffic-control/
├── config/                     # SUMO Configuration Files
│   ├── network.net.xml         # Road network topology
│   ├── routes.rou.xml          # Traffic demand patterns
│   └── view.xml                # SUMO-GUI visualization settings
├── src/                        # Source Code
│   ├── simple_env.py           # Custom Gymnasium Environment
│   ├── train.py                # Training script (CLI)
│   ├── evaluate.py             # Evaluation script (CLI)
│   └── app.py                  # Streamlit Dashboard (GUI)
├── .gitignore                  # Git ignore patterns
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Ideas for Contributions
- Add more RL algorithms (DQN, A2C, SAC)
- Implement multi-intersection scenarios
- Add performance comparison tools
- Improve reward function design
- Create more complex traffic patterns

## 👨‍💻 Author

**Nikhil Jose**
- GitHub: [@nikhiljose7](https://github.com/nikhiljose7)

## 🙏 Acknowledgments

- [Eclipse SUMO](https://eclipse.dev/sumo/) - Traffic simulation platform
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) - RL algorithms library
- [Gymnasium](https://gymnasium.farama.org/) - Standard API for RL environments
- [Streamlit](https://streamlit.io/) - Web app framework

## 📚 References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐

