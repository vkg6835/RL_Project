# 🧠 Reinforcement Learning for EEG-based Music Emotion Management

Welcome to the **RL_FULL** workspace. This project explores the use of tabular Reinforcement Learning algorithms to manage a listener's emotional state through intelligent music selection, based on EEG signals and the Valence-Arousal model.

---

## 📂 Workspace Structure

This repository contains multiple iterations and experiments of the Reinforcement Learning pipeline.

- **[RL_LAST](./RL_LAST)**: 🚀 **Primary Development Directory**. Contains the latest refined algorithms, Reward Shaping logic, and LOSO cross-validation framework.
- **[Rl_Project_Final](./Rl_Project_Final)**: A stable version of the project used for initial finalization.
- **[Rl_Final](./Rl_Final)** / **[Rl_Second](./Rl_Second)** / **[Rl_Project](./Rl_Project)**: Previous iterations and experimental versions.
- **[RLLL](./RLLL)**: Experimental playground for reinforcement learning logic.

---

## 🎯 Project Objective

The goal is to steer a listener's emotional state toward a target emotion (e.g., **"Happy"**) using a sequence of music clips. The system uses the **Geneva Emotion Wheel (GEW)** as the state space and **EEG-based ratings** (Valence & Arousal) to determine transitions.

### Key Algorithms Implemented
1. **Q-Learning**: Off-policy tabular RL (Baseline).
2. **SARSA**: On-policy tabular RL.
3. **Double Q-Learning**: Dual-table approach to reduce maximization bias.
4. **Double SARSA**: Hybrid approach combining dual-tables with on-policy updates.

---

## 🚀 Getting Started (RL_LAST)

For the most up-to-date implementation, navigate to the `RL_LAST` directory.

### 📋 Prerequisites
Ensure you have the following installed:
```bash
pip install numpy pandas matplotlib
```

### 🏃 Running the Pipeline
To run a comparison across all algorithms with default settings:
```bash
cd RL_LAST
python music_emotion_compare.py --data_root ./data/Metacsv --episodes 10000
```

To run a specific algorithm (e.g., SARSA):
```bash
python music_emotion_sarsa.py --episodes 10000
```

---

## 📊 Evaluation & Results

The project evaluates performance using:
- **Angular Error**: Distance from the target emotion on the wheel.
- **Convergence Rate**: Percentage of subjects reaching the target within 20° or 30°.
- **Leave-One-Subject-Out (LOSO)**: Ensuring robustness across different users (32 subjects).

### Recent Improvements
- **Reward Shaping**: Implementation of $\lambda$ penalties (e.g., -10, -100) to penalize stagnant states and improve convergence speed.
- **DENSE Dataset Support**: Scaling and normalization of DENSE emotion vectors for higher granularity.

---

## 📜 References
- **Dutta, S. et al. (2020)**: *EEG-based music emotion management using reinforcement learning*.
- **van Hasselt, H. (2010)**: *Double Q-Learning*.

---

## 📝 License
This project is for research and academic purposes.
# RL_Project
