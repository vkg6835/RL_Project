# 🎵 EEG Music Emotion Management Using Reinforcement Learning

> **Replication & Extension of Dutta et al. (2020)** — Comparing tabular RL algorithms for steering a listener's emotional state toward a target emotion ("happy") through intelligent music playlist generation.

---

## 📖 Overview

This project implements and compares **four tabular Reinforcement Learning algorithms** for EEG-based music emotion management on the **Geneva Emotion Wheel (GEW)**. An RL agent selects a sequence of 6 music clips (a playlist) to transition a listener's emotional state from an arbitrary starting emotion toward the target emotion **"happy"**.

The system uses **DEAP-dataset** participant EEG ratings (Valence–Arousal) and evaluates convergence using **angular error** on the 2-D emotion wheel.

### Key Features

- 🧠 **Four RL Algorithms**: Q-Learning, SARSA, Double Q-Learning, Double SARSA
- 🔄 **Leave-One-Subject-Out (LOSO)** cross-validation (32 subjects)
- 🎯 **Two-tier convergence criteria**: < 20° and < 30° angular error
- ⚡ **Reward shaping** with λ = −100 (with/without ablation)
- 🔍 **Hyperparameter grid search** via k-fold CV
- 📊 **Comprehensive visualization suite**: error curves, box plots, trajectory maps, and comparison charts

---

## 🏗️ Project Structure

```
RL_LAST/
├── music_emotion_ql.py            # Q-Learning (off-policy, Dutta et al. 2020 replication)
├── music_emotion_sarsa.py         # SARSA (on-policy TD)
├── music_emotion_double_ql.py     # Double Q-Learning (van Hasselt 2010)
├── music_emotion_double_sarsa.py  # Double SARSA (on-policy + dual Q-tables)
├── music_emotion_compare.py       # Run all algorithms & generate comparison plots
├── results_compare/               # Comparison plots output directory
├── results_double_sarsa/          # Double SARSA individual results
├── upload/                        # Curated plots for presentation/publication
├── output.png                     # Sample visualization output
├── RL_EEG_Updated_3000ep.pptx     # Presentation slides
└── README.md                      # This file
```

---

## 🔬 Algorithms

### 1. Q-Learning (Off-Policy)
Standard tabular Q-Learning as described in Dutta et al. (2020). Updates use the **greedy maximum** over the next state's Q-values:

```
Q(s,a) ← Q(s,a) + α [r + γ · max_a' Q(s',a') + φ(s,s') − Q(s,a)]
```

### 2. SARSA (On-Policy)
On-policy temporal difference learning. Updates use the **actual next action** chosen by the ε-greedy policy:

```
Q(s,a) ← Q(s,a) + α [r + γ · Q(s',a') + φ(s,s') − Q(s,a)]
```

### 3. Double Q-Learning (Off-Policy)
Maintains **two Q-tables** (QA, QB) to reduce maximization bias (van Hasselt, 2010):
- Action selection: ε-greedy on QA + QB
- Update: randomly pick one table, evaluate with the other

### 4. Double SARSA (On-Policy)
Combines Double Q-Learning's dual-table approach with SARSA's on-policy updates:
- Action selection: ε-greedy on QA + QB
- Update: randomly pick one table, evaluate the **actual next action** using the other

---

## 🌀 Environment Design

### Emotion Wheel (State Space)
- **8 grouped emotions**: happy, pleasant, exciting, calm, sad, negative, anger, fear
- Emotions are projected onto a **unit circle** in 2-D (Valence, Arousal) space
- Continuous states are **discretized** into a 20×20 grid (400 states)

### Actions
- Each action corresponds to a music clip represented as a 2-D (Valence, Arousal) vector
- Action pool: constructed from the training subjects' clip ratings

### Reward Function
- **Primary reward**: +1 if clip direction satisfies angular constraint (θ₁ + θ₂ ≤ π)
- **Terminal bonus**: +10 if angular error to target < 10°
- **Reward shaping**: φ(s,s') = λ (−100) if state unchanged, else 0

### State Transition
- Additive transition: `s' = s + clip_vec`
- Boundary projection to unit circle if ‖s'‖ > 1

---

## 📋 Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib

Install dependencies:
```bash
pip install numpy pandas matplotlib
```

---

## 🚀 Usage

### Run Individual Algorithms

```bash
# Q-Learning
python music_emotion_ql.py --data_root /path/to/data/Metacsv --episodes 10000

# SARSA
python music_emotion_sarsa.py --data_root /path/to/data/Metacsv --episodes 10000

# Double Q-Learning
python music_emotion_double_ql.py --data_root /path/to/data/Metacsv --episodes 10000

# Double SARSA
python music_emotion_double_sarsa.py --data_root /path/to/data/Metacsv --episodes 10000
```

### Run Full Comparison (All Algorithms)

```bash
python music_emotion_compare.py --data_root /path/to/data/Metacsv --episodes 10000
```

### With Grid Search for Hyperparameters

```bash
python music_emotion_compare.py --data_root /path/to/data/Metacsv --episodes 10000 --grid_search
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_root` | `/home/sahil/Desktop/Rl_Project_Final/data/Metacsv` | Directory containing `emotion_summary.csv` and `participant_ratings.csv` |
| `--output_dir` | `results_<algo>` | Output directory for plots |
| `--episodes` | `10000` | Number of training episodes |
| `--alpha` | `0.1` | Learning rate |
| `--gamma` | `0.6` | Discount factor |
| `--epsilon` | `0.1` | Exploration rate (ε-greedy) |
| `--grid_search` | `False` | Enable hyperparameter grid search |
| `--grid_folds` | `10` | Number of folds for grid search CV |
| `--grid_episodes` | `300` | Episodes per grid search trial |

---

## 📊 Output Visualizations

The comparison script generates the following plots:

| Plot | Description |
|---|---|
| `compare_angular_error_with.png` | Mean angular error per iteration — all algorithms (with shaping) |
| `compare_convergence_bar_with.png` | Convergence bar chart (< 20° / < 30°) — with shaping |
| `compare_final_error_box_with.png` | Box plot of final angular errors — with shaping |
| `compare_shaping_effect.png` | Reward shaping ablation — side-by-side per algorithm |
| `compare_shaping_convergence.png` | Shaping impact on convergence count — all algorithms |
| `*_angular_error.png` | Per-algorithm mean angular error curves |
| `*_all_trajectories.png` | Emotion wheel trajectories for all 32 subjects |
| `*_reward_shaping.png` | Per-algorithm reward shaping comparison |

---

## 📈 Evaluation Protocol

1. **Leave-One-Subject-Out (LOSO)**: Train on 31 subjects, evaluate on the held-out subject
2. **Playlist length**: 6 clips per evaluation episode
3. **Convergence criteria**:
   - **Primary (< 20°)**: Final clip angular error < 20° (strict)
   - **Secondary (< 30°)**: Final clip angular error < 30° (relaxed)
4. **Baseline**: Paper reports 19/32 convergence and 57.0° ± 2.8° mean error

---

## 📂 Data Requirements

The project expects two CSV files in the `--data_root` directory:

| File | Description |
|---|---|
| `emotion_summary.csv` | Emotion labels with Valence/Arousal values for the GEW |
| `participant_ratings.csv` | Per-subject, per-trial Valence/Arousal ratings (32 subjects × 40 clips) |

> **Note**: If data files are not found, the scripts will fall back to synthetic (randomly generated) data for demonstration purposes.

---

## 📚 References

- **Dutta, S. et al. (2020)** — EEG-based music emotion management using reinforcement learning
- **van Hasselt, H. (2010)** — Double Q-Learning

---

## 📝 License

This project is for academic and research purposes.
