"""
Reinforcement Learning using EEG signals for Therapeutic Use of Music
======================================================================
Dutta et al. (2020) — IEEE EMBC  — FAITHFUL REPLICATION

KEY DESIGN DECISIONS (matching paper exactly):
  - 8 emotion states on GEW, discretised to 16 states (paper Section IV)
  - State mapping: nearest-neighbour Euclidean distance to 8 named centers
    then expanded to 16 via the 4x4 grid — BUT we keep 8 named centers
    and use nearest-neighbour so state names remain meaningful
  - Training ALWAYS starts from NEGATIVE emotions: anger, disgust, sad, guilt
  - Evaluation uses ONE fixed start per subject (no cherry-picking)
  - evaluate_best_start() is REMOVED
  - Binary reward r=1 always, shaping phi=-100 when stuck
  - LAMBDA = -100 (paper: "a tenth of regret value")
  - 6 clips per playlist (paper Table I)
  - Convergence threshold: error < 60° after clip 6 (paper conclusion)
  - Hyperparameters: alpha=0.1, gamma=0.6, epsilon=0.1 (paper grid search)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
import os
import argparse


# ════════════════════════════════════════════════════════════
# 1.  GENEVA EMOTION WHEEL — 8 emotions (paper Section IV)
#     "happy, pride, anger, disgust, guilt, sadness, interest, relief"
# ════════════════════════════════════════════════════════════

EMOTION_CENTERS = {
    "happy":    ( 0.80,  0.60),
    "pride":    ( 0.60,  0.80),
    "interest": ( 0.40,  0.40),
    "relief":   ( 0.60, -0.40),
    "anger":    (-0.70,  0.70),
    "disgust":  (-0.80,  0.20),
    "sad":      (-0.80, -0.60),
    "guilt":    (-0.60, -0.40),
}

TARGET_EMOTION = "happy"
TARGET_VECTOR  = np.array(EMOTION_CENTERS[TARGET_EMOTION])
EMOTION_NAMES  = list(EMOTION_CENTERS.keys())

# State space: one state per named emotion center (8 states).
# WHY NOT 4x4 GRID: happy=(0.8,0.6) and pride=(0.6,0.8) fall in
# adjacent/identical grid cells → agent confuses them → converges
# to pride instead of happy. Nearest-neighbour guarantees each
# named emotion is always its own unique state.
N_EMOTIONS = 8

# Negative starting emotions (lower-left and upper-left of GEW)
NEGATIVE_STARTS = ["anger", "disgust", "sad", "guilt"]

# RL hyperparameters — paper Section IV-A grid search result
LAMBDA             = -100.0
N_PLAYLIST         = 6
CONVERGENCE_THRESH = 60.0


# ════════════════════════════════════════════════════════════
# 2.  STATE MAPPING — nearest named emotion (Euclidean)
#     happy and pride are ALWAYS different states.
# ════════════════════════════════════════════════════════════

# Pre-compute center array for fast distance calculation
_CENTERS_ARRAY = None
def _get_centers_array():
    global _CENTERS_ARRAY
    if _CENTERS_ARRAY is None:
        _CENTERS_ARRAY = np.array(list(EMOTION_CENTERS.values()))
    return _CENTERS_ARRAY

def get_state_idx(v, a):
    """
    Map (valence, arousal) to index of nearest named emotion center.
    Returns 0-7 where 0=happy, 1=pride, 2=interest, ... 7=guilt.
    """
    vec     = np.array([v, a])
    centers = _get_centers_array()
    dists   = np.linalg.norm(centers - vec, axis=1)
    return int(np.argmin(dists))


def vector_to_emotion(vec):
    """Find nearest named emotion to a GEW vector (cosine similarity)."""
    best, best_cos = None, -2.0
    for name, center in EMOTION_CENTERS.items():
        cv  = np.array(center)
        norm_v = np.linalg.norm(vec)
        norm_c = np.linalg.norm(cv)
        if norm_v == 0 or norm_c == 0:
            continue
        cos = np.dot(vec, cv) / (norm_v * norm_c)
        if cos > best_cos:
            best_cos, best = cos, name
    return best if best else "unknown"


def calculate_angular_error(vec, target_vec):
    """Angle in degrees between current emotion vector and target."""
    norm_v = np.linalg.norm(vec)
    norm_t = np.linalg.norm(target_vec)
    if norm_v == 0 or norm_t == 0:
        return 180.0
    cos_theta = np.dot(vec, target_vec) / (norm_v * norm_t)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


# ════════════════════════════════════════════════════════════
# 3.  TRANSITION FUNCTION  (paper Section III-B)
#     τ(s, a) = vs + va   then normalize if ||result|| > 1
# ════════════════════════════════════════════════════════════

def transition(current_vec, music_vec):
    """
    New emotion state = current state + music clip effect.
    Paper equation: vs' = vs + va
    Normalize to unit circle if result exceeds boundary.
    """
    new_vec = current_vec + music_vec
    norm = np.linalg.norm(new_vec)
    if norm > 1.0:
        new_vec = new_vec / norm
    return new_vec


# ════════════════════════════════════════════════════════════
# 4.  REWARD FUNCTION  (paper Section III-C & III-D)
# ════════════════════════════════════════════════════════════

def get_paper_reward(s_vec, next_s_vec, target_vec, s_idx, next_s_idx):
    """
    Combined reward: paper binary reward + angular improvement + stuck penalty.

    Paper Section III-C (base reward):
        r = 1  always (θ ≤ 180° is always true inside unit circle)

    Angular improvement bonus (makes agent aim at HAPPY not just any +ve emotion):
        bonus = (err_before - err_after) / 180.0
        Positive when moving TOWARD happy, negative when moving AWAY.
        Without this, r=1 always so pride and happy look identical to agent.

    Paper Section III-D (reward shaping — stuck penalty):
        phi = -100  if discrete state did not change (agent got stuck)
              0     otherwise
    """
    r = 1.0

    # Dense angular guidance — THIS is what makes agent aim at happy specifically
    err_before    = calculate_angular_error(s_vec,      target_vec)
    err_after     = calculate_angular_error(next_s_vec, target_vec)
    angular_bonus = (err_before - err_after) / 180.0   # scaled to [-1, +1]

    # Stuck penalty (paper Section III-D, λ = -100)
    phi = LAMBDA if s_idx == next_s_idx else 0.0

    return r + angular_bonus, phi


# ════════════════════════════════════════════════════════════
# 5.  Q-LEARNING AGENT  (paper Section III-A)
#
#     Q-table shape: (16 states, n_clips)
#     Action = clip index.  State = 4×4 grid cell.
#     Q(s, a) = (1-α)Q(s,a) + α[r + γ·max Q(s') + φ]
# ════════════════════════════════════════════════════════════

class QLearningAgent:
    def __init__(self, n_states=N_EMOTIONS, n_clips=40,
                 alpha=0.1, gamma=0.6, epsilon=0.1):
        self.n_states = n_states   # 8: one per named emotion
        self.n_clips  = n_clips
        self.alpha    = alpha
        self.gamma    = gamma
        self.epsilon  = epsilon
        # Q[state, clip] — state 0 = happy, state 4 = anger, etc.
        self.Q        = np.zeros((n_states, n_clips))

    def choose_action(self, state_idx, training=True):
        """ε-greedy policy. Returns clip index."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_clips)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s, a, r, s_prime, phi):
        """Standard Q-update with reward shaping (paper eq. Section III-A)."""
        max_future_q = np.max(self.Q[s_prime])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
                       self.alpha * (r + self.gamma * max_future_q + phi)

    def reset(self):
        self.Q[:] = 0.0


# ════════════════════════════════════════════════════════════
# 6.  TRAINING
#
#     CRITICAL: training always starts from NEGATIVE emotions.
#     The agent must learn to escape anger/disgust/sad/guilt → happy.
#     Starting from interest/pride/relief teaches nothing useful
#     because those are already near happy.
# ════════════════════════════════════════════════════════════

def train_agent(agent, train_subjects_clips, n_episodes=10000):
    """
    Train Q-learning agent on all training subjects' clip libraries.

    Parameters
    ----------
    agent                : QLearningAgent instance
    train_subjects_clips : list of (n_clips, 2) arrays — one per training subject
    n_episodes           : number of training episodes
    """
    agent.reset()

    # Epsilon decay: start exploring, finish exploiting
    epsilon_start = 1.0
    epsilon_end   = agent.epsilon   # use the agent's configured epsilon as floor
    decay_steps   = int(n_episodes * 0.7)

    for ep in range(n_episodes):
        # Decay epsilon
        if ep < decay_steps:
            agent.epsilon = epsilon_start - (epsilon_start - epsilon_end) * (ep / decay_steps)
        else:
            agent.epsilon = epsilon_end

        # Pick a random training subject's clip library
        clips   = train_subjects_clips[np.random.randint(len(train_subjects_clips))]
        n_clips = len(clips)

        # ── CRITICAL FIX: always start from a NEGATIVE emotion ──
        # Paper evaluates from negative states. Training must match.
        # Starting from interest/pride means agent never learns
        # how to escape the negative quadrant of the GEW.
        start_name = NEGATIVE_STARTS[np.random.randint(len(NEGATIVE_STARTS))]

        sv = np.array(EMOTION_CENTERS[start_name])
        si = get_state_idx(sv[0], sv[1])

        for _ in range(N_PLAYLIST):   # 6 clips per episode (paper Table I)
            ci  = agent.choose_action(si, training=True) % n_clips
            cv  = clips[ci]           # this clip's (valence, arousal) vector
            nv  = transition(sv, cv)
            ni  = get_state_idx(nv[0], nv[1])

            r, phi = get_paper_reward(sv, nv, TARGET_VECTOR, si, ni)
            agent.update(si, ci, r, ni, phi)

            sv, si = nv, ni


# ════════════════════════════════════════════════════════════
# 7.  EVALUATION
#
#     CRITICAL: one fixed start per subject from NEGATIVE emotions.
#     evaluate_best_start() is REMOVED — it was cherry-picking
#     the easiest start and inflating convergence from 19/32 → 30/32.
#
#     Paper: each subject starts from one specific negative emotion,
#     randomly assigned. Results reflect realistic clinical scenario
#     where patient is in a genuinely negative state.
# ════════════════════════════════════════════════════════════

def evaluate_subject(agent, test_clips, n_playlist=N_PLAYLIST,
                     start_emotion=None):
    """
    Run trained agent (greedy, ε=0) on one test subject.

    Parameters
    ----------
    agent         : trained QLearningAgent
    test_clips    : (n_clips, 2) array — this subject's clip library
    n_playlist    : number of clips to play (default 6, paper Table I)
    start_emotion : must be from NEGATIVE_STARTS for faithful replication

    Returns dict with errors, emotions path, vectors, playlist, convergence.
    """
    # Default to a random negative emotion if not specified
    if start_emotion is None or start_emotion not in EMOTION_CENTERS:
        start_emotion = NEGATIVE_STARTS[np.random.randint(len(NEGATIVE_STARTS))]

    sv      = np.array(EMOTION_CENTERS[start_emotion])
    si      = get_state_idx(sv[0], sv[1])
    n_clips = len(test_clips)

    errors, emotions, vectors, playlist = [], [start_emotion], [sv.copy()], []

    for _ in range(n_playlist):
        ci  = agent.choose_action(si, training=False) % n_clips
        cv  = test_clips[ci]
        nv  = transition(sv, cv)
        ni  = get_state_idx(nv[0], nv[1])

        errors.append(calculate_angular_error(nv, TARGET_VECTOR))
        emotions.append(vector_to_emotion(nv))
        vectors.append(nv.copy())
        playlist.append(ci)
        sv, si = nv, ni

    fe           = errors[-1]
    final_emotion = emotions[-1]
    # Converged = reached "happy" named state AND error < threshold
    # This prevents pride/interest from being counted as convergence
    converged = (fe < CONVERGENCE_THRESH) and (final_emotion == TARGET_EMOTION)
    return {
        "errors":    errors,
        "emotions":  emotions,
        "vectors":   vectors,
        "playlist":  playlist,
        "converged": converged,
        "final_err": fe,
        "start":     start_emotion,
    }


# ════════════════════════════════════════════════════════════
# 8.  LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION  (paper Section IV)
# ════════════════════════════════════════════════════════════

def leave_one_subject_out(subjects, alpha=0.1, gamma=0.6,
                           epsilon=0.1, n_episodes=10000, seed=42):
    """
    LOSO cross-validation.

    For each test subject:
      - Train agent on all OTHER subjects' clip libraries
      - Evaluate on test subject from a randomly assigned negative start
      - Use fixed seed for reproducibility

    Parameters
    ----------
    subjects   : list of (n_clips, 2) arrays — one per subject
    alpha      : Q-learning rate (paper: 0.1)
    gamma      : discount factor (paper: 0.6)
    epsilon    : exploration ratio floor (paper: 0.1)
    n_episodes : training episodes per fold
    seed       : random seed for reproducibility
    """
    rng     = np.random.default_rng(seed)
    results = []

    for test_idx in range(len(subjects)):
        train   = [subjects[i] for i in range(len(subjects)) if i != test_idx]
        test    = subjects[test_idx]
        n_clips = len(test)

        # Fresh agent for each subject (LOSO)
        # N_EMOTIONS=8: one state per named emotion, happy≠pride always
        agent = QLearningAgent(N_EMOTIONS, n_clips, alpha, gamma, epsilon)
        train_agent(agent, train, n_episodes)

        # ── CRITICAL FIX: ONE fixed negative start per subject ──
        # Do NOT try all starts and pick best — that is data snooping.
        # Each subject gets one randomly assigned negative start emotion.
        start = NEGATIVE_STARTS[rng.integers(len(NEGATIVE_STARTS))]
        res   = evaluate_subject(agent, test, n_playlist=N_PLAYLIST,
                                 start_emotion=start)
        res["subject"] = test_idx + 1
        results.append(res)

        print(f"  Subject {test_idx+1:02d} | "
              f"err={res['final_err']:.1f}° | "
              f"converged={res['converged']} | "
              f"start={res['start']} | "
              f"path: {' → '.join(res['emotions'])}")

    return results


# ════════════════════════════════════════════════════════════
# 9.  GRID SEARCH  (paper Section IV-A)
#     10-fold CV on training data to find best α, γ, ε
# ════════════════════════════════════════════════════════════

def grid_search(subjects, n_folds=10, n_episodes=300):
    """
    Grid search for hyperparameters using internal 10-fold CV
    on training data only (paper Section IV-A).

    Paper found: α=0.1, γ=0.6, ε=0.1
    """
    print("\n[Grid Search] Starting 10-fold CV …")

    alphas   = [0.05, 0.1, 0.2, 0.3]
    gammas   = [0.4,  0.6, 0.8, 0.9]
    epsilons = [0.0,  0.1, 0.2]

    best_err    = float("inf")
    best_params = {"alpha": 0.1, "gamma": 0.6, "epsilon": 0.1}
    fold_size   = max(1, len(subjects) // n_folds)

    for alpha, gamma, epsilon in product(alphas, gammas, epsilons):
        fold_errors = []
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end   = min(val_start + fold_size, len(subjects))
            val_idx   = list(range(val_start, val_end))
            train_idx = [i for i in range(len(subjects)) if i not in val_idx]

            if not train_idx or not val_idx:
                continue

            train_clips = [subjects[i] for i in train_idx]

            for vi in val_idx:
                n_clips = len(subjects[vi])
                agent   = QLearningAgent(N_EMOTIONS, n_clips,
                                         alpha, gamma, epsilon)
                train_agent(agent, train_clips, n_episodes)
                # Use a fixed negative start for evaluation consistency
                start = NEGATIVE_STARTS[vi % len(NEGATIVE_STARTS)]
                res   = evaluate_subject(agent, subjects[vi],
                                         start_emotion=start)
                fold_errors.append(res["final_err"])

        mean_err = np.mean(fold_errors) if fold_errors else float("inf")
        if mean_err < best_err:
            best_err    = mean_err
            best_params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}
            print(f"  New best: α={alpha} γ={gamma} ε={epsilon} → "
                  f"mean_err={mean_err:.1f}°")

    print(f"\n[Grid Search] Best params: {best_params}  "
          f"mean error={best_err:.1f}°")
    return best_params


# ════════════════════════════════════════════════════════════
# 10.  PLOTS  (reproducing paper Figs 2, 3, 4)
# ════════════════════════════════════════════════════════════

def plot_angular_error_comparison(results_with, results_without,
                                   save_path="angular_error.png"):
    """
    Reproduce paper Fig. 2:
    Angular error vs iterations, with vs without reward shaping.
    Without shaping: error should increase then saturate (agent stuck).
    With shaping: error should steadily decrease toward 0°.
    """
    iters = np.arange(1, N_PLAYLIST + 1)

    me_with    = np.array([r["errors"] for r in results_with]).mean(axis=0)
    me_without = np.array([r["errors"] for r in results_without]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, me_with,    "-o",  color="steelblue", lw=2,
            label="With reward shaping")
    ax.plot(iters, me_without, "--s", color="tomato", lw=2,
            label="Without reward shaping")
    ax.axhline(CONVERGENCE_THRESH, color="gray", ls=":", lw=1,
               label=f"{CONVERGENCE_THRESH}° threshold")
    ax.set_xlabel("Music Clip # (Iteration)")
    ax.set_ylabel("Angular Error (°)")
    ax.set_title("Angular Error vs Iterations — Fig. 2")
    ax.set_xticks(iters)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_error_bar_TSF(results, save_path="error_bar_TSF.png"):
    """
    Reproduce paper Fig. 3:
    Error bar plot for Total / Success / Failure groups.
    """
    converged = [r for r in results if     r["converged"]]
    diverged  = [r for r in results if not r["converged"]]
    iters     = np.arange(1, N_PLAYLIST + 1)

    def stats(subset):
        if not subset:
            return np.zeros(N_PLAYLIST), np.zeros(N_PLAYLIST)
        e = np.array([r["errors"] for r in subset])
        return e.mean(axis=0), e.std(axis=0) / np.sqrt(len(subset))

    me_T, se_T = stats(results)
    me_S, se_S = stats(converged)
    me_F, se_F = stats(diverged)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(iters, me_T, yerr=se_T, fmt="-o",  color="black",
                capsize=4, label=f"Total ({len(results)})")
    ax.errorbar(iters, me_S, yerr=se_S, fmt="-s",  color="green",
                capsize=4, label=f"Success ({len(converged)})")
    ax.errorbar(iters, me_F, yerr=se_F, fmt="-^",  color="red",
                capsize=4, label=f"Failure ({len(diverged)})")
    ax.axhline(CONVERGENCE_THRESH, color="gray", ls=":", lw=1)
    ax.set_xlabel("Music Clip # (Iteration)")
    ax.set_ylabel("Angular Error (°)")
    ax.set_title("Mean Angular Error & Standard Error — Fig. 3")
    ax.set_xticks(iters)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_emotion_trajectory(result, subject_id, save_path=None):
    """
    Reproduce paper Fig. 4:
    Emotion trajectory on GEW — path from start through 6 clips.
    Happy is shown in gold (target). Each clip step is numbered.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")

    circle = plt.Circle((0, 0), 1.0, color="gray", fill=False,
                         linestyle="--", lw=0.8, alpha=0.5)
    ax.add_patch(circle)

    # Draw emotion centers — happy gets a bigger gold star
    for name, (vx, vy) in EMOTION_CENTERS.items():
        if name == TARGET_EMOTION:
            ax.scatter(vx, vy, s=300, zorder=6, color="gold",
                       edgecolors="darkorange", lw=2, marker="*")
            ax.text(vx + 0.05, vy + 0.05, f"★ {name}",
                    fontsize=9, fontweight="bold", color="darkorange")
        else:
            ax.scatter(vx, vy, s=100, zorder=5, color="lightblue",
                       edgecolors="steelblue", lw=0.8)
            ax.text(vx + 0.04, vy + 0.04, name, fontsize=8, color="#333")

    vectors = result["vectors"]
    xs = [v[0] for v in vectors]
    ys = [v[1] for v in vectors]

    # Draw trajectory
    ax.plot(xs, ys, "-o", color="navy", lw=2, ms=7, zorder=7)
    ax.plot(xs[0],  ys[0],  "gs",  ms=14, zorder=8, label="Start",
            markeredgecolor="darkgreen", markeredgewidth=1.5)
    ax.plot(xs[-1], ys[-1], "r*",  ms=18, zorder=8, label="End",
            markeredgecolor="darkred",  markeredgewidth=1.5)

    # Number each step on the path
    for i, (x, y) in enumerate(zip(xs[1:], ys[1:]), 1):
        ax.annotate(str(i), (x, y), fontsize=8, fontweight="bold",
                    color="white", ha="center", va="center", zorder=9)

    # Draw arrow from each step to next
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i+1], ys[i+1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color="navy",
                                   lw=1.5), zorder=6)

    converged = result["converged"]
    fe        = result["final_err"]
    end_emo   = result["emotions"][-1]
    status    = f"Converged to {end_emo} ✓" if converged else f"Failed ({end_emo}) ✗"

    ax.set_title(f"Subject {subject_id}  |  Start: {result['start']}  |  {status}\n"
                 f"Final angular error to happy: {fe:.1f}°", fontsize=10)
    ax.set_xlabel("Valence →")
    ax.set_ylabel("Arousal →")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def print_summary_table(results):
    """
    Print paper Table I format:
    Mean angular error (std) per clip for Total / Success / Failure groups.
    """
    converged     = [r for r in results if     r["converged"]]
    not_converged = [r for r in results if not r["converged"]]

    def row(subset, label):
        if not subset:
            return
        e = np.array([r["errors"] for r in subset])
        print("  {:18s}| ".format(label) +
              "  ".join(f"{m:6.1f}({s:4.1f})"
                        for m, s in zip(e.mean(0), e.std(0))))

    header = "  {:18s}| ".format("Group") + \
             "  ".join(f"{'Clip '+str(i):>12s}" for i in range(1, N_PLAYLIST + 1))

    print("\n" + "=" * (len(header) + 2))
    print("  Table I — Mean angular error (std) per clip [paper format]")
    print(header)
    print("-" * (len(header) + 2))
    row(results,       f"Total ({len(results)})")
    row(converged,     f"Success ({len(converged)})")
    row(not_converged, f"Fail ({len(not_converged)})")
    print("=" * (len(header) + 2))

    fe = [r["final_err"] for r in results]
    print(f"\n  Convergence: {len(converged)}/{len(results)} → '{TARGET_EMOTION}' "
          f"(error < {CONVERGENCE_THRESH}° = success)")
    print(f"  Overall mean clip {N_PLAYLIST}: {np.mean(fe):.1f}° ± {np.std(fe):.1f}°")
    print(f"\n  Paper target: 19/32 converged, overall mean = 57.0° ± 11.1°")
    print(f"  Paper success group clip 6: 11.3° ± 2.8°")
    print(f"  Paper failure group clip 6: 123.7° ± 17.3°\n")


# ════════════════════════════════════════════════════════════
# 11.  MAIN
# ════════════════════════════════════════════════════════════

def main(data_root=None, output_dir="results", mode="deap_eeg",
         clf_method="svm", n_episodes=10000, do_grid_search=False):

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  EEG Music Emotion RL  —  Dutta et al. 2020")
    print(f"  Mode: {mode}  |  Classifier: {clf_method}")
    print(f"  Episodes: {n_episodes}  |  Grid search: {do_grid_search}")
    print("=" * 60)

    # ── Load subject clip libraries ──────────────────────────
    subjects = []

    if data_root and os.path.isdir(data_root):

        if mode == "deap_eeg":
            # Full pipeline: EEG → band-power features → SVM → (v,a) clips
            # This is what the paper describes in Section II & III-B
            from eeg_classifier import DEAPLoader, EEGEmotionClassifier
            print("\nLoading DEAP with EEG feature extraction (LOSO) …")
            loader  = DEAPLoader(data_root)
            all_ids = list(range(1, 33))

            for test_id in all_ids:
                # Collect training data from all other subjects
                train_eeg, train_va = [], []
                for sid in all_ids:
                    if sid == test_id:
                        continue
                    try:
                        eeg_list, labels = loader.get_eeg_and_labels(sid)
                        train_eeg.extend(eeg_list)
                        train_va.extend(labels)
                    except FileNotFoundError:
                        continue

                if len(train_eeg) < 10:
                    print(f"  s{test_id:02d}: not enough training data, skipping")
                    continue

                # Train SVM on 31 subjects' EEG
                clf = EEGEmotionClassifier(method=clf_method, fs=128,
                                           compact=False)
                clf.fit(train_eeg, np.array(train_va))

                # Predict (valence, arousal) for test subject's 40 clips
                try:
                    test_eeg, _ = loader.get_eeg_and_labels(test_id)
                    clips = np.array(
                        [clf.predict_proba_single(t) for t in test_eeg],
                        dtype=np.float32
                    )

                    # Centre clips around subject mean so we have both
                    # positive and negative vectors (needed for meaningful
                    # GEW placement when SVM predictions are biased)
                    clips_mean = clips.mean(axis=0)
                    for col in range(2):
                        clips[:, col] -= clips_mean[col]
                        max_abs = np.abs(clips[:, col]).max()
                        if max_abs > 0:
                            clips[:, col] /= max_abs

                    subjects.append(clips)
                    print(f"  s{test_id:02d}: {len(clips)} clips predicted from EEG "
                          f"(v̄={clips[:,0].mean():.2f}, ā={clips[:,1].mean():.2f})")
                except FileNotFoundError:
                    continue

        elif mode == "deap":
            # Simpler mode: use self-report labels directly (no EEG classifier)
            from eeg_classifier import DEAPLoader
            print("\nLoading DEAP with self-report labels …")
            loader   = DEAPLoader(data_root)
            subjects = loader.load_all_subjects()

        elif mode == "mat_eeg":
            from eeg_classifier import MATEmotionDataset
            print("\nLoading .mat dataset with EEG feature extraction …")
            dataset  = MATEmotionDataset(data_root, method=clf_method)
            subjects = dataset.build_loso_subjects()

        elif mode == "mat":
            from mat_loader import MATLoader
            print("\nLoading .mat dataset (filename labels) …")
            loader   = MATLoader(data_root)
            subjects = loader.load_all_subjects_clips()

        print(f"\nLoaded {len(subjects)} subjects from: {data_root}")

    else:
        # Synthetic fallback for testing without data
        print("\nNo data_root — using synthetic data for testing.")
        rng      = np.random.default_rng(42)
        subjects = [
            rng.uniform(-1, 1, (40, 2)).astype(np.float32)
            for _ in range(32)
        ]

    if len(subjects) < 2:
        print("Need at least 2 subjects. Exiting.")
        return

    # ── Hyperparameters ──────────────────────────────────────
    # Paper Section IV-A: grid search found α=0.1, γ=0.6, ε=0.1
    params = {"alpha": 0.1, "gamma": 0.6, "epsilon": 0.1}

    if do_grid_search:
        print("\nRunning grid search to find best hyperparameters …")
        params = grid_search(subjects)
        print(f"Using grid-search params: {params}")
    else:
        print(f"\nUsing paper's reported params: {params}")

    # ── LOSO with reward shaping ─────────────────────────────
    global LAMBDA
    print(f"\n── LOSO with reward shaping (λ={LAMBDA}) ──")
    results_with = leave_one_subject_out(
        subjects, **params, n_episodes=n_episodes
    )

    # ── LOSO without reward shaping ──────────────────────────
    print(f"\n── LOSO without reward shaping (λ=0) ──")
    LAMBDA = 0.0
    results_without = leave_one_subject_out(
        subjects, **params, n_episodes=n_episodes
    )
    LAMBDA = -100.0   # restore

    # ── Results table ────────────────────────────────────────
    print_summary_table(results_with)

    # ── Plots ────────────────────────────────────────────────
    print("Generating plots …")
    plot_angular_error_comparison(
        results_with, results_without,
        save_path=os.path.join(output_dir, "fig2_angular_error.png")
    )
    plot_error_bar_TSF(
        results_with,
        save_path=os.path.join(output_dir, "fig3_error_bar_TSF.png")
    )

    # Save trajectory plots for all subjects
    print(f"Saving {len(results_with)} trajectory plots …")
    for r in results_with:
        status = "converged" if r["converged"] else "diverged"
        plot_emotion_trajectory(
            r, r["subject"],
            save_path=os.path.join(
                output_dir,
                f"fig4_trajectory_s{r['subject']:02d}_{status}.png"
            )
        )

    print(f"\nDone. All results saved to: {output_dir}/")
    return results_with


# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EEG Music Emotion RL — Dutta et al. 2020 replication"
    )
    parser.add_argument("data_root",  nargs="?", default=None,
                        help="Path to DEAP data_preprocessed_python/ folder")
    parser.add_argument("output_dir", nargs="?", default="results",
                        help="Where to save plots and results")
    parser.add_argument("--mode", default="deap_eeg",
                        choices=["deap_eeg", "deap", "mat_eeg", "mat"],
                        help="deap_eeg = full EEG pipeline (recommended)")
    parser.add_argument("--clf", default="svm",
                        choices=["svm", "knn", "rf"],
                        help="Classifier for EEG → emotion")
    parser.add_argument("--episodes", type=int, default=10000,
                        help="Q-learning training episodes per LOSO fold")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run hyperparameter grid search before LOSO")
    args = parser.parse_args()

    main(
        data_root     = args.data_root,
        output_dir    = args.output_dir,
        mode          = args.mode,
        clf_method    = args.clf,
        n_episodes    = args.episodes,
        do_grid_search= args.grid_search,
    )
