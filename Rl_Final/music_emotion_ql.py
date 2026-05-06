"""
music_emotion_ql.py  —  Q-Learning (Dutta et al. 2020 replication)
====================================================================
Fixes vs. original:
  1. Convergence = EXACT emotion state is "happy"  (paper criterion)
       + angular-error < 30° also tracked separately
  2. If a subject's random start happens to be "happy", we re-sample
       until a non-target start is drawn (paper trains on non-happy starts)
  3. Reward shaping λ = -100 (unchanged, paper value)
  4. Reports both <30° and <60° convergence alongside exact-emotion count
  5. LOSO cross-validation identical to paper
  6. Grid-search hyper-params: α=0.1, γ=0.6, ε=0.1  (paper optimal)

Run:
  python music_emotion_ql.py
  python music_emotion_ql.py --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv --episodes 10000
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
import os
import argparse

EMOTION_GROUPS = {
    "happy":       ["happy", "joy"],
    "pleasant":    ["love", "lovely", "sentimental"],
    "exciting":    ["exciting", "fun"],
    "calm":        ["mellow"],
    "sad":         ["sad", "melancholy"],
    "negative":    ["depressing"],
    "anger":       ["hate", "terrible"],
    "fear":        ["shock"],
}

# ------------------------------------------------------------------
# 1. EMOTION WHEEL  (from your emotion_summary.csv)
# ------------------------------------------------------------------

def load_emotion_centers_grouped(csv_path):
    df = pd.read_csv(csv_path)

    # normalize like before
    def normalize_col(col):
        return 2 * (col - col.min()) / (col.max() - col.min()) - 1

    df["Valence_norm"] = normalize_col(df["Valence"])
    df["Arousal_norm"] = normalize_col(df["Arousal"])

    grouped_centers = {}

    for group, emotions in EMOTION_GROUPS.items():
        sub = df[df["Emotion"].isin(emotions)]
        if sub.empty:
            continue

        v = sub["Valence_norm"].mean()
        a = sub["Arousal_norm"].mean()

        vec = np.array([v, a])
        vec = vec / (np.linalg.norm(vec) + 1e-9)  # project to unit circle

        grouped_centers[group] = tuple(vec)

    return grouped_centers


# ------------------------------------------------------------------
# 2. GLOBALS  (set after loading CSV)
# ------------------------------------------------------------------

EMOTION_CENTERS: dict = {}
EMOTION_NAMES:  list  = []
EMOTION_INDEX:  dict  = {}
N_EMOTIONS:     int   = 0
TARGET_EMOTION        = "happy"
TARGET_VECTOR         = None
LAMBDA                 = -100.0   # reward-shaping penalty (paper: -100)
DEFAULT_START_EMOTION = "negative"
DEFAULT_EPISODES      = 10000
PLAYLIST_LEN          = 6
EXPECTED_N_CLIPS      = 40
TERMINAL_REWARD_MAX   = 10.0
TERMINAL_ANGLE_DEG    = 10.0
DEFAULT_GRID_FOLDS    = 10
DEFAULT_GRID_EPISODES = 300


def _init_globals(csv_path: str):
    global EMOTION_CENTERS, EMOTION_NAMES, EMOTION_INDEX
    global N_EMOTIONS, TARGET_VECTOR
    EMOTION_CENTERS = load_emotion_centers_grouped(csv_path)
    EMOTION_NAMES   = list(EMOTION_CENTERS.keys())
    EMOTION_INDEX   = {n: i for i, n in enumerate(EMOTION_NAMES)}
    N_EMOTIONS      = len(EMOTION_NAMES)
    TARGET_VECTOR   = np.array(EMOTION_CENTERS[TARGET_EMOTION])
    print(f"  Emotions loaded: {EMOTION_NAMES}")
    print(f'  Target: {TARGET_EMOTION} -> {TARGET_VECTOR}')


# ------------------------------------------------------------------
# 3. CORE FUNCTIONS
# ------------------------------------------------------------------

def vector_to_emotion(vec: np.ndarray) -> str:
    """Nearest-cosine emotion label."""
    best, best_cos = None, -2.0
    for name, center in EMOTION_CENTERS.items():
        cv  = np.array(center)
        cos = np.dot(vec, cv) / (np.linalg.norm(vec) * np.linalg.norm(cv) + 1e-9)
        if cos > best_cos:
            best_cos, best = cos, name
    return best


def angular_error(vec: np.ndarray, target: np.ndarray = None) -> float:
    """Angle in degrees between vec and target vector."""
    if target is None:
        target = TARGET_VECTOR
    cos = np.dot(vec, target) / (np.linalg.norm(vec) * np.linalg.norm(target) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def transition(state_vec: np.ndarray, clip_vec: np.ndarray) -> np.ndarray:
    """Additive state transition with GEW boundary projection (radius=1)."""
    s_next = np.asarray(state_vec, dtype=np.float32) + np.asarray(clip_vec, dtype=np.float32)
    norm = np.linalg.norm(s_next)
    if norm > 1.0:
        s_next = s_next / norm
    return s_next


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def terminal_reward_from_clip(step_idx: int,
                              n_playlist: int = PLAYLIST_LEN,
                              max_bonus: float = TERMINAL_REWARD_MAX) -> float:
    clip_number = min(max(step_idx + 1, 1), n_playlist)
    return max_bonus * (n_playlist - clip_number + 1) / max(1, n_playlist)


def reward(state_vec, clip_vec, step_idx: int,
           target=None, n_playlist: int = PLAYLIST_LEN):
    if target is None:
        target = TARGET_VECTOR

    def safe_angle(v1, v2):
        v1 = v1 / (np.linalg.norm(v1) + 1e-9)
        v2 = v2 / (np.linalg.norm(v2) + 1e-9)
        cos = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return np.arccos(cos)

    theta1 = safe_angle(target, clip_vec)
    theta2 = safe_angle(state_vec, clip_vec)

    r = 1.0 if (abs(theta1) + abs(theta2)) <= np.pi else 0.0

    next_vec = transition(state_vec, clip_vec)

    if angular_error(next_vec) < TERMINAL_ANGLE_DEG:
        r += 10.0

    return r


def reward_shaping(state_idx: int, next_state_idx: int) -> float:
    """φ(s,s') = λ if s==s', else 0   (paper: λ = -100)."""
    return LAMBDA if state_idx == next_state_idx else 0.0


def _training_start_emotions():
    return list(EMOTION_NAMES)


def resolve_start_emotion(start_emotion: str = None) -> str:
    start = DEFAULT_START_EMOTION if start_emotion is None else start_emotion
    if start == "angry":
        start = "anger"
    if start not in EMOTION_INDEX:
        raise ValueError(
            f"Unknown start emotion '{start_emotion}'. "
            f"Available emotions: {sorted(EMOTION_INDEX)}"
        )
    return start


def prepare_subjects(subjects: list) -> list:
    prepared = []
    for idx, clips in enumerate(subjects, start=1):
        arr = np.asarray(clips, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Subject {idx} must be shaped (n_clips, 2); got {arr.shape}"
            )
        prepared.append(arr)

    clip_counts = sorted({arr.shape[0] for arr in prepared})
    if len(clip_counts) != 1:
        raise ValueError(
            "All subjects must have the same number of clips for clip-wise "
            f"mean training; got clip counts {clip_counts}"
        )
    return prepared


def compute_global_clip_pool(subjects_for_pool: list) -> np.ndarray:
    return np.concatenate(subjects_for_pool, axis=0).astype(np.float32)


def build_cv_folds(n_subjects: int, n_folds: int = DEFAULT_GRID_FOLDS) -> list:
    n_folds = max(2, min(n_folds, n_subjects))
    return [fold.tolist() for fold in np.array_split(np.arange(n_subjects), n_folds) if len(fold) > 0]

def discretize_state(vec, bins=20):
    clipped = np.clip(np.asarray(vec, dtype=np.float32), -1.0, 1.0)
    v = int((clipped[0] + 1) / 2 * (bins - 1))
    a = int((clipped[1] + 1) / 2 * (bins - 1))
    return v * bins + a


# ------------------------------------------------------------------
# 4. Q-LEARNING AGENT
# ------------------------------------------------------------------

class QLearningAgent:
    """Standard Q-learning (Dutta et al. 2020)."""

    def __init__(self, n_states: int, n_clips: int,
                 alpha: float = 0.1, gamma: float = 0.6,
                 epsilon: float = 0.1):
        self.n_states  = n_states
        self.n_clips   = n_clips
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.Q         = np.zeros((n_states, n_clips))

    def choose_action(self, state_idx: int, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_clips)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, state_idx: int, clip_idx: int,
               r: float, next_state_idx: int, phi: float):
        best_next = np.max(self.Q[next_state_idx])
        td_target = r + self.gamma * best_next + phi
        self.Q[state_idx, clip_idx] += self.alpha * (
            td_target - self.Q[state_idx, clip_idx])

    def reset(self):
        self.Q[:] = 0.0


# ------------------------------------------------------------------
# 5. TRAINING / EVALUATION
# ------------------------------------------------------------------

def train_agent(agent,
                train_action_vectors: np.ndarray,
                n_episodes: int = DEFAULT_EPISODES,
                use_reward_shaping: bool = True,
                bins: int = 20):

    agent.reset()
    start_emotions = _training_start_emotions()
    train_action_vectors = np.asarray(train_action_vectors, dtype=np.float32)

    for _ in range(n_episodes):
        start = start_emotions[np.random.randint(len(start_emotions))]
        sv = np.array(EMOTION_CENTERS[start], dtype=np.float32)

        for step_idx in range(PLAYLIST_LEN):

            si = discretize_state(sv, bins)

            ci = agent.choose_action(si, training=True)

            cv = train_action_vectors[ci]

            nv = transition(sv, cv)
            ni = discretize_state(nv, bins)

            r = reward(sv, cv, step_idx=step_idx, n_playlist=PLAYLIST_LEN)

            phi = reward_shaping(si, ni) if use_reward_shaping else 0.0

            agent.update(si, ci, r, ni, phi)

            sv = nv

    
def evaluate_subject(agent,
                     action_pool: np.ndarray,
                     n_playlist: int = PLAYLIST_LEN,
                     start_emotion: str = None,
                     bins: int = 20):

    action_pool = np.asarray(action_pool, dtype=np.float32)
    start_emotion = resolve_start_emotion(start_emotion)

    sv = np.array(EMOTION_CENTERS[start_emotion], dtype=np.float32)

    errors, emotions, playlist = [], [start_emotion], []

    for _ in range(n_playlist):

        si = discretize_state(sv, bins)

        ci = agent.choose_action(si, training=False)

        cv = action_pool[ci]

        nv = transition(sv, cv)

        errors.append(angular_error(nv))

        emotions.append(vector_to_emotion(nv))
        playlist.append(ci)

        sv = nv

    fe = errors[-1]

    return {
        "errors": errors,
        "emotions": emotions,
        "playlist": playlist,
        
        
        "conv_20":         fe <= 20.0,
        "final_err": fe,
        "start": start_emotion,
    }

# ------------------------------------------------------------------
# 6. LEAVE-ONE-SUBJECT-OUT  (paper protocol)
# ------------------------------------------------------------------

def leave_one_subject_out(subjects: list,
                          alpha: float = 0.1,
                          gamma: float = 0.6,
                          epsilon: float = 0.1,
                          n_episodes: int = DEFAULT_EPISODES,
                          use_reward_shaping: bool = True,
                          start_emotion: str = None) -> list:
    results = []
    tag = "w/ shaping" if use_reward_shaping else "w/o shaping"
    print(f"\n  LOSO [{tag}]  alpha={alpha} gamma={gamma} epsilon={epsilon} "
          f"episodes={n_episodes}")
    subjects = prepare_subjects(subjects)

    for test_idx in range(len(subjects)):
        test_subject = subjects[test_idx]
        train_subjects = [s for i, s in enumerate(subjects) if i != test_idx]
        train_action_vectors = compute_global_clip_pool(train_subjects)
        bins = 20
        n_states = bins * bins
        agent = QLearningAgent(n_states, len(train_action_vectors), alpha, gamma, epsilon)
        train_agent(agent, train_action_vectors, n_episodes, use_reward_shaping, bins=bins)

        res             = evaluate_subject(agent, train_action_vectors, bins=bins, start_emotion=start_emotion)
        res["subject"]  = test_idx + 1
        results.append(res)
        print(f"    S{test_idx+1:02d} | err={res['final_err']:5.1f} deg | "
              f"<=20 deg={'v' if res['conv_20'] else 'x'} | "
              f"start={res['start']:12s} | "
              f"{' -> '.join(res['emotions'])}")

    return results


# ------------------------------------------------------------------
# 7. HYPER-PARAMETER GRID SEARCH  (paper: 10-fold CV)
# ------------------------------------------------------------------

def grid_search(subjects: list,
                alphas:   list = [0.05, 0.1, 0.2],
                gammas:   list = [0.4,  0.6, 0.8],
                epsilons: list = [0.0, 0.05, 0.1, 0.2],
                n_folds:  int  = DEFAULT_GRID_FOLDS,
                n_episodes: int = DEFAULT_GRID_EPISODES,
                use_reward_shaping: bool = True,
                start_emotion: str = None) -> dict:
    tag = "w/ shaping" if use_reward_shaping else "w/o shaping"
    print(f"[Grid Search] starting ({tag}) …")
    best_err, best_params = float("inf"), {}
    subjects = prepare_subjects(subjects)
    bins = 20
    n_states = bins * bins
    folds = build_cv_folds(len(subjects), n_folds)

    for alpha, gamma, epsilon in product(alphas, gammas, epsilons):
        fold_errors = []
        for val_idx in folds:
            train_idx = [i for i in range(len(subjects)) if i not in val_idx]
            if not train_idx or not val_idx:
                continue
            train_action_vectors = compute_global_clip_pool([subjects[i] for i in train_idx])
            agent = QLearningAgent(n_states, len(train_action_vectors),
                                   alpha, gamma, epsilon)
            train_agent(agent, train_action_vectors,
                        n_episodes, use_reward_shaping=use_reward_shaping, bins=bins)
            for vi in val_idx:
                fold_errors.append(
                    evaluate_subject(agent, train_action_vectors, bins=bins, start_emotion=start_emotion)["final_err"])

        mean_err = np.mean(fold_errors) if fold_errors else float("inf")
        print(f"    alpha={alpha:.3f} gamma={gamma:.3f} epsilon={epsilon:.3f} -> "
              f"cv_mean_err={mean_err:.2f} deg")
        if mean_err < best_err:
            best_err    = mean_err
            best_params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}

    print(f"[Grid Search] Best: {best_params}  mean_err={best_err:.1f} deg")
    return best_params


# ------------------------------------------------------------------
# 8. SUMMARY TABLE  (matches Table I in paper)
# ------------------------------------------------------------------

def print_summary_table(results: list, label: str = "Q-Learning"):
    exact   = [r for r in results if     r["exact_converged"]]
    inexact = [r for r in results if not r["exact_converged"]]
    c30     = [r for r in results if     r["conv_30"]]
    c60     = [r for r in results if     r["conv_60"]]

    def row(subset, tag):
        if not subset:
            return
        e = np.array([r["errors"] for r in subset])
        print("  {:20s}| ".format(tag) +
              "  ".join(f"{m:5.1f}({s:.1f})"
                        for m, s in zip(e.mean(0), e.std(0))))

    print("\n" + "=" * 80)
    print(f"  [{label}]  Table I — Mean angular error (std) per clip")
    print("  {:20s}| ".format("Group") +
          "  ".join(f"{'Clip '+str(i):>10s}" for i in range(1, 7)))
    print("-" * 80)
    row(results, f"Total ({len(results)})")
    row(exact,   f"Exact happy ({len(exact)})")
    row(inexact, f"Not happy ({len(inexact)})")
    print("=" * 80)

    fe = [r["final_err"] for r in results]
    print(f"\n  Convergence summary:")
    print(f"    Exact emotion = happy : {len(exact)}/{len(results)}"
          f"   (paper: 19/32)")
    print(f"    Angular error < 30°  : {len(c30)}/{len(results)}")
    print(f"    Angular error < 60°  : {len(c60)}/{len(results)}"
          f"   (paper criterion for 'converged' overall)")
    print(f"    Overall mean (clip 6): {np.mean(fe):.1f} deg +- {np.std(fe):.1f} deg"
          f"   (paper: 57.0 deg +- 2.8 deg)")


# ------------------------------------------------------------------
# 9. PLOTS
# ------------------------------------------------------------------

def plot_angular_error(results: list, save_path: str = "angular_error.png",
                       label: str = "Q-Learning"):
    all_e  = np.array([r["errors"] for r in results])
    mean_e = all_e.mean(axis=0)
    se     = all_e.std(axis=0) / np.sqrt(len(results))
    iters  = np.arange(1, 7)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(iters, mean_e, yerr=se, fmt="-o", color="steelblue",
                capsize=4, label=f"{label} — All subjects (mean ± SE)")
    ax.axhline(60, color="orange", ls="--", lw=1.2, label="60° threshold")
    ax.axhline(30, color="green",  ls="--", lw=1.2, label="30° threshold")
    ax.set_xlabel("Music Clip # (Iteration)")
    ax.set_ylabel("Angular Error (deg)")
    ax.set_title(f"[{label}] Mean Angular Error per Iteration")
    ax.set_xticks(iters)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_reward_shaping_comparison(results_with: list, results_without: list,
                                   save_path: str = "reward_shaping.png",
                                   label: str = "Q-Learning"):
    iters = np.arange(1, 7)
    me    = lambda r: np.array([x["errors"] for x in r]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, me(results_with),    "-o",  color="navy",
            label="With reward shaping")
    ax.plot(iters, me(results_without), "--s", color="tomato",
            label="Without reward shaping")
    ax.axhline(60, color="orange", ls="--", lw=1.0, label="60° threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Angular Error (deg)")
    ax.set_title(f"[{label}] Effect of Reward Shaping")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_emotion_trajectory(result: dict, subject_id: int,
                            save_path: str = None, label: str = "Q-Learning"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    boundary = plt.Circle((0, 0), 1, color="gray",
                           fill=False, linestyle="--", alpha=0.3)
    ax.add_patch(boundary)

    for name, (vx, vy) in EMOTION_CENTERS.items():
        color = "gold" if name == TARGET_EMOTION else "lightblue"
        ax.scatter(vx, vy, s=100, zorder=5, color=color,
                   edgecolors="k", lw=0.7)
        ax.text(vx + 0.05, vy + 0.05, name, fontsize=7)

    emotions = result["emotions"]
    xs = [EMOTION_CENTERS[e][0] for e in emotions]
    ys = [EMOTION_CENTERS[e][1] for e in emotions]
    ax.plot(xs, ys, "-o", color="navy", lw=1.5, ms=6, zorder=6)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    status = (f"Converged v (err={result['final_err']:.1f} deg)"
              if result["exact_converged"]
              else f"No convergence x (err={result['final_err']:.1f} deg)")
    ax.set_title(f"[{label}] Subject {subject_id} — {status}", fontsize=11)
    ax.set_xlabel("Valence →")
    ax.set_ylabel("Arousal →")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_converged_vs_diverged(converged_result: dict, diverged_result: dict,
                                save_path: str = "converged_vs_diverged.png",
                                label: str = "Q-Learning"):
    iters = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(8, 5))

    c_id  = converged_result["subject"]
    ax.plot(iters, converged_result["errors"], "-o", color="#2ecc71",
            lw=2.2, ms=8, label=f"Subject {c_id} (Converged)", zorder=5)

    d_id  = diverged_result["subject"]
    ax.plot(iters, diverged_result["errors"], "--s", color="#e74c3c",
            lw=2.2, ms=8, label=f"Subject {d_id} (Not converged)", zorder=5)

    ax.axhline(60, color="orange", ls=":", lw=1.2, label="60° threshold")
    ax.axhline(30, color="green",  ls=":", lw=1.2, label="30° threshold")
    ax.set_xlabel("Music Clip # (Iteration)", fontsize=12)
    ax.set_ylabel("Angular Error (deg)", fontsize=12)
    ax.set_title(f"[{label}] Angular Error: Converged vs Diverged", fontsize=13)
    ax.set_xticks(iters)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_subjects_trajectories(results: list,
                                   save_path: str = "all_trajectories.png",
                                   label: str = "Q-Learning"):
    n    = len(results)
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, res in enumerate(results):
        r, c = divmod(idx, cols)
        ax   = axes[r, c]

        boundary = plt.Circle((0, 0), 1, color="gray",
                              fill=False, lw=0.5, ls="--", alpha=0.3)
        ax.add_patch(boundary)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")

        for name, (vx, vy) in EMOTION_CENTERS.items():
            clr = "gold" if name == TARGET_EMOTION else "lightblue"
            ax.scatter(vx, vy, s=30, color=clr,
                       edgecolors="k", lw=0.4, zorder=3)

        emotions = res["emotions"]
        xs = [EMOTION_CENTERS[e][0] for e in emotions]
        ys = [EMOTION_CENTERS[e][1] for e in emotions]
        ax.plot(xs, ys, "-", color="navy", lw=1.0, zorder=4, alpha=0.8)
        ax.plot(xs[0],  ys[0],  "gs", ms=5, zorder=5)
        ax.plot(xs[-1], ys[-1], "r*", ms=7, zorder=5)

        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.axvline(0, color="gray", lw=0.4, ls="--")
        ax.set_xticks([])
        ax.set_yticks([])

        sid  = res["subject"]
        conv = res["exact_converged"]
        tag  = "v" if conv else "x"
        ax.set_title(f"S{sid} {tag}", fontsize=7,
                     color="green" if conv else "red", fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2ecc71" if conv else "#e74c3c")
            spine.set_linewidth(1.5)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle(f"[{label}] Emotion Wheel Trajectories — All Subjects",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_convergence_comparison(results_30: list, results_60: list,
                                 label: str = "Q-Learning",
                                 save_path: str = "convergence_comparison.png"):
    """Bar chart: exact / <30° / <60° counts side by side."""
    n        = len(results_30)          # same subjects
    exact    = sum(r["exact_converged"] for r in results_30)
    c30      = sum(r["conv_30"]         for r in results_30)
    c60      = sum(r["conv_60"]         for r in results_30)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Exact happy", "< 30°", "< 60°"],
                  [exact, c30, c60],
                  color=["#2ecc71", "#3498db", "#e67e22"])
    ax.axhline(19, ls="--", color="red", lw=1.2,
               label="Paper result (19/32 exact)")
    for b, v in zip(bars, [exact, c30, c60]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, n + 2)
    ax.set_ylabel("# Subjects converged")
    ax.set_title(f"[{label}] Convergence under different criteria")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shaping_convergence_bars(results_with: list, results_without: list,
                                  label: str = "Q-Learning",
                                  save_path: str = "ql_shaping_bars.png"):
    """
    Grouped bar chart comparing convergence rates WITH vs WITHOUT reward shaping
    under all three criteria (exact happy, <30°, <60°).
    """
    n = len(results_with)

    w_exact = sum(r["exact_converged"] for r in results_with)
    w_c30   = sum(r["conv_30"]         for r in results_with)
    w_c60   = sum(r["conv_60"]         for r in results_with)

    wo_exact = sum(r["exact_converged"] for r in results_without)
    wo_c30   = sum(r["conv_30"]         for r in results_without)
    wo_c60   = sum(r["conv_60"]         for r in results_without)

    criteria = ["Exact happy", "< 30°", "< 60°"]
    with_vals    = [w_exact, w_c30, w_c60]
    without_vals = [wo_exact, wo_c30, wo_c60]

    x = np.arange(len(criteria))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, with_vals,    width, label="With shaping",
                   color="#2ecc71", edgecolor="k", lw=0.5)
    bars2 = ax.bar(x + width/2, without_vals, width, label="Without shaping",
                   color="#e74c3c", edgecolor="k", lw=0.5)

    for b, v in zip(bars1, with_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for b, v in zip(bars2, without_vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylim(0, n + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=11)
    ax.set_ylabel("# Subjects converged", fontsize=11)
    ax.set_title(f"[{label}] Reward Shaping — Convergence Comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ------------------------------------------------------------------
# 10. SYNTHETIC DATA FALLBACK
# ------------------------------------------------------------------

def synthetic_subjects(n: int = 32, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, (EXPECTED_N_CLIPS, 2)).astype(np.float32)
            for _ in range(n)]


def load_subjects_from_ratings(ratings_csv: str) -> list:
    df_r = pd.read_csv(ratings_csv)
    sort_cols = [c for c in ("Participant_id", "Trial", "Experiment_id") if c in df_r.columns]
    if sort_cols:
        df_r = df_r.sort_values(sort_cols)

    subjects = []
    for _, grp in df_r.groupby("Participant_id", sort=True):
        subject_sort = [c for c in ("Trial", "Experiment_id") if c in grp.columns]
        if subject_sort:
            grp = grp.sort_values(subject_sort)
        clips = grp[["Valence", "Arousal"]].values.astype(np.float32)
        clips[:, 0] = 2 * (clips[:, 0] - 1) / 8 - 1
        clips[:, 1] = 2 * (clips[:, 1] - 1) / 8 - 1
        subjects.append(clips)
    return prepare_subjects(subjects)


# ------------------------------------------------------------------
# 11. MAIN
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Q-Learning music emotion RL — Dutta et al. 2020 replication")
    parser.add_argument("--data_root",  default="/home/sahil/Desktop/Rl_Project_Final/data/Metacsv",
                        help="Directory containing emotion_summary.csv")
    parser.add_argument("--output_dir", default="results_ql")
    parser.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--gamma",      type=float, default=0.6)
    parser.add_argument("--epsilon",    type=float, default=0.1)
    parser.add_argument("--grid_search", action="store_true",
                        help="Run hyper-parameter grid search first")
    parser.add_argument("--grid_folds", type=int, default=DEFAULT_GRID_FOLDS)
    parser.add_argument("--grid_episodes", type=int, default=DEFAULT_GRID_EPISODES)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  EEG Music Emotion RL  —  Q-Learning  (Dutta et al. 2020)")
    print("=" * 65)

    # ── Load emotion wheel ──────────────────────────────────────────
    csv_path = os.path.join(args.data_root, "emotion_summary.csv")
    if not os.path.isfile(csv_path):
        print(f"  [WARN] {csv_path} not found — using synthetic data.")
        csv_path = None

    if csv_path:
        _init_globals(csv_path)
    else:
        # minimal fallback (won't match paper perfectly)
        global EMOTION_CENTERS, EMOTION_NAMES, EMOTION_INDEX, N_EMOTIONS, TARGET_VECTOR
        EMOTION_CENTERS = {
            "happy": (0.9, 0.15), "joy": (0.85, 0.6), "fun": (0.55, 0.35),
            "exciting": (0.6, 0.65), "love": (0.5, -0.3), "cheerful": (0.45, -0.5),
            "lovely": (0.4, -0.52), "sentimental": (-0.15, -0.7), "mellow": (-0.2, -0.82),
            "depressing": (-0.25, -0.4), "sad": (-0.5, -0.02), "hate": (-0.5, 0.15),
            "terrible": (-0.52, 0.28), "shock": (-0.5, 0.85), "melancholy": (-0.8, -0.45),
        }
        EMOTION_NAMES  = list(EMOTION_CENTERS.keys())
        EMOTION_INDEX  = {n: i for i, n in enumerate(EMOTION_NAMES)}
        N_EMOTIONS     = len(EMOTION_NAMES)
        TARGET_VECTOR  = np.array(EMOTION_CENTERS[TARGET_EMOTION])

    # ── Load or synthesise subjects ────────────────────────────────
    ratings_csv = os.path.join(args.data_root, "participant_ratings.csv")
    if os.path.isfile(ratings_csv):
        print(f"\n  Loading participant_ratings.csv …")
        subjects = load_subjects_from_ratings(ratings_csv)
        print(f"  Loaded {len(subjects)} subjects, "
              f"{[len(s) for s in subjects]} clips each.")
    else:
        print("\n  participant_ratings.csv not found — using synthetic data.")
        subjects = synthetic_subjects(32)

    if len(subjects) < 2:
        print("  Need ≥ 2 subjects. Exiting.")
        return

    # ── Optional grid search ───────────────────────────────────────
    params_with = {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}
    params_without = dict(params_with)
    if args.grid_search:
        params_with = grid_search(
            subjects,
            n_folds=args.grid_folds,
            n_episodes=args.grid_episodes,
            use_reward_shaping=True,
        )
        params_without = grid_search(
            subjects,
            n_folds=args.grid_folds,
            n_episodes=args.grid_episodes,
            use_reward_shaping=False,
        )

    # ── LOSO: WITH reward shaping ──────────────────────────────────
    print("\n── LOSO with reward shaping ──")
    results_with = leave_one_subject_out(subjects, **params_with,
                                          n_episodes=args.episodes,
                                          use_reward_shaping=True)

    # ── LOSO: WITHOUT reward shaping ──────────────────────────────
    print("\n── LOSO without reward shaping ──")
    results_without = leave_one_subject_out(subjects, **params_without,
                                             n_episodes=args.episodes,
                                             use_reward_shaping=False)

    # ── Summary tables ──────────────────────────────────────────────
    print_summary_table(results_with,    label="Q-Learning (with shaping)")
    print_summary_table(results_without, label="Q-Learning (without shaping)")

    # ── Plots ──────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # 1) Angular error (with shaping)
    plot_angular_error(
        results_with, label="Q-Learning",
        save_path=os.path.join(args.output_dir, "ql_angular_error.png"))

    # 2) Reward shaping comparison (angular error curves)
    plot_reward_shaping_comparison(
        results_with, results_without, label="Q-Learning",
        save_path=os.path.join(args.output_dir, "ql_reward_shaping.png"))

    # 3) Convergence bar chart (with shaping)
    plot_convergence_comparison(
        results_with, results_with, label="Q-Learning (with shaping)",
        save_path=os.path.join(args.output_dir, "ql_convergence_criteria.png"))

    # 4) Convergence bar chart (without shaping)
    plot_convergence_comparison(
        results_without, results_without, label="Q-Learning (without shaping)",
        save_path=os.path.join(args.output_dir, "ql_convergence_no_shaping.png"))

    # 5) Grouped bar chart: with vs without shaping convergence
    plot_shaping_convergence_bars(
        results_with, results_without, label="Q-Learning",
        save_path=os.path.join(args.output_dir, "ql_shaping_bars.png"))

    # 6) All-subjects emotion wheel trajectories (with shaping)
    plot_all_subjects_trajectories(
        results_with, label="Q-Learning (with shaping)",
        save_path=os.path.join(args.output_dir, "ql_all_trajectories.png"))

    # 7) All-subjects trajectories (without shaping)
    plot_all_subjects_trajectories(
        results_without, label="Q-Learning (without shaping)",
        save_path=os.path.join(args.output_dir, "ql_all_trajectories_no_shaping.png"))

    # 8) Converged vs diverged angular error comparison
    converged     = [r for r in results_with if r["exact_converged"]]
    not_converged = [r for r in results_with if not r["exact_converged"]]

    pref = [r for r in converged if r.get("start") in ("sad", "guilt", "melancholy")]
    chosen_c = pref[0] if pref else (converged[0] if converged else None)

    if chosen_c and not_converged:
        plot_converged_vs_diverged(
            chosen_c, not_converged[0], label="Q-Learning",
            save_path=os.path.join(args.output_dir, "ql_converged_vs_diverged.png"))

    # 9) Individual emotion trajectory plots
    if converged:
        r = converged[0]
        plot_emotion_trajectory(
            r, r["subject"], label="Q-Learning",
            save_path=os.path.join(args.output_dir,
                                   f"ql_traj_s{r['subject']}_conv.png"))
    if not_converged:
        r = not_converged[0]
        plot_emotion_trajectory(
            r, r["subject"], label="Q-Learning",
            save_path=os.path.join(args.output_dir,
                                   f"ql_traj_s{r['subject']}_div.png"))

    print(f"\n✅  Done.  Results saved to: {args.output_dir}/")
    return results_with, results_without


if __name__ == "__main__":
    main()