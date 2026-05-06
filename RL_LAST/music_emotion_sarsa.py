"""
music_emotion_sarsa.py  —  SARSA (on-policy TD) for Music Emotion RL
=====================================================================
SARSA vs Q-Learning difference:
  Q-Learning  →  off-policy: updates using max_a' Q(s',a')   (greedy next)
  SARSA       →  on-policy:  updates using Q(s',a')           (actual next action)

Same environment, reward, reward-shaping, and LOSO protocol as Dutta et al. 2020.
Two convergence criteria:
  • conv_20         : final angular error < 20°
  • conv_30         : final angular error < 30°

Run:
  python music_emotion_sarsa.py
  python music_emotion_sarsa.py --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv --episodes 10000
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
    "happy":       ["happy", "joy", "joyous", "delighted", "amused"],
    "pleasant":    ["love", "lovely", "sentimental", "passionate", "lust"],
    "exciting":    ["exciting", "fun", "adventurous", "adventorous", "aroused"],
    "calm":        ["mellow"],
    "sad":         ["sad", "melancholy", "melancholic", "depressing", "depressed", "miserable", "despondent"],
    "negative":    ["dissatisfied", "disgust"],
    "anger":       ["hate", "terrible", "angry", "tense", "distress"],
    "fear":        ["shock", "afraid", "alarmed", "startled", "taken aback"],
}

# ------------------------------------------------------------------
# 1. MAT LOADER (For DENSE Dataset)
# ------------------------------------------------------------------

import scipy.io
import re

DENSE_EMOTION_MAPPING = {
    "Adventurous":  ( 0.70,  0.75), "Adventorous":  ( 0.70,  0.75),
    "Afraid":       (-0.60,  0.80), "Alarmed":      (-0.20,  0.90),
    "Amused":       ( 0.70,  0.50), "Angry":        (-0.70,  0.70),
    "Aroused":      ( 0.10,  0.90), "Delighted":    ( 0.80,  0.70),
    "Depressed":    (-0.90, -0.70), "Despondent":   (-0.80, -0.50),
    "Disgust":      (-0.80,  0.20), "Dissatisfied": (-0.60, -0.20),
    "Distress":     (-0.50,  0.80), "Excited":      ( 0.70,  0.90),
    "Happy":        ( 0.80,  0.60), "Hate":         (-0.80,  0.60),
    "Joyous":       ( 0.90,  0.80), "Lust":         ( 0.60,  0.90),
    "Melancholic":  (-0.70, -0.50), "Miserable":    (-0.90, -0.40),
    "Passionate":   ( 0.80,  0.90), "Sad":          (-0.80, -0.60),
    "Startled":     (-0.10,  1.00), "Taken Aback":  (-0.30,  0.70),
    "Tense":        (-0.40,  0.80),
}

class MATLoader:
    def __init__(self, data_root: str):
        self.data_root = data_root
        self.subjects  = {}
        self._discover()

    def _discover(self):
        for root, _, files in os.walk(self.data_root):
            for f in sorted(files):
                if not f.lower().endswith(".mat"): continue
                meta = self._parse_filename(f)
                if meta:
                    self.subjects.setdefault(meta["subject_id"], []).append(meta)

    def _parse_filename(self, filename: str):
        name = os.path.splitext(os.path.basename(filename))[0]
        parts = name.split("_", 1)
        if len(parts) < 2: return None
        emotion, rest = parts[0], parts[1]
        sub_match = re.search(r'(mit\d+)', rest, re.IGNORECASE)
        if not sub_match: return None
        sid = sub_match.group(1).lower()
        va = DENSE_EMOTION_MAPPING.get(emotion, (0.0, 0.0))
        return {"subject_id": sid, "valence": va[0], "arousal": va[1]}

    def load_all_subjects_clips(self) -> list:
        # Scale DENSE clips by 0.5 to allow for more granular transitions
        return [np.array([[t["valence"]*0.5, t["arousal"]*0.5] for t in trials], dtype=np.float32)
                for sid, trials in sorted(self.subjects.items())]


# ------------------------------------------------------------------
# 1. EMOTION WHEEL
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
# 2. GLOBALS
# ------------------------------------------------------------------

EMOTION_CENTERS: dict = {}
EMOTION_NAMES:   list = []
EMOTION_INDEX:   dict = {}
N_EMOTIONS:      int  = 0
TARGET_EMOTION         = "happy"
TARGET_VECTOR          = None
LAMBDA                 = -10.0
DEFAULT_START_EMOTION  = "anger"
DEFAULT_EPISODES       = 10000
PLAYLIST_LEN           = 6
EXPECTED_N_CLIPS       = 40
TERMINAL_REWARD_MAX    = 10.0
TERMINAL_ANGLE_DEG     = 10.0
DEFAULT_GRID_FOLDS     = 10
DEFAULT_GRID_EPISODES  = 300


def _init_globals(csv_path: str):
    global EMOTION_CENTERS, EMOTION_NAMES, EMOTION_INDEX
    global N_EMOTIONS, TARGET_VECTOR
    EMOTION_CENTERS = load_emotion_centers_grouped(csv_path)
    EMOTION_NAMES   = list(EMOTION_CENTERS.keys())
    EMOTION_INDEX   = {n: i for i, n in enumerate(EMOTION_NAMES)}
    N_EMOTIONS      = len(EMOTION_NAMES)
    TARGET_VECTOR   = np.array(EMOTION_CENTERS[TARGET_EMOTION])
    print(f"  Emotions: {EMOTION_NAMES}")
    print(f"  Target  : '{TARGET_EMOTION}'  -> {TARGET_VECTOR}")

def discretize_state(vec, bins=20):
    clipped = np.clip(np.asarray(vec, dtype=np.float32), -1.0, 1.0)
    v = int((clipped[0] + 1) / 2 * (bins - 1))
    a = int((clipped[1] + 1) / 2 * (bins - 1))
    return v * bins + a


# ------------------------------------------------------------------
# 3. CORE ENVIRONMENT FUNCTIONS  (identical to Q-Learning script)
# ------------------------------------------------------------------

def vector_to_emotion(vec: np.ndarray) -> str:
    best, best_cos = None, -2.0
    for name, center in EMOTION_CENTERS.items():
        cv  = np.array(center)
        cos = np.dot(vec, cv) / (np.linalg.norm(vec) * np.linalg.norm(cv) + 1e-9)
        if cos > best_cos:
            best_cos, best = cos, name
    return best


def angular_error(vec: np.ndarray, target: np.ndarray = None) -> float:
    if target is None:
        target = TARGET_VECTOR
    cos = np.dot(vec, target) / (np.linalg.norm(vec) * np.linalg.norm(target) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def transition(state_vec: np.ndarray, clip_vec: np.ndarray) -> np.ndarray:
    return np.asarray(state_vec, dtype=np.float32) + np.asarray(clip_vec, dtype=np.float32)


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


def prepare_subjects(subjects: list, dataset_name: str = "deap") -> list:
    prepared = []
    for idx, clips in enumerate(subjects, start=1):
        arr = np.asarray(clips, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Subject {idx} must be shaped (n_clips, 2); got {arr.shape}")
        prepared.append(arr)

    if dataset_name == "deap":
        clip_counts = sorted({arr.shape[0] for arr in prepared})
        if len(clip_counts) != 1:
            print(f"  [Warning] Inconsistent clip counts for DEAP: {clip_counts}")
    return prepared


def compute_global_clip_pool(subjects_for_pool: list) -> np.ndarray:
    return np.concatenate(subjects_for_pool, axis=0).astype(np.float32)


def build_cv_folds(n_subjects: int, n_folds: int = DEFAULT_GRID_FOLDS) -> list:
    n_folds = max(2, min(n_folds, n_subjects))
    return [fold.tolist() for fold in np.array_split(np.arange(n_subjects), n_folds) if len(fold) > 0]


# ------------------------------------------------------------------
# 4. SARSA AGENT
# ------------------------------------------------------------------

class SARSAAgent:
    """
    On-policy SARSA.

    SARSA update rule:
      Q(s,a) ← Q(s,a) + α [r + γ·Q(s',a') + φ(s,s') - Q(s,a)]

    where a' is the *actual* next action chosen by the ε-greedy policy,
    NOT the greedy maximum as in Q-Learning.
    """

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
        """ε-greedy policy (same for both train and eval for on-policy)."""
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_clips)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, state_idx: int, clip_idx: int,
               r: float, next_state_idx: int, next_clip_idx: int,
               phi: float):
        """SARSA update using actual next action (not greedy max)."""
        q_next  = self.Q[next_state_idx, next_clip_idx]   # ← KEY SARSA DIFFERENCE
        td_target = r + self.gamma * q_next + phi
        self.Q[state_idx, clip_idx] += self.alpha * (
            td_target - self.Q[state_idx, clip_idx])

    def reset(self):
        self.Q[:] = 0.0


# ------------------------------------------------------------------
# 5. TRAINING (SARSA on-policy loop)
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

        # 🔥 initial state + action (IMPORTANT for SARSA)
        si = discretize_state(sv, bins)
        ci = agent.choose_action(si, training=True)

        for step_idx in range(PLAYLIST_LEN):

            cv = train_action_vectors[ci]

            # 🔹 transition
            nv = transition(sv, cv)

            # 🔹 next state
            ni = discretize_state(nv, bins)

            # 🔹 reward (paper)
            r = reward(sv, cv, step_idx=step_idx, n_playlist=PLAYLIST_LEN)

            # 🔹 next action (SARSA key step)
            nci = agent.choose_action(ni, training=True)

            # 🔹 shaping
            phi = reward_shaping(si, ni) if use_reward_shaping else 0.0

            # 🔥 SARSA update
            agent.update(si, ci, r, ni, nci, phi)

            # 🔹 move forward
            sv = nv
            si = ni
            ci = nci
# ------------------------------------------------------------------
# 6. EVALUATION
# ------------------------------------------------------------------

def evaluate_subject(agent,
                     action_pool: np.ndarray,
                     n_playlist: int = PLAYLIST_LEN,
                     start_emotion: str = None,
                     bins: int = 20) -> dict:

    action_pool = np.asarray(action_pool, dtype=np.float32)
    start_emotion = resolve_start_emotion(start_emotion)

    # 🔹 continuous state
    sv = np.array(EMOTION_CENTERS[start_emotion], dtype=np.float32)

    errors = []
    emotions = [start_emotion]
    playlist = []

    for _ in range(n_playlist):

        # 🔹 discretize ONLY for Q-table
        si = discretize_state(sv, bins)

        # 🔹 action selection
        ci = agent.choose_action(si, training=False)

        cv = action_pool[ci]

        # 🔹 transition (paper)
        nv = transition(sv, cv)

        # 🔹 compute metrics (continuous!)
        err = angular_error(nv)
        errors.append(err)

        # 🔹 ONLY for logging (not for RL state)
        emotion_label = vector_to_emotion(nv)
        emotions.append(emotion_label)

        playlist.append(ci)

        # move forward
        sv = nv

    fe = errors[-1]

    return {
        "errors":          errors,
        "emotions":        emotions,
        "playlist":        playlist,
        "conv_20":         fe < 20.0,
        "conv_30":         fe < 30.0,
        "converged":       fe < 20.0,
        "final_err":       fe,
        "start":           start_emotion,
    }

# ------------------------------------------------------------------
# 7. LOSO
# ------------------------------------------------------------------

def leave_one_subject_out(subjects: list,
                          alpha: float = 0.1,
                          gamma: float = 0.6,
                          epsilon: float = 0.1,
                          n_episodes: int = DEFAULT_EPISODES,
                          use_reward_shaping: bool = True,
                          dataset_name: str = "deap") -> list:
    results = []
    tag = "w/ shaping" if use_reward_shaping else "w/o shaping"
    print(f"\n  LOSO-SARSA [{tag}] [{dataset_name.upper()}] alpha={alpha} gamma={gamma} epsilon={epsilon} "
          f"episodes={n_episodes}")
    subjects = prepare_subjects(subjects, dataset_name=dataset_name)

    for test_idx in range(len(subjects)):
        test_subject = subjects[test_idx]
        train_subjects = [s for i, s in enumerate(subjects) if i != test_idx]
        train_action_vectors = compute_global_clip_pool(train_subjects)
        bins = 20
        n_states = bins * bins   # 400

        agent = SARSAAgent(n_states, len(train_action_vectors), alpha, gamma, epsilon)
        train_agent(agent, train_action_vectors, n_episodes, use_reward_shaping, bins=bins)

        res            = evaluate_subject(agent, train_action_vectors, bins=bins)
        res["subject"] = test_idx + 1
        results.append(res)

        c20_sym = "v" if res["conv_20"] else "x"
        print(f"    S{test_idx+1:02d} | err={res['final_err']:5.1f} deg | "
              f"<20 deg={c20_sym} | <30 deg={'v' if res['conv_30'] else 'x'} | "
              f"start={res['start']:12s} | "
              f"{' -> '.join(res['emotions'])}")

    return results


# ------------------------------------------------------------------
# 8. GRID SEARCH
# ------------------------------------------------------------------

def grid_search(subjects: list,
                alphas:   list = [0.05, 0.1, 0.2],
                gammas:   list = [0.4,  0.6, 0.8],
                epsilons: list = [0.0, 0.05, 0.1, 0.2],
                n_folds:  int  = DEFAULT_GRID_FOLDS,
                n_episodes: int = DEFAULT_GRID_EPISODES,
                use_reward_shaping: bool = True,
                dataset_name: str = "deap") -> dict:
    tag = "w/ shaping" if use_reward_shaping else "w/o shaping"
    print(f"[Grid Search] SARSA starting ({tag}) [{dataset_name.upper()}] …")
    best_err, best_params = float("inf"), {}
    subjects = prepare_subjects(subjects, dataset_name=dataset_name)
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
            agent = SARSAAgent(n_states, len(train_action_vectors), alpha, gamma, epsilon)
            train_agent(agent, train_action_vectors,
                        n_episodes, use_reward_shaping=use_reward_shaping, bins=bins)
            for vi in val_idx:
                fold_errors.append(
                    evaluate_subject(agent, train_action_vectors, bins=bins)["final_err"])

        mean_err = np.mean(fold_errors) if fold_errors else float("inf")
        print(f"    alpha={alpha:.3f} gamma={gamma:.3f} epsilon={epsilon:.3f} -> "
              f"cv_mean_err={mean_err:.2f} deg")
        if mean_err < best_err:
            best_err    = mean_err
            best_params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}

    print(f"[Grid Search] Best: {best_params}  mean_err={best_err:.1f} deg")
    return best_params


# ------------------------------------------------------------------
# 9. SUMMARY + PLOTS
# ------------------------------------------------------------------

def print_summary_table(results: list, label: str = "SARSA"):
    c20     = [r for r in results if     r["conv_20"]]
    not_c20 = [r for r in results if not r["conv_20"]]
    c30     = [r for r in results if     r["conv_30"]]

    def row(subset, tag):
        if not subset:
            return
        e = np.array([r["errors"] for r in subset])
        print("  {:20s}| ".format(tag) +
              "  ".join(f"{m:5.1f}({s:.1f})"
                        for m, s in zip(e.mean(0), e.std(0))))

    print("\n" + "=" * 80)
    print(f"  [{label}]  Mean angular error (std) per clip")
    print("  {:20s}| ".format("Group") +
          "  ".join(f"{'Clip '+str(i):>10s}" for i in range(1, 7)))
    print("-" * 80)
    row(results, f"Total ({len(results)})")
    row(c20,     f"<20 deg ({len(c20)})")
    row(not_c20, f">=20 deg ({len(not_c20)})")
    print("=" * 80)

    fe = [r["final_err"] for r in results]
    print(f"\n  Convergence summary:")
    print(f"    Angular error < 20 deg  : {len(c20)}/{len(results)}")
    print(f"    Angular error < 30 deg  : {len(c30)}/{len(results)}")
    print(f"    Overall mean (clip 6): {np.mean(fe):.1f} deg +- {np.std(fe):.1f} deg")


def plot_angular_error(results: list, save_path: str,
                       label: str = "SARSA"):
    all_e  = np.array([r["errors"] for r in results])
    mean_e = all_e.mean(axis=0)
    se     = all_e.std(axis=0) / np.sqrt(len(results))
    iters  = np.arange(1, 7)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(iters, mean_e, yerr=se, fmt="-o", color="darkorange",
                capsize=4, label=f"{label} — All subjects")
    ax.axhline(20, color="blue",   ls="--", lw=1.2, label="20° threshold")
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
                                   save_path: str, label: str = "SARSA"):
    iters = np.arange(1, 7)
    me    = lambda r: np.array([x["errors"] for x in r]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, me(results_with),    "-o",  color="darkorange",
            label="With reward shaping")
    ax.plot(iters, me(results_without), "--s", color="tomato",
            label="Without reward shaping")
    ax.axhline(20, color="blue",  ls="--", lw=1.0, label="20° threshold")
    ax.axhline(30, color="green", ls="--", lw=1.0, label="30° threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Angular Error (deg)")
    ax.set_title(f"[{label}] Effect of Reward Shaping")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_convergence_comparison(results: list, label: str = "SARSA",
                                 save_path: str = "sarsa_convergence.png"):
    n     = len(results)
    c20   = sum(r["conv_20"]         for r in results)
    c30   = sum(r["conv_30"]         for r in results)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["< 20°", "< 30°"],
                  [c20, c30],
                  color=["#3498db", "#2ecc71"])
    ax.axhline(19, ls="--", color="red", lw=1.2,
               label="Q-Learning paper result (19/32)")
    for b, v in zip(bars, [c20, c30]):
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


def plot_emotion_trajectory(result: dict, subject_id: int,
                            save_path: str = None, label: str = "SARSA"):
    """Individual subject trajectory on the 2-D emotion wheel."""
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
    ax.plot(xs, ys, "-o", color="darkorange", lw=1.5, ms=6, zorder=6)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    status = (f"Converged ✓ (err={result['final_err']:.1f}°)"
              if result["conv_20"]
              else f"No convergence ✗ (err={result['final_err']:.1f}°)")
    ax.set_title(f"[{label}] Subject {subject_id} — {status}", fontsize=11)
    ax.set_xlabel("Valence →")
    ax.set_ylabel("Arousal →")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_converged_vs_diverged(converged_result: dict, diverged_result: dict,
                                save_path: str = "converged_vs_diverged.png",
                                label: str = "SARSA"):
    """Side-by-side angular error curves for one converged and one diverged subject."""
    iters = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(8, 5))

    c_id  = converged_result["subject"]
    ax.plot(iters, converged_result["errors"], "-o", color="#2ecc71",
            lw=2.2, ms=8, label=f"Subject {c_id} (Converged)", zorder=5)

    d_id  = diverged_result["subject"]
    ax.plot(iters, diverged_result["errors"], "--s", color="#e74c3c",
            lw=2.2, ms=8, label=f"Subject {d_id} (Not converged)", zorder=5)

    ax.axhline(20, color="blue",  ls=":", lw=1.2, label="20° threshold")
    ax.axhline(30, color="green", ls=":", lw=1.2, label="30° threshold")
    ax.set_xlabel("Music Clip # (Iteration)", fontsize=12)
    ax.set_ylabel("Angular Error (°)", fontsize=12)
    ax.set_title(f"[{label}] Angular Error: Converged vs Diverged", fontsize=13)
    ax.set_xticks(iters)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shaping_convergence_bars(results_with: list, results_without: list,
                                  label: str = "SARSA",
                                  save_path: str = "sarsa_shaping_bars.png"):
    """
    Grouped bar chart comparing convergence rates WITH vs WITHOUT reward shaping
    under two criteria (<20°, <30°).
    """
    n = len(results_with)

    w_c20   = sum(r["conv_20"]         for r in results_with)
    w_c30   = sum(r["conv_30"]         for r in results_with)

    wo_c20   = sum(r["conv_20"]         for r in results_without)
    wo_c30   = sum(r["conv_30"]         for r in results_without)

    criteria = ["< 20°", "< 30°"]
    with_vals    = [w_c20, w_c30]
    without_vals = [wo_c20, wo_c30]

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


def plot_all_subjects_trajectories(results: list, save_path: str,
                                   label: str = "SARSA"):
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
            clr = "gold" if name == TARGET_EMOTION else "lightyellow"
            ax.scatter(vx, vy, s=30, color=clr, edgecolors="k", lw=0.4, zorder=3)

        emotions = res["emotions"]
        xs = [EMOTION_CENTERS[e][0] for e in emotions]
        ys = [EMOTION_CENTERS[e][1] for e in emotions]
        ax.plot(xs, ys, "-", color="darkorange", lw=1.0, zorder=4, alpha=0.8)
        ax.plot(xs[0],  ys[0],  "gs", ms=5, zorder=5)
        ax.plot(xs[-1], ys[-1], "r*", ms=7, zorder=5)

        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.axvline(0, color="gray", lw=0.4, ls="--")
        ax.set_xticks([])
        ax.set_yticks([])

        conv = res["conv_20"]
        ax.set_title(f"S{res['subject']} {'✓' if conv else '✗'}",
                     fontsize=7, color="green" if conv else "red",
                     fontweight="bold")
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


# ------------------------------------------------------------------
# 10. MAIN
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

def main():
    parser = argparse.ArgumentParser(
        description="SARSA music emotion RL")
    parser.add_argument("--data_root",   default="./data/Metacsv")
    parser.add_argument("--dataset",     choices=["deap", "dense"], default="deap")
    parser.add_argument("--output_dir",  default="results_sarsa")
    parser.add_argument("--episodes",    type=int,   default=DEFAULT_EPISODES)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--gamma",       type=float, default=0.6)
    parser.add_argument("--epsilon",     type=float, default=0.1)
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--grid_folds", type=int, default=DEFAULT_GRID_FOLDS)
    parser.add_argument("--grid_episodes", type=int, default=DEFAULT_GRID_EPISODES)
    args = parser.parse_args()

    # Auto-adjust data_root if defaults are used but directory doesn't exist
    if args.dataset == "dense" and args.data_root == "./data/Metacsv":
        if os.path.isdir("./EEG_Data"):
            args.data_root = "./EEG_Data"

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(f"  EEG Music Emotion RL  —  SARSA  ({args.dataset.upper()} Dataset)")
    print("=" * 65)

    # For DEAP, we need emotion_summary.csv; for DENSE we use the internal mapping
    if args.dataset == "deap":
        csv_path = os.path.join(args.data_root, "emotion_summary.csv")
        if not os.path.isfile(csv_path):
            csv_path = "./data/Metacsv/emotion_summary.csv"
        _init_globals(csv_path)
    else:
        global EMOTION_CENTERS, EMOTION_NAMES, EMOTION_INDEX, N_EMOTIONS, TARGET_VECTOR
        EMOTION_CENTERS = {}
        # Lowercase mapping for robust lookup
        dense_map_low = {k.lower(): v for k, v in DENSE_EMOTION_MAPPING.items()}
        for group, emotions in EMOTION_GROUPS.items():
            vals = [dense_map_low[e.lower()] for e in emotions if e.lower() in dense_map_low]
            if not vals: continue
            vec = np.mean(vals, axis=0)
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            EMOTION_CENTERS[group] = tuple(vec)
        
        EMOTION_NAMES = list(EMOTION_CENTERS.keys())
        EMOTION_INDEX = {n: i for i, n in enumerate(EMOTION_NAMES)}
        N_EMOTIONS = len(EMOTION_NAMES)
        TARGET_VECTOR = np.array(EMOTION_CENTERS[TARGET_EMOTION])
        print(f"  Emotions: {EMOTION_NAMES}")
        print(f"  Target  : '{TARGET_EMOTION}'  -> {TARGET_VECTOR}")

    if args.dataset == "deap":
        ratings_csv = os.path.join(args.data_root, "participant_ratings.csv")
        if not os.path.isfile(ratings_csv):
             ratings_csv = "./data/Metacsv/participant_ratings.csv"
        
        if os.path.isfile(ratings_csv):
            subjects = load_subjects_from_ratings(ratings_csv)
            print(f"  Loaded {len(subjects)} DEAP subjects.")
        else:
            subjects = synthetic_subjects(32)
            print("  Using synthetic data (DEAP ratings not found).")
    else:
        loader = MATLoader(args.data_root)
        subjects = loader.load_all_subjects_clips()
        if subjects:
            print(f"  Loaded {len(subjects)} DENSE subjects from {args.data_root}.")
        else:
            subjects = synthetic_subjects(39)
            print("  Using synthetic data (DENSE .mat files not found).")

    params_with = {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}
    params_without = dict(params_with)
    if args.grid_search:
        params_with = grid_search(
            subjects,
            n_folds=args.grid_folds,
            n_episodes=args.grid_episodes,
            use_reward_shaping=True,
            dataset_name=args.dataset
        )
        params_without = grid_search(
            subjects,
            n_folds=args.grid_folds,
            n_episodes=args.grid_episodes,
            use_reward_shaping=False,
            dataset_name=args.dataset
        )

    print("\n── LOSO with reward shaping ──")
    results_with = leave_one_subject_out(subjects, **params_with,
                                          n_episodes=args.episodes,
                                          use_reward_shaping=True,
                                          dataset_name=args.dataset)

    print("\n── LOSO without reward shaping ──")
    results_without = leave_one_subject_out(subjects, **params_without,
                                             n_episodes=args.episodes,
                                             use_reward_shaping=False,
                                             dataset_name=args.dataset)

    # ── Summary tables ──────────────────────────────────────────────
    print_summary_table(results_with,    label="SARSA (with shaping)")
    print_summary_table(results_without, label="SARSA (without shaping)")

    # ── Plots ──────────────────────────────────────────────────────
    print("\nGenerating plots …")

    # 1) Angular error (with shaping)
    plot_angular_error(
        results_with, label="SARSA",
        save_path=os.path.join(args.output_dir, "sarsa_angular_error.png"))

    # 2) Reward shaping comparison (angular error curves)
    plot_reward_shaping_comparison(
        results_with, results_without, label="SARSA",
        save_path=os.path.join(args.output_dir, "sarsa_reward_shaping.png"))

    # 3) Convergence bar chart (with shaping)
    plot_convergence_comparison(
        results_with, label="SARSA (with shaping)",
        save_path=os.path.join(args.output_dir, "sarsa_convergence_criteria.png"))

    # 4) Convergence bar chart (without shaping)
    plot_convergence_comparison(
        results_without, label="SARSA (without shaping)",
        save_path=os.path.join(args.output_dir, "sarsa_convergence_no_shaping.png"))

    # 5) Grouped bar chart: with vs without shaping convergence
    plot_shaping_convergence_bars(
        results_with, results_without, label="SARSA",
        save_path=os.path.join(args.output_dir, "sarsa_shaping_bars.png"))

    # 6) All-subjects emotion wheel trajectories (with shaping)
    plot_all_subjects_trajectories(
        results_with, label="SARSA (with shaping)",
        save_path=os.path.join(args.output_dir, "sarsa_all_trajectories.png"))

    # 7) All-subjects trajectories (without shaping)
    plot_all_subjects_trajectories(
        results_without, label="SARSA (without shaping)",
        save_path=os.path.join(args.output_dir, "sarsa_all_trajectories_no_shaping.png"))

    # 8) Converged vs diverged angular error comparison
    converged     = [r for r in results_with if r["conv_20"]]
    not_converged = [r for r in results_with if not r["conv_20"]]

    pref = [r for r in converged if r.get("start") in ("sad", "negative", "fear")]
    chosen_c = pref[0] if pref else (converged[0] if converged else None)

    if chosen_c and not_converged:
        plot_converged_vs_diverged(
            chosen_c, not_converged[0], label="SARSA",
            save_path=os.path.join(args.output_dir, "sarsa_converged_vs_diverged.png"))

    # 9) Individual emotion trajectory plots
    if converged:
        r = converged[0]
        plot_emotion_trajectory(
            r, r["subject"], label="SARSA",
            save_path=os.path.join(args.output_dir,
                                   f"sarsa_traj_s{r['subject']}_conv.png"))
    if not_converged:
        r = not_converged[0]
        plot_emotion_trajectory(
            r, r["subject"], label="SARSA",
            save_path=os.path.join(args.output_dir,
                                   f"sarsa_traj_s{r['subject']}_div.png"))

    print(f"\n✅  Done.  Results saved to: {args.output_dir}/")
    return results_with, results_without


if __name__ == "__main__":
    main()