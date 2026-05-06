"""
music_emotion_sarsa.py  —  SARSA (on-policy TD) for Music Emotion RL
=====================================================================
SARSA vs Q-Learning difference:
  Q-Learning  →  off-policy: updates using max_a' Q(s',a')   (greedy next)
  SARSA       →  on-policy:  updates using Q(s',a')           (actual next action)

Same environment, reward, reward-shaping, and LOSO protocol as Dutta et al. 2020.
Same three convergence criteria:
  • exact_converged : final emotion == "happy"          (paper primary)
  • conv_30         : final angular error < 30°
  • conv_60         : final angular error < 60°

Run:
  python music_emotion_sarsa.py
  python music_emotion_sarsa.py --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv --episodes 2000
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
import os
import argparse

# ──────────────────────────────────────────────────────────────────
# 1. EMOTION WHEEL
# ──────────────────────────────────────────────────────────────────

def normalize(col):
    return 2 * (col - col.min()) / (col.max() - col.min()) - 1


def load_emotion_centers(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    df["Valence_norm"] = normalize(df["Valence"])
    df["Arousal_norm"] = normalize(df["Arousal"])
    r = np.sqrt(df["Valence_norm"]**2 + df["Arousal_norm"]**2)
    max_r = r.max()
    df["Valence_unit"] = df["Valence_norm"] / max_r
    df["Arousal_unit"] = df["Arousal_norm"] / max_r
    return {row["Emotion"]: (row["Valence_unit"], row["Arousal_unit"])
            for _, row in df.iterrows()}


# ──────────────────────────────────────────────────────────────────
# 2. GLOBALS
# ──────────────────────────────────────────────────────────────────

EMOTION_CENTERS: dict = {}
EMOTION_NAMES:   list = []
EMOTION_INDEX:   dict = {}
N_EMOTIONS:      int  = 0
TARGET_EMOTION         = "happy"
TARGET_VECTOR          = None
LAMBDA                 = -100.0


def _init_globals(csv_path: str):
    global EMOTION_CENTERS, EMOTION_NAMES, EMOTION_INDEX
    global N_EMOTIONS, TARGET_VECTOR
    EMOTION_CENTERS = load_emotion_centers(csv_path)
    EMOTION_NAMES   = list(EMOTION_CENTERS.keys())
    EMOTION_INDEX   = {n: i for i, n in enumerate(EMOTION_NAMES)}
    N_EMOTIONS      = len(EMOTION_NAMES)
    TARGET_VECTOR   = np.array(EMOTION_CENTERS[TARGET_EMOTION])
    print(f"  Emotions: {EMOTION_NAMES}")
    print(f"  Target  : '{TARGET_EMOTION}'  → {TARGET_VECTOR}")


# ──────────────────────────────────────────────────────────────────
# 3. CORE ENVIRONMENT FUNCTIONS  (identical to Q-Learning script)
# ──────────────────────────────────────────────────────────────────

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
    nv   = state_vec + clip_vec
    norm = np.linalg.norm(nv)
    return nv / norm if norm > 1.0 else nv


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def reward(state_vec: np.ndarray, clip_vec: np.ndarray,
           target: np.ndarray = None) -> float:
    if target is None:
        target = TARGET_VECTOR
    theta = abs(_angle(target, clip_vec)) + abs(_angle(state_vec, clip_vec))
    return 1.0 if theta <= np.pi else 0.0


def reward_shaping(state_idx: int, next_state_idx: int) -> float:
    return LAMBDA if state_idx == next_state_idx else 0.0


def _non_target():
    return [e for e in EMOTION_NAMES if e != TARGET_EMOTION]


# ──────────────────────────────────────────────────────────────────
# 4. SARSA AGENT
# ──────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────
# 5. TRAINING (SARSA on-policy loop)
# ──────────────────────────────────────────────────────────────────

def train_agent(agent: SARSAAgent,
                train_subjects_clips: list,
                n_episodes: int = 1000,
                use_reward_shaping: bool = True):
    agent.reset()
    non_target = _non_target()

    for _ in range(n_episodes):
        clips   = train_subjects_clips[np.random.randint(len(train_subjects_clips))]
        n_clips = len(clips)

        start = non_target[np.random.randint(len(non_target))]
        sv    = np.array(EMOTION_CENTERS[start])
        si    = EMOTION_INDEX[start]

        # Choose first action (SARSA requires action before loop)
        ci = agent.choose_action(si, training=True) % n_clips

        for _ in range(6):
            cv  = clips[ci]
            nv  = transition(sv, cv)
            ni  = EMOTION_INDEX[vector_to_emotion(nv)]
            r   = reward(sv, cv)
            phi = reward_shaping(si, ni) if use_reward_shaping else 0.0

            # Choose NEXT action (on-policy) before updating
            nci = agent.choose_action(ni, training=True) % n_clips

            # SARSA update: uses actual next action nci, not max
            agent.update(si, ci, r, ni, nci, phi)

            sv, si, ci = nv, ni, nci


# ──────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────────────────────────

def evaluate_subject(agent: SARSAAgent,
                     test_clips: list,
                     n_playlist: int = 6,
                     start_emotion: str = None) -> dict:
    non_target = _non_target()

    if start_emotion is None or start_emotion == TARGET_EMOTION \
            or start_emotion not in EMOTION_INDEX:
        start_emotion = non_target[np.random.randint(len(non_target))]

    sv      = np.array(EMOTION_CENTERS[start_emotion])
    si      = EMOTION_INDEX[start_emotion]
    n_clips = len(test_clips)

    errors, emotions, playlist = [], [start_emotion], []

    for _ in range(n_playlist):
        ci  = agent.choose_action(si, training=False) % n_clips
        cv  = test_clips[ci]
        nv  = transition(sv, cv)
        ni  = EMOTION_INDEX[vector_to_emotion(nv)]

        errors.append(angular_error(nv))
        emotions.append(EMOTION_NAMES[ni])
        playlist.append(ci)
        sv, si = nv, ni

    fe = errors[-1]
    return {
        "errors":          errors,
        "emotions":        emotions,
        "playlist":        playlist,
        "exact_converged": emotions[-1] == TARGET_EMOTION,
        "conv_30":         fe < 30.0,
        "conv_60":         fe < 60.0,
        "converged":       emotions[-1] == TARGET_EMOTION,
        "final_err":       fe,
        "start":           start_emotion,
    }


# ──────────────────────────────────────────────────────────────────
# 7. LOSO
# ──────────────────────────────────────────────────────────────────

def leave_one_subject_out(subjects: list,
                          alpha: float = 0.1,
                          gamma: float = 0.6,
                          epsilon: float = 0.1,
                          n_episodes: int = 2000,
                          use_reward_shaping: bool = True) -> list:
    results = []
    tag = "w/ shaping" if use_reward_shaping else "w/o shaping"
    print(f"\n  LOSO-SARSA [{tag}]  α={alpha} γ={gamma} ε={epsilon} "
          f"episodes={n_episodes}")

    for test_idx in range(len(subjects)):
        train   = [subjects[i] for i in range(len(subjects)) if i != test_idx]
        test    = subjects[test_idx]
        n_clips = len(test)

        agent = SARSAAgent(N_EMOTIONS, n_clips, alpha, gamma, epsilon)
        train_agent(agent, train, n_episodes, use_reward_shaping)

        res            = evaluate_subject(agent, test)
        res["subject"] = test_idx + 1
        results.append(res)

        conv_sym = "✓" if res["exact_converged"] else "✗"
        print(f"    S{test_idx+1:02d} | err={res['final_err']:5.1f}° | "
              f"exact={conv_sym} | <30°={'✓' if res['conv_30'] else '✗'} | "
              f"<60°={'✓' if res['conv_60'] else '✗'} | "
              f"start={res['start']:12s} | "
              f"{' → '.join(res['emotions'])}")

    return results


# ──────────────────────────────────────────────────────────────────
# 8. GRID SEARCH
# ──────────────────────────────────────────────────────────────────

def grid_search(subjects: list,
                alphas:   list = [0.05, 0.1, 0.2],
                gammas:   list = [0.4,  0.6, 0.8],
                epsilons: list = [0.05, 0.1, 0.2],
                n_folds:  int  = 10,
                n_episodes: int = 300) -> dict:
    print("[Grid Search] SARSA starting …")
    best_err, best_params = float("inf"), {}
    fold_size = max(1, len(subjects) // n_folds)

    for alpha, gamma, epsilon in product(alphas, gammas, epsilons):
        fold_errors = []
        for fold in range(n_folds):
            val_idx   = list(range(fold * fold_size,
                                   min((fold + 1) * fold_size, len(subjects))))
            train_idx = [i for i in range(len(subjects)) if i not in val_idx]
            if not train_idx or not val_idx:
                continue
            for vi in val_idx:
                n_clips = len(subjects[vi])
                agent   = SARSAAgent(N_EMOTIONS, n_clips, alpha, gamma, epsilon)
                train_agent(agent, [subjects[i] for i in train_idx],
                            n_episodes, use_reward_shaping=True)
                fold_errors.append(
                    evaluate_subject(agent, subjects[vi])["final_err"])

        mean_err = np.mean(fold_errors) if fold_errors else float("inf")
        if mean_err < best_err:
            best_err    = mean_err
            best_params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}

    print(f"[Grid Search] Best: {best_params}  mean_err={best_err:.1f}°")
    return best_params


# ──────────────────────────────────────────────────────────────────
# 9. SUMMARY + PLOTS
# ──────────────────────────────────────────────────────────────────

def print_summary_table(results: list, label: str = "SARSA"):
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
    print(f"  [{label}]  Mean angular error (std) per clip")
    print("  {:20s}| ".format("Group") +
          "  ".join(f"{'Clip '+str(i):>10s}" for i in range(1, 7)))
    print("-" * 80)
    row(results, f"Total ({len(results)})")
    row(exact,   f"Exact happy ({len(exact)})")
    row(inexact, f"Not happy ({len(inexact)})")
    print("=" * 80)

    fe = [r["final_err"] for r in results]
    print(f"\n  Convergence summary:")
    print(f"    Exact emotion = happy : {len(exact)}/{len(results)}")
    print(f"    Angular error < 30°  : {len(c30)}/{len(results)}")
    print(f"    Angular error < 60°  : {len(c60)}/{len(results)}")
    print(f"    Overall mean (clip 6): {np.mean(fe):.1f}° ± {np.std(fe):.1f}°")


def plot_angular_error(results: list, save_path: str,
                       label: str = "SARSA"):
    all_e  = np.array([r["errors"] for r in results])
    mean_e = all_e.mean(axis=0)
    se     = all_e.std(axis=0) / np.sqrt(len(results))
    iters  = np.arange(1, 7)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(iters, mean_e, yerr=se, fmt="-o", color="darkorange",
                capsize=4, label=f"{label} — All subjects")
    ax.axhline(60, color="orange", ls="--", lw=1.2, label="60° threshold")
    ax.axhline(30, color="green",  ls="--", lw=1.2, label="30° threshold")
    ax.set_xlabel("Music Clip # (Iteration)")
    ax.set_ylabel("Angular Error (°)")
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
    ax.axhline(60, color="gray", ls="--", lw=1.0, label="60° threshold")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Angular Error (°)")
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
    exact = sum(r["exact_converged"] for r in results)
    c30   = sum(r["conv_30"]         for r in results)
    c60   = sum(r["conv_60"]         for r in results)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Exact happy", "< 30°", "< 60°"],
                  [exact, c30, c60],
                  color=["#2ecc71", "#3498db", "#e67e22"])
    ax.axhline(19, ls="--", color="red", lw=1.2,
               label="Q-Learning paper result (19/32)")
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

        conv = res["exact_converged"]
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


# ──────────────────────────────────────────────────────────────────
# 10. MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SARSA music emotion RL")
    parser.add_argument("--data_root",   default="/home/sahil/Desktop/Rl_Project_Final/data/Metacsv")
    parser.add_argument("--output_dir",  default="results_sarsa")
    parser.add_argument("--episodes",    type=int,   default=2000)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--gamma",       type=float, default=0.6)
    parser.add_argument("--epsilon",     type=float, default=0.1)
    parser.add_argument("--grid_search", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  EEG Music Emotion RL  —  SARSA  (on-policy)")
    print("=" * 65)

    csv_path = os.path.join(args.data_root, "emotion_summary.csv")
    _init_globals(csv_path)

    ratings_csv = os.path.join(args.data_root, "participant_ratings.csv")
    if os.path.isfile(ratings_csv):
        df_r = pd.read_csv(ratings_csv)
        subjects = []
        for pid, grp in df_r.groupby("Participant_id"):
            clips = grp[["Valence", "Arousal"]].values.astype(np.float32)
            clips[:, 0] = 2 * (clips[:, 0] - 1) / 8 - 1
            clips[:, 1] = 2 * (clips[:, 1] - 1) / 8 - 1
            subjects.append(clips)
        print(f"  Loaded {len(subjects)} subjects.")
    else:
        rng      = np.random.default_rng(42)
        subjects = [rng.uniform(-1, 1, (np.random.randint(10, 40), 2)).astype(np.float32)
                    for _ in range(32)]
        print("  Using synthetic data (participant_ratings.csv not found).")

    params = {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}
    if args.grid_search:
        params = grid_search(subjects, n_episodes=300)

    print("\n── LOSO with reward shaping ──")
    results_with = leave_one_subject_out(subjects, **params,
                                          n_episodes=args.episodes,
                                          use_reward_shaping=True)

    print("\n── LOSO without reward shaping ──")
    results_without = leave_one_subject_out(subjects, **params,
                                             n_episodes=args.episodes,
                                             use_reward_shaping=False)

    print_summary_table(results_with, label="SARSA")

    plot_angular_error(
        results_with, label="SARSA",
        save_path=os.path.join(args.output_dir, "sarsa_angular_error.png"))

    plot_reward_shaping_comparison(
        results_with, results_without, label="SARSA",
        save_path=os.path.join(args.output_dir, "sarsa_reward_shaping.png"))

    plot_convergence_comparison(
        results_with, label="SARSA",
        save_path=os.path.join(args.output_dir, "sarsa_convergence_criteria.png"))

    plot_all_subjects_trajectories(
        results_with, label="SARSA",
        save_path=os.path.join(args.output_dir, "sarsa_all_trajectories.png"))

    print(f"\n✅  Done.  Results saved to: {args.output_dir}/")
    return results_with


if __name__ == "__main__":
    main()
