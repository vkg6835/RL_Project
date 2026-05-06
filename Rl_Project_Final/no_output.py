import os
import pickle
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

RADIUS = 9.0
N_SUBJECTS = 32
N_CLIPS = 40
STEPS_PER_EPISODE = 6
DEFAULT_TARGET = "happy"
LAMBDA = -100.0

EPS = 1e-9


# ============================================================
# EMOTION WHEEL
# ============================================================

def normalize(col: pd.Series) -> pd.Series:
    return 2 * (col - col.min()) / (col.max() - col.min() + EPS) - 1


def scale_to_radius(v: np.ndarray, radius: float = RADIUS) -> np.ndarray:
    """
    Scale vectors into a circle of given radius while keeping direction.
    """
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    if norm < EPS:
        return v.copy()
    if norm > radius:
        return v * (radius / norm)
    return v


def load_emotion_wheel(summary_csv: str, radius: float = RADIUS) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)

    required = {"Emotion", "Valence", "Arousal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"emotion_summary.csv missing columns: {missing}")

    df["Valence_norm"] = normalize(df["Valence"])
    df["Arousal_norm"] = normalize(df["Arousal"])

    # Keep the 2D geometry, then scale it to the requested wheel radius.
    raw = df[["Valence_norm", "Arousal_norm"]].to_numpy(dtype=float)
    raw_r = np.linalg.norm(raw, axis=1)
    max_r = float(raw_r.max()) if len(raw_r) else 1.0
    max_r = max(max_r, EPS)

    df["Valence_unit"] = (df["Valence_norm"] / max_r) * radius
    df["Arousal_unit"] = (df["Arousal_norm"] / max_r) * radius

    return df


def build_emotion_centers(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    centers = {
        row["Emotion"]: np.array([row["Valence_unit"], row["Arousal_unit"]], dtype=float)
        for _, row in df.iterrows()
    }
    return centers


def plot_emotion_wheel(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    boundary = plt.Circle((0, 0), RADIUS, fill=False, linewidth=1.3, linestyle="--", alpha=0.5)
    ax.add_patch(boundary)

    ax.scatter(df["Valence_unit"], df["Arousal_unit"], s=60)

    for _, row in df.iterrows():
        ax.text(row["Valence_unit"], row["Arousal_unit"], str(row["Emotion"]), fontsize=9)

    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-RADIUS * 1.1, RADIUS * 1.1)
    ax.set_ylim(-RADIUS * 1.1, RADIUS * 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"Emotion Wheel (Radius = {RADIUS})")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved emotion wheel: {save_path}")


def angle(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + EPS
    cos = np.dot(v1, v2) / denom
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def angular_error(vec: np.ndarray, target: np.ndarray) -> float:
    return float(np.degrees(angle(vec, target)))


def vector_to_emotion(vec: np.ndarray, centers: Dict[str, np.ndarray]) -> str:
    """
    IMPORTANT FIX:
    Use nearest-center Euclidean mapping, not cosine-only mapping.
    Cosine alone collapses many distinct states onto the same direction.
    """
    vec = np.asarray(vec, dtype=float)
    best_name = None
    best_dist = float("inf")

    for name, center in centers.items():
        dist = np.linalg.norm(vec - center)
        if dist < best_dist:
            best_dist = dist
            best_name = name

    return best_name


def transition(state_vec: np.ndarray, clip_vec: np.ndarray, radius: float = RADIUS) -> np.ndarray:
    """
    Paper-style transition: s' = s + a
    We only clip if the vector grows beyond the chosen GEW radius.
    """
    next_vec = np.asarray(state_vec, dtype=float) + np.asarray(clip_vec, dtype=float)
    return scale_to_radius(next_vec, radius=radius)


# ============================================================
# DEAP .DAT LOADING
# ============================================================

class DEAPLoader:
    def __init__(self, root_dir: str, n_subjects: int = N_SUBJECTS, radius: float = RADIUS):
        self.root_dir = root_dir
        self.n_subjects = n_subjects
        self.radius = radius

    def subject_path(self, sid: int) -> str:
        return os.path.join(self.root_dir, f"s{sid:02d}.dat")

    @staticmethod
    def _load_pickle(path: str):
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")

    @staticmethod
    def _extract_labels(obj):
        if isinstance(obj, dict):
            for key in ("labels", "label", "y"):
                if key in obj:
                    return np.asarray(obj[key])
        raise ValueError("Could not find labels in .dat file")

    def load_subject(self, sid: int) -> np.ndarray:
        path = self.subject_path(sid)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        obj = self._load_pickle(path)
        labels = self._extract_labels(obj)

        if labels.ndim != 2 or labels.shape[1] < 2:
            raise ValueError(f"Invalid labels shape in {path}: {labels.shape}")

        # DEAP labels are in [1..9] for valence/arousal.
        va = labels[:, :2].astype(float)
        v = 2 * (va[:, 0] - 1) / 8.0 - 1.0
        a = 2 * (va[:, 1] - 1) / 8.0 - 1.0

        clips = np.column_stack([v, a]).astype(float)

        # Scale to radius so the transition math is consistent with the wheel.
        clipped = []
        for vec in clips:
            clipped.append(scale_to_radius(vec * self.radius, radius=self.radius))
        return np.asarray(clipped, dtype=float)

    def load_all_subjects(self) -> List[np.ndarray]:
        subjects = []
        for sid in range(1, self.n_subjects + 1):
            clips = self.load_subject(sid)
            subjects.append(clips)
            print(f"Loaded s{sid:02d}: {len(clips)} clips")
        return subjects


# ============================================================
# SARSA AGENT
# ============================================================

class SARSAAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.6,
        epsilon: float = 0.1,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=float)

    def reset(self):
        self.Q[:] = 0.0

    def choose_action(self, state_idx: int, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s: int, a: int, r: float, s2: int, a2: int, phi: float):
        td_target = r + self.gamma * self.Q[s2, a2] + phi
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error


# ============================================================
# REWARD
# ============================================================

def reward(state_vec: np.ndarray, clip_vec: np.ndarray, target_vec: np.ndarray) -> float:
    """
    Paper reward: 1 if theta(s,a) <= 180°, else 0
    Plus a small directional bonus so the learner can actually move.
    """
    theta = abs(angle(target_vec, clip_vec)) + abs(angle(state_vec, clip_vec))
    base = 1.0 if theta <= np.pi else 0.0

    next_vec = transition(state_vec, clip_vec)
    dist_bonus = np.linalg.norm(state_vec - target_vec) - np.linalg.norm(next_vec - target_vec)

    return base + 0.5 * dist_bonus


def reward_shaping(state_idx: int, next_state_idx: int) -> float:
    return LAMBDA if state_idx == next_state_idx else 0.0


# ============================================================
# TRAINING / EVALUATION
# ============================================================

def start_emotion_random(emotion_names: List[str], target_emotion: Optional[str] = None) -> str:
    if target_emotion is None:
        return random.choice(emotion_names)
    return random.choice(emotion_names)


def train_agent(
    agent: SARSAAgent,
    train_subjects_clips: List[np.ndarray],
    centers: Dict[str, np.ndarray],
    target_emotion: str = DEFAULT_TARGET,
    n_episodes: int = 5000,
    steps_per_episode: int = STEPS_PER_EPISODE,
):
    agent.reset()

    emotion_names = list(centers.keys())
    emotion_index = {name: i for i, name in enumerate(emotion_names)}
    target_vec = centers[target_emotion]

    for _ in range(n_episodes):
        clips = train_subjects_clips[np.random.randint(len(train_subjects_clips))]
        n_clips = len(clips)

        # Random start from ANY emotion state.
        start = random.choice(emotion_names)
        sv = centers[start].copy()
        si = emotion_index[start]

        # SARSA starts with an actual action.
        a = agent.choose_action(si, training=True)

        for _ in range(steps_per_episode):
            clip_vec = clips[a % n_clips]

            nv = transition(sv, clip_vec)
            next_emotion = vector_to_emotion(nv, centers)
            s2 = emotion_index[next_emotion]

            r = reward(sv, clip_vec, target_vec)
            phi = reward_shaping(si, s2)

            a2 = agent.choose_action(s2, training=True)
            agent.update(si, a, r, s2, a2, phi)

            sv, si, a = nv, s2, a2


def evaluate_episode(
    agent: SARSAAgent,
    test_clips: np.ndarray,
    centers: Dict[str, np.ndarray],
    target_emotion: str = DEFAULT_TARGET,
    start_emotion: Optional[str] = None,
    steps: int = STEPS_PER_EPISODE,
):
    emotion_names = list(centers.keys())
    emotion_index = {name: i for i, name in enumerate(emotion_names)}
    target_vec = centers[target_emotion]
    n_clips = len(test_clips)

    if start_emotion is None or start_emotion not in emotion_index:
        start_emotion = random.choice(emotion_names)

    sv = centers[start_emotion].copy()
    si = emotion_index[start_emotion]
    a = agent.choose_action(si, training=False)

    errors = []
    emotions = [start_emotion]
    playlist = []

    for _ in range(steps):
        clip_vec = test_clips[a % n_clips]
        nv = transition(sv, clip_vec)
        next_emotion = vector_to_emotion(nv, centers)
        s2 = emotion_index[next_emotion]

        errors.append(angular_error(nv, target_vec))
        emotions.append(next_emotion)
        playlist.append(int(a))

        sv, si = nv, s2
        a = agent.choose_action(si, training=False)

    return {
        "start": start_emotion,
        "emotions": emotions,
        "playlist": playlist,
        "errors": errors,
        "final_err": errors[-1] if errors else None,
        "converged": emotions[-1] == target_emotion,
    }


def evaluate_best_start(
    agent: SARSAAgent,
    test_clips: np.ndarray,
    centers: Dict[str, np.ndarray],
    target_emotion: str = DEFAULT_TARGET,
    steps: int = STEPS_PER_EPISODE,
):
    best = None
    best_err = float("inf")
    for start in centers.keys():
        if start == target_emotion:
            continue
        res = evaluate_episode(agent, test_clips, centers, target_emotion, start, steps)
        if res["final_err"] < best_err:
            best_err = res["final_err"]
            best = res
    return best


def leave_one_subject_out(
    subjects: List[np.ndarray],
    centers: Dict[str, np.ndarray],
    target_emotion: str = DEFAULT_TARGET,
    alpha: float = 0.1,
    gamma: float = 0.6,
    epsilon: float = 0.1,
    n_episodes: int = 5000,
    eval_mode: str = "random",
):
    results = []
    n_states = len(centers)
    n_actions = N_CLIPS

    for test_idx in range(len(subjects)):
        train_subjects = [subjects[i] for i in range(len(subjects)) if i != test_idx]
        test_subject = subjects[test_idx]

        agent = SARSAAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

        train_agent(
            agent,
            train_subjects,
            centers=centers,
            target_emotion=target_emotion,
            n_episodes=n_episodes,
        )

        if eval_mode == "best":
            res = evaluate_best_start(agent, test_subject, centers, target_emotion, steps=STEPS_PER_EPISODE)
        else:
            res = evaluate_episode(agent, test_subject, centers, target_emotion, start_emotion=None, steps=STEPS_PER_EPISODE)

        res["subject"] = test_idx + 1
        results.append(res)

        print(
            f"Subject {test_idx+1:02d} | "
            f"err={res['final_err']:.1f}° | "
            f"converged={res['converged']} | "
            f"start={res['start']} | "
            f"path: {' -> '.join(res['emotions'])}"
        )

    return results


# ============================================================
# PLOTS / SUMMARY
# ============================================================

def plot_angular_error(results, save_path: str):
    all_e = np.array([r["errors"] for r in results], dtype=float)
    mean_e = all_e.mean(axis=0)
    se = all_e.std(axis=0) / np.sqrt(len(results))

    iters = np.arange(1, len(mean_e) + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(iters, mean_e, yerr=se, fmt="-o", capsize=4, label="All subjects (mean ± SE)")
    ax.set_xlabel("Music Clip # (Iteration)")
    ax.set_ylabel("Angular Error (°)")
    ax.set_title("Mean Angular Error per Iteration")
    ax.set_xticks(iters)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_reward_shaping_comparison(results_with, results_without, save_path: str):
    iters = np.arange(1, STEPS_PER_EPISODE + 1)

    def mean_errors(results):
        return np.array([r["errors"] for r in results], dtype=float).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, mean_errors(results_with), "-o", label="With reward shaping")
    ax.plot(iters, mean_errors(results_without), "--s", label="Without reward shaping")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Angular Error (°)")
    ax.set_title("Effect of Reward Shaping")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_emotion_trajectory(result, centers, target_emotion=DEFAULT_TARGET, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")

    boundary = plt.Circle((0, 0), RADIUS, color="gray", fill=False, linestyle="--", alpha=0.3)
    ax.add_patch(boundary)

    for name, center in centers.items():
        color = "gold" if name == target_emotion else "lightblue"
        ax.scatter(center[0], center[1], s=90, zorder=5, color=color, edgecolors="k", lw=0.7)
        ax.text(center[0] + 0.15, center[1] + 0.15, name, fontsize=8)

    emotions = result["emotions"]
    xs = [centers[e][0] for e in emotions]
    ys = [centers[e][1] for e in emotions]

    ax.plot(xs, ys, "-o", color="navy", lw=1.5, ms=6, zorder=6)

    ax.set_xlim(-RADIUS * 1.1, RADIUS * 1.1)
    ax.set_ylim(-RADIUS * 1.1, RADIUS * 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"Subject {result['subject']} — {'Converged' if result['converged'] else 'Not Converged'}")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def plot_all_subjects_trajectories(results, centers, target_emotion=DEFAULT_TARGET, save_path: str = "all_trajectories.png"):
    n = len(results)
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
        ax = axes[r, c]

        boundary = plt.Circle((0, 0), RADIUS, color="gray", fill=False, lw=0.5, ls="--", alpha=0.3)
        ax.add_patch(boundary)

        for name, center in centers.items():
            clr = "gold" if name == target_emotion else "lightblue"
            ax.scatter(center[0], center[1], s=25, color=clr, edgecolors="k", lw=0.4, zorder=3)

        emotions = res["emotions"]
        xs = [centers[e][0] for e in emotions]
        ys = [centers[e][1] for e in emotions]

        ax.plot(xs, ys, "-", color="navy", lw=1.0, alpha=0.85, zorder=4)
        ax.plot(xs[0], ys[0], "gs", ms=4, zorder=5)
        ax.plot(xs[-1], ys[-1], "r*", ms=7, zorder=5)

        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.axvline(0, color="gray", lw=0.4, ls="--")

        ax.set_xlim(-RADIUS * 1.1, RADIUS * 1.1)
        ax.set_ylim(-RADIUS * 1.1, RADIUS * 1.1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        sid = res["subject"]
        conv = res["converged"]
        ax.set_title(f"S{sid} {'✓' if conv else '✗'}", fontsize=8,
                     color="green" if conv else "red", fontweight="bold")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Emotion Wheel Trajectories — All Subjects", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(results):
    converged = [r for r in results if r["converged"]]
    not_converged = [r for r in results if not r["converged"]]

    def row(subset, label):
        if not subset:
            return
        e = np.array([r["errors"] for r in subset], dtype=float)
        print("  {:16s}| ".format(label) + "  ".join(f"{m:5.1f}({s:.1f})" for m, s in zip(e.mean(0), e.std(0))))

    print("\n" + "=" * 76)
    print("Table I — Mean angular error (std) per clip")
    print("  {:16s}| ".format("Group") + "  ".join(f"{'Clip '+str(i):>10s}" for i in range(1, STEPS_PER_EPISODE + 1)))
    print("-" * 76)
    row(results, f"Total ({len(results)})")
    row(converged, f"Success ({len(converged)})")
    row(not_converged, f"Fail ({len(not_converged)})")
    print("=" * 76)
    print(f"\n  Convergence: {len(converged)}/{len(results)} → '{DEFAULT_TARGET}'")
    fe = [r["final_err"] for r in results]
    print(f"  Overall mean (clip {STEPS_PER_EPISODE}): {np.mean(fe):.1f}° ± {np.std(fe):.1f}°\n")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="data/Metacsv/emotion_summary.csv")
    parser.add_argument("--data_root", type=str, default="data/data_preprocessed_python")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_mode", type=str, default="random", choices=["random", "best"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)

    print("=" * 60)
    print("  EEG Music Emotion RL  —  SARSA + GEW")
    print(f"  Target: {args.target} | Radius: {RADIUS}")
    print("=" * 60)

    gew_df = load_emotion_wheel(args.summary_csv, radius=RADIUS)
    centers = build_emotion_centers(gew_df)

    print("\nEmotion centers:")
    for k, v in centers.items():
        print(f"  {k:12s} -> ({v[0]: .3f}, {v[1]: .3f})")

    plot_emotion_wheel(gew_df, save_path=os.path.join(args.output_dir, "emotion_wheel.png"))

    loader = DEAPLoader(args.data_root, n_subjects=N_SUBJECTS, radius=RADIUS)
    subjects = loader.load_all_subjects()

    if len(subjects) < 2:
        raise RuntimeError("Need at least 2 subjects.")

    print(f"\nLoaded {len(subjects)} subjects successfully.")

    params = {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}

    print("\n── LOSO with reward shaping ──")
    results_with = leave_one_subject_out(
        subjects,
        centers=centers,
        target_emotion=args.target,
        n_episodes=args.episodes,
        eval_mode=args.eval_mode,
        **params,
    )

    print("\n── LOSO without reward shaping ──")
    global LAMBDA
    LAMBDA = 0.0
    results_without = leave_one_subject_out(
        subjects,
        centers=centers,
        target_emotion=args.target,
        n_episodes=args.episodes,
        eval_mode=args.eval_mode,
        **params,
    )
    LAMBDA = -100.0

    print_summary_table(results_with)

    print("Generating plots ...")
    plot_angular_error(results_with, save_path=os.path.join(args.output_dir, "angular_error.png"))
    plot_reward_shaping_comparison(results_with, results_without, save_path=os.path.join(args.output_dir, "reward_shaping.png"))

    converged = [r for r in results_with if r["converged"]]
    not_converged = [r for r in results_with if not r["converged"]]

    if converged:
        plot_emotion_trajectory(
            converged[0],
            centers,
            target_emotion=args.target,
            save_path=os.path.join(args.output_dir, f"trajectory_s{converged[0]['subject']}_converged.png"),
        )

    if not_converged:
        plot_emotion_trajectory(
            not_converged[0],
            centers,
            target_emotion=args.target,
            save_path=os.path.join(args.output_dir, f"trajectory_s{not_converged[0]['subject']}_diverged.png"),
        )

    plot_all_subjects_trajectories(
        results_with,
        centers,
        target_emotion=args.target,
        save_path=os.path.join(args.output_dir, "all_trajectories.png"),
    )

    summary_rows = []
    for r in results_with:
        summary_rows.append(
            {
                "subject": r["subject"],
                "start": r["start"],
                "final_err": r["final_err"],
                "converged": r["converged"],
                "path": " -> ".join(r["emotions"]),
                "playlist": ",".join(map(str, r["playlist"])),
            }
        )

    out_csv = os.path.join(args.output_dir, "rl_results_summary.csv")
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    print(f"\nDone. Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()