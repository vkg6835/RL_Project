"""
Emotion Wheel + RL Agent on DEAP-style .dat files
==================================================

Pipeline:
1) Build a normalized Geneva Emotion Wheel (GEW) from emotion_summary.csv
2) Load all 32 subject .dat files
3) Convert each subject's trial labels into emotion-space clip vectors
4) Train a Q-learning agent using the wheel + subject clip vectors
5) Evaluate with LOSO (leave-one-subject-out)
6) Save plots and results

Assumption:
- Each .dat file is DEAP-style and contains:
    obj["labels"] -> shape (40, 4) or at least (40, 2)
    obj["data"]   -> optional, not required here
- labels columns 0 and 1 are Valence and Arousal ratings in [1..9]
"""

import os
import pickle
import argparse
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1) EMOTION WHEEL (GEW)
# ============================================================

def normalize_col(col: pd.Series) -> pd.Series:
    return 2 * (col - col.min()) / (col.max() - col.min() + 1e-9) - 1


def load_emotion_summary(summary_csv: str) -> pd.DataFrame:
    """
    Expects columns like:
        Emotion, Valence, Arousal
    """
    df = pd.read_csv(summary_csv)

    required = {"Emotion", "Valence", "Arousal"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"emotion_summary.csv missing columns: {missing}")

    df["Valence_norm"] = normalize_col(df["Valence"])
    df["Arousal_norm"] = normalize_col(df["Arousal"])

    # Scale to unit circle
    r = np.sqrt(df["Valence_norm"] ** 2 + df["Arousal_norm"] ** 2)
    max_r = float(r.max()) if len(r) else 1.0
    max_r = max(max_r, 1e-9)

    df["Valence_unit"] = df["Valence_norm"] / max_r
    df["Arousal_unit"] = df["Arousal_norm"] / max_r

    return df


def build_emotion_centers(df: pd.DataFrame):
    return {
        row["Emotion"]: (float(row["Valence_unit"]), float(row["Arousal_unit"]))
        for _, row in df.iterrows()
    }


def plot_emotion_wheel(df: pd.DataFrame, save_path: str = "results/emotion_wheel.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    circle = plt.Circle((0, 0), 1, fill=False, linewidth=1.3)
    ax.add_patch(circle)

    ax.scatter(df["Valence_unit"], df["Arousal_unit"], s=50)

    for _, row in df.iterrows():
        ax.text(row["Valence_unit"], row["Arousal_unit"], str(row["Emotion"]), fontsize=9)

    ax.axhline(0)
    ax.axvline(0)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title("Emotion Wheel (Normalized Unit Circle)")
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved emotion wheel: {save_path}")


def vector_to_emotion(vec: np.ndarray, centers: dict) -> str:
    best_name, best_cos = None, -2.0
    v = np.array(vec, dtype=float)

    for name, center in centers.items():
        c = np.array(center, dtype=float)
        denom = (np.linalg.norm(v) * np.linalg.norm(c)) + 1e-9
        cos = float(np.dot(v, c) / denom)
        if cos > best_cos:
            best_cos = cos
            best_name = name
    return best_name


def angular_error(vec: np.ndarray, target: np.ndarray) -> float:
    v = np.array(vec, dtype=float)
    t = np.array(target, dtype=float)
    denom = (np.linalg.norm(v) * np.linalg.norm(t)) + 1e-9
    cos = np.dot(v, t) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def transition(state_vec: np.ndarray, clip_vec: np.ndarray, radius: float = 1.0) -> np.ndarray:
    next_vec = state_vec + clip_vec
    norm = np.linalg.norm(next_vec)
    if norm > radius:
        next_vec = next_vec * (radius / (norm + 1e-9))
    return next_vec


def _angle(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cos = np.dot(v1, v2) / denom
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


# ============================================================
# 2) LOAD DEAP .DAT SUBJECT FILES
# ============================================================

class DEAPDatLoader:
    def __init__(self, root_dir: str, n_subjects: int = 32):
        self.root_dir = root_dir
        self.n_subjects = n_subjects

    def _subject_path(self, subject_id: int) -> str:
        return os.path.join(self.root_dir, f"s{subject_id:02d}.dat")

    @staticmethod
    def _load_pickle(path: str):
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")

    @staticmethod
    def _extract_labels(obj):
        if isinstance(obj, dict):
            for key in ["labels", "label", "y"]:
                if key in obj:
                    return np.array(obj[key])
        raise ValueError("Could not find labels in .dat file")

    @staticmethod
    def labels_to_clip_vectors(labels: np.ndarray) -> np.ndarray:
        """
        Convert DEAP ratings to normalized emotion-space vectors.

        labels[:, 0] = Valence in [1..9]
        labels[:, 1] = Arousal in [1..9]
        """
        if labels.shape[1] < 2:
            raise ValueError("labels must have at least 2 columns: valence, arousal")

        va = labels[:, :2].astype(float)

        # Map [1..9] -> [-1..1]
        v_norm = 2 * (va[:, 0] - 1) / 8.0 - 1.0
        a_norm = 2 * (va[:, 1] - 1) / 8.0 - 1.0

        # Project to unit circle by global max radius
        r = np.sqrt(v_norm ** 2 + a_norm ** 2)
        max_r = float(np.max(r)) if len(r) else 1.0
        max_r = max(max_r, 1e-9)

        v_unit = v_norm / max_r
        a_unit = a_norm / max_r

        return np.column_stack([v_unit, a_unit]).astype(np.float32)

    def load_subject(self, subject_id: int) -> np.ndarray:
        path = self._subject_path(subject_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

        obj = self._load_pickle(path)
        labels = self._extract_labels(obj)
        clips = self.labels_to_clip_vectors(labels)
        return clips

    def load_all_subjects(self):
        subjects = []
        for sid in range(1, self.n_subjects + 1):
            try:
                clips = self.load_subject(sid)
                subjects.append(clips)
                print(f"Loaded subject {sid:02d}: {len(clips)} clips")
            except Exception as e:
                print(f"Skipping subject {sid:02d}: {e}")
        return subjects


# ============================================================
# 3) Q-LEARNING AGENT
# ============================================================

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.1, gamma: float = 0.6, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def choose_action(self, state_idx: int, training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(self.n_actions))
        return int(np.argmax(self.Q[state_idx]))

    def update(self, state_idx: int, action_idx: int, reward: float, next_state_idx: int, phi: float):
        td_target = reward + self.gamma * np.max(self.Q[next_state_idx]) + phi
        self.Q[state_idx, action_idx] += self.alpha * (
            td_target - self.Q[state_idx, action_idx]
        )

    def reset(self):
        self.Q[:] = 0.0


# ============================================================
# 4) REWARD / TRAINING / EVALUATION
# ============================================================

def reward(state_vec: np.ndarray, clip_vec: np.ndarray, target_vec: np.ndarray) -> float:
    """
    Same style as your RL code:
    reward = 1 if θ(target, clip) + θ(state, clip) <= π else 0
    """
    theta = abs(_angle(target_vec, clip_vec)) + abs(_angle(state_vec, clip_vec))
    return 1.0 if theta <= np.pi else 0.0


LAMBDA = -100.0

def reward_shaping(state_idx: int, next_state_idx: int) -> float:
    return LAMBDA if state_idx == next_state_idx else 0.0


def train_agent(agent: QLearningAgent,
                train_subjects_clips,
                emotion_centers: dict,
                target_emotion: str = "happy",
                n_episodes: int = 2000,
                steps_per_episode: int = 6):
    agent.reset()

    emotion_names = list(emotion_centers.keys())
    emotion_index = {name: i for i, name in enumerate(emotion_names)}
    target_vec = np.array(emotion_centers[target_emotion], dtype=float)

    non_target = [e for e in emotion_names if e != target_emotion]

    for _ in range(n_episodes):
        clips = train_subjects_clips[np.random.randint(len(train_subjects_clips))]
        n_clips = len(clips)

        start_emotion = non_target[np.random.randint(len(non_target))]
        sv = np.array(emotion_centers[start_emotion], dtype=float)
        si = emotion_index[start_emotion]

        for _ in range(steps_per_episode):
            ai = agent.choose_action(si, training=True) % n_clips
            clip_vec = clips[ai]

            nv = transition(sv, clip_vec, radius=1.0)
            next_emotion = vector_to_emotion(nv, emotion_centers)
            ni = emotion_index[next_emotion]

            r = reward(sv, clip_vec, target_vec)
            phi = reward_shaping(si, ni)

            agent.update(si, ai, r, ni, phi)
            sv, si = nv, ni


def evaluate_subject(agent: QLearningAgent,
                     test_clips: np.ndarray,
                     emotion_centers: dict,
                     target_emotion: str = "happy",
                     start_emotion: str = "mellow",
                     steps: int = 6):
    emotion_names = list(emotion_centers.keys())
    emotion_index = {name: i for i, name in enumerate(emotion_names)}
    target_vec = np.array(emotion_centers[target_emotion], dtype=float)

    if start_emotion not in emotion_index:
        start_emotion = emotion_names[0]

    sv = np.array(emotion_centers[start_emotion], dtype=float)
    si = emotion_index[start_emotion]

    errors = []
    emotions = [start_emotion]
    playlist = []

    n_clips = len(test_clips)

    for _ in range(steps):
        ai = agent.choose_action(si, training=False) % n_clips
        clip_vec = test_clips[ai]

        nv = transition(sv, clip_vec, radius=1.0)
        next_emotion = vector_to_emotion(nv, emotion_centers)
        ni = emotion_index[next_emotion]

        errors.append(angular_error(nv, target_vec))
        emotions.append(next_emotion)
        playlist.append(ai)

        sv, si = nv, ni

    return {
        "start": start_emotion,
        "emotions": emotions,
        "playlist": playlist,
        "errors": errors,
        "final_err": errors[-1] if errors else None,
        "converged": (emotions[-1] == target_emotion),
    }


def evaluate_best_start(agent: QLearningAgent,
                        test_clips: np.ndarray,
                        emotion_centers: dict,
                        target_emotion: str = "happy",
                        steps: int = 6):
    best = None
    best_err = float("inf")

    for start in emotion_centers.keys():
        if start == target_emotion:
            continue
        res = evaluate_subject(agent, test_clips, emotion_centers, target_emotion, start, steps)
        if res["final_err"] < best_err:
            best_err = res["final_err"]
            best = res
    return best


def leave_one_subject_out(subjects,
                          emotion_centers: dict,
                          target_emotion: str = "happy",
                          alpha: float = 0.1,
                          gamma: float = 0.6,
                          epsilon: float = 0.1,
                          n_episodes: int = 2000):
    results = []
    emotion_names = list(emotion_centers.keys())
    n_states = len(emotion_names)
    n_actions = max(len(s) for s in subjects)

    for test_idx in range(len(subjects)):
        train_subjects = [subjects[i] for i in range(len(subjects)) if i != test_idx]
        test_subject = subjects[test_idx]

        agent = QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon
        )

        train_agent(
            agent,
            train_subjects,
            emotion_centers=emotion_centers,
            target_emotion=target_emotion,
            n_episodes=n_episodes
        )

        res = evaluate_best_start(
            agent,
            test_subject,
            emotion_centers=emotion_centers,
            target_emotion=target_emotion,
            steps=6
        )
        res["subject"] = test_idx + 1
        results.append(res)

        print(
            f"Subject {test_idx+1:02d} | "
            f"final_err={res['final_err']:.2f}° | "
            f"converged={res['converged']} | "
            f"start={res['start']} | "
            f"path={' -> '.join(res['emotions'])}"
        )

    return results


# ============================================================
# 5) PLOTS
# ============================================================

def plot_angular_error(results, save_path="results/angular_error.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    all_err = np.array([r["errors"] for r in results], dtype=float)
    mean_e = all_err.mean(axis=0)
    se = all_err.std(axis=0) / np.sqrt(len(results) + 1e-9)
    iters = np.arange(1, len(mean_e) + 1)

    plt.figure(figsize=(7, 4))
    plt.errorbar(iters, mean_e, yerr=se, fmt="-o", capsize=4)
    plt.xlabel("Music Clip # (Iteration)")
    plt.ylabel("Angular Error (°)")
    plt.title("Mean Angular Error per Iteration")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved angular error plot: {save_path}")


def plot_emotion_trajectory(result, emotion_centers, target_emotion="happy", save_path=None):
    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")

    boundary = plt.Circle((0, 0), 1, color="gray", fill=False, linestyle="--", alpha=0.35)
    ax.add_patch(boundary)

    for name, (vx, vy) in emotion_centers.items():
        color = "gold" if name == target_emotion else "lightblue"
        ax.scatter(vx, vy, s=90, zorder=5, color=color, edgecolors="k", linewidths=0.6)
        ax.text(vx + 0.03, vy + 0.03, name, fontsize=8)

    emotions = result["emotions"]
    xs = [emotion_centers[e][0] for e in emotions]
    ys = [emotion_centers[e][1] for e in emotions]

    ax.plot(xs, ys, "-o", color="navy", lw=1.6, ms=6, zorder=6)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")

    status = "Converged" if result["converged"] else "Not Converged"
    ax.set_title(f"Subject {result['subject']} — {status}")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)

    plt.close()


def plot_all_subjects_trajectories(results, emotion_centers, target_emotion="happy",
                                   save_path="results/all_trajectories.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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

        boundary = plt.Circle((0, 0), 1, color="gray", fill=False, lw=0.6, ls="--", alpha=0.3)
        ax.add_patch(boundary)

        for name, (vx, vy) in emotion_centers.items():
            clr = "gold" if name == target_emotion else "lightblue"
            ax.scatter(vx, vy, s=25, color=clr, edgecolors="k", linewidths=0.3, zorder=3)

        emotions = res["emotions"]
        xs = [emotion_centers[e][0] for e in emotions]
        ys = [emotion_centers[e][1] for e in emotions]

        ax.plot(xs, ys, "-", color="navy", lw=1.0, alpha=0.85, zorder=4)
        ax.plot(xs[0], ys[0], "gs", ms=4, zorder=5)
        ax.plot(xs[-1], ys[-1], "r*", ms=7, zorder=5)

        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.axvline(0, color="gray", lw=0.4, ls="--")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
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

    fig.suptitle("Emotion Wheel Trajectories — All Subjects", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved all-trajectories plot: {save_path}")


# ============================================================
# 6) MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, default="data/Metacsv/emotion_summary.csv")
    parser.add_argument("--data_root", type=str, default="data/data_preprocessed_python")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--target", type=str, default="happy")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.6)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--n_subjects", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Emotion Wheel + RL Agent")
    print("=" * 70)

    # 1) Build wheel
    gew_df = load_emotion_summary(args.summary_csv)
    emotion_centers = build_emotion_centers(gew_df)

    print("\nEmotion centers:")
    for k, v in emotion_centers.items():
        print(f"  {k:12s} -> ({v[0]: .3f}, {v[1]: .3f})")

    plot_emotion_wheel(gew_df, save_path=os.path.join(args.output_dir, "emotion_wheel.png"))

    # 2) Load subjects
    loader = DEAPDatLoader(args.data_root, n_subjects=args.n_subjects)
    subjects = loader.load_all_subjects()

    if len(subjects) < 2:
        raise RuntimeError("Need at least 2 subject files to train/evaluate.")

    print(f"\nLoaded {len(subjects)} subjects successfully.")

    # 3) LOSO training + evaluation
    print("\nStarting LOSO training...")
    results = leave_one_subject_out(
        subjects,
        emotion_centers=emotion_centers,
        target_emotion=args.target,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        n_episodes=args.episodes
    )

    # 4) Save result summary
    summary_rows = []
    for r in results:
        summary_rows.append({
            "subject": r["subject"],
            "start": r["start"],
            "final_err": r["final_err"],
            "converged": r["converged"],
            "path": " -> ".join(r["emotions"]),
            "playlist": ",".join(map(str, r["playlist"])),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.output_dir, "rl_results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV: {summary_path}")

    # 5) Plots
    plot_angular_error(results, save_path=os.path.join(args.output_dir, "angular_error.png"))
    plot_all_subjects_trajectories(results, emotion_centers, target_emotion=args.target,
                                   save_path=os.path.join(args.output_dir, "all_trajectories.png"))

    converged = [r for r in results if r["converged"]]
    not_converged = [r for r in results if not r["converged"]]

    if converged:
        plot_emotion_trajectory(
            converged[0],
            emotion_centers,
            target_emotion=args.target,
            save_path=os.path.join(args.output_dir, f"trajectory_s{converged[0]['subject']}_converged.png")
        )

    if not_converged:
        plot_emotion_trajectory(
            not_converged[0],
            emotion_centers,
            target_emotion=args.target,
            save_path=os.path.join(args.output_dir, f"trajectory_s{not_converged[0]['subject']}_diverged.png")
        )

    print("\nDone.")
    print(f"Results folder: {args.output_dir}")


if __name__ == "__main__":
    main()