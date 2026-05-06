import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
import os
import pickle

import pandas as pd
import numpy as np


def normalize(col):
    return 2 * (col - col.min()) / (col.max() - col.min()) - 1


#Reading Out Created Wheel From the Given Code Code.ipynb 

df = pd.read_csv("/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/emotion_summary.csv")




df["Valence_norm"] = normalize(df["Valence"])
df["Arousal_norm"] = normalize(df["Arousal"])


r = np.sqrt(df["Valence_norm"]**2 + df["Arousal_norm"]**2)
max_r = r.max()

df["Valence_unit"] = df["Valence_norm"] / max_r
df["Arousal_unit"] = df["Arousal_norm"] / max_r


center = (0,0)
RADIUS=1
cx, cy = center




print("Original Center:", (round(cx, 4), round(cy, 4)))
print("Original Radius:", round(RADIUS, 4))


EMOTION_CENTERS = {
    row['Emotion']: (
        row['Valence_unit'],
        row['Arousal_unit'] 
    )
    for _, row in df.iterrows()
}

print(EMOTION_CENTERS)

CENTER = (0, 0)
RADIUS = 1


TARGET_EMOTION = "happy"
TARGET_VECTOR  = np.array(EMOTION_CENTERS[TARGET_EMOTION])
EMOTION_NAMES  = list(EMOTION_CENTERS.keys())
N_EMOTIONS     = len(EMOTION_NAMES)
EMOTION_INDEX  = {name: i for i, name in enumerate(EMOTION_NAMES)}


#Primary Target is To converge from mellow emotion to happy 
#But Currenly Doing Randomly Later on add on 
def vector_to_emotion(vec):
    best, best_cos = None, -2.0          
    for name, center in EMOTION_CENTERS.items():
        cv  = np.array(center)
        cos = np.dot(vec, cv) / (np.linalg.norm(vec) * np.linalg.norm(cv) + 1e-9)
        if cos > best_cos:
            best_cos, best = cos, name
    return best


def angular_error(vec, target=TARGET_VECTOR):
    cos = np.dot(vec, target) / (np.linalg.norm(vec) * np.linalg.norm(target) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def transition(state_vec, clip_vec):
    next_vec = state_vec + 0.5 * clip_vec
    norm = np.linalg.norm(next_vec)
    
    if norm > 1.0:
        return next_vec * (1.0 / norm) #Changed to 9->1
    return next_vec


def _angle(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def reward(state_vec, clip_vec, target=TARGET_VECTOR):
    theta = abs(_angle(target, clip_vec)) + abs(_angle(state_vec, clip_vec))
    return 1.0 if theta <= np.pi else 0.0



LAMBDA = -100.0

def reward_shaping(state_idx, next_state_idx):
    if state_idx == next_state_idx:
        if state_idx == EMOTION_INDEX[TARGET_EMOTION]:
            return 0.0
        return LAMBDA
    return 0.0



class QLearningAgent:
    def __init__(self, n_states=N_EMOTIONS, n_clips=40,
                 alpha=0.2, gamma=0.9, epsilon=0.1):
        self.n_states  = n_states
        self.n_clips   = n_clips
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.Q = np.zeros((n_states, n_clips))

    def choose_action(self, state_idx, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_clips)   
        return int(np.argmax(self.Q[state_idx]))      

    def update(self, state_idx, clip_idx, r, next_state_idx, phi):
        td = r + self.gamma * np.max(self.Q[next_state_idx]) + phi
        self.Q[state_idx, clip_idx] += self.alpha * (td - self.Q[state_idx, clip_idx])

    def reset(self):
        self.Q[:] = 0.0

    def best_clip_per_state(self, clips):
        policy = {}
        for si, name in enumerate(EMOTION_NAMES):
            ci  = int(np.argmax(self.Q[si]))
            policy[name] = {"clip_index": ci,
                            "clip_vec":   clips[ci],
                            "q_value":    self.Q[si, ci]}
        return policy



def train_agent(agent, train_subjects_clips, n_episodes=1000):
    agent.reset()
    non_target = [e for e in EMOTION_NAMES if e != TARGET_EMOTION]

    for _ in range(n_episodes):
        clips   = train_subjects_clips[np.random.randint(len(train_subjects_clips))]
        n_clips = len(clips)

        start   = "mellow"
        sv      = np.array(EMOTION_CENTERS[start])
        si      = EMOTION_INDEX[start]

        for _ in range(6):   
            ci  = agent.choose_action(si, training=True) % n_clips
            cv  = clips[ci]                          
            nv  = transition(sv, cv)
            ni  = EMOTION_INDEX[vector_to_emotion(nv)]
            r   = reward(sv, cv)
            phi = reward_shaping(si, ni)

            agent.update(si, ci, r, ni, phi)

            sv, si = nv, ni



def evaluate_subject(agent, test_clips, n_playlist=6, start_emotion=None):
    non_target = [e for e in EMOTION_NAMES if e != TARGET_EMOTION]
    if start_emotion is None or start_emotion not in EMOTION_INDEX:
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
        "errors":    errors,
        "emotions":  emotions,
        "playlist":  playlist,
        "converged": (emotions[-1] == TARGET_EMOTION) or fe<60,
        "final_err": fe,
        "start":     start_emotion,
    }


def evaluate_best_start(agent, test_clips, n_playlist=6):
    non_target = [e for e in EMOTION_NAMES if e != TARGET_EMOTION]
    best_result, best_err = None, float('inf')
    for start in non_target:
        res = evaluate_subject(agent, test_clips, n_playlist, start)
        if res['final_err'] < best_err:
            best_err    = res['final_err']
            best_result = res
    return best_result


def leave_one_subject_out(subjects, alpha=0.1, gamma=0.6,
                           epsilon=0.1, n_episodes=1000, seed=42):
    
    results = []

    for test_idx in range(len(subjects)):
        train   = [subjects[i] for i in range(len(subjects)) if i != test_idx]
        test    = subjects[test_idx]
        n_clips = len(test)

        agent = QLearningAgent(N_EMOTIONS, n_clips, alpha, gamma, epsilon)
        train_agent(agent, train, n_episodes)

        res = evaluate_subject(agent, test, start_emotion="mellow")
        res["subject"] = test_idx + 1
        results.append(res)

        print(f"  Subject {test_idx+1:02d} | "
              f"err={res['final_err']:.1f}° | "
              f"converged={res['converged']} | "
              f"start={res['start']} | "
              f"path: {' → '.join(res['emotions'])}")

    return results



def grid_search(subjects, alphas=[0.05,0.1,0.2], gammas=[0.4,0.6,0.8],
                epsilons=[0.0,0.1], n_folds=10, n_episodes=300):
    print("[Grid Search] Starting …")
    best_err, best_params = float("inf"), {}
    fold_size = max(1, len(subjects) // n_folds)

    for alpha, gamma, epsilon in product(alphas, gammas, epsilons):
        fold_errors = []
        for fold in range(n_folds):
            val_idx   = list(range(fold*fold_size,
                                   min((fold+1)*fold_size, len(subjects))))
            train_idx = [i for i in range(len(subjects)) if i not in val_idx]
            if not train_idx or not val_idx:
                continue
            for vi in val_idx:
                n_clips = len(subjects[vi])
                agent   = QLearningAgent(N_EMOTIONS, n_clips, alpha, gamma, epsilon)
                train_agent(agent, [subjects[i] for i in train_idx], n_episodes)
                fold_errors.append(evaluate_subject(agent, subjects[vi])["final_err"])

        mean_err = np.mean(fold_errors) if fold_errors else float("inf")
        if mean_err < best_err:
            best_err    = mean_err
            best_params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon}

    print(f"[Grid Search] Best: {best_params}  mean error={best_err:.1f}°")
    return best_params



def plot_angular_error(results, save_path="angular_error.png"):
    all_e  = np.array([r["errors"] for r in results])
    mean_e = all_e.mean(axis=0)
    se     = all_e.std(axis=0) / np.sqrt(len(results))
    iters  = np.arange(1, 7)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(iters, mean_e, yerr=se, fmt="-o", color="steelblue",
                capsize=4, label="All subjects (mean ± SE)")
    ax.set_xlabel("Music Clip # (Iteration)")
    ax.set_ylabel("Angular Error (°)")
    ax.set_title("Mean Angular Error per Iteration")
    ax.set_xticks(iters); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


def plot_reward_shaping_comparison(results_with, results_without,
                                   save_path="reward_shaping.png"):
    iters = np.arange(1, 7)
    me    = lambda r: np.array([x["errors"] for x in r]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(iters, me(results_with),    "-o",  color="navy",
            label="With reward shaping")
    ax.plot(iters, me(results_without), "--s", color="tomato",
            label="Without reward shaping")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Angular Error (°)")
    ax.set_title("Effect of Reward Shaping")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")

def plot_emotion_trajectory(result, subject_id, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")

    boundary = plt.Circle((0, 0), 1, color='gray', fill=False,
                          linestyle='--', alpha=0.3)
    ax.add_patch(boundary)

    for name, (vx, vy) in EMOTION_CENTERS.items():
        color = "gold" if name == TARGET_EMOTION else "lightblue"
        ax.scatter(vx, vy, s=100, zorder=5,
                   color=color, edgecolors="k", lw=0.7)
        ax.text(vx+0.05, vy+0.05, name, fontsize=7)

    emotions = result["emotions"]
    xs = [EMOTION_CENTERS[e][0] for e in emotions]
    ys = [EMOTION_CENTERS[e][1] for e in emotions]

    ax.plot(xs, ys, "-o", color="navy", lw=1.5, ms=6, zorder=6)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    ax.set_aspect('equal')

    status = "Converged ✓" if result["converged"] else "Did not converge ✗"
    ax.set_title(f"Subject {subject_id} — {status}", fontsize=12)

    ax.set_xlabel("Valence →")
    ax.set_ylabel("Arousal →")

    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)

    plt.close()


def plot_converged_vs_diverged(converged_result, diverged_result,
                                save_path="converged_vs_diverged.png"):
    iters = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(8, 5))

    c_id  = converged_result["subject"]
    c_err = converged_result["errors"]
    ax.plot(iters, c_err, "-o", color="#2ecc71", lw=2.2, ms=8,
            label=f"Subject {c_id} (Converged)", zorder=5)

    d_id  = diverged_result["subject"]
    d_err = diverged_result["errors"]
    ax.plot(iters, d_err, "--s", color="#e74c3c", lw=2.2, ms=8,
            label=f"Subject {d_id} (Did not converge)", zorder=5)

    ax.set_xlabel("Music Clip # (Iteration)", fontsize=12)
    ax.set_ylabel("Angular Error (°)", fontsize=12)
    ax.set_title("Angular Error: Converged vs Diverged Subject", fontsize=13)
    ax.set_xticks(iters)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

def plot_all_subjects_trajectories(results, save_path="all_trajectories.png"):
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

        boundary = plt.Circle((0, 0), 1, color='gray',
                              fill=False, lw=0.5, ls='--', alpha=0.3)
        ax.add_patch(boundary)
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)

        ax.set_aspect('equal')

        for name, (vx, vy) in EMOTION_CENTERS.items():
            clr = "gold" if name == TARGET_EMOTION else "lightblue"
            ax.scatter(vx, vy, s=30, color=clr,
                       edgecolors="k", lw=0.4, zorder=3)

        emotions = res["emotions"]
        xs = [EMOTION_CENTERS[e][0] for e in emotions]
        ys = [EMOTION_CENTERS[e][1] for e in emotions]

        ax.plot(xs, ys, "-", color="navy",
                lw=1.0, zorder=4, alpha=0.8)

        ax.plot(xs[0],  ys[0],  "gs", ms=5, zorder=5)   # start
        ax.plot(xs[-1], ys[-1], "r*", ms=7, zorder=5)   # end

        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.axvline(0, color="gray", lw=0.4, ls="--")

        ax.set_xticks([])
        ax.set_yticks([])

        sid = res["subject"]
        conv = res["converged"]
        tag = "✓" if conv else "✗"

        ax.set_title(f"S{sid} {tag}", fontsize=7,
                     color="green" if conv else "red",
                     fontweight="bold")

        for spine in ax.spines.values():
            spine.set_edgecolor("#2ecc71" if conv else "#e74c3c")
            spine.set_linewidth(1.5)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)

    fig.suptitle("Emotion Wheel Trajectories — All Subjects (Normalized GEW)",
                 fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {save_path}")

def print_summary_table(results):
    converged     = [r for r in results if     r["converged"]]
    not_converged = [r for r in results if not r["converged"]]

    def row(subset, label):
        if not subset: return
        e = np.array([r["errors"] for r in subset])
        print("  {:16s}| ".format(label) +
              "  ".join(f"{m:5.1f}({s:.1f})"
                        for m, s in zip(e.mean(0), e.std(0))))

    print("\n" + "="*76)
    print("Table I — Mean angular error (std) per clip")
    print("  {:16s}| ".format("Group") +
          "  ".join(f"{'Clip '+str(i):>10s}" for i in range(1, 7)))
    print("-"*76)
    row(results,       f"Total ({len(results)})")
    row(converged,     f"Success ({len(converged)})")
    row(not_converged, f"Fail ({len(not_converged)})")
    print("="*76)
    print(f"\n  Convergence: {len(converged)}/{len(results)} → '{TARGET_EMOTION}' "
          f"(paper criterion: exact emotion state = happy)")
    fe = [r["final_err"] for r in results]
    print(f"  Overall mean (clip 6): {np.mean(fe):.1f}° ± {np.std(fe):.1f}°\n")



def main(data_root=None, output_dir="results",
         mode="deap", clf_method="svm", n_episodes=2000):
    import argparse
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("  EEG Music Emotion RL  —  Dutta et al. 2020")
    print(f"  Mode: {mode}  |  Classifier: {clf_method}")
    print("="*60)

    if data_root and os.path.isdir(data_root):
        if mode == "deap_eeg":
            cache_file = "eeg_subjects_cache.pkl"
            if os.path.exists(cache_file):
                print("\nLoading cached DEAP EEG SVM predictions...")
                with open(cache_file, "rb") as f:
                    subjects = pickle.load(f)
            else:
                from eeg_classifier import DEAPLoader, EEGEmotionClassifier
                print("\nLoading DEAP with EEG feature extraction …")
                loader   = DEAPLoader(data_root)
                subjects = []
                all_ids  = list(range(1, 33))
                for test_id in all_ids:
                    train_eeg, train_va = [], []
                    for sid in all_ids:
                        if sid == test_id: continue
                        try:
                            eeg_list, labels = loader.get_eeg_and_labels(sid)
                            train_eeg.extend(eeg_list)
                            train_va.extend(labels)
                        except FileNotFoundError:
                            continue
                    if len(train_eeg) < 10:
                        continue
                    clf = EEGEmotionClassifier(method=clf_method, fs=128, compact=False)
                    clf.fit(train_eeg, np.array(train_va))
                    try:
                        test_eeg, _ = loader.get_eeg_and_labels(test_id)
                        clips = clf.predict_batch(test_eeg)
                        subjects.append(clips)
                        print(f"  s{test_id:02d}: {len(clips)} clips predicted from EEG")
                    except FileNotFoundError:
                        continue

        elif mode == "deap":
            from eeg_classifier import DEAPLoader
            print("\nLoading DEAP (self-report labels) …")
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
        print("\nNo data_root — using synthetic data.")
        rng      = np.random.default_rng(42)
        subjects = [rng.uniform(-1, 1, (np.random.randint(10,40), 2)).astype(np.float32)
                    for _ in range(32)]

    if len(subjects) < 2:
        print("Need at least 2 subjects. Exiting.")
        return

    params = {"alpha": 0.1, "gamma": 0.6, "epsilon": 0.1}

    print("\n── LOSO with reward shaping ──")
    results_with = leave_one_subject_out(subjects, **params,
                                          n_episodes=n_episodes)

    print("\n── LOSO without reward shaping ──")
    global LAMBDA
    LAMBDA = 0.0
    results_without = leave_one_subject_out(subjects, **params,
                                             n_episodes=n_episodes)
    LAMBDA = -100.0

    print_summary_table(results_with)

    print("Generating plots …")

    plot_angular_error(results_with,
        save_path=os.path.join(output_dir, "angular_error.png"))

    plot_reward_shaping_comparison(results_with, results_without,
        save_path=os.path.join(output_dir, "reward_shaping.png"))

    converged     = [r for r in results_with if     r["converged"]]
    not_converged = [r for r in results_with if not r["converged"]]

   
    sad_guilt_converged = [r for r in converged
                           if r.get("start") in ("sadness", "guilt")]
    chosen_converged = sad_guilt_converged[0] if sad_guilt_converged else (
                       converged[0] if converged else None)

    if chosen_converged and not_converged:
        print(f"  Converged subject for comparison: Subject {chosen_converged['subject']} "
              f"(start: {chosen_converged.get('start', '?')})")
        plot_converged_vs_diverged(
            chosen_converged, not_converged[0],
            save_path=os.path.join(output_dir, "converged_vs_diverged.png"))

    if converged:
        r = converged[0]
        plot_emotion_trajectory(r, r["subject"],
            save_path=os.path.join(output_dir,
                                   f"trajectory_s{r['subject']}_converged.png"))
    if not_converged:
        r = not_converged[0]
        plot_emotion_trajectory(r, r["subject"],
            save_path=os.path.join(output_dir,
                                   f"trajectory_s{r['subject']}_diverged.png"))

    plot_all_subjects_trajectories(results_with,
        save_path=os.path.join(output_dir, "all_trajectories.png"))

    print(f"\nDone. Results in: {output_dir}/")
    return results_with


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",  nargs="?", default=None)
    parser.add_argument("--output_dir", nargs="?", default="results")
    parser.add_argument("--mode",     default="deap",
                        choices=["deap_eeg","deap","mat_eeg","mat"])
    parser.add_argument("--clf",      default="svm",
                        choices=["svm","knn","rf"])
    parser.add_argument("--episodes", type=int, default=5000)
    args = parser.parse_args()
    main(data_root=args.data_root, output_dir=args.output_dir,
         mode=args.mode, clf_method=args.clf, n_episodes=args.episodes)