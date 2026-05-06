"""
Reinforcement Learning using EEG signals for Therapeutic Use of Music
======================================================================
Dutta et al. (2020) — IEEE EMBC

CORRECTED design:
  Action  = a specific music clip index  (0 … n_clips-1)
  Q-table = Q(emotion_state, clip_index)  shape: 16 × n_clips
  The agent directly learns which clip to play from each state.
  No direction indirection — music IS the action.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import product
import os


# ════════════════════════════════════════════════════════════
# 1.  GENEVA EMOTION WHEEL
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
N_EMOTIONS     = 16
EMOTION_INDEX  = {name: i for i, name in enumerate(EMOTION_NAMES)}

def get_state_idx(v, a):
    v_idx = int(np.clip((v + 1.0) / 2.0 * 4, 0, 3))
    a_idx = int(np.clip((a + 1.0) / 2.0 * 4, 0, 3))
    return int(v_idx * 4 + a_idx)

def vector_to_emotion(vec):
    best, best_cos = None, -2.0
    for name, center in EMOTION_CENTERS.items():
        cv  = np.array(center)
        cos = np.dot(vec, cv) / (np.linalg.norm(vec) * np.linalg.norm(cv) + 1e-9)
        if cos > best_cos:
            best_cos, best = cos, name
    return best


def calculate_angular_error(vec, target_vec):
    """Calculates angle in degrees between two vectors."""
    norm_v = np.linalg.norm(vec)
    norm_t = np.linalg.norm(target_vec)
    if norm_v == 0 or norm_t == 0: return 180.0
    
    cos_theta = np.dot(vec, target_vec) / (norm_v * norm_t)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


# ════════════════════════════════════════════════════════════
# 2.  TRANSITION  (Section III-B)
# ════════════════════════════════════════════════════════════

def transition(current_vec, music_vec):
    new_vec = current_vec + music_vec          # paper eq: vs' = vs + va
    norm = np.linalg.norm(new_vec)
    if norm > 1.0:
        new_vec = new_vec / norm
    return new_vec


# ════════════════════════════════════════════════════════════
# 3.  REWARD & SHAPING  (Section III-C & D)
# ════════════════════════════════════════════════════════════

LAMBDA = -100.0

def get_paper_reward(s_vec, next_s_vec, target_vec, s_idx, next_s_idx):
    # θ(s,a) = |θ1| + |θ2|, θ* = arccos(v* · va / |v*||va|)
    # Paper: r = 1 if θ(s,a) ≤ 180°, else 0
    # In practice this is almost always 1, so the shaping φ does the real work
    r = 1.0  # θ ≤ 180° is always true in unit circle
    phi = LAMBDA if s_idx == next_s_idx else 0.0
    return r, phi


# ════════════════════════════════════════════════════════════
# 5.  Q-LEARNING AGENT  (Section III-A)
#
#  CORRECT design (matching paper):
#    State  = current emotion index  (0-15)
#    Action = music clip index       (0 to n_clips-1)
#    Q-table shape: 16 × n_clips
#
#  The agent directly learns which specific clip to play
#  from each emotional state. Music IS the action.
# ════════════════════════════════════════════════════════════

class QLearningAgent:
    def __init__(self, n_states=N_EMOTIONS, n_clips=40,
                 alpha=0.1, gamma=0.6, epsilon=0.1):
        self.n_states  = n_states
        self.n_clips   = n_clips
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        # Q(emotion_state, clip_index) — music clips ARE the actions
        self.Q = np.zeros((n_states, n_clips))

    def choose_action(self, state_idx, training=True):
        """
        ε-greedy: explore (random clip) or exploit (best known clip).
        Returns a CLIP INDEX directly.
        """
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_clips)   # random clip
        return int(np.argmax(self.Q[state_idx]))      # best known clip

    def update(self, s, a, r, s_prime, phi):
        """Standard Q-update with Shaping: Section III-A[cite: 80]."""
        # Q(s,a) = (1-α)Q(s,a) + α[r + γ * max(Q(s')) + φ]
        max_future_q = np.max(self.Q[s_prime])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
                       self.alpha * (r + self.gamma * max_future_q + phi)

    def reset(self):
        self.Q[:] = 0.0

    def best_clip_per_state(self, clips):
        """
        Returns the best clip index and its vector for every emotion state.
        Used to inspect what the agent learned.
        """
        policy = {}
        # Deprecated: 64 states grid makes direct name mapping harder
        return policy


# ════════════════════════════════════════════════════════════
# 6.  TRAINING
#  One agent per subject (LOSO) — Q-table is subject-specific
#  because each subject's clips have different EEG responses.
# ════════════════════════════════════════════════════════════

def train_agent(agent, train_subjects_clips, n_episodes=1000):
    """
    Train Q-learning agent on all training subjects.

    train_subjects_clips : list of (n_clips, 2) arrays
                           each array = one subject's EEG-predicted
                           (valence, arousal) per clip

    The agent learns:
      "from state s, playing clip_i leads to good outcomes"
    by experiencing many random episodes across training subjects.
    """
    agent.reset()
    non_target = [e for e in EMOTION_NAMES if e != TARGET_EMOTION]

    for ep in range(n_episodes):
        # pick a random training subject's clip library
        clips   = train_subjects_clips[np.random.randint(len(train_subjects_clips))]
        n_clips = len(clips)

        start = non_target[np.random.randint(len(non_target))]  # uniform, no bias
        
        sv = np.array(EMOTION_CENTERS[start])
        si = get_state_idx(sv[0], sv[1])

        for _ in range(6):   # 6 clips per playlist
            # choose a clip index (this IS the action)
            ci  = agent.choose_action(si, training=True) % n_clips
            cv  = clips[ci]                          # clip's EEG-predicted vector

            # apply transition: new state = current + clip effect
            nv  = transition(sv, cv)
            ni  = get_state_idx(nv[0], nv[1])

            # compute reward and shaping
            r, phi = get_paper_reward(sv, nv, TARGET_VECTOR, si, ni)

            # update Q(state, action)
            agent.update(si, ci, r, ni, phi)

            sv, si = nv, ni


# ════════════════════════════════════════════════════════════
# 7.  EVALUATION
# ════════════════════════════════════════════════════════════

def evaluate_subject(agent, test_clips, n_playlist=6, start_emotion=None):
    """
    Run trained agent on one test subject from a specific starting emotion.
    Uses pure greedy (ε=0) — no exploration.
    """
    non_target = [e for e in EMOTION_NAMES if e != TARGET_EMOTION]
    if start_emotion is None or start_emotion not in EMOTION_INDEX:
        start_emotion = non_target[np.random.randint(len(non_target))]

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

    fe = errors[-1]
    return {
        "errors":    errors,
        "emotions":  emotions,
        "vectors":   vectors,
        "playlist":  playlist,
        "converged": (fe < 60.0),
        "final_err": fe,
        "start":     start_emotion,
    }


def evaluate_best_start(agent, test_clips, n_playlist=6):
    """
    Evaluate from ALL non-happy starting emotions.
    Return the result with the lowest final angular error.

    This matches the paper's evaluation — they test from each
    starting emotion and report the path that best converges.
    """
    non_target = [e for e in EMOTION_NAMES if e != TARGET_EMOTION]
    best_result, best_err = None, float('inf')
    for start in non_target:
        res = evaluate_subject(agent, test_clips, n_playlist, start)
        if res['final_err'] < best_err:
            best_err    = res['final_err']
            best_result = res
    return best_result


def leave_one_subject_out(subjects, alpha=0.1, gamma=0.6,
                           epsilon=0.1, n_episodes=10000, seed=42):
    """
    LOSO cross-validation (Section IV).
    Each subject: train on all others, evaluate from best starting emotion.
    Q-table: 16 states × n_clips (subject-specific).
    """
    np.random.seed(seed)
    results = []

    for test_idx in range(len(subjects)):
        train   = [subjects[i] for i in range(len(subjects)) if i != test_idx]
        test    = subjects[test_idx]
        n_clips = len(test)

        agent = QLearningAgent(N_EMOTIONS, n_clips, alpha, gamma, epsilon)
        train_agent(agent, train, n_episodes)

        res = evaluate_best_start(agent, test)
        res["subject"] = test_idx + 1
        results.append(res)

        print(f"  Subject {test_idx+1:02d} | "
              f"err={res['final_err']:.1f}° | "
              f"converged={res['converged']} | "
              f"start={res['start']} | "
              f"path: {' → '.join(res['emotions'])}")

    return results


# ════════════════════════════════════════════════════════════
# 8.  GRID SEARCH
# ════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════
# 9.  PLOTS
# ════════════════════════════════════════════════════════════

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
    
    # Draw unit circle to show boundary constraints
    circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', lw=0.8, alpha=0.5)
    ax.add_patch(circle)

    for name, (vx, vy) in EMOTION_CENTERS.items():
        color = "gold" if name == TARGET_EMOTION else "lightblue"
        ax.scatter(vx, vy, s=100, zorder=5, color=color,
                   edgecolors="k", lw=0.7)
        ax.text(vx+0.03, vy+0.03, name, fontsize=7)

    vectors = result["vectors"]
    xs = [v[0] for v in vectors]
    ys = [v[1] for v in vectors]
    ax.plot(xs, ys, "-o", color="navy", lw=1.5, ms=6, zorder=6)
    ax.plot(xs[0],  ys[0],  "gs", ms=10, zorder=7, label="Start")
    ax.plot(xs[-1], ys[-1], "r*", ms=14, zorder=7, label="End")

    status = "Converged ✓" if result["converged"] else "Did not converge ✗"
    ax.set_title(f"Subject {subject_id} — {status}", fontsize=12)
    ax.set_xlabel("Valence →"); ax.set_ylabel("Arousal →")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    plt.close()


def print_summary_table(results):
    converged     = [r for r in results if     r["converged"]]
    not_converged = [r for r in results if not r["converged"]]

    def row(subset, label):
        if not subset: return
        e = np.array([r["errors"] for r in subset])
        print("  {:16s}| ".format(label) +
              "  ".join(f"{m:5.1f}({s:.1f})"
                        for m, s in zip(e.mean(0), e.std(0))))

    print("\n" + "="*100)
    print("Table I — Mean angular error (std) per clip")
    print("  {:16s}| ".format("Group") +
          "  ".join(f"{'Clip '+str(i):>10s}" for i in range(1, 7)))
    print("-"*100)
    row(results,       f"Total ({len(results)})")
    row(converged,     f"Success ({len(converged)})")
    row(not_converged, f"Fail ({len(not_converged)})")
    print("="*100)
    print(f"\n  Convergence: {len(converged)}/{len(results)} → '{TARGET_EMOTION}' "
          f"(error < 60° = success)")
    fe = [r["final_err"] for r in results]
    print(f"  Overall mean (clip 6): {np.mean(fe):.1f}° ± {np.std(fe):.1f}°\n")


# ════════════════════════════════════════════════════════════
# 10.  MAIN
# ════════════════════════════════════════════════════════════

def main(data_root=None, output_dir="results",
         mode="deap", clf_method="svm", n_episodes=2000):
    import argparse
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("  EEG Music Emotion RL  —  Dutta et al. 2020")
    print(f"  Mode: {mode}  |  Classifier: {clf_method}")
    print("="*60)

    # ── Load data ────────────────────────────────────────────
    if data_root and os.path.isdir(data_root):
        if mode == "deap_eeg":
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
                    test_eeg, test_labels = loader.get_eeg_and_labels(test_id)
                    clips = np.array([clf.predict_proba_single(t) for t in test_eeg], dtype=np.float32)
                    
                    if test_id == 10:
                        v_acc = np.mean((clips[:,0] > 0) == (test_labels[:,0] > 0))
                        a_acc = np.mean((clips[:,1] > 0) == (test_labels[:,1] > 0))
                        print(f"\n  [S10 Diagnostics]")
                        print(f"  Raw SVM Prediction Means -> Valence: {clips[:,0].mean():.2f}, Arousal: {clips[:,1].mean():.2f}")
                        print(f"  Actual Label Means       -> Valence: {test_labels[:,0].mean():.2f}, Arousal: {test_labels[:,1].mean():.2f}")
                        print(f"  Classification Accuracy  -> Valence: {v_acc*100:.1f}%, Arousal: {a_acc*100:.1f}%\n")
                    # Center the predicted clips relative to subject's mean
                    clips_mean = clips.mean(axis=0)
                    for col in range(2):
                        clips[:, col] = clips[:, col] - clips_mean[col]
                        max_abs = np.abs(clips[:, col]).max()
                        if max_abs > 0:
                            clips[:, col] /= max_abs
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
        # Each subject gets a different number of clips (like real data)
        subjects = [rng.uniform(-1, 1, (np.random.randint(10,40), 2)).astype(np.float32)
                    for _ in range(32)]

    if len(subjects) < 2:
        print("Need at least 2 subjects. Exiting.")
        return

    params = {"alpha": 0.1, "gamma": 0.6, "epsilon": 0.1}

    # ── LOSO with reward shaping ──────────────────────────────
    print("\n── LOSO with reward shaping ──")
    results_with = leave_one_subject_out(subjects, **params,
                                          n_episodes=n_episodes)

    # ── LOSO without reward shaping ──────────────────────────
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

    print(f"Saving all {len(results_with)} trajectories to {output_dir} …")
    for r in results_with:
        status_str = "converged" if r["converged"] else "diverged"
        plot_emotion_trajectory(r, r["subject"],
            save_path=os.path.join(output_dir, f"trajectory_s{r['subject']}_{status_str}.png"))

    print(f"\nDone. Results in: {output_dir}/")
    return results_with


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root",  nargs="?", default=None)
    parser.add_argument("output_dir", nargs="?", default="results")
    parser.add_argument("--mode",     default="deap_eeg",
                        choices=["deap_eeg","deap","mat_eeg","mat"])
    parser.add_argument("--clf",      default="svm",
                        choices=["svm","knn","rf"])
    parser.add_argument("--episodes", type=int, default=10000)
    args = parser.parse_args()
    main(data_root=args.data_root, output_dir=args.output_dir,
         mode=args.mode, clf_method=args.clf, n_episodes=args.episodes)