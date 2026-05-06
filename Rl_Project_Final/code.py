import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import product
from copy import deepcopy

DAT_FOLDER   = "/home/sahil/Desktop/Rl_Project_Final/data/data_preprocessed_python"
META_CSV     = "/home/sahil/Desktop/Rl_Project_Final/data/Metacsv/emotion_summary.csv"
OUTPUT_DIR   = "/home/sahil/Desktop/Rl_Project_Final/results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_CLIPS      = 40         
N_ITERATIONS = 6          
EPISODES     = 500        
ANG_THRESHOLD = 30        
TARGET_EMOTION = "happy"
START_EMOTION  = "mellow"

LAMBDA_SHAPE = -100        

def load_emotion_wheel(csv_path):
    df = pd.read_csv(csv_path)

    def normalize(col):
        return 2 * (col - col.min()) / (col.max() - col.min()) - 1

    df["V_norm"] = normalize(df["Valence"])
    df["A_norm"] = normalize(df["Arousal"])

    r      = np.sqrt(df["V_norm"]**2 + df["A_norm"]**2)
    max_r  = r.max()
    df["V_unit"] = df["V_norm"] / max_r
    df["A_unit"] = df["A_norm"] / max_r

    emotion_to_vec = {
        row["Emotion"]: np.array([row["V_unit"], row["A_unit"]])
        for _, row in df.iterrows()
    }
    emotions = list(emotion_to_vec.keys())
    return emotion_to_vec, emotions, df


# ═════════════════════════════════════════════
# 2.  LOAD DEAP .dat FILES
# ═════════════════════════════════════════════

def load_subject(dat_folder, subject_id):
    """
    Returns dict with keys 'labels' (40×4) containing
    [valence, arousal, dominance, liking] per clip.
    valence/arousal are in [1,9].
    """
    fname = os.path.join(dat_folder, f"s{subject_id:02d}.dat")
    with open(fname, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data   # keys: 'data', 'labels'


def get_subject_clip_vectors(subject_data, n_clips=40):
    """
    Convert each clip's (valence, arousal) from [1,9] to [-1,1].
    Returns array of shape (n_clips, 2).
    """
    labels = subject_data["labels"][:n_clips]   # (40, 4)
    valence = labels[:, 0]
    arousal = labels[:, 1]
    # normalise [1,9] → [-1,1]
    v_norm = 2 * (valence - 1) / 8 - 1
    a_norm = 2 * (arousal - 1) / 8 - 1
    return np.stack([v_norm, a_norm], axis=1)   # (40, 2)


# ═════════════════════════════════════════════
# 3.  ANGULAR HELPERS
# ═════════════════════════════════════════════

def angle_between(v1, v2):
    """Angle in degrees between two 2D vectors."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def angular_error(current_vec, target_vec):
    """θ(s,a) = |θ1| + |θ2| as per paper (simplified to angle between vectors)."""
    return angle_between(current_vec, target_vec)


# ═════════════════════════════════════════════
# 4.  TRANSITION FUNCTION  τ(s, a)
# ═════════════════════════════════════════════

def transition(state_vec, action_vec):
    """
    τ(s,a) = v_s + v_a   (paper eq.)
    Clip to unit circle.
    """
    new_vec = state_vec + action_vec
    norm    = np.linalg.norm(new_vec)
    if norm > 1.0:
        new_vec = new_vec / norm
    return new_vec


# ═════════════════════════════════════════════
# 5.  REWARD FUNCTIONS
# ═════════════════════════════════════════════

def reward(state_vec, action_vec, target_vec):
    """Basic reward: +1 if action moves closer to target, else -1."""
    new_vec = transition(state_vec, action_vec)
    err_before = angular_error(state_vec, target_vec)
    err_after  = angular_error(new_vec,   target_vec)
    return 1 if err_after < err_before else -1


def phi(state_idx, next_idx, lam=LAMBDA_SHAPE):
    """Reward shaping: penalise staying in same emotional state."""
    if state_idx == next_idx:
        return lam
    return 0.0


# ═════════════════════════════════════════════
# 6.  Q-LEARNING AGENT
# ═════════════════════════════════════════════

class QAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.6, epsilon=0.1):
        self.n_states  = n_states
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.Q         = np.zeros((n_states, n_actions))

    def select_action(self, state_idx):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state_idx]))

    def update(self, s, a, r_val, s_next, phi_val):
        best_next = np.max(self.Q[s_next])
        td_target = r_val + self.gamma * best_next + phi_val
        self.Q[s, a] += self.alpha * (td_target - self.Q[s, a])


# ═════════════════════════════════════════════
# 7.  CLOSEST STATE INDEX
# ═════════════════════════════════════════════

def vec_to_state_idx(vec, emotion_vecs):
    """Map a 2D vector to the nearest emotion state index."""
    dists = [np.linalg.norm(vec - ev) for ev in emotion_vecs]
    return int(np.argmin(dists))


# ═════════════════════════════════════════════
# 8.  TRAIN ONE SUBJECT
# ═════════════════════════════════════════════

def train_subject(clip_vecs, emotion_vecs, target_vec, start_vec,
                  alpha, gamma, epsilon, episodes=EPISODES,
                  n_iter=N_ITERATIONS):
    """
    clip_vecs   : (40, 2) array OR list of (40, 2) arrays.
    emotion_vecs: list of 2D arrays (one per emotion state)
    Returns Q-table and per-episode angular errors (list of lists).
    """
    is_multi = isinstance(clip_vecs, list)
    n_states  = len(emotion_vecs)
    n_actions = len(clip_vecs[0]) if is_multi else len(clip_vecs)
    agent     = QAgent(n_states, n_actions, alpha, gamma, epsilon)

    ep_errors = []   # (episodes, n_iter)

    for ep in range(episodes):
        state_vec = start_vec.copy()
        state_idx = vec_to_state_idx(state_vec, emotion_vecs)
        errors    = []
        
        if is_multi:
            curr_clip_vecs = clip_vecs[np.random.randint(len(clip_vecs))]
        else:
            curr_clip_vecs = clip_vecs

        for _ in range(n_iter):
            action  = agent.select_action(state_idx)
            act_vec = curr_clip_vecs[action]

            next_vec  = transition(state_vec, act_vec)
            next_idx  = vec_to_state_idx(next_vec, emotion_vecs)

            r_val  = reward(state_vec, act_vec, target_vec)
            phi_v  = phi(state_idx, next_idx)

            agent.update(state_idx, action, r_val, next_idx, phi_v)

            err = angular_error(next_vec, target_vec)
            errors.append(err)

            state_vec = next_vec
            state_idx = next_idx

        ep_errors.append(errors)

    return agent.Q, ep_errors


# ═════════════════════════════════════════════
# 9.  EVALUATE ONE SUBJECT (greedy, no exploration)
# ═════════════════════════════════════════════

def evaluate_subject(Q, clip_vecs, emotion_vecs, target_vec, start_vec,
                     n_iter=N_ITERATIONS):
    """
    Run greedy policy for n_iter steps.
    Returns trajectory of 2D vectors and angular errors.
    """
    state_vec  = start_vec.copy()
    trajectory = [state_vec.copy()]
    errors     = []

    for _ in range(n_iter):
        state_idx = vec_to_state_idx(state_vec, emotion_vecs)
        action    = int(np.argmax(Q[state_idx]))
        act_vec   = clip_vecs[action]
        next_vec  = transition(state_vec, act_vec)
        err       = angular_error(next_vec, target_vec)
        errors.append(err)
        trajectory.append(next_vec.copy())
        state_vec = next_vec

    return trajectory, errors


# ═════════════════════════════════════════════
# 10. GRID SEARCH  (Leave-One-Subject-Out CV)
# ═════════════════════════════════════════════

def grid_search(all_clip_vecs, emotion_vecs, target_vec, start_vec,
                param_grid, episodes=200, n_iter=N_ITERATIONS,
                threshold=ANG_THRESHOLD):
    """
    param_grid: dict with lists for 'alpha', 'gamma', 'epsilon'
    Returns best params and results table.
    """
    combos = list(product(
        param_grid["alpha"],
        param_grid["gamma"],
        param_grid["epsilon"]
    ))

    best_score  = -1
    best_params = None
    results     = []

    n_subjects = len(all_clip_vecs)
    print(f"\n🔍 Grid search: {len(combos)} combos × {n_subjects} subjects …\n")

    for alpha, gamma, epsilon in combos:
        converged = 0
        final_errors = []

        for subj_idx, clip_vecs in enumerate(all_clip_vecs):
            # Train on ALL other subjects (LOSO)
            train_clips = [c for i, c in enumerate(all_clip_vecs) if i != subj_idx]

            Q, _ = train_subject(
                train_clips, emotion_vecs, target_vec, start_vec,
                alpha, gamma, epsilon, episodes, n_iter
            )

            # Evaluate on held-out subject
            _, errs = evaluate_subject(Q, clip_vecs, emotion_vecs, target_vec, start_vec, n_iter)
            final_err = errs[-1] if errs else 180.0
            final_errors.append(final_err)
            if final_err <= threshold:
                converged += 1

        mean_err = np.mean(final_errors)
        results.append({
            "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
            "converged": converged, "mean_final_err": mean_err
        })
        print(f"  α={alpha}  γ={gamma}  ε={epsilon}  → converged={converged}/{n_subjects}  mean_err={mean_err:.1f}°")

        if converged > best_score or (converged == best_score and mean_err < (best_params[3] if best_params else 999)):
            best_score  = converged
            best_params = (alpha, gamma, epsilon, mean_err)

    print(f"\n✅ Best params: α={best_params[0]}  γ={best_params[1]}  ε={best_params[2]}  "
          f"→ {best_score}/{n_subjects} converged,  mean_err={best_params[3]:.1f}°")

    return best_params[:3], pd.DataFrame(results)


# ═════════════════════════════════════════════
# 11. RUN EXPERIMENT (one start condition)
# ═════════════════════════════════════════════

def run_experiment(label, start_vec, all_clip_vecs, emotion_vecs,
                   target_vec, best_alpha, best_gamma, best_epsilon,
                   episodes=EPISODES, n_iter=N_ITERATIONS,
                   threshold=ANG_THRESHOLD, out_dir=OUTPUT_DIR):
    """
    Train each subject with best params and collect results.
    """
    exp_dir = os.path.join(out_dir, label)
    os.makedirs(exp_dir, exist_ok=True)

    n_subjects    = len(all_clip_vecs)
    all_errors    = []       # (subjects, n_iter)
    all_trajs     = []
    converged_ids = []
    diverged_ids  = []

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {label}")
    print(f"{'='*60}")

    for sid, clip_vecs in enumerate(all_clip_vecs):
        subj_num = sid + 1
        Q, ep_errs = train_subject(
            clip_vecs, emotion_vecs, target_vec, start_vec,
            best_alpha, best_gamma, best_epsilon, episodes, n_iter
        )
        traj, eval_errs = evaluate_subject(
            Q, clip_vecs, emotion_vecs, target_vec, start_vec, n_iter
        )
        all_errors.append(eval_errs)
        all_trajs.append(traj)

        final_err = eval_errs[-1]
        status    = "✅" if final_err <= threshold else "❌"
        if final_err <= threshold:
            converged_ids.append(subj_num)
        else:
            diverged_ids.append(subj_num)

        print(f"  Subject {subj_num:2d}: final_err={final_err:.1f}°  {status}")

    all_errors_arr = np.array(all_errors)   # (subjects, n_iter)

    # ── Save trajectory data ──
    np.save(os.path.join(exp_dir, "all_errors.npy"), all_errors_arr)

    # ── Plot 1: Angular Error Bar Plot (like paper Fig. 3) ──
    mean_err = all_errors_arr.mean(axis=0)
    std_err  = all_errors_arr.std(axis=0) / np.sqrt(n_subjects)
    iters    = np.arange(1, n_iter + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, mean_err, "b-o", label="Mean Angular Error")
    ax.fill_between(iters, mean_err - std_err, mean_err + std_err, alpha=0.3)
    ax.set_xlabel("# of Iterations (Music Clips)")
    ax.set_ylabel("Angular Error (degrees)")
    ax.set_title(f"[{label}] Mean Angular Error & Standard Error")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "angular_error_bar.png"), dpi=150)
    plt.close()

    # ── Plot 2: Convergent vs Divergent subjects (like paper Fig. 4) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_c = cm.Greens(np.linspace(0.4, 0.9, len(converged_ids)))
    colors_d = cm.Reds(np.linspace(0.4, 0.9, len(diverged_ids)))

    for i, sid in enumerate(converged_ids):
        ax.plot(iters, all_errors_arr[sid - 1], color=colors_c[i],
                alpha=0.6, label=f"S{sid} (conv)" if i < 3 else "")
    for i, sid in enumerate(diverged_ids):
        ax.plot(iters, all_errors_arr[sid - 1], "--", color=colors_d[i],
                alpha=0.6, label=f"S{sid} (div)" if i < 3 else "")

    ax.axhline(threshold, color="k", linestyle=":", label=f"Threshold {threshold}°")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Angular Error (°)")
    ax.set_title(f"[{label}] Per-Subject Angular Error Trajectories")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, "subject_trajectories.png"), dpi=150)
    plt.close()

    # ── Plot 3: Emotion wheel trajectories (2 examples) ──
    example_ids = []
    if converged_ids:
        example_ids.append(converged_ids[0])
    if diverged_ids:
        example_ids.append(diverged_ids[0])

    if example_ids:
        fig, axes = plt.subplots(1, len(example_ids), figsize=(6 * len(example_ids), 6))
        if len(example_ids) == 1:
            axes = [axes]

        for ax, sid in zip(axes, example_ids):
            traj = all_trajs[sid - 1]
            circle = plt.Circle((0, 0), 1, fill=False, color="gray")
            ax.add_patch(circle)
            xs = [v[0] for v in traj]
            ys = [v[1] for v in traj]
            ax.plot(xs, ys, "b-o", markersize=5)
            ax.plot(xs[0], ys[0], "gs", markersize=10, label="Start")
            ax.plot(target_vec[0], target_vec[1], "r*", markersize=14, label="Target (happy)")
            ax.set_xlim(-1.15, 1.15)
            ax.set_ylim(-1.15, 1.15)
            ax.set_xlabel("Valence")
            ax.set_ylabel("Arousal")
            conv = "Converged" if sid in converged_ids else "Did NOT converge"
            ax.set_title(f"Subject {sid} – {conv}")
            ax.legend(fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_aspect("equal")
            ax.grid(True)

        plt.suptitle(f"[{label}] Emotion Wheel Trajectories", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, "wheel_trajectories.png"), dpi=150)
        plt.close()

    # ── Print summary ──
    print(f"\n  📊 Summary ({label}):")
    print(f"     Converged : {len(converged_ids)}/{n_subjects}  →  {converged_ids}")
    print(f"     Diverged  : {len(diverged_ids)}/{n_subjects}  →  {diverged_ids}")
    print(f"     Mean error (iter 6): {mean_err[-1]:.1f}° ± {std_err[-1]:.1f}°")

    return {
        "label": label,
        "all_errors": all_errors_arr,
        "converged": converged_ids,
        "diverged": diverged_ids,
        "mean_err": mean_err,
        "std_err": std_err
    }


# ═════════════════════════════════════════════
# 12. COMPARISON PLOT
# ═════════════════════════════════════════════

def plot_comparison(res_mellow, res_random, out_dir=OUTPUT_DIR):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iters = np.arange(1, len(res_mellow["mean_err"]) + 1)

    for ax, res in zip(axes, [res_mellow, res_random]):
        ax.plot(iters, res["mean_err"], "b-o")
        ax.fill_between(iters,
                        res["mean_err"] - res["std_err"],
                        res["mean_err"] + res["std_err"], alpha=0.3)
        n_conv = len(res["converged"])
        n_tot  = n_conv + len(res["diverged"])
        ax.set_title(f"{res['label']}\n({n_conv}/{n_tot} converged)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Angular Error (°)")
        ax.grid(True)
        ax.set_ylim(0, 180)

    plt.suptitle("Mellow Start vs Random Start – Angular Error Comparison", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_mellow_vs_random.png"), dpi=150)
    plt.close()
    print(f"\n💾 Comparison plot saved.")


# ═════════════════════════════════════════════
# 13. TABLE I  (like paper)
# ═════════════════════════════════════════════

def print_table(res, threshold=ANG_THRESHOLD):
    errors = res["all_errors"]           # (n_subjects, n_iter)
    n_iter = errors.shape[1]

    conv_mask = np.zeros(errors.shape[0], dtype=bool)
    for sid in res["converged"]:
        conv_mask[sid - 1] = True

    rows = {}
    for group, mask in [("T (All)", np.ones(len(conv_mask), dtype=bool)),
                        ("S (Success)", conv_mask),
                        ("F (Fail)", ~conv_mask)]:
        subset = errors[mask]
        if len(subset) == 0:
            continue
        means = subset.mean(axis=0)
        stds  = subset.std(axis=0)
        row   = "  ".join([f"{m:.1f}({s:.1f})" for m, s in zip(means, stds)])
        rows[group] = row

    print(f"\n📋 Table – [{res['label']}]  Mean angular error (std)")
    header = "  ".join([f"Clip {i+1}" for i in range(n_iter)])
    print(f"{'Group':<14}  {header}")
    for g, r in rows.items():
        print(f"{g:<14}  {r}")


# ═════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════

def main():
    # ── Load emotion wheel ──
    print("📊 Loading emotion wheel …")
    emotion_to_vec, emotions, wheel_df = load_emotion_wheel(META_CSV)
    emotion_vecs = [emotion_to_vec[e] for e in emotions]
    target_vec   = emotion_to_vec[TARGET_EMOTION]
    start_mellow = emotion_to_vec[START_EMOTION]

    print(f"   Emotions : {emotions}")
    print(f"   Target   : {TARGET_EMOTION}  →  {target_vec}")
    print(f"   Mellow   : {start_mellow}")

    # ── Load DEAP subjects ──
    print("\n📂 Loading DEAP .dat files …")
    all_clip_vecs = []
    loaded = 0
    for sid in range(1, 33):
        try:
            subj_data = load_subject(DAT_FOLDER, sid)
            clip_vecs = get_subject_clip_vectors(subj_data, N_CLIPS)
            all_clip_vecs.append(clip_vecs)
            loaded += 1
        except FileNotFoundError:
            print(f"   ⚠️  s{sid:02d}.dat not found – skipping")
        except Exception as e:
            print(f"   ⚠️  s{sid:02d}.dat error: {e} – skipping")

    print(f"   Loaded {loaded} subjects")
    if loaded == 0:
        print("❌ No subjects loaded. Check DAT_FOLDER path.")
        return

    # ── Grid Search ──
    param_grid = {
        "alpha"  : [0.1, 0.3, 0.5],
        "gamma"  : [0.4, 0.6, 0.8],
        "epsilon": [0.0, 0.1, 0.2],
    }

    print("\n🔍 Running grid search CV (mellow start) …")
    best_params, gs_results = grid_search(
        all_clip_vecs, emotion_vecs, target_vec, start_mellow,
        param_grid, episodes=1000, n_iter=N_ITERATIONS, threshold=ANG_THRESHOLD
    )
    best_alpha, best_gamma, best_epsilon = best_params

    # Save grid search table
    gs_results.to_csv(os.path.join(OUTPUT_DIR, "grid_search_results.csv"), index=False)
    print(f"💾 Grid search results saved.")

    # ── Experiment 1: Start from MELLOW ──
    res_mellow = run_experiment(
        label       = "Mellow_Start",
        start_vec   = start_mellow,
        all_clip_vecs = all_clip_vecs,
        emotion_vecs  = emotion_vecs,
        target_vec    = target_vec,
        best_alpha    = best_alpha,
        best_gamma    = best_gamma,
        best_epsilon  = best_epsilon,
        episodes      = EPISODES,
        n_iter        = N_ITERATIONS,
        threshold     = ANG_THRESHOLD,
    )
    print_table(res_mellow)

    # ── Experiment 2: Start from RANDOM ──
    # Random start = uniform random vector inside unit circle per subject
    np.random.seed(42)
    random_starts = []
    for _ in range(len(all_clip_vecs)):
        angle = np.random.uniform(0, 2 * np.pi)
        r     = np.random.uniform(0.3, 1.0)
        random_starts.append(np.array([r * np.cos(angle), r * np.sin(angle)]))

    # For each subject use its own random start → we run individually
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: Random_Start")
    print(f"{'='*60}")

    exp_dir_r = os.path.join(OUTPUT_DIR, "Random_Start")
    os.makedirs(exp_dir_r, exist_ok=True)

    all_errors_r   = []
    all_trajs_r    = []
    converged_r    = []
    diverged_r     = []

    for sid, (clip_vecs, rstart) in enumerate(zip(all_clip_vecs, random_starts)):
        subj_num = sid + 1
        Q, _ = train_subject(
            clip_vecs, emotion_vecs, target_vec, rstart,
            best_alpha, best_gamma, best_epsilon, EPISODES, N_ITERATIONS
        )
        traj, eval_errs = evaluate_subject(Q, clip_vecs, emotion_vecs, target_vec, rstart, N_ITERATIONS)
        all_errors_r.append(eval_errs)
        all_trajs_r.append(traj)
        final_err = eval_errs[-1]
        status = "✅" if final_err <= ANG_THRESHOLD else "❌"
        if final_err <= ANG_THRESHOLD:
            converged_r.append(subj_num)
        else:
            diverged_r.append(subj_num)
        print(f"  Subject {subj_num:2d}: final_err={final_err:.1f}°  {status}")

    all_errors_r_arr = np.array(all_errors_r)
    np.save(os.path.join(exp_dir_r, "all_errors.npy"), all_errors_r_arr)

    iters = np.arange(1, N_ITERATIONS + 1)
    mean_r = all_errors_r_arr.mean(axis=0)
    std_r  = all_errors_r_arr.std(axis=0) / np.sqrt(len(all_clip_vecs))

    # Bar plot random
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(iters, mean_r, "r-o", label="Mean Angular Error")
    ax.fill_between(iters, mean_r - std_r, mean_r + std_r, alpha=0.3, color="red")
    ax.set_xlabel("# of Iterations")
    ax.set_ylabel("Angular Error (°)")
    ax.set_title("Random Start – Mean Angular Error & Std Error")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir_r, "angular_error_bar.png"), dpi=150)
    plt.close()

    # Per-subject trajectories random
    n_subj = len(all_clip_vecs)
    colors_c = cm.Greens(np.linspace(0.4, 0.9, max(len(converged_r), 1)))
    colors_d = cm.Reds(np.linspace(0.4, 0.9, max(len(diverged_r), 1)))
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sid in enumerate(converged_r):
        ax.plot(iters, all_errors_r_arr[sid - 1], color=colors_c[i], alpha=0.6)
    for i, sid in enumerate(diverged_r):
        ax.plot(iters, all_errors_r_arr[sid - 1], "--", color=colors_d[i], alpha=0.6)
    ax.axhline(ANG_THRESHOLD, color="k", linestyle=":", label=f"Threshold {ANG_THRESHOLD}°")
    ax.set_title("Random Start – Per-Subject Trajectories")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Angular Error (°)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir_r, "subject_trajectories.png"), dpi=150)
    plt.close()

    res_random = {
        "label"     : "Random_Start",
        "all_errors": all_errors_r_arr,
        "converged" : converged_r,
        "diverged"  : diverged_r,
        "mean_err"  : mean_r,
        "std_err"   : std_r,
    }
    print_table(res_random)

    print(f"\n  📊 Summary (Random_Start):")
    print(f"     Converged : {len(converged_r)}/{n_subj}  →  {converged_r}")
    print(f"     Diverged  : {len(diverged_r)}/{n_subj}  →  {diverged_r}")
    print(f"     Mean error (iter 6): {mean_r[-1]:.1f}° ± {std_r[-1]:.1f}°")

    # ── Comparison Plot ──
    plot_comparison(res_mellow, res_random, OUTPUT_DIR)

    # ── Final comparison bar chart ──
    fig, ax = plt.subplots(figsize=(7, 5))
    labels_x = [f"Iter {i}" for i in range(1, N_ITERATIONS + 1)]
    x = np.arange(N_ITERATIONS)
    w = 0.35
    ax.bar(x - w/2, res_mellow["mean_err"], w, label="Mellow Start",
           yerr=res_mellow["std_err"], capsize=4, color="steelblue")
    ax.bar(x + w/2, res_random["mean_err"], w, label="Random Start",
           yerr=res_random["std_err"], capsize=4, color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_x)
    ax.set_ylabel("Angular Error (°)")
    ax.set_title("Mellow Start vs Random Start – Angular Error per Iteration")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "bar_comparison.png"), dpi=150)
    plt.close()

    print(f"\n✅ ALL DONE.  Results saved in: {OUTPUT_DIR}")
    print(f"   Plots: angular_error_bar.png, subject_trajectories.png,")
    print(f"          wheel_trajectories.png, comparison_mellow_vs_random.png,")
    print(f"          bar_comparison.png")


if __name__ == "__main__":
    main()