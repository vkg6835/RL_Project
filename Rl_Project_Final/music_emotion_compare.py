"""
music_emotion_compare.py  —  Run ALL THREE algorithms and compare
=================================================================
Runs Q-Learning, SARSA, and Double Q-Learning under identical conditions
then produces side-by-side comparison plots.

Run:
  python music_emotion_compare.py
  python music_emotion_compare.py --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv --episodes 2000
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse

# ── Import each algorithm's module ────────────────────────────────
import importlib.util, sys

def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

HERE = os.path.dirname(os.path.abspath(__file__))

ql_mod  = _import(os.path.join(HERE, "music_emotion_ql.py"),        "ql")
sa_mod  = _import(os.path.join(HERE, "music_emotion_sarsa.py"),     "sarsa")
dql_mod = _import(os.path.join(HERE, "music_emotion_double_ql.py"), "dql")


# ──────────────────────────────────────────────────────────────────
# DATA LOADING (shared)
# ──────────────────────────────────────────────────────────────────

def load_subjects(data_root: str):
    ratings_csv = os.path.join(data_root, "participant_ratings.csv")
    if os.path.isfile(ratings_csv):
        df_r     = pd.read_csv(ratings_csv)
        subjects = []
        for pid, grp in df_r.groupby("Participant_id"):
            clips = grp[["Valence", "Arousal"]].values.astype(np.float32)
            clips[:, 0] = 2 * (clips[:, 0] - 1) / 8 - 1
            clips[:, 1] = 2 * (clips[:, 1] - 1) / 8 - 1
            subjects.append(clips)
        return subjects
    else:
        rng = np.random.default_rng(42)
        return [rng.uniform(-1, 1, (np.random.randint(10, 40), 2)).astype(np.float32)
                for _ in range(32)]


# ──────────────────────────────────────────────────────────────────
# COMPARISON PLOTS
# ──────────────────────────────────────────────────────────────────

COLORS = {
    "Q-Learning":        ("steelblue",  "-o"),
    "SARSA":             ("darkorange", "-s"),
    "Double Q-Learning": ("purple",     "-^"),
}


def plot_comparison_angular_error(all_results: dict, save_path: str):
    """One plot, all three algorithms, with SE bars."""
    iters = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, results in all_results.items():
        all_e  = np.array([r["errors"] for r in results])
        mean_e = all_e.mean(axis=0)
        se     = all_e.std(axis=0) / np.sqrt(len(results))
        color, fmt = COLORS[name]
        ax.errorbar(iters, mean_e, yerr=se, fmt=fmt, color=color,
                    capsize=4, lw=2, ms=7, label=name)

    ax.axhline(57.0, ls=":", color="gray",   lw=1.5, label="Paper result (57°)")
    ax.axhline(60.0, ls="--", color="orange", lw=1.2, label="60° threshold")
    ax.axhline(30.0, ls="--", color="green",  lw=1.2, label="30° threshold")
    ax.set_xlabel("Music Clip # (Iteration)", fontsize=12)
    ax.set_ylabel("Angular Error (°)", fontsize=12)
    ax.set_title("Algorithm Comparison — Mean Angular Error per Iteration",
                 fontsize=13)
    ax.set_xticks(iters)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_convergence_bar(all_results: dict, save_path: str):
    """Grouped bar chart of all three convergence criteria for each algorithm."""
    names     = list(all_results.keys())
    criteria  = ["Exact happy", "< 30°", "< 60°"]
    x         = np.arange(len(criteria))
    width     = 0.25
    n         = len(list(all_results.values())[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (algo, results) in enumerate(all_results.items()):
        exact = sum(r["exact_converged"] for r in results)
        c30   = sum(r["conv_30"]         for r in results)
        c60   = sum(r["conv_60"]         for r in results)
        vals  = [exact, c30, c60]
        color, _ = COLORS[algo]
        bars = ax.bar(x + i * width, vals, width, label=algo,
                      color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2,
                    str(v), ha="center", va="bottom", fontsize=9)

    ax.axhline(19, ls="--", color="red", lw=1.5,
               label="Paper Q-Learning (19/32 exact)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(criteria, fontsize=11)
    ax.set_ylabel("# Subjects converged", fontsize=11)
    ax.set_ylim(0, n + 3)
    ax.set_title("Algorithm Comparison — Convergence Criteria", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_final_error_box(all_results: dict, save_path: str):
    """Box plot of final angular errors for each algorithm."""
    data   = [np.array([r["final_err"] for r in v])
              for v in all_results.values()]
    labels = list(all_results.keys())
    colors = [COLORS[n][0] for n in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(57.0, ls="--", color="gray",  lw=1.5, label="Paper mean (57°)")
    ax.axhline(60.0, ls=":",  color="orange", lw=1.2, label="60° threshold")
    ax.axhline(30.0, ls=":",  color="green",  lw=1.2, label="30° threshold")
    ax.set_ylabel("Final Angular Error — Clip 6 (°)", fontsize=11)
    ax.set_title("Final Error Distribution — All Algorithms", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def print_master_summary(all_results: dict):
    print("\n" + "=" * 85)
    print("  MASTER COMPARISON TABLE")
    print("  {:22s} | {:6s} | {:6s} | {:6s} | {:14s}".format(
        "Algorithm", "Exact", "<30°", "<60°", "Mean err clip6"))
    print("-" * 85)
    for name, results in all_results.items():
        n     = len(results)
        exact = sum(r["exact_converged"] for r in results)
        c30   = sum(r["conv_30"]         for r in results)
        c60   = sum(r["conv_60"]         for r in results)
        fe    = np.array([r["final_err"] for r in results])
        print(f"  {name:22s} | {exact:3d}/{n} | {c30:3d}/{n} | {c60:3d}/{n} | "
              f"{fe.mean():5.1f}° ± {fe.std():4.1f}°")
    print("-" * 85)
    print("  Paper (Q-Learning)      |  19/32 |    N/A |    N/A |  57.0° ±  2.8°")
    print("=" * 85)


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare Q-Learning, SARSA, and Double Q-Learning")
    parser.add_argument("--data_root",  default="/home/sahil/Desktop/Rl_Project_Final/data/Metacsv")
    parser.add_argument("--output_dir", default="results_compare")
    parser.add_argument("--episodes",   type=int,   default=2000)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--gamma",      type=float, default=0.6)
    parser.add_argument("--epsilon",    type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  EEG Music Emotion RL  —  Algorithm Comparison")
    print("  Q-Learning | SARSA | Double Q-Learning")
    print("=" * 65)

    # ── Init emotion wheel (all modules share the same wheel) ──────
    csv_path = os.path.join(args.data_root, "emotion_summary.csv")
    ql_mod._init_globals(csv_path)
    sa_mod._init_globals(csv_path)
    dql_mod._init_globals(csv_path)

    # ── Load subjects ──────────────────────────────────────────────
    subjects = load_subjects(args.data_root)
    print(f"\n  Subjects: {len(subjects)}")

    params = dict(alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon,
                  n_episodes=args.episodes, use_reward_shaping=True)

    # ── Run each algorithm ─────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  Running Q-Learning …")
    print("─" * 50)
    res_ql = ql_mod.leave_one_subject_out(subjects, **params)

    print("\n" + "─" * 50)
    print("  Running SARSA …")
    print("─" * 50)
    res_sarsa = sa_mod.leave_one_subject_out(subjects, **params)

    print("\n" + "─" * 50)
    print("  Running Double Q-Learning …")
    print("─" * 50)
    res_dql = dql_mod.leave_one_subject_out(subjects, **params)

    # ── Collect & summarise ────────────────────────────────────────
    all_results = {
        "Q-Learning":        res_ql,
        "SARSA":             res_sarsa,
        "Double Q-Learning": res_dql,
    }
    print_master_summary(all_results)

    # ── Plots ──────────────────────────────────────────────────────
    print("\nGenerating comparison plots …")

    plot_comparison_angular_error(
        all_results,
        save_path=os.path.join(args.output_dir, "compare_angular_error.png"))

    plot_comparison_convergence_bar(
        all_results,
        save_path=os.path.join(args.output_dir, "compare_convergence_bar.png"))

    plot_comparison_final_error_box(
        all_results,
        save_path=os.path.join(args.output_dir, "compare_final_error_box.png"))

    # Individual detailed plots for each algorithm
    for name, results, mod, prefix in [
        ("Q-Learning",        res_ql,    ql_mod,  "ql"),
        ("SARSA",             res_sarsa, sa_mod,  "sarsa"),
        ("Double Q-Learning", res_dql,   dql_mod, "dql"),
    ]:
        mod.plot_angular_error(
            results, label=name,
            save_path=os.path.join(args.output_dir, f"{prefix}_angular_error.png"))
        mod.plot_all_subjects_trajectories(
            results, label=name,
            save_path=os.path.join(args.output_dir, f"{prefix}_all_trajectories.png"))

    print(f"\n✅  Done.  All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
