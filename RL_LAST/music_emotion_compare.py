"""
music_emotion_compare.py  —  Run ALL THREE algorithms and compare
=================================================================
Runs Q-Learning, SARSA, and Double Q-Learning under identical conditions
then produces side-by-side comparison plots.

Run:
  python music_emotion_compare.py
  python music_emotion_compare.py --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv --episodes 10000
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse

# -- Import each algorithm's module --------------------------------
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
DEFAULT_EPISODES = 10000
EXPECTED_N_CLIPS = 40
DEFAULT_GRID_FOLDS = 10
DEFAULT_GRID_EPISODES = 300

# ------------------------------------------------------------------
# DATA LOADING (shared)
# ──────────────────────────────────────────────────────────────────

def load_subjects(data_root: str, dataset_name: str = "deap"):
    if dataset_name == "deap":
        ratings_csv = os.path.join(data_root, "participant_ratings.csv")
        if not os.path.isfile(ratings_csv):
             ratings_csv = "./data/Metacsv/participant_ratings.csv"
        
        if os.path.isfile(ratings_csv):
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
            return subjects
        else:
            rng = np.random.default_rng(42)
            return [rng.uniform(-1, 1, (EXPECTED_N_CLIPS, 2)).astype(np.float32)
                    for _ in range(32)]
    else:
        # DENSE dataset
        loader = dql_mod.MATLoader(data_root)
        subjects = loader.load_all_subjects_clips()
        if not subjects:
            rng = np.random.default_rng(42)
            return [rng.uniform(-1, 1, (EXPECTED_N_CLIPS, 2)).astype(np.float32)
                    for _ in range(39)]
        # Note: loader.load_all_subjects_clips() already scales by 0.5 in the dql_mod
        return subjects


# ------------------------------------------------------------------
# COMPARISON PLOTS
# ──────────────────────────────────────────────────────────────────

COLORS = {
    "Q-Learning":        ("steelblue",  "-o"),
    "SARSA":             ("darkorange", "-s"),
    "Double Q-Learning": ("purple",     "-^"),
}


def plot_comparison_angular_error(all_results: dict, save_path: str,
                                  title_suffix: str = ""):
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

    ax.axhline(57.0, ls=":", color="gray",   lw=1.5, label="Paper result (57 deg)")
    ax.axhline(20.0, ls="--", color="blue",   lw=1.2, label="20 deg threshold")
    ax.axhline(30.0, ls="--", color="green",  lw=1.2, label="30 deg threshold")
    ax.set_xlabel("Music Clip # (Iteration)", fontsize=12)
    ax.set_ylabel("Angular Error (deg)", fontsize=12)
    title = "Algorithm Comparison — Mean Angular Error per Iteration"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13)
    ax.set_xticks(iters)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_convergence_bar(all_results: dict, save_path: str,
                                    title_suffix: str = ""):
    """Grouped bar chart of convergence criteria for each algorithm."""
    names     = list(all_results.keys())
    criteria  = ["< 20 deg", "< 30 deg"]
    x         = np.arange(len(criteria))
    width     = 0.25
    n         = len(list(all_results.values())[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (algo, results) in enumerate(all_results.items()):
        c20   = sum(1 for r in results if r["final_err"] < 20.0)
        c30   = sum(r["conv_30"]         for r in results)
        vals  = [c20, c30]
        color, _ = COLORS[algo]
        bars = ax.bar(x + i * width, vals, width, label=algo,
                      color=color, alpha=0.85)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2,
                    str(v), ha="center", va="bottom", fontsize=9)

    ax.axhline(19, ls="--", color="red", lw=1.5,
               label="Paper Q-Learning (19/32)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(criteria, fontsize=11)
    ax.set_ylabel("# Subjects converged", fontsize=11)
    ax.set_ylim(0, n + 3)
    title = "Algorithm Comparison — Convergence Criteria"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_comparison_final_error_box(all_results: dict, save_path: str,
                                    title_suffix: str = ""):
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

    ax.axhline(57.0, ls="--", color="gray",  lw=1.5, label="Paper mean (57 deg)")
    ax.axhline(20.0, ls=":",  color="blue",   lw=1.2, label="20 deg threshold")
    ax.axhline(30.0, ls=":",  color="green",  lw=1.2, label="30 deg threshold")
    ax.set_ylabel("Final Angular Error — Clip 6 (deg)", fontsize=11)
    title = "Final Error Distribution — All Algorithms"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shaping_effect_per_algo(with_results: dict, without_results: dict,
                                 save_path: str):
    """
    Side-by-side angular error curves for each algorithm:
    WITH vs WITHOUT reward shaping.
    """
    algos = list(with_results.keys())
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5), sharey=True)
    if len(algos) == 1:
        axes = [axes]
    iters = np.arange(1, 7)

    for ax, algo in zip(axes, algos):
        color, _ = COLORS[algo]

        me_w  = np.array([r["errors"] for r in with_results[algo]]).mean(axis=0)
        me_wo = np.array([r["errors"] for r in without_results[algo]]).mean(axis=0)

        ax.plot(iters, me_w,  "-o",  color=color,   lw=2, ms=7,
                label="With shaping")
        ax.plot(iters, me_wo, "--s", color="tomato", lw=2, ms=7,
                label="Without shaping")
        ax.axhline(20, color="blue",  ls=":", lw=1)
        ax.axhline(30, color="green", ls=":", lw=1)
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_title(algo, fontsize=12, fontweight="bold")
        ax.set_xticks(iters)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Angular Error (deg)", fontsize=11)
    fig.suptitle("Reward Shaping Effect — All Algorithms", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_shaping_convergence_comparison(with_results: dict, without_results: dict,
                                        save_path: str):
    """
    Grouped bar chart: for each algorithm, show <20 deg convergence
    count WITH and WITHOUT reward shaping.
    """
    algos = list(with_results.keys())
    n     = len(list(with_results.values())[0])
    x     = np.arange(len(algos))
    width = 0.35

    w_counts  = [sum(1 for r in with_results[a]    if r["final_err"] < 20.0) for a in algos]
    wo_counts = [sum(1 for r in without_results[a] if r["final_err"] < 20.0) for a in algos]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, w_counts,  width, label="With shaping",
                   color="#2ecc71", edgecolor="k", lw=0.5)
    bars2 = ax.bar(x + width/2, wo_counts, width, label="Without shaping",
                   color="#e74c3c", edgecolor="k", lw=0.5)

    for b, v in zip(bars1, w_counts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for b, v in zip(bars2, wo_counts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(19, ls="--", color="gray", lw=1.2, label="Paper (19/32)")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=11)
    ax.set_ylim(0, n + 3)
    ax.set_ylabel("# Subjects (< 20 deg)", fontsize=11)
    ax.set_title("Reward Shaping Impact on Convergence — All Algorithms", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def print_master_summary(all_results: dict, tag: str = ""):
    header = f"  MASTER COMPARISON TABLE"
    if tag:
        header += f"  [{tag}]"
    print("\n" + "=" * 90)
    print(header)
    print("  {:30s} | {:7s} | {:7s} | {:14s}".format(
        "Algorithm", "<20 deg", "<30 deg", "Mean err clip6"))
    print("-" * 90)
    for name, results in all_results.items():
        n     = len(results)
        c20   = sum(1 for r in results if r["final_err"] < 20.0)
        c30   = sum(r["conv_30"]         for r in results)
        fe    = np.array([r["final_err"] for r in results])
        print(f"  {name:30s} | {c20:3d}/{n} | {c30:3d}/{n} | "
              f"{fe.mean():5.1f} deg +- {fe.std():4.1f} deg")
    print("-" * 90)
    print("  Paper (Q-Learning)                |    N/A |    N/A |  57.0 deg +-  2.8 deg")
    print("=" * 90)


def select_params_for_mode(module, label: str, subjects: list, args,
                           use_reward_shaping: bool) -> dict:
    if not args.grid_search:
        return {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}

    mode = "with shaping" if use_reward_shaping else "without shaping"
    print(f"\n  Grid search CV: {label} ({mode}) [{args.dataset.upper()}]")
    return module.grid_search(
        subjects,
        n_folds=args.grid_folds,
        n_episodes=args.grid_episodes,
        use_reward_shaping=use_reward_shaping,
        dataset_name=args.dataset
    )


# ------------------------------------------------------------------
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare Q-Learning, SARSA, and Double Q-Learning")
    parser.add_argument("--data_root",  default="./data/Metacsv")
    parser.add_argument("--dataset",     choices=["deap", "dense"], default="deap")
    parser.add_argument("--output_dir", default="results_compare")
    parser.add_argument("--episodes",   type=int,   default=DEFAULT_EPISODES)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--gamma",      type=float, default=0.6)
    parser.add_argument("--epsilon",    type=float, default=0.1)
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
    print(f"  EEG Music Emotion RL  —  Algorithm Comparison  ({args.dataset.upper()} Dataset)")
    print("  Q-Learning | SARSA | Double Q-Learning")
    print("=" * 65)

    # -- Init emotion wheel (all modules share the same wheel) ------
    if args.dataset == "deap":
        csv_path = os.path.join(args.data_root, "emotion_summary.csv")
        if not os.path.isfile(csv_path):
            csv_path = "./data/Metacsv/emotion_summary.csv"
        ql_mod._init_globals(csv_path)
        sa_mod._init_globals(csv_path)
        dql_mod._init_globals(csv_path)
    else:
        # DENSE dataset: We need to trigger the dataset-specific global initialization in each module.
        # Actually, each module has its own global EMOTION_CENTERS.
        # We need to ensure each module knows it's using 'dense'.
        # Since I added dataset_name to _init_globals is not quite right in previous scripts (I used if/else in main).
        # Let me check how I implemented it in other scripts.
        # In other scripts I did it inside `main`.
        # I should probably expose a function or just repeat the logic.
        
        # A cleaner way: each module's main() does it. But here we import them.
        # I'll just manually set the globals for them if it's 'dense'.
        for mod in [ql_mod, sa_mod, dql_mod]:
            mod.EMOTION_CENTERS = {}
            dense_map_low = {k.lower(): v for k, v in mod.DENSE_EMOTION_MAPPING.items()}
            for group, emotions in mod.EMOTION_GROUPS.items():
                vals = [dense_map_low[e.lower()] for e in emotions if e.lower() in dense_map_low]
                if not vals: continue
                vec = np.mean(vals, axis=0)
                vec = vec / (np.linalg.norm(vec) + 1e-9)
                mod.EMOTION_CENTERS[group] = tuple(vec)
            
            mod.EMOTION_NAMES = list(mod.EMOTION_CENTERS.keys())
            mod.EMOTION_INDEX = {n: i for i, n in enumerate(mod.EMOTION_NAMES)}
            mod.N_EMOTIONS = len(mod.EMOTION_NAMES)
            mod.TARGET_VECTOR = np.array(mod.EMOTION_CENTERS[mod.TARGET_EMOTION])

    # -- Load subjects ----------------------------------------------
    subjects = load_subjects(args.data_root, dataset_name=args.dataset)
    print(f"\n  Subjects: {len(subjects)}")

    ql_params_with = select_params_for_mode(ql_mod, "Q-Learning", subjects, args, True)
    ql_params_without = select_params_for_mode(ql_mod, "Q-Learning", subjects, args, False)
    sarsa_params_with = select_params_for_mode(sa_mod, "SARSA", subjects, args, True)
    sarsa_params_without = select_params_for_mode(sa_mod, "SARSA", subjects, args, False)
    dql_params_with = select_params_for_mode(dql_mod, "Double Q-Learning", subjects, args, True)
    dql_params_without = select_params_for_mode(dql_mod, "Double Q-Learning", subjects, args, False)

    # ══════════════════════════════════════════════════════════════
    # WITH REWARD SHAPING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("  WITH REWARD SHAPING")
    print("=" * 50)

    print("\n" + "-" * 50)
    print("  Running Q-Learning (with shaping) ...")
    print("-" * 50)
    res_ql_w = ql_mod.leave_one_subject_out(
        subjects, **ql_params_with, n_episodes=args.episodes, use_reward_shaping=True, dataset_name=args.dataset)

    print("\n" + "-" * 50)
    print("  Running SARSA (with shaping) ...")
    print("-" * 50)
    res_sarsa_w = sa_mod.leave_one_subject_out(
        subjects, **sarsa_params_with, n_episodes=args.episodes, use_reward_shaping=True, dataset_name=args.dataset)

    print("\n" + "-" * 50)
    print("  Running Double Q-Learning (with shaping) ...")
    print("-" * 50)
    res_dql_w = dql_mod.leave_one_subject_out(
        subjects, **dql_params_with, n_episodes=args.episodes, use_reward_shaping=True, dataset_name=args.dataset)

    # ══════════════════════════════════════════════════════════════
    # WITHOUT REWARD SHAPING
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("  WITHOUT REWARD SHAPING")
    print("=" * 50)

    print("\n" + "-" * 50)
    print("  Running Q-Learning (without shaping) ...")
    print("-" * 50)
    res_ql_wo = ql_mod.leave_one_subject_out(
        subjects, **ql_params_without, n_episodes=args.episodes, use_reward_shaping=False, dataset_name=args.dataset)

    print("\n" + "-" * 50)
    print("  Running SARSA (without shaping) ...")
    print("-" * 50)
    res_sarsa_wo = sa_mod.leave_one_subject_out(
        subjects, **sarsa_params_without, n_episodes=args.episodes, use_reward_shaping=False, dataset_name=args.dataset)

    print("\n" + "-" * 50)
    print("  Running Double Q-Learning (without shaping) ...")
    print("-" * 50)
    res_dql_wo = dql_mod.leave_one_subject_out(
        subjects, **dql_params_without, n_episodes=args.episodes, use_reward_shaping=False, dataset_name=args.dataset)

    # -- Collect results --------------------------------------------
    all_with = {
        "Q-Learning":        res_ql_w,
        "SARSA":             res_sarsa_w,
        "Double Q-Learning": res_dql_w,
    }
    all_without = {
        "Q-Learning":        res_ql_wo,
        "SARSA":             res_sarsa_wo,
        "Double Q-Learning": res_dql_wo,
    }

    # -- Summary tables ---------------------------------------------
    print_master_summary(all_with,    tag="With Reward Shaping")
    print_master_summary(all_without, tag="Without Reward Shaping")

    # ══════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\nGenerating comparison plots ...")

    # --- WITH shaping comparison plots ---
    plot_comparison_angular_error(
        all_with,
        title_suffix="With Shaping",
        save_path=os.path.join(args.output_dir, "compare_angular_error_with.png"))

    plot_comparison_convergence_bar(
        all_with,
        title_suffix="With Shaping",
        save_path=os.path.join(args.output_dir, "compare_convergence_bar_with.png"))

    plot_comparison_final_error_box(
        all_with,
        title_suffix="With Shaping",
        save_path=os.path.join(args.output_dir, "compare_final_error_box_with.png"))

    # --- WITHOUT shaping comparison plots ---
    plot_comparison_angular_error(
        all_without,
        title_suffix="Without Shaping",
        save_path=os.path.join(args.output_dir, "compare_angular_error_without.png"))

    plot_comparison_convergence_bar(
        all_without,
        title_suffix="Without Shaping",
        save_path=os.path.join(args.output_dir, "compare_convergence_bar_without.png"))

    plot_comparison_final_error_box(
        all_without,
        title_suffix="Without Shaping",
        save_path=os.path.join(args.output_dir, "compare_final_error_box_without.png"))

    # --- Cross-condition comparison plots ---
    plot_shaping_effect_per_algo(
        all_with, all_without,
        save_path=os.path.join(args.output_dir, "compare_shaping_effect.png"))

    plot_shaping_convergence_comparison(
        all_with, all_without,
        save_path=os.path.join(args.output_dir, "compare_shaping_convergence.png"))

    # --- Per-algorithm reward shaping comparison ---
    for name, mod, prefix in [
        ("Q-Learning",        ql_mod,  "ql"),
        ("SARSA",             sa_mod,  "sarsa"),
        ("Double Q-Learning", dql_mod, "dql"),
    ]:
        mod.plot_reward_shaping_comparison(
            all_with[name], all_without[name], label=name,
            save_path=os.path.join(args.output_dir, f"{prefix}_reward_shaping.png"))

    # --- Individual detailed plots for each algorithm (with shaping) ---
    for name, results, mod, prefix in [
        ("Q-Learning",        res_ql_w,    ql_mod,  "ql"),
        ("SARSA",             res_sarsa_w, sa_mod,  "sarsa"),
        ("Double Q-Learning", res_dql_w,   dql_mod, "dql"),
    ]:
        mod.plot_angular_error(
            results, label=f"{name} (with shaping)",
            save_path=os.path.join(args.output_dir, f"{prefix}_angular_error.png"))
        mod.plot_all_subjects_trajectories(
            results, label=f"{name} (with shaping)",
            save_path=os.path.join(args.output_dir, f"{prefix}_all_trajectories.png"))

    # --- Individual detailed plots (without shaping) ---
    for name, results, mod, prefix in [
        ("Q-Learning",        res_ql_wo,    ql_mod,  "ql"),
        ("SARSA",             res_sarsa_wo, sa_mod,  "sarsa"),
        ("Double Q-Learning", res_dql_wo,   dql_mod, "dql"),
    ]:
        mod.plot_angular_error(
            results, label=f"{name} (without shaping)",
            save_path=os.path.join(args.output_dir, f"{prefix}_angular_error_no_shaping.png"))
        mod.plot_all_subjects_trajectories(
            results, label=f"{name} (without shaping)",
            save_path=os.path.join(args.output_dir, f"{prefix}_all_trajectories_no_shaping.png"))

    print(f"\nDONE. All results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()