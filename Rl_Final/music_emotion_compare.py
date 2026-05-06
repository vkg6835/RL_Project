"""
music_emotion_compare.py  —  Run ALL FOUR algorithms and compare across ALL start emotions
=================================================================
Runs Q-Learning, SARSA, Double Q-Learning, and Double SARSA under
identical conditions then produces comparison plots for EACH starting emotion.

Convergence criterion: angular error ≤ 20°

Run:
  python music_emotion_compare.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import argparse

import importlib.util, sys

def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

HERE = os.path.dirname(os.path.abspath(__file__))

ql_mod   = _import(os.path.join(HERE, "music_emotion_ql.py"),           "ql")
sa_mod   = _import(os.path.join(HERE, "music_emotion_sarsa.py"),        "sarsa")
dql_mod  = _import(os.path.join(HERE, "music_emotion_double_ql.py"),    "dql")
dsa_mod  = _import(os.path.join(HERE, "music_emotion_double_sarsa.py"), "dsa")

DEFAULT_EPISODES = 10000
EXPECTED_N_CLIPS = 40
DEFAULT_GRID_FOLDS = 10
DEFAULT_GRID_EPISODES = 300
CONV_THRESHOLD = 20.0  # Only criterion is <= 20 degrees


# ------------------------------------------------------------------
# DATA LOADING (shared)
# ──────────────────────────────────────────────────────────────────

def load_subjects(data_root: str):
    ratings_csv = os.path.join(data_root, "participant_ratings.csv")
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
        clip_counts = sorted({s.shape[0] for s in subjects})
        if len(clip_counts) != 1:
            raise ValueError(
                "All subjects must have the same number of clips for clip-wise "
                f"fold means; got clip counts {clip_counts}"
            )
        return subjects
    else:
        rng = np.random.default_rng(42)
        return [rng.uniform(-1, 1, (EXPECTED_N_CLIPS, 2)).astype(np.float32)
                for _ in range(32)]


# ------------------------------------------------------------------
# COMPARISON PLOTS
# ──────────────────────────────────────────────────────────────────

COLORS = {
    "Q-Learning":        ("steelblue",  "-o"),
    "SARSA":             ("darkorange", "-s"),
    "Double Q-Learning": ("purple",     "-^"),
    "Double SARSA":      ("teal",       "-D"),
}


def plot_comparison_angular_error(all_results: dict, save_path: str, title_suffix: str = ""):
    iters = np.arange(1, 7)
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, results in all_results.items():
        all_e  = np.array([r["errors"] for r in results])
        mean_e = all_e.mean(axis=0)
        se     = all_e.std(axis=0) / np.sqrt(len(results))
        color, fmt = COLORS[name]
        ax.errorbar(iters, mean_e, yerr=se, fmt=fmt, color=color,
                    capsize=4, lw=2, ms=7, label=name)

    ax.axhline(CONV_THRESHOLD, ls="--", color="green",  lw=1.5, label=f"Converged (≤{int(CONV_THRESHOLD)}°)")
    ax.set_xlabel("Music Clip # (Iteration)", fontsize=12)
    ax.set_ylabel("Angular Error (deg)", fontsize=12)
    title = f"Algorithm Comparison — Mean Angular Error per Iteration"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13)
    ax.set_xticks(iters)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_comparison_convergence_bar(all_results: dict, save_path: str, title_suffix: str = ""):
    names     = list(all_results.keys())
    x         = np.arange(len(names))
    width     = 0.6
    n         = len(list(all_results.values())[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [sum(1 for r in results if r["final_err"] <= CONV_THRESHOLD) for results in all_results.values()]
    colors = [COLORS[a][0] for a in names]
    
    bars = ax.bar(x, vals, width, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel(f"# Subjects Converged (≤{int(CONV_THRESHOLD)}°)", fontsize=11)
    ax.set_ylim(0, n + 3)
    title = f"Algorithm Comparison — Convergence (≤{int(CONV_THRESHOLD)}°)"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_comparison_final_error_box(all_results: dict, save_path: str, title_suffix: str = ""):
    data   = [np.array([r["final_err"] for r in v]) for v in all_results.values()]
    labels = list(all_results.keys())
    colors = [COLORS[n][0] for n in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(CONV_THRESHOLD, ls="--", color="green", lw=2, label=f"Converged (≤{int(CONV_THRESHOLD)}°)")
    ax.set_ylabel("Final Angular Error — Clip 6 (deg)", fontsize=11)
    title = "Final Error Distribution — All Algorithms"
    if title_suffix:
        title += f" ({title_suffix})"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_shaping_effect_per_algo(with_results: dict, without_results: dict, save_path: str):
    algos = list(with_results.keys())
    fig, axes = plt.subplots(1, len(algos), figsize=(4 * len(algos), 4), sharey=True)
    if len(algos) == 1:
        axes = [axes]
    iters = np.arange(1, 7)

    for ax, algo in zip(axes, algos):
        color, _ = COLORS[algo]
        me_w  = np.array([r["errors"] for r in with_results[algo]]).mean(axis=0)
        me_wo = np.array([r["errors"] for r in without_results[algo]]).mean(axis=0)
        ax.plot(iters, me_w,  "-o",  color=color,   lw=2, ms=7, label="With shaping")
        ax.plot(iters, me_wo, "--s", color="tomato", lw=2, ms=7, label="Without shaping")
        ax.axhline(CONV_THRESHOLD, color="green", ls=":", lw=1.5, label=f"≤{int(CONV_THRESHOLD)}°")
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_title(algo, fontsize=11, fontweight="bold")
        ax.set_xticks(iters)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Angular Error (deg)", fontsize=11)
    fig.suptitle("Reward Shaping Effect — All Algorithms", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_shaping_convergence_comparison(with_results: dict, without_results: dict, save_path: str):
    algos = list(with_results.keys())
    n     = len(list(with_results.values())[0])
    x     = np.arange(len(algos))
    width = 0.35

    w_counts  = [sum(1 for r in with_results[a]    if r["final_err"] <= CONV_THRESHOLD) for a in algos]
    wo_counts = [sum(1 for r in without_results[a] if r["final_err"] <= CONV_THRESHOLD) for a in algos]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, w_counts,  width, label="With shaping", color="#2ecc71", edgecolor="k", lw=0.5)
    bars2 = ax.bar(x + width/2, wo_counts, width, label="Without shaping", color="#e74c3c", edgecolor="k", lw=0.5)

    for b, v in zip(bars1, w_counts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for b, v in zip(bars2, wo_counts):
        ax.text(b.get_x() + b.get_width()/2, v + 0.3,
                f"{v}/{n}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=10)
    ax.set_ylim(0, n + 3)
    ax.set_ylabel(f"# Subjects (≤ {int(CONV_THRESHOLD)}°)", fontsize=11)
    ax.set_title(f"Reward Shaping Impact on Convergence (≤{int(CONV_THRESHOLD)}°) — All Algorithms", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def print_master_summary(all_results: dict, tag: str = "", start_emotion: str = ""):
    header = f"  MASTER COMPARISON TABLE  [Start: {start_emotion.upper()}]"
    if tag:
        header += f"  [{tag}]"
    print("\n" + "=" * 80)
    print(header)
    print("  {:30s} | {:10s} | {:14s}".format(
        "Algorithm", f"≤{int(CONV_THRESHOLD)} deg", "Mean err clip6"))
    print("-" * 80)
    for name, results in all_results.items():
        n     = len(results)
        c20   = sum(1 for r in results if r["final_err"] <= CONV_THRESHOLD)
        fe    = np.array([r["final_err"] for r in results])
        print(f"  {name:30s} | {c20:4d}/{n} | {fe.mean():5.1f} deg +- {fe.std():4.1f} deg")
    print("=" * 80)


def select_params_for_mode(module, label: str, subjects: list, args, use_reward_shaping: bool) -> dict:
    if not args.grid_search:
        return {"alpha": args.alpha, "gamma": args.gamma, "epsilon": args.epsilon}

    mode = "with shaping" if use_reward_shaping else "without shaping"
    print(f"\n  Grid search CV: {label} ({mode})")
    return module.grid_search(
        subjects,
        n_folds=args.grid_folds,
        n_episodes=args.grid_episodes,
        use_reward_shaping=use_reward_shaping,
    )


# ------------------------------------------------------------------
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate RL algorithms for all emotion start states.")
    parser.add_argument("--data_root",  default="data/Metacsv")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--episodes",   type=int,   default=DEFAULT_EPISODES)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--gamma",      type=float, default=0.6)
    parser.add_argument("--epsilon",    type=float, default=0.1)
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--grid_folds", type=int, default=DEFAULT_GRID_FOLDS)
    parser.add_argument("--grid_episodes", type=int, default=DEFAULT_GRID_EPISODES)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  EEG Music Emotion RL  —  Algorithm Evaluation Over ALL States")
    print("=" * 70)

    # -- Init emotion wheel (all modules share the same wheel) ------
    csv_path = os.path.join(args.data_root, "emotion_summary.csv")
    ql_mod._init_globals(csv_path)
    sa_mod._init_globals(csv_path)
    dql_mod._init_globals(csv_path)
    dsa_mod._init_globals(csv_path)

    # -- Load subjects ----------------------------------------------
    subjects = load_subjects(args.data_root)
    print(f"\n  Subjects: {len(subjects)}")

    # -- Get optimal parameters (run once without start_emotion specific) --
    ql_params_with    = select_params_for_mode(ql_mod,  "Q-Learning",        subjects, args, True)
    ql_params_without = select_params_for_mode(ql_mod,  "Q-Learning",        subjects, args, False)
    sarsa_params_with    = select_params_for_mode(sa_mod,  "SARSA",           subjects, args, True)
    sarsa_params_without = select_params_for_mode(sa_mod,  "SARSA",           subjects, args, False)
    dql_params_with    = select_params_for_mode(dql_mod, "Double Q-Learning", subjects, args, True)
    dql_params_without = select_params_for_mode(dql_mod, "Double Q-Learning", subjects, args, False)
    dsa_params_with    = select_params_for_mode(dsa_mod, "Double SARSA",      subjects, args, True)
    dsa_params_without = select_params_for_mode(dsa_mod, "Double SARSA",      subjects, args, False)

    # Track overall convergence safely
    summary_file = os.path.join(args.output_dir, "overall_convergence_summary.csv")
    summary_data = []

    emotions_to_test = ql_mod.EMOTION_NAMES

    for emotion in emotions_to_test:
        print("\n\n" + "#" * 70)
        print(f"### EVALUATING WITH START EMOTION: {emotion.upper()}")
        print("#" * 70)

        emotion_dir = os.path.join(args.output_dir, emotion)
        os.makedirs(emotion_dir, exist_ok=True)

        res_ql_w    = ql_mod.leave_one_subject_out(subjects, **ql_params_with, n_episodes=args.episodes, use_reward_shaping=True, start_emotion=emotion)
        res_sarsa_w = sa_mod.leave_one_subject_out(subjects, **sarsa_params_with, n_episodes=args.episodes, use_reward_shaping=True, start_emotion=emotion)
        res_dql_w   = dql_mod.leave_one_subject_out(subjects, **dql_params_with, n_episodes=args.episodes, use_reward_shaping=True, start_emotion=emotion)
        res_dsa_w   = dsa_mod.leave_one_subject_out(subjects, **dsa_params_with, n_episodes=args.episodes, use_reward_shaping=True, start_emotion=emotion)

        res_ql_wo    = ql_mod.leave_one_subject_out(subjects, **ql_params_without, n_episodes=args.episodes, use_reward_shaping=False, start_emotion=emotion)
        res_sarsa_wo = sa_mod.leave_one_subject_out(subjects, **sarsa_params_without, n_episodes=args.episodes, use_reward_shaping=False, start_emotion=emotion)
        res_dql_wo   = dql_mod.leave_one_subject_out(subjects, **dql_params_without, n_episodes=args.episodes, use_reward_shaping=False, start_emotion=emotion)
        res_dsa_wo   = dsa_mod.leave_one_subject_out(subjects, **dsa_params_without, n_episodes=args.episodes, use_reward_shaping=False, start_emotion=emotion)

        all_with = {
            "Q-Learning":        res_ql_w,
            "SARSA":             res_sarsa_w,
            "Double Q-Learning": res_dql_w,
            "Double SARSA":      res_dsa_w,
        }
        all_without = {
            "Q-Learning":        res_ql_wo,
            "SARSA":             res_sarsa_wo,
            "Double Q-Learning": res_dql_wo,
            "Double SARSA":      res_dsa_wo,
        }

        print_master_summary(all_with,    tag="With Reward Shaping",    start_emotion=emotion)
        print_master_summary(all_without, tag="Without Reward Shaping", start_emotion=emotion)

        # Save to summary data
        row = {"Start Emotion": emotion}
        for algo, results in all_with.items():
            n = len(results)
            c20 = sum(1 for r in results if r["final_err"] <= CONV_THRESHOLD)
            row[f"{algo} (Shaping)"] = f"{c20}/{n}"
        for algo, results in all_without.items():
            n = len(results)
            c20 = sum(1 for r in results if r["final_err"] <= CONV_THRESHOLD)
            row[f"{algo} (No Shaping)"] = f"{c20}/{n}"
            
        summary_data.append(row)

        plot_comparison_angular_error(all_with, title_suffix=f"Start={emotion}, With Shaping",
                                      save_path=os.path.join(emotion_dir, "compare_angular_error_with.png"))
        plot_comparison_convergence_bar(all_with, title_suffix=f"Start={emotion}, With Shaping",
                                        save_path=os.path.join(emotion_dir, "compare_convergence_bar_with.png"))
        plot_comparison_final_error_box(all_with, title_suffix=f"Start={emotion}, With Shaping",
                                        save_path=os.path.join(emotion_dir, "compare_final_error_box_with.png"))

        plot_comparison_angular_error(all_without, title_suffix=f"Start={emotion}, Without Shaping",
                                      save_path=os.path.join(emotion_dir, "compare_angular_error_without.png"))
        plot_comparison_convergence_bar(all_without, title_suffix=f"Start={emotion}, Without Shaping",
                                        save_path=os.path.join(emotion_dir, "compare_convergence_bar_without.png"))
        plot_comparison_final_error_box(all_without, title_suffix=f"Start={emotion}, Without Shaping",
                                        save_path=os.path.join(emotion_dir, "compare_final_error_box_without.png"))

        plot_shaping_effect_per_algo(all_with, all_without,
                                     save_path=os.path.join(emotion_dir, "compare_shaping_effect.png"))
        plot_shaping_convergence_comparison(all_with, all_without,
                                            save_path=os.path.join(emotion_dir, "compare_shaping_convergence.png"))

        # Detailed individual plots could be written but keep it concise for 8 emotions
        
        print(f"-> Saved {emotion} plots in directory: {emotion_dir}/")

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_file, index=False)
    print(f"\nSaved overall convergence summary to: {summary_file}")
    
    print("\n\nOVERALL RAW CONVERGENCE SUMMARY (≤20°):")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    main()