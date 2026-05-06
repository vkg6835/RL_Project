import argparse
import json
import os

from emotion_classifier import run_benchmark
from music_emotion_rl import run_rl_benchmark


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Run the full LOSO EEG emotion benchmark and RL recommendation study."
    )
    parser.add_argument("--data_root", default="data/Metacsv")
    parser.add_argument("--eeg_root", default="data/data_preprocessed_python")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--eeg_channels", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--playlist_length", type=int, default=6)
    parser.add_argument("--bins", type=int, default=21)
    parser.add_argument("--target_emotion", default="happy")
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--convergence_deg", type=float, default=20.0)
    parser.add_argument("--max_train_starts", type=int, default=256)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--skip_lstm", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    benchmark = run_benchmark(
        metadata_root=args.data_root,
        eeg_root=args.eeg_root,
        output_dir=output_dir,
        eeg_channels=args.eeg_channels,
        random_state=args.random_state,
        include_lstm=not args.skip_lstm,
    )

    best_regressor_predictions = benchmark["regressor"]["predictions"]
    best_regressor_name = benchmark["regressor"]["best_model_name"]
    best_regressor_predictions = best_regressor_predictions[
        best_regressor_predictions["model_name"] == best_regressor_name
    ].copy()

    rl_results = run_rl_benchmark(
        dataset=benchmark["dataset"],
        prediction_frame=best_regressor_predictions,
        output_dir=output_dir,
        target_emotion=args.target_emotion,
        episodes=args.episodes,
        playlist_length=args.playlist_length,
        bins=args.bins,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        convergence_threshold_deg=args.convergence_deg,
        random_state=args.random_state,
        max_train_starts=args.max_train_starts,
    )

    summary = {
        "classifier": benchmark["summary"]["best_classifier_model"],
        "regressor": benchmark["summary"]["best_regressor_model"],
        "rl_algorithm": rl_results["best_algorithm"],
        "target_emotion": args.target_emotion,
        "convergence_deg": args.convergence_deg,
        "output_dir": os.path.abspath(output_dir),
    }
    with open(os.path.join(output_dir, "overall_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("=" * 76)
    print("Full EEG Emotion + RL Experiment Complete")
    print("=" * 76)
    print(f"Best classifier : {summary['classifier']}")
    print(f"Best regressor  : {summary['regressor']}")
    print(f"Best RL agent   : {summary['rl_algorithm']}")
    print(f"Target emotion  : {summary['target_emotion']}")
    print(f"Convergence     : <= {summary['convergence_deg']:.1f} deg")
    print(f"Artifacts saved : {summary['output_dir']}")


if __name__ == "__main__":
    main()
