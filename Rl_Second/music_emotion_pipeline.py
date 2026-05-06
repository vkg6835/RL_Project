import argparse
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import numpy as np
import pandas as pd

from emotion_classifier import predict_eeg_sample, predict_subject_trial
from music_emotion_data import DEFAULT_DEAP_ROOT, DEFAULT_METADATA_ROOT, load_eeg_feature_dataset
from music_emotion_rl import recommend_playlist_from_state


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def playlist_to_frame(playlist: list) -> pd.DataFrame:
    return pd.DataFrame(playlist)


def main():
    parser = argparse.ArgumentParser(
        description="Run the saved EEG -> emotion -> RL music recommendation pipeline."
    )
    parser.add_argument("--data_root", default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--eeg_root", default=DEFAULT_DEAP_ROOT)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--subject_id", type=int)
    parser.add_argument("--trial", type=int, default=1)
    parser.add_argument("--eeg_npy")
    parser.add_argument(
        "--algorithm",
        choices=[
            "q_learning",
            "sarsa",
            "expected_sarsa",
            "double_q_learning",
            "deep_q_learning",
        ],
    )
    parser.add_argument("--playlist_length", type=int)
    args = parser.parse_args()

    pipeline_dir = ensure_dir(os.path.join(args.output_dir, "pipeline"))

    if args.eeg_npy:
        eeg_trial = np.load(args.eeg_npy)
        if eeg_trial.ndim != 2:
            raise ValueError("The .npy EEG input must have shape (channels, samples).")
        prediction = predict_eeg_sample(eeg_trial, artifacts_dir=args.output_dir)
        subject_label = "custom"
        trial_label = "custom"
    elif args.subject_id is not None:
        dataset = load_eeg_feature_dataset(
            metadata_root=args.data_root,
            eeg_root=args.eeg_root,
        )
        prediction = predict_subject_trial(
            dataset,
            subject_id=args.subject_id,
            trial_number=args.trial,
            artifacts_dir=args.output_dir,
        )
        subject_label = f"s{int(args.subject_id):02d}"
        trial_label = f"t{int(args.trial):02d}"
    else:
        raise ValueError("Provide either --subject_id/--trial or --eeg_npy.")

    recommendation = recommend_playlist_from_state(
        prediction["predicted_state_vector"],
        artifacts_dir=args.output_dir,
        algorithm_name=args.algorithm,
        playlist_length=args.playlist_length,
    )

    playlist_df = playlist_to_frame(recommendation["playlist"])
    playlist_path = os.path.join(
        pipeline_dir,
        f"playlist_{subject_label}_{trial_label}_{recommendation['algorithm']}.csv",
    )
    summary_path = os.path.join(
        pipeline_dir,
        f"pipeline_summary_{subject_label}_{trial_label}_{recommendation['algorithm']}.json",
    )
    playlist_df.to_csv(playlist_path, index=False)

    summary_payload = {
        "prediction": {
            key: (
                value.tolist()
                if isinstance(value, np.ndarray)
                else float(value)
                if isinstance(value, np.floating)
                else int(value)
                if isinstance(value, np.integer)
                else value
            )
            for key, value in prediction.items()
        },
        "recommendation": recommendation,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary_payload, fh, indent=2)

    print("=" * 76)
    print("Saved EEG -> Emotion -> RL Pipeline")
    print("=" * 76)
    if args.subject_id is not None:
        print(f"Subject / trial       : S{int(args.subject_id):02d} / {int(args.trial)}")
        print(f"True EEG emotion      : {prediction['true_emotion']}")
    else:
        print(f"Custom EEG input      : {os.path.abspath(args.eeg_npy)}")
    print(f"Classifier emotion    : {prediction.get('classifier_emotion', prediction.get('classifier_model', ''))}")
    print(f"Regressor emotion     : {prediction['regressor_emotion']}")
    print(f"Predicted valence     : {prediction['predicted_valence']:.3f}")
    print(f"Predicted arousal     : {prediction['predicted_arousal']:.3f}")
    print(f"RL algorithm          : {recommendation['algorithm']}")
    print(f"RL target emotion     : {recommendation['target_emotion']}")
    print(f"Initial error         : {recommendation['initial_error_deg']:.2f} deg")
    print(f"Final error           : {recommendation['final_error_deg']:.2f} deg")
    print(f"Converged <= 20 deg   : {recommendation['converged']}")
    print(f"Playlist csv          : {playlist_path}")
    print(f"Summary json          : {summary_path}")
    print("\nEmotion path:")
    print("  " + " -> ".join(recommendation["emotion_path"]))
    print("\nPlaylist:")
    for row in playlist_df.itertuples(index=False):
        print(
            f"  {row.rank}. clip {int(row.experiment_id):02d} | "
            f"{row.artist} - {row.title} [{row.canonical_emotion}]"
        )


if __name__ == "__main__":
    main()
