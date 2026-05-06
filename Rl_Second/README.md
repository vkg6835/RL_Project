# EEG Emotion Classification and RL Music Recommendation

This repo now runs the full DEAP workflow end to end with leave-one-subject-out (LOSO) validation over all 32 subjects:

1. EEG features are extracted from every DEAP trial in `data/data_preprocessed_python/`.
2. EEG supervision comes directly from the DEAP `.dat` labels for valence/arousal.
3. `participant_ratings.csv` and `video_list.csv` are used to build the music clip catalog and RL action metadata through `Experiment_id`.
4. Every emotion state is mapped into a fixed 8-emotion label space.
5. Multiple emotion classifiers and valence/arousal regressors are compared with LOSO.
6. The best classifier and best regressor are trained on the full dataset and saved.
7. RL agents start from the held-out EEG prediction instead of a random hard-coded emotion.
8. Q-Learning, SARSA, Expected SARSA, Double Q-Learning, and Deep Q-Learning are benchmarked and saved.

## Important data note

`participant_ratings.csv` is not aligned row-for-row with the DEAP `.dat` trial order, so it is no longer used as the EEG training target.

- EEG models use the per-trial labels stored in each `sXX.dat` file.
- Music recommendation metadata still comes from `video_list.csv` and the aggregated clip ratings.
- This avoids training EEG against mismatched labels.

## 8-emotion space

The pipeline uses 8 canonical emotions on the valence-arousal circumplex:

- `happy`
- `exciting`
- `shock`
- `terrible`
- `melancholy`
- `sentimental`
- `mellow`
- `love`

This is deterministic and shared by the classifier, regressor, and RL code. The convergence rule for RL is now strictly:

- converged only if final angular error `<= 20` degrees

All angle calculations are done in degrees in the reporting layer.

## Main scripts

- `music_emotion_data.py`
  Shared metadata join, 8-emotion mapping, EEG feature extraction, and LOSO-ready dataset loading.
- `emotion_classifier.py`
  LOSO benchmark for classifiers and regressors, plus saved inference artifacts.
- `music_emotion_rl.py`
  RL training/evaluation module for Q-Learning, SARSA, Expected SARSA, Double Q-Learning, and Deep Q-Learning.
- `research_experiment.py`
  Runs the full benchmark pipeline and saves all outputs to `output/`.
- `music_emotion_pipeline.py`
  Loads the saved models and predicts emotion from EEG, then generates a playlist with a saved RL policy.
- `run_all.py`
  Convenience launcher for `research_experiment.py`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- `torch` is listed for the optional LSTM classifier.
- If `torch` is not installed, the rest of the workflow still runs and the README/output will say that the LSTM benchmark was skipped.

## Run the full LOSO benchmark

```bash
./.venv/bin/python research_experiment.py
```

Useful options:

```bash
./.venv/bin/python research_experiment.py --output_dir output --target_emotion happy
./.venv/bin/python research_experiment.py --episodes 15 --playlist_length 6
./.venv/bin/python research_experiment.py --skip_lstm
```

You can also launch the same workflow with:

```bash
./.venv/bin/python run_all.py
```

## Predict one DEAP trial with the saved pipeline

Run the full benchmark first so the saved artifacts exist, then:

```bash
./.venv/bin/python music_emotion_pipeline.py --subject_id 1 --trial 1
```

Choose a specific RL policy:

```bash
./.venv/bin/python music_emotion_pipeline.py --subject_id 1 --trial 1 --algorithm deep_q_learning
```

Use a custom EEG trial stored as `.npy` with shape `(channels, samples)`:

```bash
./.venv/bin/python music_emotion_pipeline.py --eeg_npy /path/to/eeg_trial.npy
```

## Run only the classifier/regressor benchmark

```bash
./.venv/bin/python emotion_classifier.py
```

Predict a saved DEAP sample with the saved classifier/regressor artifacts:

```bash
./.venv/bin/python emotion_classifier.py --subject_id 1 --trial 1
```

## Output structure

After `research_experiment.py` runs, the main artifacts are:

- `output/models/clip_catalog.csv`
- `output/models/emotion_space.json`
- `output/models/feature_info.json`
- `output/models/best_classifier.joblib`
- `output/models/best_regressor.joblib`
- `output/models/policy_q_learning.joblib`
- `output/models/policy_sarsa.joblib`
- `output/models/policy_expected_sarsa.joblib`
- `output/models/policy_double_q_learning.joblib`
- `output/models/policy_deep_q_learning.joblib`
- `output/models/model_selection.json`
- `output/models/rl_model_selection.json`

Classifier outputs:

- `output/classifier/classifier_summary.csv`
- `output/classifier/classifier_fold_metrics.csv`
- `output/classifier/classifier_predictions.csv`
- `output/classifier/best_classifier_confusion.png`
- `output/classifier/best_classifier_subject_accuracy.png`
- `output/classifier/classifier_accuracy_boxplot.png`

Regressor outputs:

- `output/regressor/regressor_summary.csv`
- `output/regressor/regressor_fold_metrics.csv`
- `output/regressor/regressor_predictions.csv`
- `output/regressor/best_regressor_predictions.csv`
- `output/regressor/regressor_emotion_accuracy_boxplot.png`
- `output/regressor/regressor_angle_error_boxplot.png`
- `output/regressor/best_regressor_true_vs_predicted.png`

RL outputs:

- `output/rl/rl_summary.csv`
- `output/rl/rl_trial_results.csv`
- `output/rl/rl_training_history.csv`
- `output/rl/rl_algorithm_comparison.png`
- `output/rl/<algorithm>_training_curves.png`
- `output/rl/<algorithm>_mean_error_by_step.png`
- `output/rl/<algorithm>_sample_trajectories.png`

Pipeline outputs:

- `output/pipeline/playlist_<sample>_<algorithm>.csv`
- `output/pipeline/pipeline_summary_<sample>_<algorithm>.json`

## What changed

- The old random start emotion inside the RL scripts is no longer used in the new workflow.
- EEG supervision now uses the DEAP `.dat` labels instead of the misaligned `participant_ratings.csv` trial rows.
- LOSO is now run across all 32 subjects using the full dataset.
- The classifier side now compares multiple direct classifiers and multiple regressors.
- The RL side now evaluates 5 algorithms and uses held-out EEG predictions as the start state.
- All major artifacts, metrics tables, and plots are written to `output/`.
