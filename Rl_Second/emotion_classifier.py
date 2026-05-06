import argparse
import json
import os
import warnings
from typing import Dict

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from music_emotion_data import (
    DEFAULT_DEAP_ROOT,
    DEFAULT_METADATA_ROOT,
    RATING_MAX,
    RATING_MIN,
    angular_distance_deg,
    build_canonical_emotion_space,
    extract_eeg_trial_features,
    extract_eeg_trial_sequence,
    label_vector_to_emotion,
    load_eeg_feature_dataset,
    normalize_va,
    save_emotion_space,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

try:  # Optional torch support for a real LSTM classifier.
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False


class TorchLSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X, y):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is required for TorchLSTMClassifier.")

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        class_to_idx = {label: idx for idx, label in enumerate(self.classes_)}
        y_encoded = np.asarray([class_to_idx[label] for label in y], dtype=np.int64)

        torch.manual_seed(self.random_state)
        self.input_size_ = int(X.shape[-1])
        self.model_ = nn.Sequential()
        self.lstm_ = nn.LSTM(
            input_size=self.input_size_,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head_ = nn.Linear(self.hidden_size, len(self.classes_))
        optimizer = torch.optim.Adam(
            list(self.lstm_.parameters()) + list(self.head_.parameters()),
            lr=self.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_encoded, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.lstm_.train()
        self.head_.train()
        for _ in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                output, _ = self.lstm_(batch_X)
                logits = self.head_(output[:, -1, :])
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return self

    def _forward_logits(self, X):
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        self.lstm_.eval()
        self.head_.eval()
        with torch.no_grad():
            output, _ = self.lstm_(X_tensor)
            return self.head_(output[:, -1, :]).cpu().numpy()

    def predict_proba(self, X):
        logits = self._forward_logits(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        indices = np.argmax(probabilities, axis=1)
        return self.classes_[indices]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def get_feature_matrix(dataset: dict, feature_key: str, mask=None) -> np.ndarray:
    if feature_key == "X_tabular":
        X = dataset["X_tabular"]
    elif feature_key == "X_sequence":
        X = dataset["X_sequence"]
    elif feature_key == "X_sequence_flat":
        X = dataset["X_sequence"].reshape(dataset["X_sequence"].shape[0], -1)
    else:
        raise KeyError(f"Unknown feature key: {feature_key}")

    return X if mask is None else X[mask]


def build_classifier_registry(random_state: int = 42, include_lstm: bool = True) -> Dict[str, dict]:
    registry = {
        "knn_classifier": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=9, weights="distance")),
                ]
            ),
        },
        "logistic_regression": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=3000,
                            multi_class="auto",
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        },
        "svm_rbf": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            kernel="rbf",
                            C=3.0,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        },
        "mlp_classifier": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            early_stopping=True,
                            max_iter=200,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        },
        "sequence_mlp_classifier": {
            "feature_key": "X_sequence_flat",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            early_stopping=True,
                            max_iter=200,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        },
    }

    if include_lstm and TORCH_AVAILABLE:
        registry["lstm_classifier"] = {
            "feature_key": "X_sequence",
            "build": lambda: TorchLSTMClassifier(random_state=random_state),
        }

    return registry


def build_regressor_registry(random_state: int = 42) -> Dict[str, dict]:
    return {
        "knn_regressor": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=9, weights="distance")),
                ]
            ),
        },
        "ridge_regression": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=2.0, random_state=random_state)),
                ]
            ),
        },
        "svr_rbf": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MultiOutputRegressor(
                            SVR(kernel="rbf", C=8.0, epsilon=0.15, gamma="scale")
                        ),
                    ),
                ]
            ),
        },
        "mlp_regressor": {
            "feature_key": "X_tabular",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            early_stopping=True,
                            max_iter=200,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        },
        "sequence_mlp_regressor": {
            "feature_key": "X_sequence_flat",
            "build": lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(128, 64),
                            activation="relu",
                            early_stopping=True,
                            max_iter=200,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        },
    }


def plot_metric_boxplot(frame: pd.DataFrame, x: str, y: str, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=frame, x=x, y=y, ax=ax, color="#8ecae6")
    sns.stripplot(data=frame, x=x, y=y, ax=ax, color="#023047", alpha=0.5, size=3)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_confusion(y_true, y_pred, labels, title: str, save_path: str) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_subject_accuracy(frame: pd.DataFrame, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ordered = frame.sort_values("subject_id")
    ax.bar(ordered["subject_id"].astype(str), ordered["accuracy"], color="#219ebc")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_regression_scatter(predictions: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(predictions["valence"], predictions["predicted_valence"], alpha=0.45, s=18)
    axes[0].plot([RATING_MIN, RATING_MAX], [RATING_MIN, RATING_MAX], "k--", lw=1)
    axes[0].set_title("Valence")
    axes[0].set_xlabel("True")
    axes[0].set_ylabel("Predicted")
    axes[1].scatter(predictions["arousal"], predictions["predicted_arousal"], alpha=0.45, s=18)
    axes[1].plot([RATING_MIN, RATING_MAX], [RATING_MIN, RATING_MAX], "k--", lw=1)
    axes[1].set_title("Arousal")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Predicted")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()


def save_model_artifact(artifact: dict, save_path: str) -> None:
    joblib.dump(artifact, save_path)


def load_model_artifact(save_path: str) -> dict:
    artifact = joblib.load(save_path)
    artifact["emotion_space"] = {
        label: tuple(values) for label, values in artifact["emotion_space"].items()
    }
    return artifact


def evaluate_classifier_models(dataset: dict, output_dir: str, random_state: int = 42, include_lstm: bool = True) -> dict:
    classifier_dir = ensure_dir(os.path.join(output_dir, "classifier"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))
    emotion_space = dataset["emotion_space"]
    subject_ids = dataset["subject_ids"]
    y_true = dataset["y_emotion"]
    registry = build_classifier_registry(random_state=random_state, include_lstm=include_lstm)

    fold_rows = []
    prediction_rows = []

    for model_name, spec in registry.items():
        print(f"[Classifier] Evaluating {model_name}")
        feature_key = spec["feature_key"]
        for holdout_subject in np.unique(subject_ids):
            train_mask = subject_ids != holdout_subject
            test_mask = subject_ids == holdout_subject

            model = spec["build"]()
            X_train = get_feature_matrix(dataset, feature_key, train_mask)
            X_test = get_feature_matrix(dataset, feature_key, test_mask)
            y_train = y_true[train_mask]
            y_test = y_true[test_mask]

            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test))

            fold_rows.append(
                {
                    "model_name": model_name,
                    "subject_id": int(holdout_subject),
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                    "f1_weighted": float(
                        f1_score(y_test, y_pred, average="weighted", zero_division=0)
                    ),
                }
            )

            test_indices = np.flatnonzero(test_mask)
            for local_idx, sample_idx in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "model_name": model_name,
                        "subject_id": int(dataset["subject_ids"][sample_idx]),
                        "trial": int(dataset["trial_numbers"][sample_idx]),
                        "experiment_id": int(dataset["experiment_ids"][sample_idx]),
                        "true_emotion": str(y_test[local_idx]),
                        "predicted_emotion": str(y_pred[local_idx]),
                        "correct": bool(y_test[local_idx] == y_pred[local_idx]),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    summary_df = (
        fold_df.groupby("model_name", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_f1_weighted=("f1_weighted", "mean"),
        )
        .sort_values(
            ["mean_accuracy", "mean_balanced_accuracy", "mean_f1_weighted"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    best_model_name = str(summary_df.iloc[0]["model_name"])
    best_spec = registry[best_model_name]
    best_predictions = predictions_df[predictions_df["model_name"] == best_model_name].copy()
    best_subject_accuracy = (
        best_predictions.groupby("subject_id", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )

    fold_df.to_csv(os.path.join(classifier_dir, "classifier_fold_metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(classifier_dir, "classifier_predictions.csv"), index=False)
    summary_df.to_csv(os.path.join(classifier_dir, "classifier_summary.csv"), index=False)
    best_predictions.to_csv(
        os.path.join(classifier_dir, "best_classifier_predictions.csv"), index=False
    )
    best_subject_accuracy.to_csv(
        os.path.join(classifier_dir, "best_classifier_subject_accuracy.csv"),
        index=False,
    )

    plot_metric_boxplot(
        fold_df,
        x="model_name",
        y="accuracy",
        title="Classifier LOSO Accuracy by Model",
        save_path=os.path.join(classifier_dir, "classifier_accuracy_boxplot.png"),
    )
    plot_confusion(
        best_predictions["true_emotion"],
        best_predictions["predicted_emotion"],
        labels=list(emotion_space.keys()),
        title=f"Best Classifier Confusion Matrix: {best_model_name}",
        save_path=os.path.join(classifier_dir, "best_classifier_confusion.png"),
    )
    plot_subject_accuracy(
        best_subject_accuracy,
        title=f"Best Classifier Subject Accuracy: {best_model_name}",
        save_path=os.path.join(classifier_dir, "best_classifier_subject_accuracy.png"),
    )

    full_model = best_spec["build"]()
    full_model.fit(get_feature_matrix(dataset, best_spec["feature_key"]), y_true)
    classifier_artifact = {
        "task": "classifier",
        "model_name": best_model_name,
        "feature_key": best_spec["feature_key"],
        "model": full_model,
        "emotion_space": {
            label: [float(values[0]), float(values[1])]
            for label, values in emotion_space.items()
        },
        "feature_info": dataset["feature_info"],
    }
    save_model_artifact(classifier_artifact, os.path.join(models_dir, "best_classifier.joblib"))

    return {
        "summary": summary_df,
        "fold_metrics": fold_df,
        "predictions": predictions_df,
        "best_model_name": best_model_name,
        "artifact_path": os.path.join(models_dir, "best_classifier.joblib"),
    }


def evaluate_regressor_models(dataset: dict, output_dir: str, random_state: int = 42) -> dict:
    regressor_dir = ensure_dir(os.path.join(output_dir, "regressor"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))
    emotion_space = dataset["emotion_space"]
    subject_ids = dataset["subject_ids"]
    y_true = np.column_stack([dataset["valence"], dataset["arousal"]]).astype(np.float32)
    registry = build_regressor_registry(random_state=random_state)

    fold_rows = []
    prediction_rows = []

    for model_name, spec in registry.items():
        print(f"[Regressor] Evaluating {model_name}")
        feature_key = spec["feature_key"]
        for holdout_subject in np.unique(subject_ids):
            train_mask = subject_ids != holdout_subject
            test_mask = subject_ids == holdout_subject

            model = spec["build"]()
            X_train = get_feature_matrix(dataset, feature_key, train_mask)
            X_test = get_feature_matrix(dataset, feature_key, test_mask)
            y_train = y_true[train_mask]
            y_test = y_true[test_mask]

            model.fit(X_train, y_train)
            y_pred = np.asarray(model.predict(X_test), dtype=np.float32)
            y_pred = np.clip(y_pred, RATING_MIN, RATING_MAX)

            predicted_emotions = np.asarray(
                [label_vector_to_emotion(row, emotion_space) for row in y_pred]
            )
            true_emotions = dataset["y_emotion"][test_mask]
            angle_errors = np.asarray(
                [
                    angular_distance_deg(normalize_va(row_true), normalize_va(row_pred))
                    for row_true, row_pred in zip(y_test, y_pred)
                ],
                dtype=np.float32,
            )

            fold_rows.append(
                {
                    "model_name": model_name,
                    "subject_id": int(holdout_subject),
                    "emotion_accuracy": float(accuracy_score(true_emotions, predicted_emotions)),
                    "valence_mae": float(mean_absolute_error(y_test[:, 0], y_pred[:, 0])),
                    "arousal_mae": float(mean_absolute_error(y_test[:, 1], y_pred[:, 1])),
                    "mean_angle_error_deg": float(np.mean(angle_errors)),
                }
            )

            test_indices = np.flatnonzero(test_mask)
            for local_idx, sample_idx in enumerate(test_indices):
                prediction_rows.append(
                    {
                        "model_name": model_name,
                        "subject_id": int(dataset["subject_ids"][sample_idx]),
                        "trial": int(dataset["trial_numbers"][sample_idx]),
                        "experiment_id": int(dataset["experiment_ids"][sample_idx]),
                        "valence": float(y_test[local_idx, 0]),
                        "arousal": float(y_test[local_idx, 1]),
                        "predicted_valence": float(y_pred[local_idx, 0]),
                        "predicted_arousal": float(y_pred[local_idx, 1]),
                        "true_emotion": str(true_emotions[local_idx]),
                        "predicted_emotion": str(predicted_emotions[local_idx]),
                        "angle_error_deg": float(angle_errors[local_idx]),
                        "correct": bool(true_emotions[local_idx] == predicted_emotions[local_idx]),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    summary_df = (
        fold_df.groupby("model_name", as_index=False)
        .agg(
            mean_emotion_accuracy=("emotion_accuracy", "mean"),
            std_emotion_accuracy=("emotion_accuracy", "std"),
            mean_valence_mae=("valence_mae", "mean"),
            mean_arousal_mae=("arousal_mae", "mean"),
            mean_angle_error_deg=("mean_angle_error_deg", "mean"),
        )
        .sort_values(
            ["mean_angle_error_deg", "mean_emotion_accuracy", "mean_valence_mae", "mean_arousal_mae"],
            ascending=[True, False, True, True],
        )
        .reset_index(drop=True)
    )

    best_model_name = str(summary_df.iloc[0]["model_name"])
    best_spec = registry[best_model_name]
    best_predictions = predictions_df[predictions_df["model_name"] == best_model_name].copy()
    best_subject_accuracy = (
        best_predictions.groupby("subject_id", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )

    fold_df.to_csv(os.path.join(regressor_dir, "regressor_fold_metrics.csv"), index=False)
    predictions_df.to_csv(os.path.join(regressor_dir, "regressor_predictions.csv"), index=False)
    summary_df.to_csv(os.path.join(regressor_dir, "regressor_summary.csv"), index=False)
    best_predictions.to_csv(
        os.path.join(regressor_dir, "best_regressor_predictions.csv"), index=False
    )
    best_subject_accuracy.to_csv(
        os.path.join(regressor_dir, "best_regressor_subject_accuracy.csv"),
        index=False,
    )

    plot_metric_boxplot(
        fold_df,
        x="model_name",
        y="emotion_accuracy",
        title="Regressor-Derived Emotion Accuracy by Model",
        save_path=os.path.join(regressor_dir, "regressor_emotion_accuracy_boxplot.png"),
    )
    plot_metric_boxplot(
        fold_df,
        x="model_name",
        y="mean_angle_error_deg",
        title="Regressor Angular Error by Model",
        save_path=os.path.join(regressor_dir, "regressor_angle_error_boxplot.png"),
    )
    plot_subject_accuracy(
        best_subject_accuracy,
        title=f"Best Regressor Subject Emotion Accuracy: {best_model_name}",
        save_path=os.path.join(regressor_dir, "best_regressor_subject_accuracy.png"),
    )
    plot_regression_scatter(
        best_predictions,
        save_path=os.path.join(regressor_dir, "best_regressor_true_vs_predicted.png"),
    )

    full_model = best_spec["build"]()
    full_model.fit(get_feature_matrix(dataset, best_spec["feature_key"]), y_true)
    regressor_artifact = {
        "task": "regressor",
        "model_name": best_model_name,
        "feature_key": best_spec["feature_key"],
        "model": full_model,
        "emotion_space": {
            label: [float(values[0]), float(values[1])]
            for label, values in emotion_space.items()
        },
        "feature_info": dataset["feature_info"],
    }
    save_model_artifact(regressor_artifact, os.path.join(models_dir, "best_regressor.joblib"))

    return {
        "summary": summary_df,
        "fold_metrics": fold_df,
        "predictions": predictions_df,
        "best_model_name": best_model_name,
        "artifact_path": os.path.join(models_dir, "best_regressor.joblib"),
    }


def run_benchmark(
    metadata_root: str = DEFAULT_METADATA_ROOT,
    eeg_root: str = DEFAULT_DEAP_ROOT,
    output_dir: str = "output",
    eeg_channels: int = 32,
    random_state: int = 42,
    include_lstm: bool = True,
) -> dict:
    output_dir = ensure_dir(output_dir)
    models_dir = ensure_dir(os.path.join(output_dir, "models"))
    dataset = load_eeg_feature_dataset(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        eeg_channels=eeg_channels,
    )

    dataset["clip_catalog"].to_csv(os.path.join(models_dir, "clip_catalog.csv"), index=False)
    save_emotion_space(dataset["emotion_space"], os.path.join(models_dir, "emotion_space.json"))
    with open(os.path.join(models_dir, "feature_info.json"), "w", encoding="utf-8") as fh:
        json.dump(dataset["feature_info"], fh, indent=2)

    classifier_results = evaluate_classifier_models(
        dataset,
        output_dir=output_dir,
        random_state=random_state,
        include_lstm=include_lstm,
    )
    regressor_results = evaluate_regressor_models(
        dataset,
        output_dir=output_dir,
        random_state=random_state,
    )

    summary = {
        "best_classifier_model": classifier_results["best_model_name"],
        "best_regressor_model": regressor_results["best_model_name"],
        "classifier_artifact": classifier_results["artifact_path"],
        "regressor_artifact": regressor_results["artifact_path"],
        "torch_available": bool(TORCH_AVAILABLE),
    }
    with open(os.path.join(models_dir, "model_selection.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return {
        "dataset": dataset,
        "classifier": classifier_results,
        "regressor": regressor_results,
        "summary": summary,
    }


def _features_from_raw_eeg(eeg_trial: np.ndarray, feature_info: dict) -> dict:
    tabular = extract_eeg_trial_features(
        eeg_trial,
        eeg_channels=feature_info["eeg_channels"],
        baseline_seconds=feature_info["baseline_seconds"],
        sampling_rate=feature_info["sampling_rate"],
    )[None, :]
    sequence = extract_eeg_trial_sequence(
        eeg_trial,
        eeg_channels=feature_info["eeg_channels"],
        baseline_seconds=feature_info["baseline_seconds"],
        sampling_rate=feature_info["sampling_rate"],
        n_windows=feature_info["n_windows"],
    )[None, :, :]
    return {
        "X_tabular": tabular,
        "X_sequence": sequence,
        "X_sequence_flat": sequence.reshape(sequence.shape[0], -1),
    }


def predict_eeg_sample(eeg_trial: np.ndarray, artifacts_dir: str = "output") -> dict:
    classifier_artifact = load_model_artifact(os.path.join(artifacts_dir, "models", "best_classifier.joblib"))
    regressor_artifact = load_model_artifact(os.path.join(artifacts_dir, "models", "best_regressor.joblib"))
    feature_info = classifier_artifact["feature_info"]
    sample = _features_from_raw_eeg(eeg_trial, feature_info)

    classifier_input = sample[classifier_artifact["feature_key"]]
    regressor_input = sample[regressor_artifact["feature_key"]]
    classifier_emotion = str(classifier_artifact["model"].predict(classifier_input)[0])
    predicted_va = np.asarray(regressor_artifact["model"].predict(regressor_input)[0], dtype=np.float32)
    predicted_va = np.clip(predicted_va, RATING_MIN, RATING_MAX)
    regressor_emotion = label_vector_to_emotion(predicted_va, regressor_artifact["emotion_space"])

    return {
        "classifier_model": classifier_artifact["model_name"],
        "regressor_model": regressor_artifact["model_name"],
        "classifier_emotion": classifier_emotion,
        "predicted_valence": float(predicted_va[0]),
        "predicted_arousal": float(predicted_va[1]),
        "regressor_emotion": regressor_emotion,
        "predicted_state_vector": normalize_va(predicted_va).astype(np.float32),
    }


def predict_subject_trial(
    dataset: dict,
    subject_id: int,
    trial_number: int,
    artifacts_dir: str = "output",
) -> dict:
    mask = (dataset["subject_ids"] == int(subject_id)) & (
        dataset["trial_numbers"] == int(trial_number)
    )
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise KeyError(f"Subject {subject_id} trial {trial_number} was not found.")

    sample_index = int(indices[0])
    eeg_features = {
        "X_tabular": dataset["X_tabular"][sample_index : sample_index + 1],
        "X_sequence": dataset["X_sequence"][sample_index : sample_index + 1],
        "X_sequence_flat": dataset["X_sequence"][sample_index : sample_index + 1].reshape(1, -1),
    }

    classifier_artifact = load_model_artifact(os.path.join(artifacts_dir, "models", "best_classifier.joblib"))
    regressor_artifact = load_model_artifact(os.path.join(artifacts_dir, "models", "best_regressor.joblib"))

    classifier_prediction = str(
        classifier_artifact["model"].predict(eeg_features[classifier_artifact["feature_key"]])[0]
    )
    predicted_va = np.asarray(
        regressor_artifact["model"].predict(eeg_features[regressor_artifact["feature_key"]])[0],
        dtype=np.float32,
    )
    predicted_va = np.clip(predicted_va, RATING_MIN, RATING_MAX)
    regressor_emotion = label_vector_to_emotion(predicted_va, regressor_artifact["emotion_space"])

    return {
        "subject_id": int(dataset["subject_ids"][sample_index]),
        "trial": int(dataset["trial_numbers"][sample_index]),
        "experiment_id": int(dataset["experiment_ids"][sample_index]),
        "true_emotion": str(dataset["y_emotion"][sample_index]),
        "subjective_emotion": str(dataset["y_subjective_emotion"][sample_index]),
        "classifier_emotion": classifier_prediction,
        "regressor_emotion": regressor_emotion,
        "true_valence": float(dataset["valence"][sample_index]),
        "true_arousal": float(dataset["arousal"][sample_index]),
        "predicted_valence": float(predicted_va[0]),
        "predicted_arousal": float(predicted_va[1]),
        "predicted_state_vector": normalize_va(predicted_va).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="LOSO EEG emotion classifier and valence/arousal regressor benchmark."
    )
    parser.add_argument("--data_root", default=DEFAULT_METADATA_ROOT)
    parser.add_argument("--eeg_root", default=DEFAULT_DEAP_ROOT)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--eeg_channels", type=int, default=32)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--skip_lstm", action="store_true")
    parser.add_argument("--subject_id", type=int)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()

    if args.subject_id is not None:
        dataset = load_eeg_feature_dataset(
            metadata_root=args.data_root,
            eeg_root=args.eeg_root,
            eeg_channels=args.eeg_channels,
        )
        prediction = predict_subject_trial(
            dataset,
            subject_id=args.subject_id,
            trial_number=args.trial,
            artifacts_dir=args.output_dir,
        )
        print("=" * 72)
        print("Saved EEG Emotion Pipeline Prediction")
        print("=" * 72)
        print(f"Subject / trial       : S{prediction['subject_id']:02d} / {prediction['trial']}")
        print(f"Trial index           : {prediction['experiment_id']}")
        print(f"True EEG emotion      : {prediction['true_emotion']}")
        print(f"Classifier emotion    : {prediction['classifier_emotion']}")
        print(f"Regressor emotion     : {prediction['regressor_emotion']}")
        print(f"True valence/arousal  : {prediction['true_valence']:.3f} / {prediction['true_arousal']:.3f}")
        print(f"Pred valence/arousal  : {prediction['predicted_valence']:.3f} / {prediction['predicted_arousal']:.3f}")
        return

    results = run_benchmark(
        metadata_root=args.data_root,
        eeg_root=args.eeg_root,
        output_dir=args.output_dir,
        eeg_channels=args.eeg_channels,
        random_state=args.random_state,
        include_lstm=not args.skip_lstm,
    )
    print("=" * 72)
    print("EEG Emotion Benchmark")
    print("=" * 72)
    print(f"Best classifier : {results['summary']['best_classifier_model']}")
    print(f"Best regressor  : {results['summary']['best_regressor_model']}")
    print(f"Output folder   : {os.path.abspath(args.output_dir)}")
    if not TORCH_AVAILABLE:
        print("Torch was not available, so the optional LSTM benchmark was skipped.")


if __name__ == "__main__":
    main()
