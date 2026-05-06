import json
import os
import pickle
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

try:
    from scipy.signal import welch
except Exception:  # pragma: no cover - graceful fallback if scipy is absent
    welch = None


RATING_MIN = 1.0
RATING_MAX = 9.0
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_METADATA_ROOT = os.path.join(PROJECT_ROOT, "data", "Metacsv")
DEFAULT_DEAP_ROOT = os.path.join(PROJECT_ROOT, "data", "data_preprocessed_python")
DEFAULT_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "output")
DEFAULT_CLIP_CATALOG = os.path.join(DEFAULT_METADATA_ROOT, "video_list.csv")
DEFAULT_RAW_EMOTION_CSV = os.path.join(DEFAULT_METADATA_ROOT, "emotion_summary.csv")

# Eight fixed circumplex sectors used everywhere in the new pipeline.
CANONICAL_EMOTION_LAYOUT = (
    ("happy", 22.5),
    ("exciting", 67.5),
    ("shock", 112.5),
    ("terrible", 157.5),
    ("melancholy", 202.5),
    ("sentimental", 247.5),
    ("mellow", 292.5),
    ("love", 337.5),
)


def resolve_metadata_root(data_root: Optional[str] = None) -> str:
    return os.path.abspath(data_root or DEFAULT_METADATA_ROOT)


def resolve_eeg_root(eeg_root: Optional[str] = None) -> str:
    return os.path.abspath(eeg_root or DEFAULT_DEAP_ROOT)


def resolve_output_root(output_root: Optional[str] = None) -> str:
    return os.path.abspath(output_root or DEFAULT_OUTPUT_ROOT)


def normalize_va(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return 2.0 * (arr - RATING_MIN) / (RATING_MAX - RATING_MIN) - 1.0


def denormalize_va(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return 0.5 * (arr + 1.0) * (RATING_MAX - RATING_MIN) + RATING_MIN


def project_to_unit_circle(vec: Iterable[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / (np.linalg.norm(arr) + 1e-9)


def va_to_unit_vector(values: Iterable[float], normalized_input: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if not normalized_input:
        arr = normalize_va(arr)
    return project_to_unit_circle(arr)


def vector_angle_deg(vec: Iterable[float]) -> float:
    arr = project_to_unit_circle(vec)
    return float((np.degrees(np.arctan2(arr[1], arr[0])) + 360.0) % 360.0)


def angular_distance_deg(vec_a: Iterable[float], vec_b: Iterable[float]) -> float:
    a = project_to_unit_circle(vec_a)
    b = project_to_unit_circle(vec_b)
    cosine = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def build_canonical_emotion_space() -> Dict[str, tuple]:
    emotion_space = {}
    for label, angle_deg in CANONICAL_EMOTION_LAYOUT:
        radians = np.deg2rad(angle_deg)
        emotion_space[label] = (
            float(np.cos(radians)),
            float(np.sin(radians)),
        )
    return emotion_space


def serialize_emotion_space(emotion_space: Dict[str, Iterable[float]]) -> dict:
    return {label: [float(values[0]), float(values[1])] for label, values in emotion_space.items()}


def save_emotion_space(emotion_space: Dict[str, Iterable[float]], save_path: str) -> None:
    payload = serialize_emotion_space(emotion_space)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def load_emotion_centers(csv_path: Optional[str] = None) -> Dict[str, tuple]:
    csv_path = csv_path or DEFAULT_RAW_EMOTION_CSV
    df = pd.read_csv(csv_path)
    values = normalize_va(df[["Valence", "Arousal"]].to_numpy(np.float32))

    centers = {}
    for emotion, vec in zip(df["Emotion"], values):
        centers[str(emotion).strip().lower()] = tuple(project_to_unit_circle(vec))
    return centers


def emotion_from_vector(
    vector: Iterable[float],
    emotion_centers: Optional[Dict[str, Iterable[float]]] = None,
) -> str:
    emotion_centers = emotion_centers or build_canonical_emotion_space()
    vec = project_to_unit_circle(vector)

    best_emotion = None
    best_cosine = -2.0
    for emotion, center in emotion_centers.items():
        center_vec = np.asarray(center, dtype=np.float32)
        cosine = float(np.dot(vec, center_vec))
        if cosine > best_cosine:
            best_cosine = cosine
            best_emotion = emotion
    return str(best_emotion)


def label_vector_to_emotion(
    values: Iterable[float],
    emotion_centers: Optional[Dict[str, Iterable[float]]] = None,
    normalized_input: bool = False,
) -> str:
    return emotion_from_vector(
        va_to_unit_vector(values, normalized_input=normalized_input),
        emotion_centers=emotion_centers,
    )


def nearest_raw_emotion_label(values: Iterable[float], raw_emotion_centers: Dict[str, Iterable[float]]) -> str:
    return label_vector_to_emotion(values, emotion_centers=raw_emotion_centers)


def canonicalize_emotion_label(
    emotion_label: Optional[str],
    raw_emotion_centers: Optional[Dict[str, Iterable[float]]] = None,
    canonical_emotion_space: Optional[Dict[str, Iterable[float]]] = None,
    fallback_va: Optional[Iterable[float]] = None,
) -> str:
    canonical_emotion_space = canonical_emotion_space or build_canonical_emotion_space()
    raw_emotion_centers = raw_emotion_centers or load_emotion_centers()

    label = str(emotion_label).strip().lower() if emotion_label is not None else ""
    if label and label != "nan":
        if label in canonical_emotion_space:
            return label
        if label in raw_emotion_centers:
            return emotion_from_vector(raw_emotion_centers[label], canonical_emotion_space)

    if fallback_va is None:
        raise KeyError("A fallback valence/arousal vector is required for missing emotion labels.")

    fallback_raw = nearest_raw_emotion_label(fallback_va, raw_emotion_centers)
    return emotion_from_vector(raw_emotion_centers[fallback_raw], canonical_emotion_space)


def load_clip_catalog(
    metadata_root: Optional[str] = None,
    raw_emotion_csv: Optional[str] = None,
) -> pd.DataFrame:
    metadata_root = resolve_metadata_root(metadata_root)
    raw_emotion_csv = raw_emotion_csv or os.path.join(metadata_root, "emotion_summary.csv")

    ratings = pd.read_csv(os.path.join(metadata_root, "participant_ratings.csv"))
    videos = pd.read_csv(os.path.join(metadata_root, "video_list.csv"))
    emotion_space = build_canonical_emotion_space()
    raw_emotion_centers = load_emotion_centers(raw_emotion_csv)

    videos = videos[videos["Experiment_id"].notna()].copy()
    videos["Experiment_id"] = videos["Experiment_id"].astype(np.int32)
    videos = (
        videos.sort_values(["Experiment_id", "Num_ratings"], ascending=[True, False])
        .drop_duplicates("Experiment_id")
        .reset_index(drop=True)
    )

    rating_summary = (
        ratings.groupby("Experiment_id", as_index=False)
        .agg(
            rating_mean_valence=("Valence", "mean"),
            rating_std_valence=("Valence", "std"),
            rating_mean_arousal=("Arousal", "mean"),
            rating_std_arousal=("Arousal", "std"),
            rating_mean_dominance=("Dominance", "mean"),
            rating_mean_liking=("Liking", "mean"),
            rating_mean_familiarity=("Familiarity", "mean"),
            num_subject_ratings=("Participant_id", "count"),
        )
    )

    clip_catalog = rating_summary.merge(videos, on="Experiment_id", how="left")

    representative_valence = clip_catalog["AVG_Valence"].fillna(clip_catalog["rating_mean_valence"])
    representative_arousal = clip_catalog["AVG_Arousal"].fillna(clip_catalog["rating_mean_arousal"])
    clip_catalog["clip_valence"] = representative_valence.astype(np.float32)
    clip_catalog["clip_arousal"] = representative_arousal.astype(np.float32)

    raw_labels = []
    canonical_labels = []
    clip_vectors = []
    clip_angles = []

    for row in clip_catalog.itertuples(index=False):
        va = [row.clip_valence, row.clip_arousal]
        raw_label = str(getattr(row, "Lastfm_tag", "")).strip().lower()
        canonical_label = canonicalize_emotion_label(
            raw_label,
            raw_emotion_centers=raw_emotion_centers,
            canonical_emotion_space=emotion_space,
            fallback_va=va,
        )
        clip_vector = va_to_unit_vector(va)
        raw_labels.append(
            raw_label
            if raw_label and raw_label != "nan"
            else nearest_raw_emotion_label(va, raw_emotion_centers)
        )
        canonical_labels.append(canonical_label)
        clip_vectors.append(clip_vector)
        clip_angles.append(vector_angle_deg(clip_vector))

    clip_catalog["raw_emotion"] = raw_labels
    clip_catalog["canonical_emotion"] = canonical_labels
    clip_catalog["clip_vector_x"] = [float(vector[0]) for vector in clip_vectors]
    clip_catalog["clip_vector_y"] = [float(vector[1]) for vector in clip_vectors]
    clip_catalog["clip_angle_deg"] = clip_angles

    return clip_catalog.sort_values("Experiment_id").reset_index(drop=True)


def load_subjects(metadata_root: Optional[str] = None) -> list:
    metadata_root = resolve_metadata_root(metadata_root)
    ratings_csv = os.path.join(metadata_root, "participant_ratings.csv")
    if not os.path.isfile(ratings_csv):
        return None

    df = pd.read_csv(ratings_csv).sort_values(["Participant_id", "Trial"])
    clip_ids = np.array(sorted(df["Experiment_id"].unique()), dtype=np.int32)
    clip_index = {clip_id: idx for idx, clip_id in enumerate(clip_ids.tolist())}

    subjects = []
    for participant_id, group in df.groupby("Participant_id", sort=True):
        clip_vectors = np.zeros((len(clip_ids), 2), dtype=np.float32)
        seen = np.zeros(len(clip_ids), dtype=bool)

        for row in group.itertuples(index=False):
            idx = clip_index[int(row.Experiment_id)]
            clip_vectors[idx] = normalize_va([row.Valence, row.Arousal])
            seen[idx] = True

        if not np.all(seen):
            missing = clip_ids[~seen].tolist()
            raise ValueError(
                f"Participant {participant_id} is missing Experiment_id values: {missing}"
            )

        subjects.append(
            {
                "participant_id": int(participant_id),
                "clip_ids": clip_ids.copy(),
                "trial_clip_ids": group["Experiment_id"].astype(np.int32).to_numpy(),
                "clip_vectors": clip_vectors,
            }
        )

    return subjects


def compute_prototype_actions(subjects: list) -> dict:
    if not subjects:
        raise ValueError("Need at least one subject to compute prototype actions.")

    clip_ids = subjects[0]["clip_ids"].copy()
    stack = np.stack([subject["clip_vectors"] for subject in subjects], axis=0)
    return {
        "clip_ids": clip_ids,
        "clip_vectors": stack.mean(axis=0).astype(np.float32),
    }


def _signal_statistics(eeg: np.ndarray) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=np.float32)
    mean = eeg.mean(axis=1)
    std = eeg.std(axis=1)
    minimum = eeg.min(axis=1)
    maximum = eeg.max(axis=1)
    peak_to_peak = maximum - minimum
    rms = np.sqrt(np.mean(np.square(eeg), axis=1))
    abs_mean = np.mean(np.abs(eeg), axis=1)
    zero_crossing = np.mean((eeg[:, 1:] * eeg[:, :-1]) < 0.0, axis=1).astype(np.float32)
    return np.concatenate(
        [mean, std, minimum, maximum, peak_to_peak, rms, abs_mean, zero_crossing],
        axis=0,
    ).astype(np.float32)


def _bandpower_features(eeg: np.ndarray, sampling_rate: int = 128) -> np.ndarray:
    eeg = np.asarray(eeg, dtype=np.float32)
    if welch is None or eeg.shape[1] < 16:
        return np.zeros(eeg.shape[0] * 4, dtype=np.float32)

    freqs, power = welch(
        eeg,
        fs=sampling_rate,
        axis=1,
        nperseg=min(256, eeg.shape[1]),
    )
    bands = ((4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))
    features = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            band_power = np.zeros(eeg.shape[0], dtype=np.float32)
        else:
            band_power = np.log10(
                np.trapz(power[:, mask], freqs[mask], axis=1) + 1e-8
            ).astype(np.float32)
        features.append(band_power)
    return np.concatenate(features, axis=0).astype(np.float32)


def extract_eeg_trial_features(
    trial_data: np.ndarray,
    eeg_channels: int = 32,
    baseline_seconds: int = 3,
    sampling_rate: int = 128,
) -> np.ndarray:
    eeg = np.asarray(trial_data[:eeg_channels], dtype=np.float32)
    baseline_samples = min(int(baseline_seconds * sampling_rate), eeg.shape[1] - 1)

    baseline = eeg[:, :baseline_samples]
    stimulus = eeg[:, baseline_samples:]

    full_stats = _signal_statistics(eeg)
    baseline_stats = _signal_statistics(baseline)
    stimulus_stats = _signal_statistics(stimulus)
    delta_mean = stimulus.mean(axis=1) - baseline.mean(axis=1)
    delta_std = stimulus.std(axis=1) - baseline.std(axis=1)
    baseline_bands = _bandpower_features(baseline, sampling_rate=sampling_rate)
    stimulus_bands = _bandpower_features(stimulus, sampling_rate=sampling_rate)

    return np.concatenate(
        [
            full_stats,
            baseline_stats,
            stimulus_stats,
            delta_mean.astype(np.float32),
            delta_std.astype(np.float32),
            baseline_bands,
            stimulus_bands,
            (stimulus_bands - baseline_bands).astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def extract_eeg_trial_sequence(
    trial_data: np.ndarray,
    eeg_channels: int = 32,
    baseline_seconds: int = 3,
    sampling_rate: int = 128,
    n_windows: int = 24,
) -> np.ndarray:
    eeg = np.asarray(trial_data[:eeg_channels], dtype=np.float32)
    baseline_samples = min(int(baseline_seconds * sampling_rate), eeg.shape[1] - 1)
    baseline = eeg[:, :baseline_samples]
    stimulus = eeg[:, baseline_samples:]
    baseline_mean = baseline.mean(axis=1, keepdims=True)

    edges = np.linspace(0, stimulus.shape[1], n_windows + 1, dtype=np.int32)
    steps = []
    for start, end in zip(edges[:-1], edges[1:]):
        if end <= start:
            end = min(start + 1, stimulus.shape[1])
        segment = stimulus[:, start:end]
        centered = segment - baseline_mean
        step = np.concatenate(
            [
                centered.mean(axis=1),
                centered.std(axis=1),
                np.sqrt(np.mean(np.square(centered), axis=1)),
                np.mean(np.abs(centered), axis=1),
            ],
            axis=0,
        )
        steps.append(step.astype(np.float32))

    return np.stack(steps, axis=0).astype(np.float32)


def load_trial_eeg(
    subject_id: int,
    trial_number: int,
    eeg_root: Optional[str] = None,
    eeg_channels: int = 32,
) -> np.ndarray:
    eeg_root = resolve_eeg_root(eeg_root)
    subject_path = os.path.join(eeg_root, f"s{int(subject_id):02d}.dat")
    if not os.path.isfile(subject_path):
        raise FileNotFoundError(f"Missing EEG subject file: {subject_path}")

    with open(subject_path, "rb") as fh:
        subject_record = pickle.load(fh, encoding="latin1")

    trials = np.asarray(subject_record["data"], dtype=np.float32)
    trial_index = int(trial_number) - 1
    if trial_index < 0 or trial_index >= trials.shape[0]:
        raise IndexError(
            f"Subject {subject_id} trial {trial_number} is outside [1, {trials.shape[0]}]."
        )

    return trials[trial_index, :eeg_channels].astype(np.float32)


def load_eeg_feature_dataset(
    metadata_root: Optional[str] = None,
    eeg_root: Optional[str] = None,
    raw_emotion_csv: Optional[str] = None,
    eeg_channels: int = 32,
    baseline_seconds: int = 3,
    sampling_rate: int = 128,
    n_windows: int = 24,
) -> dict:
    metadata_root = resolve_metadata_root(metadata_root)
    eeg_root = resolve_eeg_root(eeg_root)
    raw_emotion_csv = raw_emotion_csv or os.path.join(metadata_root, "emotion_summary.csv")

    clip_catalog = load_clip_catalog(metadata_root=metadata_root, raw_emotion_csv=raw_emotion_csv)
    clip_lookup = clip_catalog.set_index("Experiment_id")
    emotion_space = build_canonical_emotion_space()

    tabular_features = []
    sequence_features = []
    trial_emotions = []
    clip_emotions = []
    raw_emotions = []
    subject_ids = []
    trial_numbers = []
    experiment_ids = []
    valence = []
    arousal = []
    dominance = []
    liking = []
    familiarity = []
    normalized_va = []
    metadata_rows = []

    subject_ids_available = []
    for filename in sorted(os.listdir(eeg_root)):
        if filename.startswith("s") and filename.endswith(".dat"):
            subject_ids_available.append(int(filename[1:3]))

    for participant_id in subject_ids_available:
        subject_path = os.path.join(eeg_root, f"s{int(participant_id):02d}.dat")
        if not os.path.isfile(subject_path):
            raise FileNotFoundError(f"Missing EEG subject file: {subject_path}")

        with open(subject_path, "rb") as fh:
            subject_record = pickle.load(fh, encoding="latin1")

        trials = np.asarray(subject_record["data"], dtype=np.float32)
        labels = np.asarray(subject_record["labels"], dtype=np.float32)
        if labels.shape[0] != trials.shape[0]:
            raise ValueError(
                f"Subject {participant_id} has {trials.shape[0]} EEG trials but "
                f"{labels.shape[0]} label rows in the DEAP file."
            )

        for trial_index in range(trials.shape[0]):
            trial_number = int(trial_index + 1)
            experiment_id = trial_number
            clip_row = clip_lookup.loc[experiment_id] if experiment_id in clip_lookup.index else None
            trial_va = np.array(labels[trial_index, :2], dtype=np.float32)
            dominance_value = (
                float(labels[trial_index, 2]) if labels.shape[1] > 2 else float("nan")
            )
            liking_value = (
                float(labels[trial_index, 3]) if labels.shape[1] > 3 else float("nan")
            )
            normalized_vector = normalize_va(trial_va)

            tabular_features.append(
                extract_eeg_trial_features(
                    trials[trial_index],
                    eeg_channels=eeg_channels,
                    baseline_seconds=baseline_seconds,
                    sampling_rate=sampling_rate,
                )
            )
            sequence_features.append(
                extract_eeg_trial_sequence(
                    trials[trial_index],
                    eeg_channels=eeg_channels,
                    baseline_seconds=baseline_seconds,
                    sampling_rate=sampling_rate,
                    n_windows=n_windows,
                )
            )
            trial_emotions.append(
                label_vector_to_emotion(trial_va, emotion_centers=emotion_space)
            )
            clip_emotions.append(
                str(clip_row["canonical_emotion"]) if clip_row is not None else "unknown"
            )
            raw_emotions.append(str(clip_row["raw_emotion"]) if clip_row is not None else "unknown")
            subject_ids.append(int(participant_id))
            trial_numbers.append(trial_number)
            experiment_ids.append(experiment_id)
            valence.append(float(trial_va[0]))
            arousal.append(float(trial_va[1]))
            dominance.append(dominance_value)
            liking.append(liking_value)
            familiarity.append(float("nan"))
            normalized_va.append(normalized_vector.astype(np.float32))
            metadata_rows.append(
                {
                    "subject_id": int(participant_id),
                    "trial": trial_number,
                    "experiment_id": experiment_id,
                    "eeg_emotion": trial_emotions[-1],
                    "clip_emotion": clip_emotions[-1],
                    "raw_emotion": raw_emotions[-1],
                    "valence": float(trial_va[0]),
                    "arousal": float(trial_va[1]),
                    "dominance": dominance_value,
                    "liking": liking_value,
                    "familiarity": float("nan"),
                    "clip_title": str(clip_row.get("Title", "")) if clip_row is not None else "",
                    "clip_artist": str(clip_row.get("Artist", "")) if clip_row is not None else "",
                    "label_source": "deap_dat",
                    "metadata_note": (
                        "EEG supervision comes from DEAP .dat labels. "
                        "Clip metadata is attached separately from the music catalog."
                    ),
                }
            )

    return {
        "X_tabular": np.vstack(tabular_features).astype(np.float32),
        "X_sequence": np.stack(sequence_features).astype(np.float32),
        "y_emotion": np.asarray(trial_emotions),
        "y_subjective_emotion": np.asarray(trial_emotions),
        "y_clip_emotion": np.asarray(clip_emotions),
        "y_raw_emotion": np.asarray(raw_emotions),
        "subject_ids": np.asarray(subject_ids, dtype=np.int32),
        "trial_numbers": np.asarray(trial_numbers, dtype=np.int32),
        "experiment_ids": np.asarray(experiment_ids, dtype=np.int32),
        "valence": np.asarray(valence, dtype=np.float32),
        "arousal": np.asarray(arousal, dtype=np.float32),
        "dominance": np.asarray(dominance, dtype=np.float32),
        "liking": np.asarray(liking, dtype=np.float32),
        "familiarity": np.asarray(familiarity, dtype=np.float32),
        "normalized_va": np.vstack(normalized_va).astype(np.float32),
        "emotion_space": emotion_space,
        "emotion_names": list(emotion_space.keys()),
        "clip_catalog": clip_catalog,
        "metadata": pd.DataFrame(metadata_rows),
        "feature_info": {
            "eeg_channels": int(eeg_channels),
            "baseline_seconds": int(baseline_seconds),
            "sampling_rate": int(sampling_rate),
            "n_windows": int(n_windows),
            "label_source": "deap_dat",
            "tabular_dim": int(np.vstack(tabular_features).shape[1]),
            "sequence_shape": list(np.stack(sequence_features).shape[1:]),
        },
    }


def get_subject_by_id(subjects: list, participant_id: int) -> dict:
    for subject in subjects:
        if int(subject["participant_id"]) == int(participant_id):
            return subject
    raise KeyError(f"Participant {participant_id} not found in subject records.")


def synthetic_subjects(n_subjects: int = 32, n_clips: int = 40, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    clip_ids = np.arange(1, n_clips + 1, dtype=np.int32)

    subjects = []
    for participant_id in range(1, n_subjects + 1):
        subjects.append(
            {
                "participant_id": participant_id,
                "clip_ids": clip_ids.copy(),
                "trial_clip_ids": rng.permutation(clip_ids),
                "clip_vectors": rng.uniform(-1, 1, (n_clips, 2)).astype(np.float32),
            }
        )
    return subjects
