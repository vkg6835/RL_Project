"""
EEG Feature Extraction & Emotion Classifier
=============================================
Extracts band-power features from raw EEG signals and trains
a classifier to predict valence & arousal.

Works with BOTH datasets:
  - DEAP dataset  (.dat files, 32 EEG channels, 128Hz)
  - Your dataset  (.mat files, 128 source channels, 128Hz)

Pipeline:
  Raw EEG (channels × samples)
       ↓
  Band-power features per channel  [δ θ α β γ]
       ↓
  Feature vector (n_channels × 5)
       ↓
  SVM classifier → predict valence class (high/low)
                 → predict arousal class (high/low)
       ↓
  (valence, arousal) in [-1, 1]  →  fed to Q-learning agent

KEY FIX vs previous version:
  _normalise() now uses simple linear map from [1,9] → [-1,1]
  instead of centering around subject mean.
  Centering was destroying the true GEW position of clips
  (a truly happy clip was getting mapped to negative valence
  just because the subject's mean was high).
"""

import numpy as np
from scipy.signal import welch
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1.  BAND-POWER FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────

BANDS = {
    "delta": (0.5,   4.0),
    "theta": (4.0,   8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
N_BANDS = len(BANDS)

_integrate = None
def _trapz(y, x):
    global _integrate
    if _integrate is None:
        _integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return _integrate(y, x)


def band_power_channel(signal: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Compute 5 band powers for a single EEG channel.
    Input : (T,)
    Output: (5,)  [delta, theta, alpha, beta, gamma]
    """
    nperseg = min(256, len(signal))
    freqs, psd = welch(signal.astype(float), fs=fs, nperseg=nperseg)
    powers = []
    for lo, hi in BANDS.values():
        idx = (freqs >= lo) & (freqs <= hi)
        powers.append(_trapz(psd[idx], freqs[idx]) if idx.any() else 0.0)
    return np.array(powers, dtype=np.float32)


def extract_features(eeg: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Extract band-power features from all EEG channels.

    Input : (n_channels, T)
    Output: (n_channels × 5,)  flattened feature vector

    For DEAP  : (32, 8064) → (160,)
    For yours : (128, 1751) → (640,)
    """
    features = [band_power_channel(eeg[ch], fs) for ch in range(eeg.shape[0])]
    return np.concatenate(features)


def extract_features_compact(eeg: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Mean band power across channels — smaller, faster.
    Input : (n_channels, T)
    Output: (5,)
    """
    return np.array([band_power_channel(eeg[ch], fs)
                     for ch in range(eeg.shape[0])]).mean(axis=0)


# ─────────────────────────────────────────────────────────────
# 2.  VALENCE / AROUSAL DISCRETISATION
# ─────────────────────────────────────────────────────────────

def continuous_to_class(value: float, threshold: float = 0.0) -> int:
    """Map continuous [-1,1] value to binary class: 1=high, 0=low."""
    return 1 if value >= threshold else 0


def class_to_continuous(cls: int, high: float = 0.6,
                         low: float = -0.6) -> float:
    """Map binary class back to a representative continuous value."""
    return high if cls == 1 else low


# ─────────────────────────────────────────────────────────────
# 3.  CLASSIFIER
# ─────────────────────────────────────────────────────────────

class EEGEmotionClassifier:
    """
    Predicts (valence, arousal) from raw EEG signals.

    Internally trains two binary SVM classifiers:
      - valence_clf  : high valence (positive) vs low valence (negative)
      - arousal_clf  : high arousal (excited)  vs low arousal (calm)

    Usage
    -----
        clf = EEGEmotionClassifier()
        clf.fit(X_train, va_train)        # va_train: (N, 2) valence/arousal [-1,1]
        valence, arousal = clf.predict_single(eeg_trial)
    """

    def __init__(self, method: str = "svm", fs: int = 128,
                 compact: bool = False):
        self.method  = method
        self.fs      = fs
        self.compact = compact
        self.valence_clf = None
        self.arousal_clf = None
        self._build_clf()

    def _build_clf(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        def make_pipeline():
            if self.method == "svm":
                from sklearn.svm import SVC
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf",    SVC(kernel="rbf", C=1.0, gamma="scale",
                                  class_weight="balanced",
                                  probability=True, random_state=42)),
                ])
            elif self.method == "knn":
                from sklearn.neighbors import KNeighborsClassifier
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf",    KNeighborsClassifier(n_neighbors=5)),
                ])
            elif self.method == "rf":
                from sklearn.ensemble import RandomForestClassifier
                return Pipeline([
                    ("clf", RandomForestClassifier(n_estimators=100,
                                                    class_weight="balanced",
                                                    random_state=42)),
                ])
            else:
                raise ValueError(f"Unknown method: {self.method}")

        self.valence_clf = make_pipeline()
        self.arousal_clf = make_pipeline()

    def _features(self, eeg: np.ndarray) -> np.ndarray:
        if self.compact:
            return extract_features_compact(eeg, self.fs)
        return extract_features(eeg, self.fs)

    def _build_feature_matrix(self, eeg_trials: list) -> np.ndarray:
        print(f"  Extracting features from {len(eeg_trials)} trials …")
        return np.array([self._features(trial) for trial in eeg_trials])

    def fit(self, eeg_trials: list, va_labels: np.ndarray):
        """
        Train valence and arousal classifiers.

        Parameters
        ----------
        eeg_trials : list of (n_channels, T) EEG arrays
        va_labels  : (N, 2) array of [valence, arousal] in [-1, 1]
        """
        X = self._build_feature_matrix(eeg_trials)

        y_valence = np.array([continuous_to_class(v) for v in va_labels[:, 0]])
        y_arousal = np.array([continuous_to_class(a) for a in va_labels[:, 1]])

        print(f"  Training {self.method.upper()} classifiers …")
        print(f"    Valence: {y_valence.sum()} high / {(y_valence==0).sum()} low")
        print(f"    Arousal: {y_arousal.sum()} high / {(y_arousal==0).sum()} low")

        self.valence_clf.fit(X, y_valence)
        self.arousal_clf.fit(X, y_arousal)
        print("  Classifiers trained.")

    def cross_val_score(self, eeg_trials: list,
                         va_labels: np.ndarray, cv: int = 5) -> dict:
        from sklearn.model_selection import cross_val_score as cvs

        X = self._build_feature_matrix(eeg_trials)
        y_v = np.array([continuous_to_class(v) for v in va_labels[:, 0]])
        y_a = np.array([continuous_to_class(a) for a in va_labels[:, 1]])

        v_scores = cvs(self.valence_clf, X, y_v, cv=cv, scoring="accuracy")
        a_scores = cvs(self.arousal_clf, X, y_a, cv=cv, scoring="accuracy")

        return {
            "valence_acc": v_scores.mean(),
            "valence_std": v_scores.std(),
            "arousal_acc": a_scores.mean(),
            "arousal_std": a_scores.std(),
        }

    def predict_single(self, eeg: np.ndarray) -> tuple:
        """
        Predict (valence, arousal) for one EEG trial.
        Returns hard class predictions mapped to ±0.70.
        """
        if self.valence_clf is None:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        feats = self._features(eeg).reshape(1, -1)
        v_cls = self.valence_clf.predict(feats)[0]
        a_cls = self.arousal_clf.predict(feats)[0]

        valence = class_to_continuous(v_cls, high=0.70, low=-0.70)
        arousal = class_to_continuous(a_cls, high=0.70, low=-0.70)
        return float(valence), float(arousal)

    def predict_batch(self, eeg_trials: list) -> np.ndarray:
        return np.array([self.predict_single(t) for t in eeg_trials],
                        dtype=np.float32)

    def predict_proba_single(self, eeg: np.ndarray) -> tuple:
        """
        Soft (valence, arousal) using class probabilities.
        Maps P(high) from [0,1] → [-1,1].
        This gives smoother GEW positions than hard class predictions.
        """
        feats  = self._features(eeg).reshape(1, -1)
        v_prob = self.valence_clf.predict_proba(feats)[0][1]
        a_prob = self.arousal_clf.predict_proba(feats)[0][1]

        valence = (v_prob * 2) - 1
        arousal = (a_prob * 2) - 1
        return float(valence), float(arousal)


# ─────────────────────────────────────────────────────────────
# 4.  DEAP DATASET INTEGRATION
# ─────────────────────────────────────────────────────────────

class DEAPLoader:
    """
    Loads the DEAP dataset (preprocessed Python .dat files).

    data_preprocessed_python/
        s01.dat … s32.dat

    Each file:
        data   : (40 trials, 40 channels, 8064 samples)
                  ch 0-31 = EEG,  ch 32-39 = peripheral
        labels : (40 trials, 4) = [valence, arousal, dominance, liking]
                  range [1, 9]
    """

    FS         = 128
    N_EEG_CH   = 32
    N_TRIALS   = 40

    def __init__(self, data_dir: str, verbose: bool = True):
        self.data_dir = data_dir
        self.verbose  = verbose

    def _load(self, subject_id: int) -> dict:
        import os
        import pickle
        path = os.path.join(self.data_dir, f"s{subject_id:02d}.dat")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Not found: {path}")
        with open(path, "rb") as f:
            return pickle.load(f, encoding="latin1")

    @staticmethod
    def _normalise(raw_va_labels: np.ndarray) -> np.ndarray:
        """
        Map DEAP self-report labels [1,9] → [-1,1] linearly.

        Formula: normalised = (raw - 5) / 4
          score=1 → -1.0  (most negative)
          score=5 →  0.0  (neutral)
          score=9 → +1.0  (most positive)

        WHY THIS MATTERS:
        Previous version subtracted the subject's own mean then
        divided by max-abs. This destroyed the true GEW position —
        a subject who consistently rated everything high would have
        even their happiest clips mapped to near-zero valence.
        Linear mapping preserves the absolute meaning of scores.
        """
        out = (raw_va_labels.astype(float) - 5.0) / 4.0
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def get_eeg_and_labels(self, subject_id: int) -> tuple:
        """
        Returns
        -------
        eeg    : list of 40 arrays, each (32, 8064)
        labels : (40, 2) [valence, arousal] in [-1, 1]
        """
        d      = self._load(subject_id)
        eeg    = [d["data"][t, :self.N_EEG_CH, :]
                  for t in range(self.N_TRIALS)]
        raw_va = d["labels"][:, :2]
        labels = self._normalise(raw_va)
        return eeg, labels

    def get_clips(self, subject_id: int) -> np.ndarray:
        """Returns (40, 2) [valence, arousal] from self-report labels."""
        d      = self._load(subject_id)
        raw_va = d["labels"][:, :2]
        return self._normalise(raw_va)

    def load_all_subjects(self, n_subjects: int = 32) -> list:
        """Returns list of (40, 2) label arrays — one per subject."""
        subjects = []
        for i in range(1, n_subjects + 1):
            try:
                clips = self.get_clips(i)
                subjects.append(clips)
                if self.verbose:
                    print(f"  Loaded s{i:02d}  "
                          f"valence: {clips[:,0].mean():.2f}  "
                          f"arousal: {clips[:,1].mean():.2f}")
            except FileNotFoundError:
                if self.verbose:
                    print(f"  Skipping s{i:02d} — file not found")
        return subjects


# ─────────────────────────────────────────────────────────────
# 5.  MAT DATASET INTEGRATION  (sLORETA dataset)
# ─────────────────────────────────────────────────────────────

class MATEmotionDataset:
    """
    Loads .mat EEG dataset and prepares it for the full pipeline.
    Uses LOSO: for each test subject, trains classifier on all others.
    """

    def __init__(self, data_root: str, fs: int = 128,
                 method: str = "svm", use_soft_proba: bool = True,
                 verbose: bool = True):
        self.data_root    = data_root
        self.fs           = fs
        self.method       = method
        self.use_soft     = use_soft_proba
        self.verbose      = verbose
        self.subjects_raw = {}
        self._load_all()

    def _load_all(self):
        import os
        import scipy.io
        from mat_loader import parse_filename

        n_loaded = n_skip = 0
        for root, _, files in os.walk(self.data_root):
            for fname in sorted(files):
                if not fname.lower().endswith(".mat"):
                    continue
                meta = parse_filename(fname)
                if meta is None:
                    n_skip += 1
                    continue

                fpath = os.path.join(root, fname)
                try:
                    mat = __import__("scipy.io", fromlist=["loadmat"]).loadmat(fpath)
                    eeg = mat["data"].astype(np.float32)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: could not load {fname}: {e}")
                    n_skip += 1
                    continue

                sid = meta["subject_id"]
                self.subjects_raw.setdefault(sid, []).append({
                    "eeg":     eeg,
                    "valence": meta["valence"],
                    "arousal": meta["arousal"],
                    "emotion": meta["emotion"],
                    "file":    fname,
                })
                n_loaded += 1

        if self.verbose:
            print(f"[MATEmotionDataset] Loaded {n_loaded} files, "
                  f"{len(self.subjects_raw)} subjects "
                  f"({n_skip} skipped)")

    def _train_classifier_for_subject(self, subject_id, other_subjects):
        eeg_list, va_list = [], []
        for sid in other_subjects:
            for trial in self.subjects_raw.get(sid, []):
                eeg_list.append(trial["eeg"])
                va_list.append([trial["valence"], trial["arousal"]])

        if len(eeg_list) < 4:
            return None

        va_array = np.array(va_list, dtype=np.float32)
        clf      = EEGEmotionClassifier(method=self.method, fs=self.fs,
                                         compact=True)
        clf.fit(eeg_list, va_array)
        return clf

    def get_subject_clips_eeg_predicted(self, subject_id, other_subjects):
        clf    = self._train_classifier_for_subject(subject_id, other_subjects)
        trials = self.subjects_raw.get(subject_id, [])

        if clf is None or not trials:
            if self.verbose:
                print(f"    {subject_id}: using filename labels (insufficient data)")
            return np.array([[t["valence"], t["arousal"]] for t in trials],
                            dtype=np.float32)

        clips = []
        for trial in trials:
            if self.use_soft:
                v, a = clf.predict_proba_single(trial["eeg"])
            else:
                v, a = clf.predict_single(trial["eeg"])
            clips.append([v, a])

        predicted = np.array(clips, dtype=np.float32)
        if self.verbose:
            print(f"    {subject_id}: {len(trials)} trials  "
                  f"v̄={predicted[:,0].mean():.2f}  "
                  f"ā={predicted[:,1].mean():.2f}")
        return predicted

    def build_loso_subjects(self) -> list:
        all_ids  = sorted(self.subjects_raw.keys())
        subjects = []
        print(f"\nBuilding EEG-predicted clips for {len(all_ids)} subjects …")
        for sid in all_ids:
            others = [s for s in all_ids if s != sid]
            clips  = self.get_subject_clips_eeg_predicted(sid, others)
            subjects.append(clips)
        return subjects

    def get_all_subject_ids(self) -> list:
        return sorted(self.subjects_raw.keys())