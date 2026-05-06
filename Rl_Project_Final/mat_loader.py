"""
mat_loader.py
=============
Loads .mat EEG files from your sLORETA dataset.

File naming convention:
    Emotion_mitSubjectIDTrial-XClick-Y-slor.mat
    e.g.  Amused_mit004Trial-3Click-1-slor.mat

Each .mat file contains:
    data : (128 channels, 1751 samples)  float32  EEG in µV
    Sampling rate: 128 Hz  →  ~13.7 sec per trial

Folder structure (flat or grouped):
    EEG_Data/
        Group1/
            Amused_mit004Trial-3Click-1-slor.mat
            Happy_mit004Trial-8Click-6-slor.mat
            ...
"""

import os
import re
import numpy as np
import scipy.io
from scipy.signal import welch

# ── Sampling rate ────────────────────────────────────────────
FS         = 128   # Hz
N_CHANNELS = 128   # EEG source channels per file

# ── Geneva Emotion Wheel mapping (valence, arousal) in [-1,1] ──
EMOTION_MAPPING = {
    "Adventurous":  ( 0.70,  0.75),
    "Adventorous":  ( 0.70,  0.75),   # typo variant in filenames
    "Afraid":       (-0.60,  0.80),
    "Alarmed":      (-0.20,  0.90),
    "Amused":       ( 0.70,  0.50),
    "Angry":        (-0.70,  0.70),
    "Aroused":      ( 0.10,  0.90),
    "Delighted":    ( 0.80,  0.70),
    "Depressed":    (-0.90, -0.70),
    "Despondent":   (-0.80, -0.50),
    "Disgust":      (-0.80,  0.20),
    "Dissatisfied": (-0.60, -0.20),
    "Distress":     (-0.50,  0.80),
    "Excited":      ( 0.70,  0.90),
    "Happy":        ( 0.80,  0.60),
    "Hate":         (-0.80,  0.60),
    "Joyous":       ( 0.90,  0.80),
    "Lust":         ( 0.60,  0.90),
    "Melancholic":  (-0.70, -0.50),
    "Miserable":    (-0.90, -0.40),
    "Passionate":   ( 0.80,  0.90),
    "Sad":          (-0.80, -0.60),
    "Startled":     (-0.10,  1.00),
    "Taken Aback":  (-0.30,  0.70),
    "Tense":        (-0.40,  0.80),
}

BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

# numpy 2.x compatibility
_integrate = None
def _trapz(y, x):
    global _integrate
    if _integrate is None:
        _integrate = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return _integrate(y, x)


# ─────────────────────────────────────────────────────────────
# EEG Feature Extraction
# ─────────────────────────────────────────────────────────────

def band_power_channel(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Compute 5 band powers for one EEG channel.
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


def extract_features(eeg: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Full per-channel band-power features.
    Input : (128, T)
    Output: (640,)  — 128 channels × 5 bands
    """
    return np.concatenate([band_power_channel(eeg[ch], fs)
                           for ch in range(eeg.shape[0])])


def extract_features_compact(eeg: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Mean band power across all channels — faster, smaller.
    Input : (128, T)
    Output: (5,)
    """
    return np.array([band_power_channel(eeg[ch], fs)
                     for ch in range(eeg.shape[0])]).mean(axis=0)


# ─────────────────────────────────────────────────────────────
# Filename Parser
# ─────────────────────────────────────────────────────────────

def parse_filename(filename: str) -> dict | None:
    """
    Parse emotion, subject ID, trial, click from filename.
    Pattern: Emotion_mitIDTrial-XClick-Y-slor.mat
    Returns dict or None if pattern doesn't match.
    """
    name  = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_", 1)
    if len(parts) < 2:
        return None

    emotion = parts[0]
    rest    = parts[1]

    sub_match = re.search(r'(mit\d+)', rest, re.IGNORECASE)
    if not sub_match:
        return None
    subject_id = sub_match.group(1).lower()

    trial_match = re.search(r'Trial-?(\d+)', rest, re.IGNORECASE)
    trial = int(trial_match.group(1)) if trial_match else 0

    click_match = re.search(r'Click-?(\d+)', rest, re.IGNORECASE)
    click = int(click_match.group(1)) if click_match else 0

    va = None
    for key, val in EMOTION_MAPPING.items():
        if key.lower() == emotion.lower():
            va = val
            break
    if va is None:
        va = (0.0, 0.0)   # unknown emotion → neutral

    return {
        "emotion":    emotion,
        "subject_id": subject_id,
        "trial":      trial,
        "click":      click,
        "valence":    va[0],
        "arousal":    va[1],
    }


# ─────────────────────────────────────────────────────────────
# MAT Loader
# ─────────────────────────────────────────────────────────────

class MATLoader:
    """
    Walks the dataset folder, finds all -slor.mat files,
    and organises them by subject.

    Usage
    -----
        loader   = MATLoader("C:/Users/.../EEG_Data")
        subjects = loader.load_all_subjects_clips()  # list of (N,2) arrays
        clips    = loader.get_subject_clips("mit004") # (N, 2)
        eeg      = loader.load_eeg("mit004", index=0) # (128, T)
    """

    def __init__(self, data_root: str, verbose: bool = True):
        self.data_root = data_root
        self.verbose   = verbose
        self.subjects  = {}   # {subject_id: [{emotion, valence, arousal, file_path}]}
        self._discover()

    def _discover(self):
        n_files = n_skip = 0
        for root, _, files in os.walk(self.data_root):
            for f in sorted(files):
                if not f.lower().endswith(".mat"):
                    continue
                meta = parse_filename(f)
                if meta is None:
                    n_skip += 1
                    continue
                meta["file_path"] = os.path.join(root, f)
                self.subjects.setdefault(meta["subject_id"], []).append(meta)
                n_files += 1

        if self.verbose:
            print(f"[MATLoader] Found {n_files} files across "
                  f"{len(self.subjects)} subjects  "
                  f"({n_skip} skipped)")
            for sid, trials in sorted(self.subjects.items()):
                emotions = sorted({t["emotion"] for t in trials})
                print(f"  {sid}: {len(trials)} trials | emotions: {emotions}")

    def get_all_subject_ids(self) -> list:
        return sorted(self.subjects.keys())

    def get_subject_clips(self, subject_id: str) -> np.ndarray:
        """
        Returns (N_clips, 2) array of [valence, arousal] from filename labels.
        Note: use MATEmotionDataset (eeg_classifier.py) for EEG-predicted values.
        """
        trials = self.subjects.get(subject_id, [])
        if not trials:
            raise KeyError(f"Subject '{subject_id}' not found.")
        return np.array([[t["valence"], t["arousal"]] for t in trials],
                        dtype=np.float32)

    def load_eeg(self, subject_id: str, index: int) -> np.ndarray:
        """
        Load raw EEG for one trial.
        Returns (128, T) float32 array.
        """
        trials = self.subjects.get(subject_id, [])
        if not trials or index >= len(trials):
            raise IndexError(f"Index {index} out of range for '{subject_id}'")
        mat = scipy.io.loadmat(trials[index]["file_path"])
        return mat["data"].astype(np.float32)

    def load_all_subjects_clips(self) -> list:
        """
        Returns list of (N_clips, 2) arrays — one per subject.
        Uses filename labels (fast). For EEG-predicted values use
        MATEmotionDataset.build_loso_subjects() from eeg_classifier.py.
        """
        return [self.get_subject_clips(sid)
                for sid in self.get_all_subject_ids()]

    def get_emotion_info(self, subject_id: str) -> list:
        """Returns list of emotion names per trial (for inspection)."""
        return [t["emotion"] for t in self.subjects.get(subject_id, [])]


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\04khu\Desktop\Rl Project\EEG_Data"
    loader = MATLoader(root)
    subs   = loader.get_all_subject_ids()
    if subs:
        sid   = subs[0]
        clips = loader.get_subject_clips(sid)
        print(f"\nSubject {sid}: {len(clips)} clips")
        print(f"Sample clip (valence, arousal): {clips[0]}")
        print(f"Emotions: {loader.get_emotion_info(sid)}")