import pickle, os
import numpy as np
from eeg_classifier import DEAPLoader, EEGEmotionClassifier

data_root = "data/data_preprocessed_python/"
out_file = "eeg_subjects_cache.pkl"

if not os.path.exists(out_file):
    print("Generating cache...")
    loader = DEAPLoader(data_root)
    subjects = []
    all_ids = list(range(1, 33))
    for test_id in all_ids:
        train_eeg, train_va = [], []
        for sid in all_ids:
            if sid == test_id: continue
            try:
                eeg_list, labels = loader.get_eeg_and_labels(sid)
                train_eeg.extend(eeg_list)
                train_va.extend(labels)
            except FileNotFoundError:
                continue
        if len(train_eeg) < 10:
            subjects.append(np.zeros((40, 2))) # dummy
            continue
        clf = EEGEmotionClassifier(method="svm", fs=128, compact=False)
        clf.fit(train_eeg, np.array(train_va))
        try:
            test_eeg, _ = loader.get_eeg_and_labels(test_id)
            clips = clf.predict_batch(test_eeg)
            subjects.append(clips)
            print(f"s{test_id:02d}: {len(clips)} clips")
        except FileNotFoundError:
            subjects.append(np.zeros((40, 2)))
    with open(out_file, "wb") as f:
        pickle.dump(subjects, f)
    print("Saved cache.")
else:
    print("Cache exists.")
