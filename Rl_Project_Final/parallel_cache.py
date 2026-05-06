import os, pickle
import numpy as np
import multiprocessing as mp
from eeg_classifier import DEAPLoader, EEGEmotionClassifier

def process_subject(args):
    test_id, data_root = args
    print(f"Starting s{test_id:02d}...")
    loader = DEAPLoader(data_root)
    all_ids = list(range(1, 33))
    train_eeg, train_va = [], []
    for sid in all_ids:
        if sid == test_id: continue
        try:
            eeg_list, labels = loader.get_eeg_and_labels(sid)
            train_eeg.extend(eeg_list)
            train_va.extend(labels)
        except Exception: pass
    if len(train_eeg) < 10: return test_id, None
    
    clf = EEGEmotionClassifier(method="svm", fs=128, compact=False)
    clf.fit(train_eeg, np.array(train_va))
    try:
        test_eeg, _ = loader.get_eeg_and_labels(test_id)
        clips = clf.predict_batch(test_eeg)
        print(f"Finished s{test_id:02d}: {len(clips)} clips")
        return test_id, clips
    except Exception:
        return test_id, None

if __name__ == '__main__':
    data_root = "data/data_preprocessed_python/"
    out_file = "eeg_subjects_cache.pkl"
    
    pool = mp.Pool(mp.cpu_count())
    args = [(i, data_root) for i in range(1, 33)]
    
    results = pool.map(process_subject, args)
    
    # Sort just to maintain order
    results.sort(key=lambda x: x[0])
    
    final_subjects = []
    for tid, clips in results:
        if clips is not None:
            final_subjects.append(clips)
        else:
            final_subjects.append(np.zeros((40, 2)))
            
    with open(out_file, "wb") as f:
        pickle.dump(final_subjects, f)
        
    print(f"Cache generated successfully. Saved to {out_file}.")
