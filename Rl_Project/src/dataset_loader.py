import os
import pickle
import numpy as np
from config import DATA_PATH


def load_dataset():

    all_emotions = []
    subjects = []

    files = sorted(os.listdir(DATA_PATH))

    for sid, file in enumerate(files):

        if file.endswith(".dat"):

            data = pickle.load(
                open(os.path.join(DATA_PATH,file),"rb"),
                encoding="latin1"
            )

            labels = data["labels"]

            valence = labels[:,0]
            arousal = labels[:,1]

            subject_emotions = np.column_stack((valence,arousal))

            all_emotions.append(subject_emotions)

            subjects.append(sid)

    return np.array(all_emotions)