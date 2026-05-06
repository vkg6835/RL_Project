import numpy as np
from emotion_utils import angle_between
from config import TARGET_EMOTION

target = np.array(TARGET_EMOTION)


def evaluate_playlist(actions, playlist):

    errors = []

    # ✅ start from neutral emotion (important)
    state = np.array([5.0, 5.0])   # center of DEAP scale

    for clip in playlist:

        action = actions[clip]

        # transition
        state = state + action

        # compute angular error
        error = angle_between(state, target)

        errors.append(error)

    return errors