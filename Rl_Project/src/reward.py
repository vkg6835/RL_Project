import numpy as np
from config import TARGET_EMOTION

target = np.array(TARGET_EMOTION)

def reward(state, action):

    next_state = state + action

    # distance to target
    dist_current = np.linalg.norm(state - target)
    dist_next = np.linalg.norm(next_state - target)

    if dist_next < dist_current:
        return 1
    else:
        return -1


def reward_shaping(state, next_state):

    # strong penalty if stuck
    if np.allclose(state, next_state):
        return -100

    return 0