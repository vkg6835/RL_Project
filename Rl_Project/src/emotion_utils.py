import numpy as np

def angle_between(v1, v2):

    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)

    dot = np.dot(v1, v2)
    dot = np.clip(dot, -1.0, 1.0)

    return np.degrees(np.arccos(dot))


def transition(state, action):
    return state + action