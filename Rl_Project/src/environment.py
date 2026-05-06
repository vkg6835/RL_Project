import numpy as np
from emotion_utils import transition
from reward import reward, reward_shaping

class EmotionEnv:

    def __init__(self, actions):
        self.actions = actions

    def step(self, state_id, action_id):

        state = self.actions[state_id]
        action = self.actions[action_id]

        next_state_vec = transition(state, action)

        # reward
        r = reward(state, action)

        # map to closest state
        next_state_id = np.argmin(
            np.linalg.norm(self.actions - next_state_vec, axis=1)
        )

        # shaping
        r += reward_shaping(state, self.actions[next_state_id])

        return next_state_id, r