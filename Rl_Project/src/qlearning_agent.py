import numpy as np
import random
from config import ALPHA, GAMMA, EPSILON, EPSILON_DECAY, EPSILON_MIN

class QLearning:

    def __init__(self, n_states, n_actions):

        self.q = np.zeros((n_states, n_actions))

        self.n_actions = n_actions
        self.epsilon = EPSILON

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        return np.argmax(self.q[state])

    def update(self, s, a, r, s_next):

        best_next = np.max(self.q[s_next])

        self.q[s, a] = (1 - ALPHA) * self.q[s, a] + \
                       ALPHA * (r + GAMMA * best_next)

    def decay_epsilon(self):

        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)