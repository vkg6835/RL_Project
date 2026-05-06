import numpy as np
from tqdm import tqdm
from config import EPISODES, MAX_STEPS
from environment import EmotionEnv
from qlearning_agent import QLearning


def train(actions):

    env = EmotionEnv(actions)

    agent = QLearning(len(actions), len(actions))

    for ep in tqdm(range(EPISODES)):

        # random starting state
        state = np.random.randint(len(actions))

        for step in range(MAX_STEPS):

            # choose action
            action = agent.choose_action(state)

            # environment step
            next_state, r = env.step(state, action)

            # update Q-table
            agent.update(state, action, r, next_state)

            # move to next state
            state = next_state

        # decay epsilon AFTER each episode
        agent.decay_epsilon()

    return agent.q