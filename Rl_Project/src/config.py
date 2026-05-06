DATA_PATH = "data/data_preprocessed_python"

TARGET_EMOTION = [8,8]

ALPHA = 0.1
GAMMA = 0.6


REWARD_SHAPING_LAMBDA = -100
EPISODES = 5000      # increase training
MAX_STEPS = 10       # more transitions

EPSILON = 1.0        # start fully random
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05