import numpy as np
import music_emotion_ql as m
import music_emotion_compare as mc
m._init_globals("data/Metacsv/emotion_summary.csv")
subjects = mc.load_subjects("data/Metacsv")
res = m.leave_one_subject_out(subjects[:1], alpha=0.1, gamma=0.6, epsilon=0.1, n_episodes=5, use_reward_shaping=True, start_emotion="happy")
