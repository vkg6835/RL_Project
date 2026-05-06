# Music Emotion Reinforcement Learning

This project implements a Reinforcement Learning (Q-learning) agent to select music clips that transition a user's emotional state toward a target emotion (e.g., "Happy/Joyful" quadrant on the Geneva Emotion Wheel). The pipeline is based on the methodology described by Dutta et al. (2020) and uses EEG signals alongside physiological data.

## Recent Improvements

The codebase was recently refactored to align closely with the paper's specific methodology:

1. **Geneva Emotion Wheel (GEW) State Mapping**
   - **Improvement**: Updated `DEAPLoader._normalise` in `eeg_classifier.py` to linearly map Valence and Arousal ratings from the original `[1, 9]` scale directly to the `[-1.0, 1.0]` GEW coordinate space. 
   - **Reason**: The original code used a relative minimum and maximum normalisation bounding to `[-1.0, 1.0]`. Linearly shifting using `(val - 5.0) / 4.0` preserves the true distance and mapping to quadrants corresponding to the wheel.
2. **Dynamic Reward Function**
   - **Improvement**: Rewrote `reward()` logic in `music_emotion_rl.py` to evaluate the agent based on reduction of Angular Error. The reward now compares `angular_error(current_state)` with `angular_error(next_state)` and yields `+1` if the state moving closer to the target quadrant, rather than static comparison logic.
3. **Reward Shaping Penalty (λ) ENFORCED**
   - **Improvement**: Strictly enforced the required penalty factor `LAMBDA = -100.0` over `-1.0` and removed the script reset that was temporarily breaking this constraint.
   - **Reason**: A steep penalty of `-100` prevents the Q-learning agent from getting stuck exploiting sequences that don't transition the state vector towards the objective.
4. **Leave-One-Subject-Out (LOSO) Pipeline**
   - The environment now efficiently traverses all states using training parameters `alpha=0.1`, `gamma=0.6`, over sequence playlists of `6` songs.

## Running the Pipeline

You can execute the pipeline via:
```bash
python music_emotion_rl.py data/data_preprocessed_python --mode deap --episodes 1000
```
