import sys
sys.path.append("src")

import numpy as np
import os
import pandas as pd

from dataset_loader import load_dataset
from trainer import train
from evaluator import evaluate_playlist
from plotter import plot_error


# -------------------------
# Create result directories
# -------------------------
os.makedirs("results/metrics", exist_ok=True)
os.makedirs("results/qtables", exist_ok=True)
os.makedirs("results/playlists", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)


# -------------------------
# Load dataset
# -------------------------
subjects = load_dataset()

print("Dataset shape:", subjects.shape)  
# Expected: (32, 40, 2)


# -------------------------
# Train on first subject
# -------------------------
actions = subjects[0]   # 40 music clips

q_table = train(actions)

np.save("results/qtables/q_table.npy", q_table)


# -------------------------
# Generate playlist
# -------------------------
start = np.random.randint(0, 40)
playlist = [start]

current = start

for i in range(5):
    next_song = np.argmax(q_table[current])
    playlist.append(next_song)
    current = next_song

# -------------------------
# Evaluate angular error
# -------------------------
errors = evaluate_playlist(actions, playlist)


# -------------------------
# Save plot
# -------------------------
plot_error(errors)


# -------------------------
# Save results table
# -------------------------
df = pd.DataFrame({
    "iteration": list(range(len(errors))),
    "angular_error": errors
})

df.to_csv("results/metrics/angular_error.csv", index=False)


# -------------------------
# Save playlist
# -------------------------
with open("results/playlists/playlist.txt", "w") as f:
    f.write(str(playlist))


print("Playlist:", playlist)
print("Angular errors:", errors)