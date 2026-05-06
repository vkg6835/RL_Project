import numpy as np

def generate_playlist(q_table,start_state,n_clips=6):

    playlist = []

    state = start_state

    for i in range(n_clips):

        action = np.argmax(q_table[state])

        playlist.append(action)

        state = action

    return playlist