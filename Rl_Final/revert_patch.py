import os
import re

files = [
    "music_emotion_ql.py",
    "music_emotion_sarsa.py",
    "music_emotion_double_ql.py",
    "music_emotion_double_sarsa.py"
]

for f in files:
    with open(f, "r") as file:
        content = file.read()
    
    # 1. Revert LAMBDA
    content = re.sub(r'LAMBDA\s*=\s*-150\.0', 'LAMBDA                 = -100.0', content)
    
    # 2. Revert Reward
    content = re.sub(r'r = 10\.0 if \(abs', 'r = 1.0 if (abs', content)
    
    with open(f, "w") as file:
        file.write(content)

print("Revert patching complete.")
