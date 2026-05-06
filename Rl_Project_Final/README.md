# Music Emotion RL — Dutta et al. 2020 Replication
## Q-Learning | SARSA | Double Q-Learning

---

## Files

| File | Description |
|------|-------------|
| `music_emotion_ql.py` | Q-Learning (corrected, paper replication) |
| `music_emotion_sarsa.py` | SARSA (on-policy TD) |
| `music_emotion_double_ql.py` | Double Q-Learning (van Hasselt 2010) |
| `music_emotion_compare.py` | Runs all three and compares side-by-side |

---

## Running Commands

### 1. Q-Learning only (paper replication)
```bash
python music_emotion_ql.py \
  --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv \
  --output_dir results_ql \
  --episodes 2000

# With grid search for best hyper-params:
python music_emotion_ql.py \
  --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv \
  --output_dir results_ql \
  --episodes 2000 \
  --grid_search
```

### 2. SARSA
```bash
python music_emotion_sarsa.py \
  --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv \
  --output_dir results_sarsa \
  --episodes 2000
```

### 3. Double Q-Learning
```bash
python music_emotion_double_ql.py \
  --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv \
  --output_dir results_double_ql \
  --episodes 2000
```

### 4. Compare all three algorithms together (recommended)
```bash
python music_emotion_compare.py \
  --data_root /home/sahil/Desktop/Rl_Project_Final/data/Metacsv \
  --output_dir results_compare \
  --episodes 2000
```

---

## Bugs Fixed in Original Code

### Bug 1 — Wrong convergence criterion
**Original:** `"converged": (emotions[-1] == TARGET_EMOTION) or fe < 60`  
This counted subjects as "converged" even if they ended at `sad` with 59° error.

**Fixed:** Three separate criteria are now tracked:
```
exact_converged = (final emotion == "happy")     ← Paper primary criterion
conv_30         = (final angular error < 30°)    ← Strict
conv_60         = (final angular error < 60°)    ← Paper's reported overall metric
```

The paper says **19/32 subjects converged** = 19 subjects whose final emotion label was exactly "happy".

### Bug 2 — Happy start state not handled
**Original:** If `start_emotion` was None, it could randomly pick "happy" as start,
and the agent would trivially stay at happy — inflating convergence count.

**Fixed:** `evaluate_subject()` now always resamples from non-target emotions:
```python
if start_emotion is None or start_emotion == TARGET_EMOTION:
    start_emotion = non_target[np.random.randint(len(non_target))]
```

The training loop already excluded happy starts — now evaluation matches.

### What happens if a subject starts at "happy"?
The paper trains the agent to guide subjects **FROM** a non-happy emotion **TO** happy.
Starting at happy is undefined behavior for this task. The fix resamples a random
non-happy starting emotion, which is consistent with the paper's experimental setup.

---

## Convergence Criteria — Both Reported

Every script prints:
```
Exact emotion = happy : XX/32    (paper primary — matches paper's "19/32")
Angular error < 30°  : XX/32    (strict criterion)
Angular error < 60°  : XX/32    (paper's overall convergence metric)
Overall mean (clip 6): XX.X° ± X.X°   (paper: 57.0° ± 2.8°)
```

---

## Algorithm Differences

| Property | Q-Learning | SARSA | Double Q-Learning |
|----------|-----------|-------|-------------------|
| Policy type | Off-policy | On-policy | Off-policy |
| Update uses | max Q(s',a') | Q(s', actual a') | QB(s', argmax QA) |
| Tables | 1 | 1 | 2 (QA + QB) |
| Overestimation bias | Yes | Less | Minimal |
| Convergence | Faster | More stable | Most stable |

---

## Hyper-parameters (paper optimal)

| Parameter | Value | Description |
|-----------|-------|-------------|
| α (alpha) | 0.1 | Learning rate |
| γ (gamma) | 0.6 | Discount factor |
| ε (epsilon) | 0.1 | Exploration rate (ε-greedy) |
| λ (lambda) | -100 | Reward shaping penalty |
| Episodes | 2000 | Training episodes per LOSO fold |
| Playlist length | 6 | Music clips per evaluation |

---

## Paper Results (target to match)

- **19/32 subjects** converged to "happy" state  
- **Mean angular error (clip 6)**: 57.0° ± 2.8° (all 32 subjects)  
- **Converged group (19)**: 11.3° ± 2.7°  
- **Failed group (13)**: 123.7° ± 17.3°  
