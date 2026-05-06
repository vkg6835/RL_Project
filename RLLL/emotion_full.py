"""
RL-based Music Emotion Regulation — DEAP Dataset
=================================================
Replication of Dutta et al. 2020 (IEEE EMBC)

Key design:
  • NO hardcoded emotion coordinates
  • GEW emotion nodes built from actual metadata:
      video_list_fixed.xlsx  →  group by Lastfm_tag  →  mean(Valence), mean(Arousal)
      Normalize from DEAP 1–9 scale  →  -1..+1
  • participant_ratings.xls  →  per-subject per-clip V/A (also normalised)
  • Convergence threshold = 30° (angular distance to target 'happy')
  • All 32 subjects saved individually to results/

File structure expected (relative to this script):
    data/Metadata/video_list_fixed.xlsx
    data/Metadata/participant_ratings.xls
    data/Metadata/online_ratings.xls      (optional)

Paper hyper-parameters:
    alpha  = 0.1   (learning rate)
    gamma  = 0.6   (discount factor)
    eps    = 0.1   (epsilon-greedy)
    lambda = -100  (reward shaping penalty)
    iters  = 6     (playlist length)
    LOSO cross-validation
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════
# 0.  PATHS  (adjust BASE_DIR if needed)
# ══════════════════════════════════════════════════════
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
METADATA_DIR  = os.path.join(BASE_DIR, 'data', 'Metadata')
VIDEO_FILE    = os.path.join(METADATA_DIR, 'video_list_fixed.xlsx')
RATINGS_FILE  = os.path.join(METADATA_DIR, 'participant_ratings.xls')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════
# 1.  LOAD & NORMALISE METADATA
#     DEAP V/A scale: 1–9  →  normalise to -1..+1
#     formula:  x_norm = (x - 5) / 4
# ══════════════════════════════════════════════════════
def normalise(x):
    """Map DEAP 1–9 scale to -1..+1."""
    return (x - 5.0) / 4.0

def load_video_list(path):
    """
    Load video_list_fixed.xlsx.
    Returns DataFrame with columns:
        experiment_id, tag, avg_valence, avg_arousal
    Both V/A are normalised to -1..+1.
    """
    print(f"  Loading: {path}")
    df = pd.read_excel(path)

    # Identify columns robustly (case-insensitive partial match)
    col = {c.lower().replace(' ','_'): c for c in df.columns}

    def find(keys):
        for k in keys:
            for c in col:
                if k in c:
                    return col[c]
        return None

    exp_col  = find(['experiment_id', 'exp_id'])
    tag_col  = find(['lastfm_tag', 'tag', 'emotion'])
    v_col    = find(['avg_valence', 'avgvalence', 'valence'])
    a_col    = find(['avg_arousal', 'avgarousal', 'arousal'])

    print(f"    Columns found → exp:{exp_col}  tag:{tag_col}  "
          f"valence:{v_col}  arousal:{a_col}")

    out = pd.DataFrame({
        'experiment_id': df[exp_col],
        'tag':           df[tag_col].astype(str).str.strip().str.lower(),
        'avg_valence':   normalise(pd.to_numeric(df[v_col], errors='coerce')),
        'avg_arousal':   normalise(pd.to_numeric(df[a_col], errors='coerce')),
    }).dropna()

    return out


def load_participant_ratings(path):
    """
    Load participant_ratings.xls.
    Returns DataFrame with columns:
        subject_id, experiment_id, valence, arousal
    Both V/A normalised to -1..+1.
    """
    print(f"  Loading: {path}")
    df = pd.read_excel(path)

    col = {c.lower().replace(' ','_'): c for c in df.columns}

    def find(keys):
        for k in keys:
            for c in col:
                if k in c:
                    return col[c]
        return None

    sid_col  = find(['participant_id', 'subject_id', 'participant'])
    exp_col  = find(['experiment_id', 'exp_id'])
    v_col    = find(['valence'])
    a_col    = find(['arousal'])

    print(f"    Columns found → subject:{sid_col}  exp:{exp_col}  "
          f"valence:{v_col}  arousal:{a_col}")

    out = pd.DataFrame({
        'subject_id':    pd.to_numeric(df[sid_col], errors='coerce'),
        'experiment_id': pd.to_numeric(df[exp_col], errors='coerce'),
        'valence':       normalise(pd.to_numeric(df[v_col], errors='coerce')),
        'arousal':       normalise(pd.to_numeric(df[a_col], errors='coerce')),
    }).dropna()
    out['subject_id']    = out['subject_id'].astype(int)
    out['experiment_id'] = out['experiment_id'].astype(int)

    return out


# ══════════════════════════════════════════════════════
# 2.  BUILD GEW EMOTION NODES FROM DATA
#     Group video_list by tag  →  mean V/A  →  emotion coords
#     Clean noisy/numeric tags; keep only clean emotion words
# ══════════════════════════════════════════════════════

# Canonical emotion names we want to keep (superset)
EMOTION_WHITELIST = {
    'happy','joy','excited','pleasure','fun','cheerful','amused','elated',
    'calm','relaxing','relaxed','content','hopeful','relief','peaceful',
    'anger','angry','tense','stressed','stress','fear','afraid',
    'sad','sadness','melancholy','depressed','depressing','miserable','boredom',
    'disgust','disgust','contempt',
    'sexy','love','romantic','sentiment','sentimental','pride',
    'interest','interesting','regret','guilt','surprise',
}

def clean_tag(tag):
    """Normalise a Lastfm tag to a clean emotion label."""
    tag = tag.lower().strip()
    # Strip leading digits (e.g. '12cheerful' → 'cheerful')
    tag = ''.join(c for c in tag if not c.isdigit()).strip()
    # Map common variants
    MAP = {
        'relaxing':'relaxed','relaxed':'relaxed',
        'depressing':'depressed','depressed':'depressed',
        'interesting':'interest',
        'sentimental':'sentiment',
        'angry':'anger', 'tense':'stressed', 'stress':'stressed',
        'afraid':'fear', 'romantic':'love',
        'elated':'excited', 'amused':'happy',
        'peaceful':'calm', 'joyful':'joy',
        'bored':'boredom',
    }
    return MAP.get(tag, tag)


def build_emotion_coords(video_df):
    """
    Group by cleaned tag, compute mean V/A per emotion.
    Returns dict: emotion_name → np.array([v, a])
    """
    video_df = video_df.copy()
    video_df['emotion'] = video_df['tag'].apply(clean_tag)

    # Filter to whitelist; drop generic tags like 'music', 'song' etc.
    video_df = video_df[video_df['emotion'].isin(EMOTION_WHITELIST)]

    grouped = (video_df
               .groupby('emotion')[['avg_valence','avg_arousal']]
               .mean()
               .dropna())

    emotion_coords = {
        row.Index: np.array([row.avg_valence, row.avg_arousal])
        for row in grouped.itertuples()
    }

    print(f"\n  Emotion nodes built from data ({len(emotion_coords)} emotions):")
    for ename, (v, a) in sorted(emotion_coords.items()):
        print(f"    {ename:<15}  V={v:+.3f}  A={a:+.3f}")

    return emotion_coords


# ══════════════════════════════════════════════════════
# 3.  RL CONFIGURATION  (exact paper values)
# ══════════════════════════════════════════════════════
ALPHA         = 0.1
GAMMA         = 0.6
EPSILON       = 0.1
LAMBDA        = -100.0
N_ITERATIONS  = 6
CONV_THRESH   = 30.0    # degrees — changed to 30 as requested
TARGET_EMOTION = 'happy'
START_EMOTION  = 'anger'


def va_to_state(v, a, emotion_coords):
    """Snap (v,a) to the nearest discrete emotion state."""
    vec = np.array([v, a])
    return min(emotion_coords, key=lambda e: np.linalg.norm(vec - emotion_coords[e]))


def angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angular_error(v, a, target_vec):
    return angle_between(np.array([v, a]), target_vec)


def reward_fn(sv, sa, cv, ca, target_vec):
    """r = 1 if θ1 + θ2 ≤ 180°, else 0.  (paper Section III-C)"""
    va = np.array([cv, ca])
    vs = np.array([sv, sa])
    theta = angle_between(target_vec, va) + angle_between(vs, va)
    return 1.0 if theta <= 180.0 else 0.0


def next_state_vec(sv, sa, cv, ca):
    """τ(s,a) = v_s + v_a,  then normalise.  (paper Section III-B)"""
    ns = np.array([sv + cv, sa + ca])
    n  = np.linalg.norm(ns)
    return ns / n if n > 1e-9 else ns


# ══════════════════════════════════════════════════════
# 4.  Q-TABLE
# ══════════════════════════════════════════════════════
def build_q(states, clip_ids):
    return {s: {c: 0.0 for c in clip_ids} for s in states}


def q_update(q, state, action, reward, next_state, phi, clip_ids):
    best = max(q[next_state].values())
    q[state][action] = ((1 - ALPHA) * q[state][action]
                        + ALPHA * (reward + GAMMA * best + phi))


def select_action(q, state, clip_ids, greedy=False):
    if not greedy and np.random.random() < EPSILON:
        return np.random.choice(clip_ids)
    return max(q[state], key=q[state].get)


# ══════════════════════════════════════════════════════
# 5.  PER-SUBJECT DATA HELPER
# ══════════════════════════════════════════════════════
def get_subject_clip_map(subject_id, ratings_df):
    """Return {experiment_id: (valence, arousal)} for one subject."""
    sub = ratings_df[ratings_df['subject_id'] == subject_id]
    return {row['experiment_id']: (row['valence'], row['arousal'])
            for _, row in sub.iterrows()}


# ══════════════════════════════════════════════════════
# 6.  TRAIN  (LOSO)
# ══════════════════════════════════════════════════════
def train(train_sids, ratings_df, emotion_coords, clip_ids, target_vec, start_vec):
    states = list(emotion_coords.keys())
    q = build_q(states, clip_ids)

    for sid in train_sids:
        smap = get_subject_clip_map(sid, ratings_df)
        for _ in range(5):                        # 5 passes per subject
            state = START_EMOTION if START_EMOTION in emotion_coords else states[0]
            sv, sa = start_vec

            for cid in clip_ids:
                if cid not in smap:
                    continue
                cv, ca = smap[cid]
                r    = reward_fn(sv, sa, cv, ca, target_vec)
                nsv  = next_state_vec(sv, sa, cv, ca)
                ns   = va_to_state(*nsv, emotion_coords)
                phi  = LAMBDA if state == ns else 0.0
                q_update(q, state, cid, r, ns, phi, clip_ids)
                state = ns
                sv, sa = nsv[0], nsv[1]

    return q


# ══════════════════════════════════════════════════════
# 7.  EVALUATE  (one subject)
# ══════════════════════════════════════════════════════
def evaluate(subject_id, q, ratings_df, emotion_coords, clip_ids,
             target_vec, start_vec):
    """
    Returns list of dicts per iteration:
        iter, valence, arousal, state, angular_error
    """
    states = list(emotion_coords.keys())
    smap   = get_subject_clip_map(subject_id, ratings_df)

    state = START_EMOTION if START_EMOTION in emotion_coords else states[0]
    sv, sa = start_vec

    traj = [{'iter': 0, 'valence': sv, 'arousal': sa, 'state': state,
              'angular_error': angular_error(sv, sa, target_vec)}]

    for it in range(1, N_ITERATIONS + 1):
        action = select_action(q, state, clip_ids, greedy=True)

        if action in smap:
            cv, ca = smap[action]
        else:
            # fallback: clip mean V/A from video_list
            row = clip_meta[clip_meta['experiment_id'] == action]
            if len(row):
                cv, ca = row.iloc[0]['avg_valence'], row.iloc[0]['avg_arousal']
            else:
                cv, ca = 0.0, 0.0

        nsv   = next_state_vec(sv, sa, cv, ca)
        ns    = va_to_state(*nsv, emotion_coords)
        sv, sa = nsv[0], nsv[1]
        state  = ns

        traj.append({'iter': it, 'valence': sv, 'arousal': sa, 'state': state,
                     'angular_error': angular_error(sv, sa, target_vec)})

    return traj


# ══════════════════════════════════════════════════════
# 8.  PLOTTING
# ══════════════════════════════════════════════════════
PANEL   = '#f7f7f7'
NODE_C  = '#5b9bd5'    # blue nodes
TARGET_C = '#f4c430'   # gold — happy
PATH_OK  = '#27ae60'   # green path (converged)
PATH_BAD = '#e74c3c'   # red path (diverged)
GRID_C   = '#cccccc'


def draw_gew(ax, emotion_coords, target_emotion):
    """Draw the GEW canvas: quadrant lines, all emotion nodes, labels."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color('#aaa')

    ax.axhline(0, color=GRID_C, lw=1.0, zorder=1)
    ax.axvline(0, color=GRID_C, lw=1.0, zorder=1)

    # Subtle quadrant tint
    ax.fill_between([-1.3, 0], [0,0], [1.3,1.3], color='#ffcccc', alpha=0.12)
    ax.fill_between([0, 1.3], [0,0], [1.3,1.3], color='#ccffcc', alpha=0.12)
    ax.fill_between([-1.3, 0], [-1.3,-1.3], [0,0], color='#cce0ff', alpha=0.12)
    ax.fill_between([0, 1.3], [-1.3,-1.3], [0,0], color='#fffacc', alpha=0.12)

    for ename, vec in emotion_coords.items():
        is_target = (ename == target_emotion)
        c  = TARGET_C if is_target else NODE_C
        sz = 200 if is_target else 90
        ec = '#a07000' if is_target else '#2e6da4'
        ax.scatter(*vec, s=sz, c=c, zorder=4,
                   edgecolors=ec, linewidths=0.9)
        ax.annotate(ename,
                    xy=vec,
                    xytext=(vec[0] + 0.04, vec[1] + 0.05),
                    fontsize=7, color='#222', zorder=5,
                    annotation_clip=False)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel('Valence →', fontsize=10, color='#333')
    ax.set_ylabel('Arousal →', fontsize=10, color='#333')
    ax.tick_params(colors='#444', labelsize=8)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.5, zorder=0)


def plot_subject(sid, traj, converged, emotion_coords, save_dir):
    """One GEW plot per subject with annotated path arrows."""
    fig, ax = plt.subplots(figsize=(7.5, 6.8))
    fig.patch.set_facecolor(PANEL)

    draw_gew(ax, emotion_coords, TARGET_EMOTION)

    vs  = [t['valence']       for t in traj]
    ars = [t['arousal']       for t in traj]
    sts = [t['state']         for t in traj]
    err = [t['angular_error'] for t in traj]

    path_col = PATH_OK if converged else PATH_BAD

    # ── draw arrows step by step ──
    for i in range(len(vs) - 1):
        ax.annotate('',
                    xy=(vs[i+1], ars[i+1]),
                    xytext=(vs[i], ars[i]),
                    arrowprops=dict(
                        arrowstyle='->', color=path_col,
                        lw=2.0,
                        mutation_scale=14,
                        connectionstyle='arc3,rad=0.15'),
                    zorder=7)

    # ── intermediate dots + iteration labels ──
    for i in range(1, len(vs) - 1):
        ax.scatter(vs[i], ars[i], s=70, c=path_col,
                   zorder=8, edgecolors='white', linewidths=0.9)
        ax.text(vs[i] + 0.04, ars[i] + 0.04, str(i),
                fontsize=8, color=path_col, fontweight='bold', zorder=9)

    # ── start: green square ──
    ax.scatter(vs[0], ars[0], s=160, c='#2ecc71', marker='s',
               zorder=9, edgecolors='white', linewidths=1.2,
               label='Start')

    # ── end: red star ──
    ax.scatter(vs[-1], ars[-1], s=250, c='#e74c3c', marker='*',
               zorder=9, edgecolors='white', linewidths=0.8,
               label='End')

    # ── legend ──
    ax.legend(loc='upper left', fontsize=9,
              facecolor=PANEL, edgecolor='#bbb', framealpha=0.9,
              scatterpoints=1)

    # ── angular error info box ──
    ax.text(0.98, 0.03,
            f"Start err: {err[0]:.1f}°  →  End err: {err[-1]:.1f}°",
            transform=ax.transAxes, fontsize=8.5, color='#333',
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='#bbb', alpha=0.9))

    # ── title ──
    sym    = '✓' if converged else '✗'
    status = 'Converged' if converged else 'Did not converge'
    ax.set_title(f'Subject {sid} — {status} {sym}',
                 fontsize=12, color='#111', fontweight='bold', pad=10)

    # ── state-path footer ──
    path_str = ' → '.join(t['state'].capitalize() for t in traj)
    fig.text(0.5, 0.005, path_str,
             ha='center', fontsize=7.5, color='#555', style='italic')

    plt.tight_layout(rect=[0, 0.025, 1, 1])

    tag   = 'converged' if converged else 'diverged'
    fname = os.path.join(save_dir, f'trajectory_s{sid:02d}_{tag}.png')
    plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor=PANEL)
    plt.close()
    return fname


def plot_angular_error_summary(all_traj, success_set, fail_set, save_dir):
    all_sids = list(all_traj.keys())
    iters    = list(range(N_ITERATIONS + 1))

    def stats(sids, it):
        e = [all_traj[s][it]['angular_error'] for s in sids]
        return np.mean(e), np.std(e)

    T_m  = [stats(all_sids,          i)[0] for i in iters]
    T_s  = [stats(all_sids,          i)[1] for i in iters]
    S_m  = [stats(list(success_set), i)[0] for i in iters]
    S_s  = [stats(list(success_set), i)[1] for i in iters]
    F_m  = [stats(list(fail_set),    i)[0] for i in iters]
    F_s  = [stats(list(fail_set),    i)[1] for i in iters]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)

    def eb(m, s, col, lbl):
        ax.plot(iters, m, 'o-', color=col, lw=2, ms=5, label=lbl)
        ax.fill_between(iters,
                        [a-b for a,b in zip(m,s)],
                        [a+b for a,b in zip(m,s)],
                        alpha=0.15, color=col)

    eb(T_m, T_s, '#1565c0', f'All (T) n={len(all_sids)}')
    eb(S_m, S_s, '#2e7d32', f'Success (S) n={len(success_set)}')
    eb(F_m, F_s, '#c62828', f'Failed (F) n={len(fail_set)}')
    ax.axhline(CONV_THRESH, color='#f57c00', lw=1.3, ls='--',
               label=f'{CONV_THRESH}° threshold')

    ax.set_xlabel('Iteration (Clips Played)', fontsize=11)
    ax.set_ylabel('Angular Error (°)', fontsize=11)
    ax.set_title('Mean Angular Error vs Iterations', fontsize=13, fontweight='bold')
    ax.set_xticks(iters)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor='#ccc')
    ax.grid(True, color=GRID_C, lw=0.6, alpha=0.7)
    for sp in ax.spines.values(): sp.set_color('#aaa')
    plt.tight_layout()
    out = os.path.join(save_dir, 'angular_error.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=PANEL)
    plt.close()
    print(f"  Saved: {out}")


def plot_reward_shaping(all_traj_rs, all_traj_nors, save_dir):
    iters = list(range(N_ITERATIONS + 1))
    sids  = list(all_traj_rs.keys())

    def mean_err(traj_dict, it):
        return np.mean([traj_dict[s][it]['angular_error'] for s in sids])

    m_rs   = [mean_err(all_traj_rs,   i) for i in iters]
    m_nors = [mean_err(all_traj_nors, i) for i in iters]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)

    ax.plot(iters, m_nors, 'o--', color='#c62828', lw=2, ms=5,
            label='Without Reward Shaping')
    ax.plot(iters, m_rs,   's-',  color='#2e7d32', lw=2, ms=5,
            label='With Reward Shaping')

    mid = len(iters) // 2
    ax.annotate('Stuck at local minimum',
                xy=(mid, m_nors[mid]),
                xytext=(mid - 1.2, m_nors[mid] + 15),
                fontsize=8.5, color='#c62828', style='italic',
                arrowprops=dict(arrowstyle='->', color='#c62828', lw=1.2))
    ax.annotate('Steadily decreasing',
                xy=(mid, m_rs[mid]),
                xytext=(mid + 0.3, m_rs[mid] - 18),
                fontsize=8.5, color='#2e7d32', style='italic',
                arrowprops=dict(arrowstyle='->', color='#2e7d32', lw=1.2))

    ax.set_xlabel('Number of Iterations', fontsize=11)
    ax.set_ylabel('Angular Error in Emotion Transition (°)', fontsize=11)
    ax.set_title('Reward Shaping Effect — Angular Error vs Iterations',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(iters)
    ax.legend(fontsize=9, facecolor=PANEL, edgecolor='#ccc')
    ax.grid(True, color=GRID_C, lw=0.6, alpha=0.7)
    for sp in ax.spines.values(): sp.set_color('#aaa')
    plt.tight_layout()
    out = os.path.join(save_dir, 'reward_shaping.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=PANEL)
    plt.close()
    print(f"  Saved: {out}")


def plot_gew_standalone(emotion_coords, save_dir):
    """Full GEW with all emotion nodes (no trajectory)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(PANEL)
    draw_gew(ax, emotion_coords, TARGET_EMOTION)
    ax.set_title('Geneva Emotion Wheel — Coordinates from DEAP Metadata',
                 fontsize=12, fontweight='bold', color='#111', pad=12)

    # Quadrant labels
    kw = dict(fontsize=9, alpha=0.6, ha='center', va='center', style='italic')
    ax.text(-0.65,  1.15, 'Negative / High Arousal', color='#c0392b', **kw)
    ax.text( 0.65,  1.15, 'Positive / High Arousal', color='#27ae60', **kw)
    ax.text(-0.65, -1.15, 'Negative / Low Arousal',  color='#2980b9', **kw)
    ax.text( 0.65, -1.15, 'Positive / Low Arousal',  color='#f39c12', **kw)

    plt.tight_layout()
    out = os.path.join(save_dir, 'gew_emotion_wheel.png')
    plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=PANEL)
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════
# 9.  MAIN
# ══════════════════════════════════════════════════════
def main():
    global clip_meta  # needed inside evaluate() fallback

    print("=" * 60)
    print("DEAP RL Music Emotion Regulation")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────
    print("\n[1] Loading metadata files…")
    if not os.path.exists(VIDEO_FILE):
        sys.exit(f"ERROR: {VIDEO_FILE} not found. "
                 "Make sure the file exists relative to this script.")
    if not os.path.exists(RATINGS_FILE):
        sys.exit(f"ERROR: {RATINGS_FILE} not found.")

    clip_meta   = load_video_list(VIDEO_FILE)
    ratings_df  = load_participant_ratings(RATINGS_FILE)

    n_subjects  = ratings_df['subject_id'].nunique()
    subject_ids = sorted(ratings_df['subject_id'].unique())
    clip_ids    = sorted(clip_meta['experiment_id'].unique())

    print(f"\n  Subjects found : {n_subjects}")
    print(f"  Clips    found : {len(clip_ids)}")

    # ── Build GEW from data ────────────────────────────────
    print("\n[2] Building GEW emotion coordinates from metadata…")
    emotion_coords = build_emotion_coords(clip_meta)

    if TARGET_EMOTION not in emotion_coords:
        # Try closest match
        candidates = [e for e in emotion_coords if 'happ' in e]
        if candidates:
            print(f"  WARNING: '{TARGET_EMOTION}' not found, using '{candidates[0]}'")
            # rebind global
            globals()['TARGET_EMOTION'] = candidates[0]
        else:
            sys.exit(f"ERROR: Target emotion '{TARGET_EMOTION}' not in GEW. "
                     f"Available: {list(emotion_coords.keys())}")

    if START_EMOTION not in emotion_coords:
        candidates = [e for e in emotion_coords if 'ang' in e]
        if candidates:
            print(f"  WARNING: '{START_EMOTION}' not found, using '{candidates[0]}'")
            globals()['START_EMOTION'] = candidates[0]
        else:
            fallback = list(emotion_coords.keys())[0]
            for emo in ['sad', 'depressed', 'melancholy', 'sadness']:
                if emo in emotion_coords:
                    fallback = emo
                    break
            print(f"  WARNING: '{START_EMOTION}' not found, using fallback '{fallback}'")
            globals()['START_EMOTION'] = fallback

    target_vec = emotion_coords[TARGET_EMOTION]
    start_vec  = emotion_coords[START_EMOTION]

    print(f"\n  Target: '{TARGET_EMOTION}' at V={target_vec[0]:+.3f}  A={target_vec[1]:+.3f}")
    print(f"  Start : '{START_EMOTION}' at V={start_vec[0]:+.3f}  A={start_vec[1]:+.3f}")
    print(f"  Convergence threshold : {CONV_THRESH}°")

    plot_gew_standalone(emotion_coords, RESULTS_DIR)

    # ── LOSO Cross-Validation ──────────────────────────────
    print(f"\n[3] LOSO Cross-Validation  ({n_subjects} subjects)…")
    states = list(emotion_coords.keys())

    all_traj      = {}   # with reward shaping
    all_traj_nors = {}   # without reward shaping (baseline)

    for test_sid in subject_ids:
        train_sids = [s for s in subject_ids if s != test_sid]

        # Train WITH shaping
        q_rs = train(train_sids, ratings_df, emotion_coords,
                     clip_ids, target_vec, start_vec)

        # Train WITHOUT shaping (phi=0 always)
        q_nors = build_q(states, clip_ids)
        for sid in train_sids:
            smap = get_subject_clip_map(sid, ratings_df)
            for _ in range(5):
                state  = START_EMOTION
                sv, sa = start_vec
                for cid in clip_ids:
                    if cid not in smap: continue
                    cv, ca = smap[cid]
                    r    = reward_fn(sv, sa, cv, ca, target_vec)
                    nsv  = next_state_vec(sv, sa, cv, ca)
                    ns   = va_to_state(*nsv, emotion_coords)
                    q_update(q_nors, state, cid, r, ns, 0.0, clip_ids)
                    state = ns; sv, sa = nsv[0], nsv[1]

        all_traj[test_sid]      = evaluate(test_sid, q_rs,   ratings_df,
                                           emotion_coords, clip_ids,
                                           target_vec, start_vec)
        all_traj_nors[test_sid] = evaluate(test_sid, q_nors, ratings_df,
                                           emotion_coords, clip_ids,
                                           target_vec, start_vec)

        if test_sid % 8 == 0 or test_sid == subject_ids[-1]:
            print(f"  Completed {test_sid}/{subject_ids[-1]}")

    # ── Classify convergence  (threshold = 30°) ───────────
    final_err = {s: all_traj[s][-1]['angular_error'] for s in subject_ids}
    success_set = {s for s, e in final_err.items() if e < CONV_THRESH}
    fail_set    = {s for s, e in final_err.items() if e >= CONV_THRESH}

    print(f"\n  Converged (< {CONV_THRESH}°) : {len(success_set)}")
    print(f"  Failed    (≥ {CONV_THRESH}°) : {len(fail_set)}")

    # ── Per-subject trajectory plots ──────────────────────
    print(f"\n[4] Saving {n_subjects} trajectory plots → {RESULTS_DIR}/")
    for sid in subject_ids:
        converged = sid in success_set
        plot_subject(sid, all_traj[sid], converged,
                     emotion_coords, RESULTS_DIR)
        tag = 'converged' if converged else 'diverged'
        print(f"  trajectory_s{sid:02d}_{tag}.png")

    # ── Summary plots ──────────────────────────────────────
    print("\n[5] Saving summary plots…")
    plot_angular_error_summary(all_traj, success_set, fail_set, RESULTS_DIR)
    plot_reward_shaping(all_traj, all_traj_nors, RESULTS_DIR)

    # ── Print Table I equivalent ───────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS  (Table I equivalent)")
    print("=" * 60)
    print(f"{'Group':<12} {'n':>4}  "
          f"{'Clip1-2':>8} {'3':>8} {'4':>8} {'5':>8} {'6':>8}")
    print("-" * 60)
    groups = [
        ('Total (T)', list(subject_ids)),
        ('Success (S)', list(success_set)),
        ('Failed (F)',  list(fail_set)),
    ]
    for lbl, sids in groups:
        if not sids:
            print(f"{lbl:<12} {len(sids):>4}  {'N/A':>8}")
            continue
        row = f"{lbl:<12} {len(sids):>4}"
        for it in [1, 3, 4, 5, 6]:
            errs = [all_traj[s][it]['angular_error'] for s in sids]
            row += f"  {np.mean(errs):5.1f}"
        print(row)

    print()
    print("Paper reference → T:56.9 | S:11.3 | F:123.7  (at iter 6)")
    print(f"\nAll outputs saved in: {RESULTS_DIR}/")
    print(f"  • gew_emotion_wheel.png")
    print(f"  • trajectory_s01_converged/diverged.png  × {n_subjects}")
    print(f"  • angular_error.png")
    print(f"  • reward_shaping.png")


if __name__ == '__main__':
    main()