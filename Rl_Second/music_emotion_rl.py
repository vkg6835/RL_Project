import copy
import json
import os
from collections import deque
from typing import Dict, Iterable, List, Optional

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neural_network import MLPRegressor

from music_emotion_data import (
    angular_distance_deg,
    build_canonical_emotion_space,
    emotion_from_vector,
    normalize_va,
    project_to_unit_circle,
)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def serialize_emotion_space(emotion_space: Dict[str, Iterable[float]]) -> dict:
    return {label: [float(values[0]), float(values[1])] for label, values in emotion_space.items()}


def discretize_state(state_vector: np.ndarray, bins: int) -> int:
    state_vector = np.asarray(state_vector, dtype=np.float32)
    x_idx = int(np.clip(np.round((state_vector[0] + 1.0) * 0.5 * (bins - 1)), 0, bins - 1))
    y_idx = int(np.clip(np.round((state_vector[1] + 1.0) * 0.5 * (bins - 1)), 0, bins - 1))
    return x_idx * bins + y_idx


def state_transition(state_vector: np.ndarray, action_vector: np.ndarray) -> np.ndarray:
    return project_to_unit_circle(np.asarray(state_vector, dtype=np.float32) + np.asarray(action_vector, dtype=np.float32))


def step_reward(
    current_state: np.ndarray,
    next_state: np.ndarray,
    target_vector: np.ndarray,
    convergence_threshold_deg: float = 20.0,
) -> tuple:
    current_error = angular_distance_deg(current_state, target_vector)
    next_error = angular_distance_deg(next_state, target_vector)
    reward = ((current_error - next_error) / 180.0) - 0.02
    if next_error <= convergence_threshold_deg:
        reward += 1.0
    return float(reward), float(current_error), float(next_error)


def build_action_space(clip_catalog: pd.DataFrame) -> dict:
    action_summary = clip_catalog.copy()
    normalized = normalize_va(
        action_summary[["clip_valence", "clip_arousal"]].to_numpy(np.float32)
    )
    action_summary["valence_norm"] = normalized[:, 0]
    action_summary["arousal_norm"] = normalized[:, 1]
    action_summary = action_summary.rename(columns={"Experiment_id": "experiment_id"})

    return {
        "action_ids": action_summary["experiment_id"].astype(np.int32).to_numpy(),
        "action_vectors": action_summary[["valence_norm", "arousal_norm"]].to_numpy(np.float32),
        "catalog": action_summary,
    }


def cap_start_vectors(start_vectors: np.ndarray, max_vectors: int) -> np.ndarray:
    if max_vectors <= 0 or len(start_vectors) <= max_vectors:
        return np.asarray(start_vectors, dtype=np.float32)
    indices = np.linspace(0, len(start_vectors) - 1, max_vectors, dtype=np.int32)
    return np.asarray(start_vectors[indices], dtype=np.float32)


class TabularAgentBase:
    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float, epsilon: float, random_state: int = 42):
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.reset()

    def reset(self):
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def _greedy_action(self, values: np.ndarray, available_actions: List[int]) -> int:
        available_values = values[np.asarray(available_actions, dtype=np.int32)]
        return int(available_actions[int(np.argmax(available_values))])

    def choose_action(self, state_idx: int, available_actions: List[int], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_actions))
        return self._greedy_action(self.Q[state_idx], available_actions)

    def expected_next_value(self, next_state_idx: int, available_actions: List[int]) -> float:
        greedy_action = self._greedy_action(self.Q[next_state_idx], available_actions)
        n_available = len(available_actions)
        expectation = 0.0
        for action in available_actions:
            probability = self.epsilon / n_available
            if action == greedy_action:
                probability += 1.0 - self.epsilon
            expectation += probability * float(self.Q[next_state_idx, action])
        return float(expectation)


class QLearningAgent(TabularAgentBase):
    name = "q_learning"

    def update(self, state_idx: int, action_idx: int, reward: float, next_state_idx: int, next_available_actions: List[int], done: bool):
        best_next = 0.0 if done else float(np.max(self.Q[next_state_idx, next_available_actions]))
        td_target = reward + self.gamma * best_next
        self.Q[state_idx, action_idx] += self.alpha * (td_target - self.Q[state_idx, action_idx])


class SARSAAgent(TabularAgentBase):
    name = "sarsa"

    def update(self, state_idx: int, action_idx: int, reward: float, next_state_idx: int, next_action_idx: Optional[int], done: bool):
        next_value = 0.0 if done or next_action_idx is None else float(self.Q[next_state_idx, next_action_idx])
        td_target = reward + self.gamma * next_value
        self.Q[state_idx, action_idx] += self.alpha * (td_target - self.Q[state_idx, action_idx])


class ExpectedSARSAAgent(TabularAgentBase):
    name = "expected_sarsa"

    def update(self, state_idx: int, action_idx: int, reward: float, next_state_idx: int, next_available_actions: List[int], done: bool):
        next_value = 0.0 if done else self.expected_next_value(next_state_idx, next_available_actions)
        td_target = reward + self.gamma * next_value
        self.Q[state_idx, action_idx] += self.alpha * (td_target - self.Q[state_idx, action_idx])


class DoubleQLearningAgent(TabularAgentBase):
    name = "double_q_learning"

    def reset(self):
        self.Q1 = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.Q2 = np.zeros((self.n_states, self.n_actions), dtype=np.float32)

    def choose_action(self, state_idx: int, available_actions: List[int], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_actions))
        combined = self.Q1[state_idx] + self.Q2[state_idx]
        available_values = combined[np.asarray(available_actions, dtype=np.int32)]
        return int(available_actions[int(np.argmax(available_values))])

    def update(self, state_idx: int, action_idx: int, reward: float, next_state_idx: int, next_available_actions: List[int], done: bool):
        if self.rng.random() < 0.5:
            next_action = int(next_available_actions[int(np.argmax(self.Q1[next_state_idx, next_available_actions]))]) if not done else None
            next_value = 0.0 if done else float(self.Q2[next_state_idx, next_action])
            td_target = reward + self.gamma * next_value
            self.Q1[state_idx, action_idx] += self.alpha * (td_target - self.Q1[state_idx, action_idx])
        else:
            next_action = int(next_available_actions[int(np.argmax(self.Q2[next_state_idx, next_available_actions]))]) if not done else None
            next_value = 0.0 if done else float(self.Q1[next_state_idx, next_action])
            td_target = reward + self.gamma * next_value
            self.Q2[state_idx, action_idx] += self.alpha * (td_target - self.Q2[state_idx, action_idx])


class DQNAgent:
    name = "deep_q_learning"

    def __init__(
        self,
        n_actions: int,
        gamma: float = 0.95,
        epsilon: float = 0.15,
        learning_rate: float = 1e-3,
        hidden_layer_sizes: tuple = (64, 64),
        replay_capacity: int = 4096,
        batch_size: int = 32,
        target_sync_interval: int = 50,
        update_every: int = 4,
        random_state: int = 42,
    ):
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.learning_rate = float(learning_rate)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.replay_capacity = int(replay_capacity)
        self.batch_size = int(batch_size)
        self.target_sync_interval = int(target_sync_interval)
        self.update_every = int(update_every)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)
        self.reset()

    def reset(self):
        self.online = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            learning_rate_init=self.learning_rate,
            max_iter=1,
            warm_start=True,
            random_state=self.random_state,
        )
        zeros_X = np.zeros((2, 2), dtype=np.float32)
        zeros_y = np.zeros((2, self.n_actions), dtype=np.float32)
        self.online.fit(zeros_X, zeros_y)
        self.target = copy.deepcopy(self.online)
        self.replay = deque(maxlen=self.replay_capacity)
        self.update_steps = 0
        self.remembered_steps = 0

    def _predict(self, model: MLPRegressor, states: np.ndarray) -> np.ndarray:
        values = np.asarray(model.predict(np.asarray(states, dtype=np.float32)), dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        return values

    def choose_action(self, state_vector: np.ndarray, available_actions: List[int], training: bool = True) -> int:
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.choice(available_actions))
        q_values = self._predict(self.online, np.asarray(state_vector, dtype=np.float32)[None, :])[0]
        masked = q_values[np.asarray(available_actions, dtype=np.int32)]
        return int(available_actions[int(np.argmax(masked))])

    def remember(self, state, action, reward, next_state, done, next_available_actions):
        self.replay.append(
            (
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
                tuple(int(action_id) for action_id in next_available_actions),
            )
        )
        self.remembered_steps += 1

    def update(self):
        if len(self.replay) < self.batch_size or (self.remembered_steps % self.update_every != 0):
            return

        batch_indices = self.rng.choice(len(self.replay), size=self.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in batch_indices]
        states = np.stack([item[0] for item in batch], axis=0)
        actions = [item[1] for item in batch]
        rewards = np.asarray([item[2] for item in batch], dtype=np.float32)
        next_states = np.stack([item[3] for item in batch], axis=0)
        dones = np.asarray([item[4] for item in batch], dtype=bool)
        next_available = [list(item[5]) for item in batch]

        targets = self._predict(self.online, states)
        next_q_values = self._predict(self.target, next_states)

        for row_idx in range(self.batch_size):
            if dones[row_idx]:
                td_target = rewards[row_idx]
            else:
                td_target = rewards[row_idx] + self.gamma * float(
                    np.max(next_q_values[row_idx, next_available[row_idx]])
                )
            targets[row_idx, actions[row_idx]] = td_target

        self.online.partial_fit(states, targets)
        self.update_steps += 1
        if self.update_steps % self.target_sync_interval == 0:
            self.target = copy.deepcopy(self.online)


def train_tabular_agent(
    agent,
    start_vectors: np.ndarray,
    action_vectors: np.ndarray,
    target_vector: np.ndarray,
    episodes: int,
    playlist_length: int,
    bins: int,
    convergence_threshold_deg: float,
) -> list:
    history = []
    agent.reset()

    for epoch in range(episodes):
        epoch_rewards = []
        epoch_errors = []
        epoch_converged = []

        for start_vector in start_vectors:
            state_vector = project_to_unit_circle(start_vector)
            available_actions = list(range(len(action_vectors)))
            total_reward = 0.0

            for _ in range(playlist_length):
                state_idx = discretize_state(state_vector, bins)
                action_idx = agent.choose_action(state_idx, available_actions, training=True)
                next_state = state_transition(state_vector, action_vectors[action_idx])
                reward, _, next_error = step_reward(
                    state_vector,
                    next_state,
                    target_vector,
                    convergence_threshold_deg=convergence_threshold_deg,
                )
                next_available = [idx for idx in available_actions if idx != action_idx]
                done = next_error <= convergence_threshold_deg or len(next_available) == 0
                next_state_idx = discretize_state(next_state, bins)

                if isinstance(agent, SARSAAgent):
                    next_action = None
                    if not done:
                        next_action = agent.choose_action(next_state_idx, next_available, training=True)
                    agent.update(state_idx, action_idx, reward, next_state_idx, next_action, done)
                elif isinstance(agent, ExpectedSARSAAgent):
                    agent.update(state_idx, action_idx, reward, next_state_idx, next_available, done)
                else:
                    agent.update(state_idx, action_idx, reward, next_state_idx, next_available, done)

                total_reward += reward
                state_vector = next_state
                available_actions = next_available
                if done:
                    break

            final_error = angular_distance_deg(state_vector, target_vector)
            epoch_rewards.append(total_reward)
            epoch_errors.append(final_error)
            epoch_converged.append(final_error <= convergence_threshold_deg)

        history.append(
            {
                "epoch": epoch + 1,
                "mean_reward": float(np.mean(epoch_rewards)),
                "mean_final_error_deg": float(np.mean(epoch_errors)),
                "convergence_rate": float(np.mean(epoch_converged)),
            }
        )

    return history


def train_dqn_agent(
    agent: DQNAgent,
    start_vectors: np.ndarray,
    action_vectors: np.ndarray,
    target_vector: np.ndarray,
    episodes: int,
    playlist_length: int,
    convergence_threshold_deg: float,
) -> list:
    history = []
    agent.reset()

    for epoch in range(episodes):
        epoch_rewards = []
        epoch_errors = []
        epoch_converged = []

        for start_vector in start_vectors:
            state_vector = project_to_unit_circle(start_vector)
            available_actions = list(range(len(action_vectors)))
            total_reward = 0.0

            for _ in range(playlist_length):
                action_idx = agent.choose_action(state_vector, available_actions, training=True)
                next_state = state_transition(state_vector, action_vectors[action_idx])
                reward, _, next_error = step_reward(
                    state_vector,
                    next_state,
                    target_vector,
                    convergence_threshold_deg=convergence_threshold_deg,
                )
                next_available = [idx for idx in available_actions if idx != action_idx]
                done = next_error <= convergence_threshold_deg or len(next_available) == 0
                agent.remember(state_vector, action_idx, reward, next_state, done, next_available)
                agent.update()
                total_reward += reward
                state_vector = next_state
                available_actions = next_available
                if done:
                    break

            final_error = angular_distance_deg(state_vector, target_vector)
            epoch_rewards.append(total_reward)
            epoch_errors.append(final_error)
            epoch_converged.append(final_error <= convergence_threshold_deg)

        history.append(
            {
                "epoch": epoch + 1,
                "mean_reward": float(np.mean(epoch_rewards)),
                "mean_final_error_deg": float(np.mean(epoch_errors)),
                "convergence_rate": float(np.mean(epoch_converged)),
            }
        )

    return history


def rollout_policy(
    agent,
    algorithm_name: str,
    start_vector: np.ndarray,
    action_ids: np.ndarray,
    action_vectors: np.ndarray,
    action_catalog: pd.DataFrame,
    emotion_space: Dict[str, Iterable[float]],
    target_vector: np.ndarray,
    target_emotion: str,
    playlist_length: int,
    bins: int,
    convergence_threshold_deg: float,
) -> dict:
    state_vector = project_to_unit_circle(start_vector)
    state_path = [state_vector.tolist()]
    emotion_path = [emotion_from_vector(state_vector, emotion_space)]
    error_path = [angular_distance_deg(state_vector, target_vector)]
    playlist = []
    available_actions = list(range(len(action_ids)))
    total_reward = 0.0

    for step_idx in range(playlist_length):
        if algorithm_name == DQNAgent.name:
            action_idx = agent.choose_action(state_vector, available_actions, training=False)
        else:
            state_idx = discretize_state(state_vector, bins)
            action_idx = agent.choose_action(state_idx, available_actions, training=False)

        experiment_id = int(action_ids[action_idx])
        clip_row = action_catalog.iloc[action_idx]
        next_state = state_transition(state_vector, action_vectors[action_idx])
        reward, _, next_error = step_reward(
            state_vector,
            next_state,
            target_vector,
            convergence_threshold_deg=convergence_threshold_deg,
        )
        total_reward += reward

        playlist.append(
            {
                "rank": step_idx + 1,
                "experiment_id": experiment_id,
                "artist": str(clip_row.get("Artist", "")),
                "title": str(clip_row.get("Title", "")),
                "tag": str(clip_row.get("Lastfm_tag", "")),
                "canonical_emotion": str(clip_row.get("canonical_emotion", "")),
                "youtube_link": str(clip_row.get("Youtube_link", "")),
            }
        )
        state_vector = next_state
        state_path.append(state_vector.tolist())
        emotion_path.append(emotion_from_vector(state_vector, emotion_space))
        error_path.append(next_error)
        available_actions = [idx for idx in available_actions if idx != action_idx]

        if next_error <= convergence_threshold_deg or len(available_actions) == 0:
            break

    return {
        "target_emotion": target_emotion,
        "start_emotion": emotion_path[0],
        "final_emotion": emotion_path[-1],
        "playlist": playlist,
        "emotion_path": emotion_path,
        "state_path": state_path,
        "error_path_deg": error_path,
        "initial_error_deg": float(error_path[0]),
        "final_error_deg": float(error_path[-1]),
        "total_reward": float(total_reward),
        "converged": bool(error_path[-1] <= convergence_threshold_deg),
    }


def build_algorithm_registry(alpha: float, gamma: float, epsilon: float, random_state: int) -> Dict[str, callable]:
    return {
        QLearningAgent.name: lambda n_states, n_actions: QLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            random_state=random_state,
        ),
        SARSAAgent.name: lambda n_states, n_actions: SARSAAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            random_state=random_state,
        ),
        ExpectedSARSAAgent.name: lambda n_states, n_actions: ExpectedSARSAAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            random_state=random_state,
        ),
        DoubleQLearningAgent.name: lambda n_states, n_actions: DoubleQLearningAgent(
            n_states=n_states,
            n_actions=n_actions,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            random_state=random_state,
        ),
        DQNAgent.name: lambda n_states, n_actions: DQNAgent(
            n_actions=n_actions,
            gamma=gamma,
            epsilon=epsilon,
            random_state=random_state,
        ),
    }


def plot_training_history(history_df: pd.DataFrame, algorithm_name: str, save_dir: str) -> None:
    subset = history_df[history_df["algorithm"] == algorithm_name]
    mean_curve = (
        subset.groupby("epoch", as_index=False)
        .agg(
            mean_reward=("mean_reward", "mean"),
            mean_final_error_deg=("mean_final_error_deg", "mean"),
            convergence_rate=("convergence_rate", "mean"),
        )
        .sort_values("epoch")
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(mean_curve["epoch"], mean_curve["mean_reward"], color="#219ebc", marker="o")
    axes[0].set_title(f"{algorithm_name}: Training Reward")
    axes[1].plot(mean_curve["epoch"], mean_curve["mean_final_error_deg"], color="#fb8500", marker="o")
    axes[1].axhline(20.0, ls="--", color="black", lw=1)
    axes[1].set_title(f"{algorithm_name}: Final Error")
    axes[2].plot(mean_curve["epoch"], mean_curve["convergence_rate"], color="#023047", marker="o")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_title(f"{algorithm_name}: Convergence Rate")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        if len(mean_curve) == 1:
            epoch_value = float(mean_curve["epoch"].iloc[0])
            ax.set_xlim(epoch_value - 0.5, epoch_value + 0.5)
            ax.set_xticks([int(epoch_value)])
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{algorithm_name}_training_curves.png"), dpi=160)
    plt.close()


def plot_error_trajectory(trial_df: pd.DataFrame, algorithm_name: str, save_dir: str) -> None:
    subset = trial_df[trial_df["algorithm"] == algorithm_name].copy()
    paths = [json.loads(path_json) for path_json in subset["error_path_deg_json"]]
    max_len = max(len(path) for path in paths)
    padded = []
    for path in paths:
        values = list(float(value) for value in path)
        values.extend([values[-1]] * (max_len - len(values)))
        padded.append(values)
    padded = np.asarray(padded, dtype=np.float32)
    mean_step = pd.DataFrame(
        {
            "step": np.arange(max_len, dtype=np.int32),
            "error_deg": padded.mean(axis=0),
        }
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mean_step["step"], mean_step["error_deg"], marker="o", color="#8ecae6")
    ax.axhline(20.0, ls="--", color="red", lw=1, label="Converged <= 20 deg")
    ax.set_title(f"{algorithm_name}: Mean Angular Error by Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("Angular Error (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{algorithm_name}_mean_error_by_step.png"), dpi=160)
    plt.close()


def plot_sample_trajectories(
    trial_df: pd.DataFrame,
    algorithm_name: str,
    emotion_space: Dict[str, Iterable[float]],
    target_emotion: str,
    save_dir: str,
) -> None:
    subset = trial_df[trial_df["algorithm"] == algorithm_name].copy()
    chosen = pd.concat(
        [
            subset.nsmallest(2, "final_error_deg"),
            subset.nlargest(2, "final_error_deg"),
        ],
        axis=0,
    ).drop_duplicates(subset=["subject_id", "trial"])
    if chosen.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    boundary = plt.Circle((0, 0), 1.0, fill=False, ls="--", color="gray", alpha=0.4)
    ax.add_patch(boundary)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.axvline(0, color="gray", lw=0.8, ls="--")
    for label, center in emotion_space.items():
        color = "#ffb703" if label == target_emotion else "#8ecae6"
        ax.scatter(center[0], center[1], s=110, color=color, edgecolors="black", linewidths=0.6)
        ax.text(center[0] + 0.03, center[1] + 0.03, label, fontsize=8)

    for row in chosen.itertuples(index=False):
        points = np.asarray(json.loads(row.state_path_json), dtype=np.float32)
        ax.plot(points[:, 0], points[:, 1], marker="o", alpha=0.75, label=f"S{int(row.subject_id):02d}T{int(row.trial):02d}")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_title(f"{algorithm_name}: Sample Emotion Trajectories")
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{algorithm_name}_sample_trajectories.png"), dpi=160)
    plt.close()


def plot_comparison_summary(summary_df: pd.DataFrame, trial_df: pd.DataFrame, save_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(summary_df["algorithm"], summary_df["convergence_rate"], color="#219ebc")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("RL Convergence Rate (<= 20 deg)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.25)

    sns.boxplot(data=trial_df, x="algorithm", y="final_error_deg", ax=axes[1], color="#ffb703")
    axes[1].axhline(20.0, ls="--", color="red", lw=1)
    axes[1].set_title("RL Final Angular Error")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(True, axis="y", alpha=0.25)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rl_algorithm_comparison.png"), dpi=160)
    plt.close()


def save_policy_artifact(artifact: dict, save_path: str) -> None:
    joblib.dump(artifact, save_path)


def load_policy_artifact(save_path: str) -> dict:
    artifact = joblib.load(save_path)
    artifact["emotion_space"] = {
        label: tuple(values) for label, values in artifact["emotion_space"].items()
    }
    artifact["target_vector"] = np.asarray(artifact["target_vector"], dtype=np.float32)
    artifact["action_ids"] = np.asarray(artifact["action_ids"], dtype=np.int32)
    artifact["action_vectors"] = np.asarray(artifact["action_vectors"], dtype=np.float32)
    artifact["action_catalog"] = pd.DataFrame(artifact["action_catalog"])
    return artifact


def run_rl_benchmark(
    dataset: dict,
    prediction_frame: pd.DataFrame,
    output_dir: str = "output",
    target_emotion: str = "happy",
    episodes: int = 12,
    playlist_length: int = 6,
    bins: int = 21,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon: float = 0.15,
    convergence_threshold_deg: float = 20.0,
    random_state: int = 42,
    max_train_starts: int = 256,
) -> dict:
    if target_emotion not in build_canonical_emotion_space():
        raise KeyError(
            f"Unknown target emotion '{target_emotion}'. "
            f"Choose from: {', '.join(build_canonical_emotion_space().keys())}"
        )

    rl_dir = ensure_dir(os.path.join(output_dir, "rl"))
    models_dir = ensure_dir(os.path.join(output_dir, "models"))
    emotion_space = dataset["emotion_space"]
    target_vector = np.asarray(emotion_space[target_emotion], dtype=np.float32)
    prediction_lookup = prediction_frame.set_index(["subject_id", "trial"])
    algorithms = build_algorithm_registry(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        random_state=random_state,
    )

    training_rows = []
    trial_rows = []

    for holdout_subject in np.unique(dataset["subject_ids"]):
        print(f"[RL] Fold subject {int(holdout_subject):02d}")
        train_mask = dataset["subject_ids"] != holdout_subject
        test_mask = dataset["subject_ids"] == holdout_subject
        action_space = build_action_space(dataset["clip_catalog"])
        start_vectors = cap_start_vectors(dataset["normalized_va"][train_mask], max_train_starts)

        for algorithm_name, builder in algorithms.items():
            print(f"  [RL] Training {algorithm_name}")
            agent = builder(n_states=bins * bins, n_actions=len(action_space["action_ids"]))
            if algorithm_name == DQNAgent.name:
                history = train_dqn_agent(
                    agent=agent,
                    start_vectors=start_vectors,
                    action_vectors=action_space["action_vectors"],
                    target_vector=target_vector,
                    episodes=episodes,
                    playlist_length=playlist_length,
                    convergence_threshold_deg=convergence_threshold_deg,
                )
            else:
                history = train_tabular_agent(
                    agent=agent,
                    start_vectors=start_vectors,
                    action_vectors=action_space["action_vectors"],
                    target_vector=target_vector,
                    episodes=episodes,
                    playlist_length=playlist_length,
                    bins=bins,
                    convergence_threshold_deg=convergence_threshold_deg,
                )

            for history_row in history:
                training_rows.append(
                    {
                        "algorithm": algorithm_name,
                        "holdout_subject": int(holdout_subject),
                        **history_row,
                    }
                )

            test_indices = np.flatnonzero(test_mask)
            for sample_idx in test_indices:
                subject_id = int(dataset["subject_ids"][sample_idx])
                trial_number = int(dataset["trial_numbers"][sample_idx])
                prediction = prediction_lookup.loc[(subject_id, trial_number)]
                predicted_vector = normalize_va(
                    [prediction["predicted_valence"], prediction["predicted_arousal"]]
                )
                rollout = rollout_policy(
                    agent=agent,
                    algorithm_name=algorithm_name,
                    start_vector=predicted_vector,
                    action_ids=action_space["action_ids"],
                    action_vectors=action_space["action_vectors"],
                    action_catalog=action_space["catalog"],
                    emotion_space=emotion_space,
                    target_vector=target_vector,
                    target_emotion=target_emotion,
                    playlist_length=playlist_length,
                    bins=bins,
                    convergence_threshold_deg=convergence_threshold_deg,
                )
                trial_rows.append(
                    {
                        "algorithm": algorithm_name,
                        "subject_id": subject_id,
                        "trial": trial_number,
                        "true_emotion": str(dataset["y_emotion"][sample_idx]),
                        "predicted_start_emotion": str(prediction["predicted_emotion"]),
                        "classifier_model": str(prediction.get("model_name", "")),
                        "initial_error_deg": rollout["initial_error_deg"],
                        "final_error_deg": rollout["final_error_deg"],
                        "converged": rollout["converged"],
                        "total_reward": rollout["total_reward"],
                        "start_emotion": rollout["start_emotion"],
                        "final_emotion": rollout["final_emotion"],
                        "emotion_path_json": json.dumps(rollout["emotion_path"]),
                        "state_path_json": json.dumps(rollout["state_path"]),
                        "error_path_deg_json": json.dumps(rollout["error_path_deg"]),
                        "playlist_json": json.dumps(rollout["playlist"]),
                    }
                )

    training_df = pd.DataFrame(training_rows)
    trial_df = pd.DataFrame(trial_rows)
    summary_df = (
        trial_df.groupby("algorithm", as_index=False)
        .agg(
            convergence_rate=("converged", "mean"),
            mean_final_error_deg=("final_error_deg", "mean"),
            std_final_error_deg=("final_error_deg", "std"),
            mean_total_reward=("total_reward", "mean"),
        )
        .sort_values(
            ["convergence_rate", "mean_final_error_deg", "mean_total_reward"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )

    best_algorithm = str(summary_df.iloc[0]["algorithm"])
    training_df.to_csv(os.path.join(rl_dir, "rl_training_history.csv"), index=False)
    trial_df.to_csv(os.path.join(rl_dir, "rl_trial_results.csv"), index=False)
    summary_df.to_csv(os.path.join(rl_dir, "rl_summary.csv"), index=False)
    plot_comparison_summary(summary_df, trial_df, rl_dir)

    for algorithm_name in summary_df["algorithm"]:
        plot_training_history(training_df, algorithm_name, rl_dir)
        plot_error_trajectory(trial_df, algorithm_name, rl_dir)
        plot_sample_trajectories(trial_df, algorithm_name, emotion_space, target_emotion, rl_dir)

    full_action_space = build_action_space(dataset["clip_catalog"])
    full_start_vectors = cap_start_vectors(dataset["normalized_va"], max_train_starts)
    best_policies = {}
    for algorithm_name, builder in algorithms.items():
        agent = builder(n_states=bins * bins, n_actions=len(full_action_space["action_ids"]))
        if algorithm_name == DQNAgent.name:
            train_dqn_agent(
                agent=agent,
                start_vectors=full_start_vectors,
                action_vectors=full_action_space["action_vectors"],
                target_vector=target_vector,
                episodes=episodes,
                playlist_length=playlist_length,
                convergence_threshold_deg=convergence_threshold_deg,
            )
        else:
            train_tabular_agent(
                agent=agent,
                start_vectors=full_start_vectors,
                action_vectors=full_action_space["action_vectors"],
                target_vector=target_vector,
                episodes=episodes,
                playlist_length=playlist_length,
                bins=bins,
                convergence_threshold_deg=convergence_threshold_deg,
            )

        artifact = {
            "algorithm": algorithm_name,
            "agent": agent,
            "action_ids": full_action_space["action_ids"].tolist(),
            "action_vectors": full_action_space["action_vectors"].tolist(),
            "action_catalog": full_action_space["catalog"].to_dict(orient="records"),
            "target_emotion": target_emotion,
            "target_vector": target_vector.tolist(),
            "emotion_space": serialize_emotion_space(emotion_space),
            "bins": int(bins),
            "playlist_length": int(playlist_length),
            "convergence_threshold_deg": float(convergence_threshold_deg),
        }
        save_path = os.path.join(models_dir, f"policy_{algorithm_name}.joblib")
        save_policy_artifact(artifact, save_path)
        best_policies[algorithm_name] = save_path

    with open(os.path.join(models_dir, "rl_model_selection.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "best_algorithm": best_algorithm,
                "target_emotion": target_emotion,
                "convergence_threshold_deg": convergence_threshold_deg,
                "policy_artifacts": best_policies,
            },
            fh,
            indent=2,
        )

    return {
        "summary": summary_df,
        "trial_results": trial_df,
        "training_history": training_df,
        "best_algorithm": best_algorithm,
        "policy_artifacts": best_policies,
    }


def recommend_playlist_from_state(
    predicted_state_vector: np.ndarray,
    artifacts_dir: str = "output",
    algorithm_name: Optional[str] = None,
    playlist_length: Optional[int] = None,
) -> dict:
    selection_path = os.path.join(artifacts_dir, "models", "rl_model_selection.json")
    with open(selection_path, "r", encoding="utf-8") as fh:
        selection = json.load(fh)

    algorithm_name = algorithm_name or selection["best_algorithm"]
    artifact = load_policy_artifact(
        os.path.join(artifacts_dir, "models", f"policy_{algorithm_name}.joblib")
    )
    if playlist_length is None:
        playlist_length = int(artifact["playlist_length"])

    rollout = rollout_policy(
        agent=artifact["agent"],
        algorithm_name=algorithm_name,
        start_vector=np.asarray(predicted_state_vector, dtype=np.float32),
        action_ids=artifact["action_ids"],
        action_vectors=artifact["action_vectors"],
        action_catalog=artifact["action_catalog"],
        emotion_space=artifact["emotion_space"],
        target_vector=artifact["target_vector"],
        target_emotion=artifact["target_emotion"],
        playlist_length=playlist_length,
        bins=int(artifact["bins"]),
        convergence_threshold_deg=float(artifact["convergence_threshold_deg"]),
    )
    rollout["algorithm"] = algorithm_name
    return rollout
