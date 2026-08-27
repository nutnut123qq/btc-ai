"""
Deep Reinforcement Learning (Q-Learning / Policy Gradient) Self-Improving Agent
for Bitcoin Market Direction Prediction & Position Optimization.
"""

import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("rl_agent")

DEFAULT_MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODELS_DIR / "q_table_state.json"


class QTradingAgent:
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 3,
        lr: float = 0.05,
        gamma: float = 0.95,
        checkpoint_path: Optional[Path] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim  # 0: Bullish (Long), 1: Bearish (Short), 2: Sideways (Hold)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 0.10  # Exploration rate
        self.q_table: Dict[str, List[float]] = {}
        self.action_names = ["Bullish", "Bearish", "Sideways"]
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self._updates_count = 0

        # Auto-load existing policy from disk on initialization
        self.load_policy()

    def _discretize_state(self, state_features: List[float]) -> str:
        """Converts continuous feature vector into discrete state hash key."""
        buckets = []
        for val in state_features:
            if val > 0.60:
                b = "H"  # High
            elif val < 0.40:
                b = "L"  # Low
            else:
                b = "M"  # Mid
            buckets.append(b)
        return "_".join(buckets)

    def get_action(self, state_features: List[float]) -> Tuple[str, float]:
        """Selects action via Epsilon-Greedy policy with Q-value Confidence."""
        state_key = self._discretize_state(state_features)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]

        q_values = self.q_table[state_key]

        if random.random() < self.epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            action_idx = q_values.index(max(q_values))

        # Softmax confidence calculation
        exp_q = [math.exp(q) for q in q_values]
        sum_exp = sum(exp_q) if sum(exp_q) > 0 else 1.0
        confidence = exp_q[action_idx] / sum_exp

        return self.action_names[action_idx], confidence

    def update_q_value(
        self,
        state_features: List[float],
        action_name: str,
        reward: float,
        next_state_features: List[float],
    ):
        """Updates Q-table state-action value based on realized reward (Bellman Equation) and auto-saves."""
        state_key = self._discretize_state(state_features)
        next_state_key = self._discretize_state(next_state_features)

        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0, 0.0, 0.0]

        action_idx = self.action_names.index(action_name) if action_name in self.action_names else 2
        best_next_q = max(self.q_table[next_state_key])

        current_q = self.q_table[state_key][action_idx]
        new_q = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

        self._updates_count += 1
        # Auto-persist state to disk
        self.save_policy()

    def save_policy(self, file_path: Optional[Path | str] = None) -> bool:
        """Atomically saves Q-table state and hyperparameters to JSON checkpoint."""
        target_path = Path(file_path) if file_path else self.checkpoint_path
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "metadata": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "lr": self.lr,
                    "gamma": self.gamma,
                    "epsilon": self.epsilon,
                    "updates_count": self._updates_count,
                    "states_count": len(self.q_table),
                },
                "q_table": self.q_table,
            }
            tmp_path = target_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(target_path)
            return True
        except Exception as e:
            logger.error(f"Failed to save Q-table checkpoint to {target_path}: {e}")
            return False

    def load_policy(self, file_path: Optional[Path | str] = None) -> bool:
        """Loads Q-table state from JSON checkpoint with corrupted file fallback."""
        target_path = Path(file_path) if file_path else self.checkpoint_path
        if not target_path.exists():
            return False

        try:
            content = target_path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, dict) and "q_table" in data and isinstance(data["q_table"], dict):
                self.q_table = data["q_table"]
                meta = data.get("metadata", {})
                self._updates_count = meta.get("updates_count", 0)
                logger.info(f"Loaded Q-table from {target_path} ({len(self.q_table)} states).")
                return True
            else:
                logger.warning(f"Invalid schema in Q-table file {target_path}, initializing fresh table.")
                self.q_table = {}
                return False
        except Exception as e:
            logger.warning(f"Corrupted Q-table file at {target_path} ({e}), initializing fresh table.")
            self.q_table = {}
            return False


# Global Singleton RL Agent Instance
global_rl_agent = QTradingAgent()
