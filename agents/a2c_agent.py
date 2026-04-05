"""Advantage Actor-Critic (A2C) agent for Catan.

Key improvements over REINFORCE:
  - Learned value function V(s) for credit assignment
  - GAE (Generalized Advantage Estimation) for smooth advantage signal
  - Entropy bonus for exploration
  - Learning rate scheduling
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent
from catan.actions import Action
from catan.decks import RESOURCE_INDEXES
from catan.state import GameState
from training.features import state_to_features, feature_size


class ActorCriticNetwork(nn.Module):
    """Shared trunk with separate policy (actor) and value (critic) heads."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_action_types: int = 17):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor heads (decomposed action scoring)
        self.action_head = nn.Linear(hidden_dim, num_action_types)
        self.vertex_head = nn.Linear(hidden_dim, 54)
        self.edge_head = nn.Linear(hidden_dim, 72)
        self.tile_head = nn.Linear(hidden_dim, 19)
        self.resource_head = nn.Linear(hidden_dim, 5)

        # Critic head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, state_features: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        h = self.shared(state_features)
        policy = {
            "action_type": self.action_head(h),
            "vertex": self.vertex_head(h),
            "edge": self.edge_head(h),
            "tile": self.tile_head(h),
            "resource": self.resource_head(h),
        }
        value = self.value_head(h).squeeze(-1)
        return policy, value


@dataclass
class StepRecord:
    features: list[float]
    action_idx: int
    actions: list[Action]


@dataclass
class EpisodeRecord:
    steps: list[StepRecord] = field(default_factory=list)
    reward: float = 0.0


class A2CAgent(Agent):
    """A2C with GAE and LR scheduling."""

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        input_dim = feature_size()
        self.network = ActorCriticNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.training = True
        self.lr = lr

        self._current_episode: Optional[EpisodeRecord] = None
        self._vertex_to_idx: Optional[dict] = None
        self._edge_to_idx: Optional[dict] = None
        self._ensure_encodings()

    def _ensure_encodings(self) -> None:
        if self._vertex_to_idx is None:
            from catan.board import Board
            board = Board.build(seed=0)
            self._vertex_to_idx = {v: i for i, v in enumerate(board.get_vertex_ids())}
            self._edge_to_idx = {e: i for i, e in enumerate(board.get_edge_ids())}

    def _score_actions(self, scores: dict[str, torch.Tensor], actions: list[Action]) -> torch.Tensor:
        logits = []
        for a in actions:
            s = scores["action_type"][a.action_type.value]
            if a.vertex is not None:
                idx = self._vertex_to_idx.get(a.vertex)
                if idx is not None:
                    s = s + scores["vertex"][idx]
            if a.edge is not None:
                idx = self._edge_to_idx.get(a.edge)
                if idx is not None:
                    s = s + scores["edge"][idx]
            if a.tile is not None:
                s = s + scores["tile"][a.tile]
            if a.resource1 is not None and a.resource1 in RESOURCE_INDEXES:
                s = s + scores["resource"][RESOURCE_INDEXES[a.resource1]]
            if a.give_resource is not None and a.give_resource in RESOURCE_INDEXES:
                s = s + scores["resource"][RESOURCE_INDEXES[a.give_resource]]
            logits.append(s)
        return torch.stack(logits)

    def start_episode(self) -> None:
        self._current_episode = EpisodeRecord()

    def finish_episode(self, reward: float) -> EpisodeRecord:
        ep = self._current_episode
        ep.reward = reward
        self._current_episode = None
        return ep

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        self._ensure_encodings()
        features = state_to_features(state)
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            policy_scores, _ = self.network(feat_tensor)
            action_scores = self._score_actions(policy_scores, legal_actions)
            probs = F.softmax(action_scores, dim=0)

            if self.training:
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = probs.argmax().item()

        if self.training and self._current_episode is not None:
            self._current_episode.steps.append(StepRecord(
                features=features,
                action_idx=idx,
                actions=legal_actions,
            ))

        return legal_actions[idx]

    def _compute_gae(self, values: list[float], reward: float) -> list[float]:
        """Compute GAE advantages.

        GAE blends 1-step TD errors across the trajectory:
          delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
          A_t = sum_{l=0}^{T-t-1} (gamma * lambda)^l * delta_{t+l}

        For Catan: r_t = 0 for all t except terminal, where r_T = reward.
        """
        T = len(values)
        advantages = [0.0] * T
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0  # terminal
                r_t = reward
            else:
                next_value = values[t + 1]
                r_t = 0.0  # no intermediate rewards in Catan

            delta = r_t + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        return advantages

    def update_from_episodes(self, episodes: list[EpisodeRecord]) -> dict[str, float]:
        """Batched A2C update with GAE."""
        if not episodes:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        total_steps = 0

        for ep in episodes:
            if not ep.steps:
                continue

            T = len(ep.steps)

            # First pass: get value estimates for all states (no grad needed here)
            with torch.no_grad():
                values_list = []
                for step in ep.steps:
                    feat = torch.tensor(step.features, dtype=torch.float32, device=self.device)
                    _, v = self.network(feat)
                    values_list.append(v.item())

            # Compute GAE advantages
            advantages = self._compute_gae(values_list, ep.reward)

            # Compute returns (advantage + value = return)
            returns = [adv + val for adv, val in zip(advantages, values_list)]

            # Normalize advantages across this episode
            adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            if T > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

            # Second pass: compute losses with gradients
            for i, step in enumerate(ep.steps):
                feat = torch.tensor(step.features, dtype=torch.float32, device=self.device)
                policy_scores, value = self.network(feat)

                # Policy loss with GAE advantage
                action_scores = self._score_actions(policy_scores, step.actions)
                log_probs = F.log_softmax(action_scores, dim=0)
                total_policy_loss = total_policy_loss - log_probs[step.action_idx] * adv_t[i]

                # Value loss against GAE returns
                total_value_loss = total_value_loss + F.smooth_l1_loss(value, returns_t[i].detach())

                # Entropy
                probs = F.softmax(action_scores, dim=0)
                entropy = -(probs * log_probs).sum()
                total_entropy = total_entropy + entropy

                total_steps += 1

        if total_steps == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "total_loss": 0}

        avg_policy = total_policy_loss / total_steps
        avg_value = total_value_loss / total_steps
        avg_entropy = total_entropy / total_steps

        loss = avg_policy + self.value_coeff * avg_value - self.entropy_coeff * avg_entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": avg_policy.item(),
            "value_loss": avg_value.item(),
            "entropy": avg_entropy.item(),
            "total_loss": loss.item(),
        }

    def adjust_lr(self, progress: float) -> None:
        """Cosine decay: progress goes from 0.0 to 1.0 over training."""
        import math
        new_lr = self.lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = new_lr

    def get_weights(self) -> dict:
        return {k: v.cpu() for k, v in self.network.state_dict().items()}

    def set_weights(self, state_dict: dict) -> None:
        self.network.load_state_dict(state_dict)

    def save(self, path: str, metadata: dict = None) -> None:
        data = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if metadata:
            data["metadata"] = metadata
        torch.save(data, path)

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("metadata", {})
