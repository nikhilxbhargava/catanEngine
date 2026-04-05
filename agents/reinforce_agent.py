"""REINFORCE (policy gradient) agent for Catan.

A simple MLP policy network that maps game state features to action
probabilities over the legal action set. Trained via REINFORCE with
baseline subtraction and batched multi-game updates.
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


class SimplePolicy(nn.Module):
    """State -> decomposed action scores.

    Maps state features to scores for each action component (type, vertex,
    edge, tile, resource). The total score for an action is the sum of its
    component scores, which are then softmaxed over the legal action set.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_action_types: int = 17):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_action_types)
        self.vertex_head = nn.Linear(hidden_dim, 54)
        self.edge_head = nn.Linear(hidden_dim, 72)
        self.tile_head = nn.Linear(hidden_dim, 19)
        self.resource_head = nn.Linear(hidden_dim, 5)

    def forward(self, state_features: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass. Accepts (input_dim,) or (batch, input_dim)."""
        h = self.shared(state_features)
        return {
            "action_type": self.action_head(h),
            "vertex": self.vertex_head(h),
            "edge": self.edge_head(h),
            "tile": self.tile_head(h),
            "resource": self.resource_head(h),
        }


# ── Trajectory recording (picklable, no tensors) ────────────────────


@dataclass
class StepRecord:
    """One decision point: features + action index chosen + action list."""
    features: list[float]
    action_idx: int
    actions: list[Action]


@dataclass
class EpisodeRecord:
    """Full episode trajectory for one player."""
    steps: list[StepRecord] = field(default_factory=list)
    reward: float = 0.0


# ── Agent ────────────────────────────────────────────────────────────


class ReinforceAgent(Agent):
    """REINFORCE agent with batched multi-game updates.

    In training mode, records trajectories as plain data (no tensors).
    After a batch of games, call `update_from_episodes()` to do a single
    batched gradient step.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        input_dim = feature_size()
        self.network = SimplePolicy(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.training = True

        # Current episode being recorded
        self._current_episode: Optional[EpisodeRecord] = None

        # Action encoding helpers (built lazily from first board seen)
        self._vertex_to_idx: Optional[dict] = None
        self._edge_to_idx: Optional[dict] = None
        self._ensure_encodings_from_scratch()

    def _ensure_encodings_from_scratch(self) -> None:
        """Build vertex/edge index maps from a fresh board.
        All standard boards have the same topology, so we can use any seed."""
        if self._vertex_to_idx is None:
            from catan.board import Board
            board = Board.build(seed=0)
            self._vertex_to_idx = {v: i for i, v in enumerate(board.get_vertex_ids())}
            self._edge_to_idx = {e: i for i, e in enumerate(board.get_edge_ids())}

    def _ensure_encodings(self, state: GameState) -> None:
        if self._vertex_to_idx is None:
            self._ensure_encodings_from_scratch()

    def _score_actions(self, scores: dict[str, torch.Tensor], actions: list[Action]) -> torch.Tensor:
        """Score a list of actions, return tensor of shape (num_actions,)."""
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
        """Call before each game to start recording."""
        self._current_episode = EpisodeRecord()

    def finish_episode(self, reward: float) -> EpisodeRecord:
        """Call after each game. Returns the recorded episode."""
        ep = self._current_episode
        ep.reward = reward
        self._current_episode = None
        return ep

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        self._ensure_encodings(state)

        features = state_to_features(state)
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            scores = self.network(feat_tensor)
            action_scores = self._score_actions(scores, legal_actions)
            probs = F.softmax(action_scores, dim=0)

            if self.training:
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = probs.argmax().item()

        # Record step (plain data, no tensors — picklable for multiprocessing)
        if self.training and self._current_episode is not None:
            self._current_episode.steps.append(StepRecord(
                features=features,
                action_idx=idx,
                actions=legal_actions,
            ))

        return legal_actions[idx]

    def update_from_episodes(self, episodes: list[EpisodeRecord]) -> float:
        """Batched policy gradient update from multiple episodes.

        Replays all steps through the network with gradients enabled,
        computes REINFORCE loss, and does a single optimizer step.

        Returns average loss.
        """
        if not episodes:
            return 0.0

        total_loss = torch.tensor(0.0, device=self.device)
        total_steps = 0

        for ep in episodes:
            if not ep.steps:
                continue

            # Compute discounted returns for each step
            T = len(ep.steps)
            returns = [0.0] * T
            R = ep.reward
            for t in reversed(range(T)):
                returns[t] = R
                R = self.gamma * R

            returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            if T > 1:
                returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

            # Replay each step with gradients
            for step, G in zip(ep.steps, returns_t):
                feat = torch.tensor(step.features, dtype=torch.float32, device=self.device)
                scores = self.network(feat)
                action_scores = self._score_actions(scores, step.actions)
                log_probs = F.log_softmax(action_scores, dim=0)
                total_loss = total_loss - log_probs[step.action_idx] * G
                total_steps += 1

        if total_steps == 0:
            return 0.0

        avg_loss = total_loss / total_steps

        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return avg_loss.item()

    def get_weights(self) -> dict:
        """Return network weights as CPU state dict (for sending to workers)."""
        return {k: v.cpu() for k, v in self.network.state_dict().items()}

    def set_weights(self, state_dict: dict) -> None:
        """Load weights (e.g., received from main process)."""
        self.network.load_state_dict(state_dict)

    def save(self, path: str) -> None:
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
