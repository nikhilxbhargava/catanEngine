"""PPO (Proximal Policy Optimization) agent for Catan.

Key improvements over A2C:
  - Clipped surrogate objective for stable policy updates
  - Multiple gradient epochs per batch of experience
  - Minibatch training within each epoch
  - Deeper residual network with layer normalization
  - Per-step shaped rewards for faster credit assignment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import Agent
from catan.actions import Action
from catan.decks import RESOURCE_INDEXES
from catan.state import GameState
from training.features import state_to_features, feature_size


class ResBlock(nn.Module):
    """Residual block with layer normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.ln(x + residual)
        return F.relu(x)


class PPOActorCriticNetwork(nn.Module):
    """Deeper network with residual blocks and separate critic trunk."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_action_types: int = 17):
        super().__init__()

        # Shared trunk: projection + 2 residual blocks
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.shared_res1 = ResBlock(hidden_dim)
        self.shared_res2 = ResBlock(hidden_dim)

        # Actor: extra layer before decomposed heads
        self.actor_pre = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.action_head = nn.Linear(hidden_dim, num_action_types)
        self.vertex_head = nn.Linear(hidden_dim, 54)
        self.edge_head = nn.Linear(hidden_dim, 72)
        self.tile_head = nn.Linear(hidden_dim, 19)
        self.resource_head = nn.Linear(hidden_dim, 5)

        # Critic: separate residual block + value output (no Tanh — unbounded for shaped rewards)
        self.critic_res = ResBlock(hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Orthogonal initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.zeros_(m.bias)

        # Small init for policy heads (near-uniform initial policy)
        for head in [self.action_head, self.vertex_head, self.edge_head,
                     self.tile_head, self.resource_head]:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)

        # Standard init for value head
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, state_features: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        h = self.projection(state_features)
        h = self.shared_res1(h)
        h = self.shared_res2(h)

        # Actor
        a = self.actor_pre(h)
        policy = {
            "action_type": self.action_head(a),
            "vertex": self.vertex_head(a),
            "edge": self.edge_head(a),
            "tile": self.tile_head(a),
            "resource": self.resource_head(a),
        }

        # Critic (separate path)
        c = self.critic_res(h)
        value = self.value_head(c).squeeze(-1)

        return policy, value


@dataclass
class StepRecord:
    features: list[float]
    action_idx: int
    actions: list[Action]
    log_prob: float = 0.0
    value: float = 0.0
    reward: float = 0.0  # intermediate shaped reward


@dataclass
class EpisodeRecord:
    steps: list[StepRecord] = field(default_factory=list)
    reward: float = 0.0  # terminal reward


class PPOAgent(Agent):
    """PPO with GAE, clipped surrogate, and reward shaping support."""

    def __init__(
        self,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        target_kl: float = 0.02,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        input_dim = feature_size()
        self.network = PPOActorCriticNetwork(input_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
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
        # Add terminal reward to last step
        if ep.steps:
            ep.steps[-1].reward += reward
        self._current_episode = None
        return ep

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        self._ensure_encodings()
        features = state_to_features(state)
        feat_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            policy_scores, value = self.network(feat_tensor)
            action_scores = self._score_actions(policy_scores, legal_actions)
            probs = F.softmax(action_scores, dim=0)

            if self.training:
                idx = torch.multinomial(probs, 1).item()
            else:
                idx = probs.argmax().item()

            log_prob = torch.log(probs[idx] + 1e-8).item()

        if self.training and self._current_episode is not None:
            self._current_episode.steps.append(StepRecord(
                features=features,
                action_idx=idx,
                actions=legal_actions,
                log_prob=log_prob,
                value=value.item(),
            ))

        return legal_actions[idx]

    def add_shaped_reward(self, reward: float) -> None:
        """Add intermediate shaped reward to the most recent step."""
        if self._current_episode and self._current_episode.steps:
            self._current_episode.steps[-1].reward += reward

    def _compute_gae(self, values: list[float], rewards: list[float]) -> tuple[list[float], list[float]]:
        """Compute GAE advantages and returns from per-step rewards."""
        T = len(values)
        advantages = [0.0] * T
        gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0  # terminal
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update_from_episodes(self, episodes: list[EpisodeRecord]) -> dict[str, float]:
        """PPO update: multiple epochs over collected experience.

        Optimized: batches the network trunk forward pass, only does per-step
        action scoring (which can't be batched due to variable action sets).
        """
        # Flatten all episodes into a single buffer
        all_features = []
        all_action_idx = []
        all_actions = []
        all_old_log_probs = []
        all_old_values = []
        all_advantages = []
        all_returns = []

        for ep in episodes:
            if not ep.steps:
                continue

            values = [s.value for s in ep.steps]
            rewards = [s.reward for s in ep.steps]

            advantages, returns = self._compute_gae(values, rewards)

            for i, step in enumerate(ep.steps):
                all_features.append(step.features)
                all_action_idx.append(step.action_idx)
                all_actions.append(step.actions)
                all_old_log_probs.append(step.log_prob)
                all_old_values.append(step.value)
                all_advantages.append(advantages[i])
                all_returns.append(returns[i])

        N = len(all_features)
        if N == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0,
                    "total_loss": 0, "clip_frac": 0, "approx_kl": 0}

        # Pre-compute all feature tensors
        features_t = torch.tensor(all_features, dtype=torch.float32, device=self.device)

        adv_t = torch.tensor(all_advantages, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        returns_t = torch.tensor(all_returns, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32, device=self.device)
        old_values_t = torch.tensor(all_old_values, dtype=torch.float32, device=self.device)

        # PPO epochs
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_approx_kl = 0.0
        num_updates = 0

        indices = list(range(N))

        for epoch in range(self.ppo_epochs):
            import random
            random.shuffle(indices)

            for start in range(0, N, self.minibatch_size):
                batch_idx = indices[start:start + self.minibatch_size]
                mb_size = len(batch_idx)
                if mb_size < 4:
                    continue

                # Batched forward pass through the network trunk
                mb_features = features_t[batch_idx]
                policy_batch, values_batch = self.network(mb_features)

                # Per-step action scoring (can't batch — variable action sets)
                mb_policy_loss = torch.tensor(0.0, device=self.device)
                mb_entropy = torch.tensor(0.0, device=self.device)
                mb_clip_count = 0
                mb_kl_sum = 0.0

                for j, i in enumerate(batch_idx):
                    # Extract per-step policy scores from batched output
                    step_scores = {k: v[j] for k, v in policy_batch.items()}
                    action_scores = self._score_actions(step_scores, all_actions[i])
                    log_probs = F.log_softmax(action_scores, dim=0)
                    new_log_prob = log_probs[all_action_idx[i]]

                    # Clipped surrogate
                    ratio = torch.exp(new_log_prob - old_log_probs_t[i])
                    clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    surr1 = ratio * adv_t[i]
                    surr2 = clipped_ratio * adv_t[i]
                    mb_policy_loss = mb_policy_loss - torch.min(surr1, surr2)

                    # Entropy
                    probs = F.softmax(action_scores, dim=0)
                    entropy = -(probs * log_probs).sum()
                    mb_entropy = mb_entropy + entropy

                    with torch.no_grad():
                        mb_clip_count += int((torch.abs(ratio - 1.0) > self.clip_eps).item())
                        mb_kl_sum += (old_log_probs_t[i] - new_log_prob).item()

                # Batched value loss (fully vectorized)
                mb_returns = returns_t[batch_idx]
                mb_old_values = old_values_t[batch_idx]
                v_clipped = mb_old_values + torch.clamp(
                    values_batch - mb_old_values, -self.clip_eps, self.clip_eps
                )
                v_loss1 = (values_batch - mb_returns).pow(2)
                v_loss2 = (v_clipped - mb_returns).pow(2)
                mb_value_loss = torch.max(v_loss1, v_loss2).mean()

                avg_policy = mb_policy_loss / mb_size
                avg_entropy = mb_entropy / mb_size

                loss = avg_policy + self.value_coeff * mb_value_loss - self.entropy_coeff * avg_entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += avg_policy.item()
                total_value_loss += mb_value_loss.item()
                total_entropy += avg_entropy.item()
                total_clip_frac += mb_clip_count / mb_size
                total_approx_kl += mb_kl_sum / mb_size
                num_updates += 1

            # Early stopping on KL divergence
            if num_updates > 0 and abs(total_approx_kl / num_updates) > self.target_kl:
                break

        if num_updates == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0,
                    "total_loss": 0, "clip_frac": 0, "approx_kl": 0}

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "total_loss": (total_policy_loss + total_value_loss) / num_updates,
            "clip_frac": total_clip_frac / num_updates,
            "approx_kl": total_approx_kl / num_updates,
        }

    def adjust_lr(self, progress: float) -> None:
        """Cosine decay: progress goes from 0.0 to 1.0 over training."""
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
