"""
PPO controller backend for centralised TRAIN policy inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributions import Categorical


@dataclass(frozen=True)
class PPOConfig:
    """
    Use:
    Store immutable PPO hyperparameters required to construct policy/value
    networks and optimiser state.

    Attributes:
    - gamma: Discount factor for future rewards.
    - gae_lambda: Lambda for GAE smoothing (wired in later update step).
    - clip_eps: PPO clipping epsilon (wired in later update step).
    - lr: Learning rate for Adam.
    - ppo_epochs: PPO epochs per rollout update (wired in later step).
    - minibatch_size: Minibatch size for PPO updates.
    - value_coef: Critic-loss weight (later update step).
    - entropy_coef: Entropy bonus weight.
    - max_grad_norm: Gradient clipping cap.
    - hidden_size: Shared MLP hidden width.
    """

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    ppo_epochs: int = 4
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_size: int = 128


@dataclass
class PPOUpdateStats:
    """
    Use:
    Carry aggregated optimisation statistics after one PPO update.

    Attributes:
    - loss: Composite optimisation loss.
    - actor_loss: Policy loss component.
    - value_loss: Critic regression loss component.
    - entropy: Mean policy entropy.
    - clip_fraction: Fraction of clipped-ratio samples.
    """

    loss: float
    actor_loss: float
    value_loss: float
    entropy: float
    clip_fraction: float


class _ActorCritic(nn.Module):
    """
    Use:
    Build a shared-backbone actor-critic network for discrete control.

    Attributes:
    - backbone: Shared MLP feature extractor.
    - actor_head: Action-logit head for categorical policy.
    - value_head: Scalar state-value head.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        """
        Use:
        Construct the actor-critic network layers.

        Inputs:
        - obs_dim: Observation vector width.
        - action_dim: Number of discrete actions.
        - hidden_size: Hidden-layer width.

        Output:
        None.
        """
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use:
        Run one forward pass and emit policy logits + values.

        Inputs:
        - observations: Tensor with shape [N, obs_dim].

        Output:
        Tuple `(logits, values)`.
        """
        features = self.backbone(observations)
        logits = self.actor_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


class PPOController:
    """
    Use:
    Own PPO policy inference state and persistence hooks.

    This first-stage implementation establishes the network, optimiser,
    action-selection path, and checkpoint save/load path. GAE + PPO update
    methods are intentionally deferred to the next backend increment.

    Attributes:
    - obs_dim: Observation feature count.
    - action_dim: Number of discrete actions.
    - config: Immutable PPO hyperparameter bundle.
    - device: Torch device used for tensors/model.
    - model: Actor-critic model instance.
    - optimiser: Adam optimiser for model parameters.
    """

    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig, device: Optional[str] = None) -> None:
        """
        Use:
        Initialise model + optimiser and choose device.

        Inputs:
        - obs_dim: Observation vector width.
        - action_dim: Number of discrete actions.
        - config: PPO configuration.
        - device: Optional explicit device string.

        Output:
        None.
        """
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.config = config

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = _ActorCritic(self.obs_dim, self.action_dim, hidden_size=int(self.config.hidden_size)).to(self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=float(self.config.lr))

    def select_actions(self, observations: List[List[float]], deterministic: bool = False) -> Tuple[List[int], List[float], List[float]]:
        """
        Use:
        Select one action per observation and return action/log-prob/value.

        Inputs:
        - observations: Batch observation vectors.
        - deterministic: Use argmax policy when true, sampled policy otherwise.

        Output:
        Tuple `(actions, log_probs, values)` as Python lists.
        """
        if not observations:
            return [], [], []

        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, values = self.model(obs_tensor)
            dist = Categorical(logits=logits)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample()
            log_probs = dist.log_prob(actions)

        return (
            actions.detach().cpu().tolist(),
            log_probs.detach().cpu().tolist(),
            values.detach().cpu().tolist(),
        )

    def save(self, path: Path, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Use:
        Save model/optimiser/config state to checkpoint file.

        Inputs:
        - path: Destination checkpoint path.
        - metadata: Optional metadata dictionary to persist alongside state.

        Output:
        None.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "config": self.config.__dict__,
            "model_state": self.model.state_dict(),
            "optimiser_state": self.optimiser.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(payload, path)

    def load(self, path: Path) -> Dict[str, Any]:
        """
        Use:
        Load model/optimiser state from checkpoint.

        Inputs:
        - path: Source checkpoint path.

        Output:
        Metadata dictionary found in checkpoint.
        """
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["model_state"])
        self.optimiser.load_state_dict(payload["optimiser_state"])
        return dict(payload.get("metadata", {}))

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[float],
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[List[float], List[float]]:
        """
        Use:
        Compute Generalised Advantage Estimation (GAE) and return targets for
        one rollout trajectory.

        Inputs:
        - rewards: Per-step reward list.
        - values: Per-step value estimates.
        - dones: Terminal flags.
        - gamma: Discount factor.
        - gae_lambda: GAE lambda.

        Output:
        Tuple of returns and advantages.
        """
        if not rewards:
            return [], []

        n_steps = len(rewards)
        safe_values = list(values[:n_steps]) + [0.0] * max(0, n_steps - len(values))
        safe_dones = list(dones[:n_steps]) + [1.0] * max(0, n_steps - len(dones))

        advantages = [0.0] * n_steps
        returns = [0.0] * n_steps
        running_advantage = 0.0

        for step in reversed(range(n_steps)):
            next_value = safe_values[step + 1] if step + 1 < n_steps else 0.0
            non_terminal = 1.0 - float(safe_dones[step])
            delta = float(rewards[step]) + (float(gamma) * next_value * non_terminal) - float(safe_values[step])
            running_advantage = delta + (float(gamma) * float(gae_lambda) * non_terminal * running_advantage)
            advantages[step] = running_advantage
            returns[step] = running_advantage + float(safe_values[step])

        return returns, advantages

    def update(
        self,
        observations: List[List[float]],
        actions: List[int],
        old_log_probs: List[float],
        returns: List[float],
        advantages: List[float],
    ) -> PPOUpdateStats:
        """
        Use:
        Run one PPO optimisation cycle over a rollout batch.

        Inputs:
        - observations: Observation batch.
        - actions: Action batch.
        - old_log_probs: Behaviour-policy log-prob batch.
        - returns: Return targets.
        - advantages: Advantage targets.

        Output:
        PPO update statistics.
        """
        if not observations:
            return PPOUpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)

        obs_tensor = torch.tensor(observations, dtype=torch.float32, device=self.device)
        act_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        old_logp_tensor = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Standardise advantages to keep PPO gradients numerically stable.
        adv_mean = adv_tensor.mean()
        adv_std = adv_tensor.std(unbiased=False)
        adv_tensor = (adv_tensor - adv_mean) / (adv_std + 1e-8)

        n_samples = int(obs_tensor.shape[0])
        minibatch_size = max(1, min(int(self.config.minibatch_size), n_samples))

        loss_acc = 0.0
        actor_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        clip_frac_acc = 0.0
        updates = 0

        for _ in range(int(self.config.ppo_epochs)):
            permutation = torch.randperm(n_samples, device=self.device)
            for start in range(0, n_samples, minibatch_size):
                batch_idx = permutation[start : start + minibatch_size]
                obs_b = obs_tensor[batch_idx]
                act_b = act_tensor[batch_idx]
                old_logp_b = old_logp_tensor[batch_idx]
                ret_b = ret_tensor[batch_idx]
                adv_b = adv_tensor[batch_idx]

                logits, values = self.model(obs_b)
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(act_b)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp_b)
                unclipped_obj = ratio * adv_b
                clipped_obj = torch.clamp(
                    ratio,
                    1.0 - float(self.config.clip_eps),
                    1.0 + float(self.config.clip_eps),
                ) * adv_b

                actor_loss = -torch.min(unclipped_obj, clipped_obj).mean()
                value_loss = torch.mean((ret_b - values) ** 2)
                loss = (
                    actor_loss
                    + (float(self.config.value_coef) * value_loss)
                    - (float(self.config.entropy_coef) * entropy)
                )

                self.optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config.max_grad_norm))
                self.optimiser.step()

                clip_fraction = ((ratio - 1.0).abs() > float(self.config.clip_eps)).float().mean()

                loss_acc += float(loss.item())
                actor_loss_acc += float(actor_loss.item())
                value_loss_acc += float(value_loss.item())
                entropy_acc += float(entropy.item())
                clip_frac_acc += float(clip_fraction.item())
                updates += 1

        if updates <= 0:
            return PPOUpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)

        inv = 1.0 / float(updates)
        return PPOUpdateStats(
            loss=loss_acc * inv,
            actor_loss=actor_loss_acc * inv,
            value_loss=value_loss_acc * inv,
            entropy=entropy_acc * inv,
            clip_fraction=clip_frac_acc * inv,
        )
