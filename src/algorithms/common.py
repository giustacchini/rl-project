"""Small helpers shared by all algorithms."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2, out_gain: float = 1.0, activation: str = "tanh") -> nn.Sequential:
    """MLP with orthogonal init (the PPO default)."""
    act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        lin = nn.Linear(d, hidden)
        nn.init.orthogonal_(lin.weight, gain=2**0.5)
        nn.init.zeros_(lin.bias)
        mods += [lin, act()]
        d = hidden
    out = nn.Linear(d, out_dim)
    nn.init.orthogonal_(out.weight, gain=out_gain)
    nn.init.zeros_(out.bias)
    mods.append(out)
    return nn.Sequential(*mods)


def compute_gae(
    rewards: torch.Tensor,  # [T, N]
    values: torch.Tensor,  # [T, N]
    next_values: torch.Tensor,  # [T, N]  V(s_{t+1}) (bootstrapped)
    terminals: torch.Tensor,  # [T, N]  1 if episode ended (terminated OR truncated) after step t
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised Advantage Estimation.

    `next_values` is supplied explicitly so truncated episodes (max_steps) can
    be bootstrapped while still cutting the GAE chain at the episode boundary.
    Truncation bootstrapping is folded into `rewards` by the caller, so here
    a terminal always zeroes the bootstrap.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        nonterminal = 1.0 - terminals[t]
        delta = rewards[t] + gamma * next_values[t] * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


class ReplayBuffer:
    """Uniform replay buffer over dict-of-array transitions (used by MADDPG)."""

    def __init__(self, capacity: int, shapes: dict[str, tuple[int, ...]], dtypes: dict[str, type] | None = None):
        self.capacity = capacity
        dtypes = dtypes or {}
        self.data = {k: np.zeros((capacity, *s), dtype=dtypes.get(k, np.float32)) for k, s in shapes.items()}
        self.ptr = 0
        self.size = 0

    def add(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.data[k][self.ptr] = v
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        idx = rng.integers(0, self.size, size=batch_size)
        return {k: v[idx] for k, v in self.data.items()}

    def __len__(self) -> int:
        return self.size
