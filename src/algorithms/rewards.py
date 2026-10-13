"""Reward shaping for MAPPO-LCR (Yang et al., Chaos Solitons Fractals 206 (2026) 117948).

Paper, Eq. 8-9:
    r_LCR(i) = N_i / 4                      N_i = number of cooperators in the focal group of agent i
                                            (agent i itself + its 4 von Neumann neighbours)
    r_i      = r_SPGG(i) + zeta * r_LCR(i)  zeta = local-cooperation weight (paper default: 3.0)

The shaping is applied to the *training* reward only; the architecture and loss of MAPPO are untouched
("The LCR is introduced as an auxiliary feedback term without modifying the MAPPO architecture").

`LocalCooperationReward` is the general form: you provide (a) which agents form the local group of each
agent and (b) which agents are currently "cooperating". `spgg_lcr` builds the paper's exact version.
Use it as `MAPPO(..., reward_transform=spgg_lcr(L=..., zeta=3.0))`.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def von_neumann_groups(L: int, periodic: bool = True) -> list[list[int]]:
    """Focal group of every site of an L x L lattice: [self, up, down, left, right] (flat indices)."""
    groups = []
    for y in range(L):
        for x in range(L):
            g = [y * L + x]
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if periodic:
                    g.append((ny % L) * L + (nx % L))
                elif 0 <= nx < L and 0 <= ny < L:
                    g.append(ny * L + nx)
            groups.append(g)
    return groups


def radius_groups(positions: np.ndarray, radius: float = 2.0) -> list[list[int]]:
    """Generic neighbourhoods for free-moving agents (e.g. plows): every agent within `radius`
    (Chebyshev distance), plus the agent itself. `positions` is [A, 2]."""
    d = np.abs(positions[:, None, :] - positions[None, :, :]).max(-1)
    return [list(np.flatnonzero(d[i] <= radius)) for i in range(len(positions))]


class LocalCooperationReward:
    """r'_i = r_i + zeta * (sum of `cooperation` over the local group of i) / norm.

    Args:
        zeta: shaping strength (paper: 3.0; its ablation sweeps 0.5 ... 10).
        groups: `info -> list of index lists` (each list must contain the agent itself).
        cooperation: `(rewards, info) -> [A]` array, 1 if that agent is cooperating now.
        norm: paper divides by 4 (number of neighbours, not group size).
    """

    def __init__(
        self,
        zeta: float,
        groups: Callable[[dict], Sequence[Sequence[int]]],
        cooperation: Callable[[np.ndarray, dict], np.ndarray],
        norm: float = 4.0,
    ):
        self.zeta, self.groups, self.cooperation, self.norm = zeta, groups, cooperation, norm

    def local_signal(self, rewards: np.ndarray, info: dict) -> np.ndarray:
        coop = np.asarray(self.cooperation(rewards, info), dtype=np.float32)
        return np.array([coop[list(g)].sum() / self.norm for g in self.groups(info)], dtype=np.float32)

    def __call__(self, rewards: np.ndarray, info: dict) -> np.ndarray:
        return rewards + self.zeta * self.local_signal(rewards, info)


def spgg_lcr(L: int, zeta: float = 3.0, periodic: bool = True) -> LocalCooperationReward:
    """The paper's LCR for the spatial public goods game on an L x L lattice.
    The env must put the flat strategy vector in `info["strategies"]` (1 = cooperator, 0 = defector)."""
    groups = von_neumann_groups(L, periodic)
    return LocalCooperationReward(
        zeta, groups=lambda info: groups, cooperation=lambda r, info: np.asarray(info["strategies"]).ravel()
    )


def plow_progress_lcr(zeta: float = 1.0, radius: float = 2.0) -> LocalCooperationReward:
    """OUR adaptation for Snowplow (NOT in the paper): a plow is "cooperating" when it just earned a
    positive reward (it removed snow), and each plow receives the density of working plows near it.
    Needs `info["positions"]` ([A,2]). Caution: it rewards being near productive neighbours, which can
    push plows to cluster -- exactly what you do NOT want for coverage. Treat as an experiment and ablate zeta."""
    return LocalCooperationReward(
        zeta,
        groups=lambda info: radius_groups(np.asarray(info["positions"]), radius),
        cooperation=lambda r, info: (np.asarray(r) > 0).astype(np.float32),
        norm=4.0,
    )
