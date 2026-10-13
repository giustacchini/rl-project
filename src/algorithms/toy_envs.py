"""Tiny PettingZoo-Parallel-style environments for testing the algorithms.

They exist so the algorithms can be developed and unit-tested BEFORE the real
SnowPlow / Soccer environments have a `step()`. They are not part of the
experiments. Action order matches `Enums.MovementActions`:
    0 DO_NOTHING, 1 UP, 2 DOWN, 3 RIGHT, 4 LEFT
"""

from __future__ import annotations

import numpy as np

MOVES = [(0, 0), (0, -1), (0, 1), (1, 0), (-1, 0)]


class _Discrete:
    def __init__(self, n: int):
        self.n = n


class ToyCoverageEnv:
    """Cooperative snow clearing: clear every cell, max 2 snow levels per visit.

    Rewards: -0.1 per move, +1 per level removed, +10 (everyone) when all clear.
    `partial=True` -> each agent sees only a 3x3 window of snow and NO position,
    so memory (recurrent policy) is useful.
    `snowfall_prob` -> hidden random snowfall, mimicking the clouds.
    """

    def __init__(self, n_agents: int = 1, size: int = 4, max_steps: int = 50, partial: bool = False, snowfall_prob: float = 0.0):
        self.size, self.max_steps, self.partial, self.snowfall_prob = size, max_steps, partial, snowfall_prob
        self.possible_agents = [f"plow_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)
        self.rng = np.random.default_rng(0)

    def action_space(self, agent) -> _Discrete:
        return _Discrete(len(MOVES))

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        s = self.size
        self.snow = self.rng.integers(0, 4, size=(s, s)) * (self.rng.random((s, s)) < 0.6)
        if self.snow.sum() == 0:
            self.snow[self.rng.integers(s), self.rng.integers(s)] = 2
        self.pos = {a: np.array([self.rng.integers(s), self.rng.integers(s)]) for a in self.possible_agents}
        self.t = 0
        self.agents = list(self.possible_agents)
        return self._obs(), {}

    def _window(self, p) -> np.ndarray:
        pad = np.pad(self.snow / 3.0, 1, constant_values=-1.0)
        return pad[p[1]: p[1] + 3, p[0]: p[0] + 3]

    def _obs(self):
        out = {}
        for a in self.possible_agents:
            if self.partial:
                out[a] = {"window": self._window(self.pos[a])}
            else:
                out[a] = {
                    "snow": self.snow / 3.0,
                    "self_pos": self.pos[a] / self.size,
                    "others": [self.pos[o] / self.size for o in self.possible_agents if o != a],
                }
        return out

    def state(self) -> np.ndarray:
        return np.concatenate([(self.snow / 3.0).ravel(), *[self.pos[a] / self.size for a in self.possible_agents]])

    def step(self, actions: dict):
        rewards = {a: 0.0 for a in self.possible_agents}
        for a in self.possible_agents:
            dx, dy = MOVES[int(actions[a])]
            new = self.pos[a] + (dx, dy)
            if 0 <= new[0] < self.size and 0 <= new[1] < self.size:
                self.pos[a] = new
            if (dx, dy) != (0, 0):
                rewards[a] -= 0.1
        for a in self.possible_agents:  # sequential clearing, max 2 levels
            x, y = self.pos[a]
            removed = min(2, int(self.snow[y, x]))
            self.snow[y, x] -= removed
            rewards[a] += float(removed)
        if self.snowfall_prob > 0 and self.rng.random() < self.snowfall_prob:
            self.snow[self.rng.integers(self.size), self.rng.integers(self.size)] = min(3, 1 + int(self.rng.integers(3)))
        self.t += 1
        success = self.snow.sum() == 0
        if success:
            for a in rewards:
                rewards[a] += 10.0
        trunc = self.t >= self.max_steps and not success
        term = {a: bool(success) for a in self.possible_agents}
        truncs = {a: bool(trunc) for a in self.possible_agents}
        if success or trunc:
            self.agents = []
        return self._obs(), rewards, term, truncs, {}


class ToyChaseEnv:
    """Competitive 2-agent game: 'chaser' wants to land on 'evader'; zero-sum-ish rewards."""

    def __init__(self, size: int = 5, max_steps: int = 30):
        self.size, self.max_steps = size, max_steps
        self.possible_agents = ["chaser", "evader"]
        self.agents = list(self.possible_agents)
        self.rng = np.random.default_rng(0)

    def action_space(self, agent) -> _Discrete:
        return _Discrete(len(MOVES))

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos = {a: self.rng.integers(0, self.size, size=2) for a in self.possible_agents}
        self.t = 0
        self.agents = list(self.possible_agents)
        return self._obs(), {}

    def _obs(self):
        a, b = (self.pos[k] / self.size for k in self.possible_agents)
        return {"chaser": np.concatenate([a, b]), "evader": np.concatenate([b, a])}

    def step(self, actions: dict):
        for a in self.possible_agents:
            new = self.pos[a] + MOVES[int(actions[a])]
            if (new >= 0).all() and (new < self.size).all():
                self.pos[a] = new
        caught = bool((self.pos["chaser"] == self.pos["evader"]).all())
        self.t += 1
        trunc = self.t >= self.max_steps and not caught
        rew = {"chaser": 1.0 if caught else -0.02, "evader": -1.0 if caught else 0.02}
        term = {a: caught for a in self.possible_agents}
        truncs = {a: trunc for a in self.possible_agents}
        if caught or trunc:
            self.agents = []
        return self._obs(), rew, term, truncs, {}


class SPGGEnv:
    """Spatial public goods game of the MAPPO-LCR paper (Sec. 2), on a periodic L x L lattice.

    Every site holds one agent that picks C (1) or D (0). Each agent belongs to 5 groups (the groups centred on
    itself and on its 4 von Neumann neighbours). In a group with N_C cooperators the pot r*N_C is split
    among the 5 members and each cooperator pays 1 (Eq. 2); the payoff is summed over the 5 groups (Eq. 3).

    Observation of agent i = (x_i, n_i/5, g): own strategy, cooperators in its focal group, global cooperation
    ratio (the paper's "encode state"). Global state for the critic = the L x L strategy matrix (Eq. 12).
    There is no terminal state: episodes are cut at `max_steps` (truncation). `info["strategies"]` feeds `spgg_lcr`.
    """

    def __init__(self, L: int = 8, r: float = 4.7, max_steps: int = 50, init: str = "half"):
        self.L, self.r, self.max_steps, self.init = L, r, max_steps, init
        self.possible_agents = [f"a{i}" for i in range(L * L)]
        self.agents = list(self.possible_agents)
        self.rng = np.random.default_rng(0)

    def action_space(self, agent) -> _Discrete:
        return _Discrete(2)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        L = self.L
        if self.init == "half":  # defectors on top, cooperators at the bottom (paper Sec. 3.2)
            s = np.zeros((L, L), dtype=np.int64)
            s[L // 2:, :] = 1
        elif self.init == "bernoulli":
            s = (self.rng.random((L, L)) < 0.5).astype(np.int64)
        elif self.init == "defect":
            s = np.zeros((L, L), dtype=np.int64)
        else:
            raise ValueError(self.init)
        self.s, self.t = s, 0
        self.agents = list(self.possible_agents)
        return self._obs(), {}

    def _group_counts(self, s: np.ndarray) -> np.ndarray:  # N_C of the group centred on each site
        return s + np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)

    def payoffs(self, s: np.ndarray) -> np.ndarray:
        n = self._group_counts(s)
        pot = self.r * n / 5.0
        total_pot = pot + np.roll(pot, 1, 0) + np.roll(pot, -1, 0) + np.roll(pot, 1, 1) + np.roll(pot, -1, 1)
        return total_pot - 5.0 * s  # a cooperator pays 1 in each of its 5 groups

    def _obs(self):
        n = self._group_counts(self.s)
        g = float(self.s.mean())
        return {a: np.array([self.s.flat[i], n.flat[i] / 5.0, g], dtype=np.float32) for i, a in enumerate(self.possible_agents)}

    def state(self) -> np.ndarray:
        return self.s.astype(np.float32).ravel()

    def step(self, actions: dict):
        self.s = np.array([int(actions[a]) for a in self.possible_agents], dtype=np.int64).reshape(self.L, self.L)
        pay = self.payoffs(self.s).ravel()
        self.t += 1
        trunc = self.t >= self.max_steps
        if trunc:
            self.agents = []
        rew = {a: float(pay[i]) for i, a in enumerate(self.possible_agents)}
        term = {a: False for a in self.possible_agents}
        truncs = {a: trunc for a in self.possible_agents}
        return self._obs(), rew, term, truncs, {"strategies": self.s.ravel().copy()}
