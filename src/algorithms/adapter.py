"""Glue between the project environments and the learning algorithms.

The algorithms never see dictionaries, enums or agent objects: they see numpy
arrays with a fixed agent order. This adapter is the *only* place that knows
how an environment's observations / actions / rewards look, so when the team's
final `SnowPlowEnv` / `SoccerEnv` is ready you only have to configure the
adapter (`obs_encoder`, `action_decoder`, `state_fn`), not the algorithms.

Expected environment API (PettingZoo Parallel style):
    obs, info = env.reset(seed=...)                 # obs: {agent_id: observation}
    obs, rew, term, trunc, info = env.step(actions) # every one a {agent_id: value} dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np


def flatten_obs(obs: Any) -> np.ndarray:
    """Recursively flatten dict / tuple / list / array / scalar into a float32 vector.

    Dict keys are visited in sorted order so the layout is stable. Good enough
    for small grids with an MLP; for big grids pass a custom `obs_encoder`.
    """
    parts: list[np.ndarray] = []

    def rec(x: Any) -> None:
        if isinstance(x, dict):
            for k in sorted(x, key=str):
                rec(x[k])
        elif isinstance(x, (tuple, list)):
            for v in x:
                rec(v)
        elif isinstance(x, Enum):
            rec(x.value)
        elif x is None:
            return
        else:
            parts.append(np.asarray(x, dtype=np.float32).ravel())

    rec(obs)
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


@dataclass
class StepResult:
    obs: np.ndarray  # [A, D]
    state: np.ndarray  # [S]
    reward: np.ndarray  # [A]
    terminated: bool  # episode ended because of the task (success / failure)
    truncated: bool  # episode ended because of the steps limit
    alive: np.ndarray  # [A] 1.0 if the agent produced an observation
    info: dict = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class ParallelEnvAdapter:
    """Wrap a Parallel-style multi-agent env into fixed-order numpy arrays.

    Args:
        env: the raw environment.
        agent_ids: agent keys, in the order used everywhere. Default:
            `env.possible_agents`, or `[a.index for a in env.agents]`.
        n_actions: size of the discrete action space (default: tries
            `env.action_space(agent).n`).
        obs_encoder: observation -> 1-D float vector (default `flatten_obs`).
        action_decoder: int -> whatever the env wants, e.g.
            `lambda a: list(MovementActions)[a]`. Default: identity.
        state_fn: `(env, obs_dict) -> 1-D vector`, the global state used by the
            centralised critics. Default: `env.state()` if it exists, else the
            concatenation of all agents' encoded observations.
    """

    def __init__(
        self,
        env: Any,
        agent_ids: list | None = None,
        n_actions: int | None = None,
        obs_encoder: Callable[[Any], np.ndarray] = flatten_obs,
        action_decoder: Callable[[int], Any] | None = None,
        state_fn: Callable[[Any, dict], np.ndarray] | None = None,
    ):
        self.env = env
        if agent_ids is None:
            if hasattr(env, "possible_agents"):
                agent_ids = list(env.possible_agents)
            else:
                agent_ids = [a.index for a in env.agents]
        self.agent_ids = list(agent_ids)
        self.n_agents = len(self.agent_ids)
        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder or (lambda a: a)
        self.state_fn = state_fn

        if n_actions is None:
            n_actions = int(env.action_space(self.agent_ids[0]).n)
        self.n_actions = n_actions

        self.obs_dim: int | None = None
        self.state_dim: int | None = None

    # ------------------------------------------------------------------ utils
    def _encode(self, obs_dict: dict) -> tuple[np.ndarray, np.ndarray]:
        encoded = {a: self.obs_encoder(obs_dict[a]) for a in self.agent_ids if a in obs_dict}
        if self.obs_dim is None:
            self.obs_dim = len(next(iter(encoded.values())))
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        alive = np.zeros(self.n_agents, dtype=np.float32)
        for i, a in enumerate(self.agent_ids):
            if a in encoded:
                obs[i] = encoded[a]
                alive[i] = 1.0
        if self.state_fn is not None:
            state = np.asarray(self.state_fn(self.env, obs_dict), dtype=np.float32).ravel()
        elif hasattr(self.env, "state") and callable(self.env.state):
            state = np.asarray(self.env.state(), dtype=np.float32).ravel()
        else:
            state = obs.ravel().copy()
        if self.state_dim is None:
            self.state_dim = len(state)
        return obs, state, alive

    # -------------------------------------------------------------------- API
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = self.env.reset(seed=seed)
        obs_dict = out[0] if isinstance(out, tuple) else out
        return self._encode(obs_dict)

    def step(self, actions: np.ndarray) -> StepResult:
        action_dict = {a: self.action_decoder(int(actions[i])) for i, a in enumerate(self.agent_ids)}
        obs_dict, rew, term, trunc, info = self.env.step(action_dict)
        obs, state, alive = self._encode(obs_dict)
        reward = np.array([float(rew.get(a, 0.0)) for a in self.agent_ids], dtype=np.float32)
        # missing agents count as terminated
        terminated = all(term.get(a, True) for a in self.agent_ids)
        truncated = any(trunc.get(a, False) for a in self.agent_ids)
        if hasattr(self.env, "agents") and len(self.env.agents) == 0:
            terminated = terminated or not truncated
        return StepResult(obs, state, reward, terminated, truncated, alive, info if isinstance(info, dict) else {})
