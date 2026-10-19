"""MADDPG (Lowe et al., 2017, "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments").

Follows the paper:
  * centralised training / decentralised execution: actor i only sees o_i, critic Q_i sees x (= all
    observations, or the env state) and the actions a_1..a_N of ALL agents (Eq. 4-6);
  * one critic PER AGENT, so rewards may differ / conflict (competitive and mixed teams);
  * replay buffer of (x, x', a_1..a_N, r_1..r_N), target networks with soft update tau (Appendix, Alg. 1);
  * Sec. 4.2 (`infer_policies=True`): each agent i learns approximations mu^_i^j of the other agents' policies by
    maximising log-prob of their actions + lambda * entropy (Eq. 7) and uses them in the target (Eq. 8)
    instead of the true target policies;
  * Sec. 4.3 (`n_subpolicies=K>1`): each agent trains an ensemble of K sub-policies; one is picked at random per
    episode, each has its own replay subset D_i^(k) (Eq. 9).

Adaptation (not in the paper): the paper's physical actions are continuous. Our grids have discrete actions, so
the actor outputs logits and actions are relaxed with the straight-through Gumbel-Softmax (the paper already uses
Gumbel-Softmax for its discrete messages). Exploration is policy sampling + epsilon-random instead of
continuous noise.

Paper hyper-parameters (Appendix, "Experimental Results"): Adam lr 0.01, tau 0.01, gamma 0.95, replay 1e6,
a gradient update after every 100 samples added to the buffer, batch 1024, 2-layer ReLU MLP with 64 units.
These are the defaults below. For short toy runs use more frequent updates (`update_every`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.adapter import ParallelEnvAdapter
from algorithms.common import ReplayBuffer, mlp, set_seed


@dataclass
class MADDPGConfig:
    # --- values from the paper -------------------------------------------------
    buffer_size: int = 1_000_000
    batch_size: int = 1024
    gamma: float = 0.95
    tau: float = 0.01
    lr_actor: float = 0.01
    lr_critic: float = 0.01
    hidden: int = 64
    layers: int = 2
    activation: str = "relu"
    update_every: int = 100  # env steps (= samples added to the buffer) between updates
    # --- paper's optional components -------------------------------------------
    n_subpolicies: int = 1  # K > 1 -> policy ensembles (Sec. 4.3)
    infer_policies: bool = False  # Sec. 4.2
    infer_lambda: float = 0.001  # entropy weight in Eq. 7 (paper value)
    # --- not specified by the paper (discrete-action adaptation) ----------------
    warmup_steps: int = 1024
    eps_start: float = 0.5
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    max_grad_norm: float = 0.5  # paper's public code clips at 0.5
    logit_reg: float = 1e-3  # paper's public code regularises policy logits
    seed: int = 0
    device: str = "cpu"


class MADDPG:
    def __init__(
        self,
        env_fn: Callable[[], object],
        config: MADDPGConfig | None = None,
        adapter_kwargs: dict | None = None,
        reward_transform: Callable[[np.ndarray, dict], np.ndarray] | None = None,
    ):
        self.cfg = c = config or MADDPGConfig()
        set_seed(c.seed)
        self.rng = np.random.default_rng(c.seed)
        self.dev = torch.device(c.device)
        self.env_fn = env_fn
        self.adapter_kwargs = adapter_kwargs or {}
        self.reward_transform = reward_transform

        self.env = ParallelEnvAdapter(env_fn(), **self.adapter_kwargs)
        self.A, self.n_actions, self.K = self.env.n_agents, self.env.n_actions, c.n_subpolicies
        self.obs, self.state, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
        self.D, self.S = self.env.obs_dim, self.env.state_dim

        def actor():
            return mlp(self.D, c.hidden, self.n_actions, c.layers, 0.01, c.activation)

        def critic():
            return mlp(self.S + self.A * self.n_actions, c.hidden, 1, c.layers, 1.0, c.activation)

        # actors[i][k]: sub-policy k of agent i;  critics[i]: Q_i
        self.actors = nn.ModuleList([nn.ModuleList([actor() for _ in range(self.K)]) for _ in range(self.A)]).to(self.dev)
        self.t_actors = nn.ModuleList([nn.ModuleList([actor() for _ in range(self.K)]) for _ in range(self.A)]).to(self.dev)
        self.critics = nn.ModuleList([critic() for _ in range(self.A)]).to(self.dev)
        self.t_critics = nn.ModuleList([critic() for _ in range(self.A)]).to(self.dev)
        self.t_actors.load_state_dict(self.actors.state_dict())
        self.t_critics.load_state_dict(self.critics.state_dict())
        self.opt_a = [[torch.optim.Adam(n.parameters(), lr=c.lr_actor) for n in row] for row in self.actors]
        self.opt_c = [torch.optim.Adam(q.parameters(), lr=c.lr_critic) for q in self.critics]

        # approximate policies mu^_i^j (agent i's model of agent j != i); Sec. 4.2
        self.approx = self.t_approx = self.opt_approx = None
        if c.infer_policies:
            def make():
                return nn.ModuleList([nn.ModuleList([actor() if j != i else nn.Identity() for j in range(self.A)]) for i in range(self.A)])
            self.approx, self.t_approx = make().to(self.dev), make().to(self.dev)
            self.t_approx.load_state_dict(self.approx.state_dict())
            self.opt_approx = [torch.optim.Adam(self.approx[i].parameters(), lr=c.lr_actor) if self.A > 1 else None for i in range(self.A)]

        self.buffer = ReplayBuffer(
            c.buffer_size,
            {"obs": (self.A, self.D), "state": (self.S,), "act": (self.A,), "rew": (self.A,), "sub": (self.A,),
             "next_obs": (self.A, self.D), "next_state": (self.S,), "done": ()},
            dtypes={"act": np.int64, "sub": np.int64},
        )
        self.sub = self._draw_sub()
        self.env_steps = 0
        self.ep_return, self.ep_len = 0.0, 0
        self.finished_returns: list[float] = []
        self.finished_lengths: list[int] = []

    # ------------------------------------------------------------------ acting
    def _draw_sub(self) -> np.ndarray:
        return self.rng.integers(0, self.K, size=self.A)

    def _eps(self) -> float:
        c = self.cfg
        return c.eps_start + min(1.0, self.env_steps / c.eps_decay_steps) * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def select_actions(self, obs: np.ndarray, explore: bool = True, deterministic: bool | None = None, sub: np.ndarray | None = None) -> np.ndarray:
        """Decentralised execution: agent i only uses obs[i] (and its own sub-policy)."""
        det = (not explore) if deterministic is None else deterministic
        sub = self.sub if sub is None else sub
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.dev)
        acts = np.zeros(self.A, dtype=np.int64)
        eps = self._eps() if explore else 0.0
        for i in range(self.A):
            logits = self.actors[i][int(sub[i])](o[i])
            if det:
                acts[i] = int(logits.argmax())
            elif explore and (self.env_steps < self.cfg.warmup_steps or self.rng.random() < eps):
                acts[i] = int(self.rng.integers(self.n_actions))
            else:
                acts[i] = int(torch.distributions.Categorical(logits=logits).sample())
        return acts

    # ---------------------------------------------------------------- learning
    def _onehot(self, a: torch.Tensor) -> torch.Tensor:
        return F.one_hot(a, self.n_actions).float()

    def _logits(self, nets, j: int, obs_j: torch.Tensor, sub_j: torch.Tensor) -> torch.Tensor:
        """Logits of agent j's sub-policy `sub_j[b]` for every batch row b."""
        if self.K == 1:
            return nets[j][0](obs_j)
        allv = torch.stack([nets[j][k](obs_j) for k in range(self.K)], dim=1)
        return allv[torch.arange(len(obs_j), device=self.dev), sub_j]

    def _sample(self, i: int, k: int) -> dict[str, torch.Tensor] | None:
        """Batch from D_i^(k): transitions where agent i was running sub-policy k."""
        buf, n = self.buffer, len(self.buffer)
        if self.K == 1:
            b = buf.sample(self.cfg.batch_size, self.rng)
        else:
            idx = np.flatnonzero(buf.data["sub"][:n, i] == k)
            if len(idx) < 32:
                return None
            idx = self.rng.choice(idx, size=min(self.cfg.batch_size, len(idx)), replace=len(idx) < self.cfg.batch_size)
            b = {key: v[idx] for key, v in buf.data.items()}
        return {key: torch.as_tensor(v, device=self.dev) for key, v in b.items()}

    def _update_agent(self, i: int, k: int) -> tuple[float, float] | None:
        c = self.cfg
        t = self._sample(i, k)
        if t is None:
            return None
        B = len(t["done"])
        act1h = self._onehot(t["act"])  # [B,A,n]

        if self.approx is not None:  # Eq. 7: fit mu^_i^j on the actions agent j actually took
            loss = 0.0
            for j in range(self.A):
                if j == i:
                    continue
                logp = F.log_softmax(self.approx[i][j](t["obs"][:, j]), dim=-1)
                ent = -(logp.exp() * logp).sum(-1).mean()
                loss = loss - (logp.gather(1, t["act"][:, j:j + 1]).mean() + c.infer_lambda * ent)
            if self.A > 1:
                self.opt_approx[i].zero_grad()
                loss.backward()
                self.opt_approx[i].step()

        with torch.no_grad():  # target y = r_i + gamma * Q'_i(x', a'_1..a'_N)   (Eq. 6 / Eq. 8)
            next_a = []
            for j in range(self.A):
                if self.approx is not None and j != i:
                    next_a.append(F.softmax(self.t_approx[i][j](t["next_obs"][:, j]), dim=-1))  # probabilities, no sampling
                else:
                    lg = self._logits(self.t_actors, j, t["next_obs"][:, j], t["sub"][:, j])
                    next_a.append(F.gumbel_softmax(lg, tau=1.0, hard=True))
            next_in = torch.cat([t["next_state"], torch.cat(next_a, dim=-1)], dim=-1)
            y = t["rew"][:, i] + c.gamma * (1.0 - t["done"]) * self.t_critics[i](next_in)[:, 0]

        in_now = torch.cat([t["state"], act1h.reshape(B, -1)], dim=-1)
        c_loss = F.mse_loss(self.critics[i](in_now)[:, 0], y)
        self.opt_c[i].zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critics[i].parameters(), c.max_grad_norm)
        self.opt_c[i].step()

        logits = self.actors[i][k](t["obs"][:, i])  # Eq. 5: ascend Q_i w.r.t. a_i = mu_i(o_i)
        joint = act1h.clone()
        joint[:, i] = F.gumbel_softmax(logits, tau=1.0, hard=True)
        a_loss = -self.critics[i](torch.cat([t["state"], joint.reshape(B, -1)], dim=-1)).mean() + c.logit_reg * logits.pow(2).mean()
        self.opt_a[i][k].zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actors[i][k].parameters(), c.max_grad_norm)
        self.opt_a[i][k].step()
        return c_loss.item(), a_loss.item()

    def _soft_update(self) -> None:
        pairs = [(self.actors, self.t_actors), (self.critics, self.t_critics)]
        if self.approx is not None:
            pairs.append((self.approx, self.t_approx))
        with torch.no_grad():
            for net, tgt in pairs:
                for p, tp in zip(net.parameters(), tgt.parameters()):
                    tp.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p)

    def _update(self) -> dict[str, float]:
        cl, al, n = 0.0, 0.0, 0
        for i in range(self.A):
            for k in range(self.K):
                out = self._update_agent(i, k)
                if out:
                    cl, al, n = cl + out[0], al + out[1], n + 1
        self._soft_update()
        return {"critic_loss": cl / max(n, 1), "actor_loss": al / max(n, 1)}

    def learn(self, total_env_steps: int, log_every: int = 5000, callback: Callable[[dict], None] | None = None) -> list[dict]:
        history, last = [], {}
        c = self.cfg
        for _ in range(total_env_steps):
            act = self.select_actions(self.obs)
            res = self.env.step(act)
            r = res.reward if self.reward_transform is None else self.reward_transform(res.reward, res.info)
            # only a real terminal cuts bootstrapping; time-limit truncation does not
            self.buffer.add(obs=self.obs, state=self.state, act=act, rew=r, sub=self.sub,
                            next_obs=res.obs, next_state=res.state, done=float(res.terminated))
            self.ep_return += float(res.reward.sum())
            self.ep_len += 1
            self.env_steps += 1
            if res.done:
                self.finished_returns.append(self.ep_return)
                self.finished_lengths.append(self.ep_len)
                self.ep_return, self.ep_len = 0.0, 0
                self.obs, self.state, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
                self.sub = self._draw_sub()  # new sub-policy per episode (Sec. 4.3)
            else:
                self.obs, self.state = res.obs, res.state

            if len(self.buffer) >= max(c.batch_size, c.warmup_steps) and self.env_steps % c.update_every == 0:
                last = self._update()
            if log_every and self.env_steps % log_every == 0:
                rec = {"env_steps": self.env_steps, "epsilon": self._eps(),
                       "mean_return": float(np.mean(self.finished_returns[-20:])) if self.finished_returns else float("nan"), **last}
                history.append(rec)
                if callback:
                    callback(rec)
                print(f"[MADDPG] steps {self.env_steps}  return {rec['mean_return']:.2f}  eps {rec['epsilon']:.2f}  " + "  ".join(f"{k} {v:.3f}" for k, v in last.items()))
        return history

    @torch.no_grad()
    def evaluate(self, n_episodes: int = 10, seed: int = 12345, deterministic: bool = False) -> dict[str, float]:
        """Decentralised execution. Samples from each actor by default; `deterministic=True` is greedy, which in
        gridworlds can get stuck oscillating and look much worse. A random sub-policy is drawn per episode."""
        env = ParallelEnvAdapter(self.env_fn(), **self.adapter_kwargs)
        rng = np.random.default_rng(seed)
        per_agent, lengths = [], []
        for _ in range(n_episodes):
            obs, _, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            sub = rng.integers(0, self.K, size=self.A)
            tot, steps, done = np.zeros(self.A), 0, False
            while not done:
                res = env.step(self.select_actions(obs, explore=False, deterministic=deterministic, sub=sub))
                obs, tot, steps, done = res.obs, tot + res.reward, steps + 1, res.done
            per_agent.append(tot)
            lengths.append(steps)
        pa = np.mean(per_agent, axis=0)
        return {"mean_return": float(pa.sum()), "per_agent_return": pa.tolist(), "mean_length": float(np.mean(lengths))}

    def save(self, path: str) -> None:
        torch.save({"actors": self.actors.state_dict(), "critics": self.critics.state_dict(), "config": asdict(self.cfg)}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.dev)
        self.actors.load_state_dict(ck["actors"])
        self.critics.load_state_dict(ck["critics"])
        self.t_actors.load_state_dict(ck["actors"])
        self.t_critics.load_state_dict(ck["critics"])
