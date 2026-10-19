"""PPO and MAPPO in one implementation.

One trainer, two flavours -- they only differ in what the *critic* sees:

    PPO     1 agent, critic sees its own observation.
    MAPPO   N agents, parameter-shared actor, ONE centralised critic that sees the
            global state (Yu et al., "The Surprising Effectiveness of PPO in
            Cooperative Multi-Agent Games"). Actors only use local observations,
            so execution stays decentralised.

Partial observability (Experiments 2 and 4): set `PPOConfig(recurrent=True)` and
actor and critic get a GRU. Sequences are trained with truncated BPTT over
`rollout_len` steps; hidden states are reset at episode boundaries.

Action masking: if the environment reports which actions are legal (the adapter's `mask`,
[A, n_actions] bool), illegal actions get logit -1e9 before sampling. The agent therefore never
samples an illegal move, and the policy gradient only acts on the legal ones.

Discrete actions only (the grid environments use 4-5 moves).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from algorithms.adapter import ParallelEnvAdapter
from algorithms.common import compute_gae, set_seed


@dataclass
class PPOConfig:
    n_envs: int = 8  # environments stepped per rollout (more = less noisy)
    rollout_len: int = 128  # steps per env per update (= BPTT length when recurrent)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    n_minibatches: int = 4  # split over *streams* (env x agent), never over time
    lr: float = 3e-4
    anneal_lr: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    layers: int = 2
    recurrent: bool = False  # GRU in actor and critic (partial observability)
    agent_id: bool = True  # append one-hot agent id (needed for a shared actor to specialise)
    reward_mode: str = "individual"  # "individual" | "team_sum" | "team_mean"
    activation: str = "tanh"  # "tanh" (PPO default) or "relu" (MAPPO-LCR paper)
    reward_scale: float = 1.0  # training rewards are multiplied by this (keeps value targets small when returns are in the 100s)
    seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        for name in ("n_envs", "rollout_len", "epochs", "n_minibatches", "hidden", "layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.lr <= 0 or self.reward_scale <= 0:
            raise ValueError("lr and reward_scale must be positive")
        if self.reward_mode not in ("individual", "team_sum", "team_mean"):
            raise ValueError("unknown reward_mode")

    @classmethod
    def mappo_lcr_paper(cls, **overrides) -> "PPOConfig":
        """Hyper-parameters reported in the MAPPO-LCR paper (Yang et al. 2026, Sec. 3.1):
        Adam lr 1e-3, clip 0.2, GAE lambda 0.95, entropy 0.001, value-loss weight 0.5,
        gamma 0.99, 3-layer (FC-ReLU-FC-ReLU-FC) actor and critic.
        The paper does not state epochs / minibatches / hidden width, so those keep our defaults."""
        base = dict(
            lr=1e-3, clip=0.2, gae_lambda=0.95, ent_coef=0.001, vf_coef=0.5, gamma=0.99, layers=2, activation="relu", anneal_lr=False
        )
        base.update(overrides)
        return cls(**base)


# ----------------------------------------------------------------------------
# networks
# ----------------------------------------------------------------------------
class _Branch(nn.Module):
    """MLP trunk, optional GRU, linear head. Operates on [T, N, D] sequences."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, layers: int, recurrent: bool, out_gain: float, activation: str = "tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            lin = nn.Linear(d, hidden)
            nn.init.orthogonal_(lin.weight, gain=2**0.5)
            nn.init.zeros_(lin.bias)
            mods += [lin, act()]
            d = hidden
        self.body = nn.Sequential(*mods)
        self.hidden = hidden
        self.recurrent = recurrent
        self.gru = nn.GRUCell(hidden, hidden) if recurrent else None
        self.head = nn.Linear(hidden, out_dim)
        nn.init.orthogonal_(self.head.weight, gain=out_gain)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor, first: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x [T,N,D], h [N,H], first [T,N] (1 -> new episode starts at this step)
        z = self.body(x)
        if self.gru is not None:
            outs = []
            for t in range(z.shape[0]):
                h = h * (1.0 - first[t]).unsqueeze(-1)
                h = self.gru(z[t], h)
                outs.append(h)
            z = torch.stack(outs)
        return self.head(z), h


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, critic_dim: int, n_actions: int, cfg: PPOConfig):
        super().__init__()
        self.actor = _Branch(obs_dim, cfg.hidden, n_actions, cfg.layers, cfg.recurrent, out_gain=0.01, activation=cfg.activation)
        self.critic = _Branch(critic_dim, cfg.hidden, 1, cfg.layers, cfg.recurrent, out_gain=1.0, activation=cfg.activation)


# ----------------------------------------------------------------------------
# trainer
# ----------------------------------------------------------------------------
class _PPOBase:
    centralized_critic = False

    def __init__(
        self,
        env_fn: Callable[[], object],
        config: PPOConfig | None = None,
        adapter_kwargs: dict | None = None,
        reward_transform: Callable[[np.ndarray, dict], np.ndarray] | None = None,
    ):
        """
        Args:
            env_fn: zero-arg function returning a *fresh* raw environment.
            adapter_kwargs: forwarded to `ParallelEnvAdapter` (obs_encoder,
                action_decoder, state_fn, n_actions, agent_ids).
            reward_transform: optional `(rewards[A], info) -> rewards[A]` applied
                to every step before training. This is the hook for reward
                shaping such as MAPPO-CLR-style local cooperation rewards.
        """
        # Here is the PPO config being read on initialization
        self.cfg = config or PPOConfig()
        set_seed(self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.dev = torch.device(self.cfg.device)
        self.env_fn = env_fn
        self.adapter_kwargs = adapter_kwargs or {}
        self.reward_transform = reward_transform

        self.envs = [ParallelEnvAdapter(env_fn(), **self.adapter_kwargs) for _ in range(self.cfg.n_envs)]
        self.E = self.cfg.n_envs
        self.A = self.envs[0].n_agents
        self.n_actions = self.envs[0].n_actions
        self.N = self.E * self.A  # number of independent streams

        self._reset_all(first=True)
        self.obs_dim = self.envs[0].obs_dim
        self.use_id = self.cfg.agent_id and self.A > 1
        id_dim = self.A if self.use_id else 0
        crit_base = self.envs[0].state_dim if self.centralized_critic else self.obs_dim
        self.model = ActorCritic(self.obs_dim + id_dim, crit_base + id_dim, self.n_actions, self.cfg).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr, eps=1e-5)

        H = self.cfg.hidden
        self.h_a = torch.zeros(self.N, H, device=self.dev)
        self.h_c = torch.zeros(self.N, H, device=self.dev)
        self.first = torch.ones(self.N, device=self.dev)

        self.ep_return = np.zeros(self.E)
        self.ep_len = np.zeros(self.E, dtype=int)
        self.finished_returns: list[float] = []
        self.finished_lengths: list[int] = []
        self.finished_metrics: list[dict] = []
        self.env_steps = 0
        self.updates = 0
        self.history: list[dict] = []
        self.run_config: dict = {}

    # ---------------------------------------------------------------- helpers
    def _reset_all(self, first: bool = False) -> None:
        obs, state, alive, mask = [], [], [], []
        for e, env in enumerate(self.envs):
            o, s, al = env.reset(seed=int(self.rng.integers(0, 2**31 - 1)))
            obs.append(o), state.append(s), alive.append(al), mask.append(env.mask.copy())
        self.mask = np.stack(mask)  # [E,A,n_actions] bool, legal actions
        self.obs = np.stack(obs)  # [E,A,D]
        self.state = np.stack(state)  # [E,S]
        self.alive = np.stack(alive)  # [E,A]

    def _ids(self, n_streams: int, n_envs: int) -> torch.Tensor:
        eye = torch.eye(self.A, device=self.dev)
        return eye.repeat(n_envs, 1)  # [E*A, A] env-major

    def _actor_in(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(obs.reshape(self.N, -1), dtype=torch.float32, device=self.dev)
        if self.use_id:
            x = torch.cat([x, self._ids(self.N, self.E)], dim=-1)
        return x

    def _critic_in(self, obs: np.ndarray, state: np.ndarray) -> torch.Tensor:
        if self.centralized_critic:
            x = torch.as_tensor(np.repeat(state, self.A, axis=0), dtype=torch.float32, device=self.dev)
        else:
            x = torch.as_tensor(obs.reshape(self.N, -1), dtype=torch.float32, device=self.dev)
        if self.use_id:
            x = torch.cat([x, self._ids(self.N, self.E)], dim=-1)
        return x

    @staticmethod
    def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Give illegal actions (mask False) a huge negative logit so they get probability ~0."""
        return logits.masked_fill(~mask, -1e9)

    def _team_reward(self, r: np.ndarray) -> np.ndarray:
        mode = self.cfg.reward_mode
        if mode == "individual":
            return r
        agg = r.sum() if mode == "team_sum" else r.mean()
        return np.full_like(r, agg)

    # ------------------------------------------------------------ rollout
    @torch.no_grad()
    def _collect(self) -> dict[str, torch.Tensor]:
        T, N = self.cfg.rollout_len, self.N
        buf = {k: [] for k in ("actor_in", "critic_in", "act", "logp", "val", "rew", "term", "first", "alive", "mask")}
        h_a0, h_c0 = self.h_a.clone(), self.h_c.clone()

        for _ in range(T):
            a_in, c_in = self._actor_in(self.obs), self._critic_in(self.obs, self.state)
            first = self.first
            logits, self.h_a = self.model.actor(a_in[None], self.h_a, first[None])
            value, self.h_c = self.model.critic(c_in[None], self.h_c, first[None])
            mask_t = torch.as_tensor(self.mask.reshape(N, -1), dtype=torch.bool, device=self.dev)
            dist = Categorical(logits=self._mask_logits(logits[0], mask_t))
            act = dist.sample()
            logp = dist.log_prob(act)

            act_np = act.view(self.E, self.A).cpu().numpy()
            rew = np.zeros((self.E, self.A), dtype=np.float32)
            done = np.zeros(self.E, dtype=bool)
            trunc_only = np.zeros(self.E, dtype=bool)
            next_obs, next_state, next_alive, next_mask = [None] * self.E, [None] * self.E, [None] * self.E, [None] * self.E
            for e, env in enumerate(self.envs):
                res = env.step(act_np[e])
                r = res.reward
                if self.reward_transform is not None:
                    r = self.reward_transform(r, res.info)
                self.ep_return[e] += float(np.sum(res.reward))  # report the *raw* reward
                self.ep_len[e] += 1
                rew[e] = self._team_reward(r) * self.cfg.reward_scale  # evaluation/reporting always use the raw reward
                done[e] = res.done
                if res.done:
                    self.finished_metrics.append(dict(res.info.get("episode", {})))
                trunc_only[e] = res.truncated and not res.terminated
                next_obs[e], next_state[e], next_alive[e], next_mask[e] = res.obs, res.state, res.alive, res.mask
            self.env_steps += self.E

            # time-limit truncation is not a real terminal state: bootstrap from V(s')
            if trunc_only.any():
                v_next = self._peek_value(np.stack(next_obs), np.stack(next_state)).view(self.E, self.A).cpu().numpy()
                rew = rew + self.cfg.gamma * v_next * trunc_only[:, None]

            buf["actor_in"].append(a_in), buf["critic_in"].append(c_in)
            buf["act"].append(act), buf["logp"].append(logp), buf["val"].append(value[0, :, 0])
            buf["rew"].append(torch.as_tensor(rew.reshape(N), device=self.dev))
            buf["term"].append(torch.as_tensor(np.repeat(done, self.A).astype(np.float32), device=self.dev))
            buf["first"].append(first)
            buf["mask"].append(mask_t)
            buf["alive"].append(torch.as_tensor(self.alive.reshape(N), dtype=torch.float32, device=self.dev))

            for e in range(self.E):
                if done[e]:
                    self.finished_returns.append(float(self.ep_return[e]))
                    self.finished_lengths.append(int(self.ep_len[e]))
                    self.ep_return[e], self.ep_len[e] = 0.0, 0
                    o, s, al = self.envs[e].reset(seed=int(self.rng.integers(0, 2**31 - 1)))
                    next_obs[e], next_state[e], next_alive[e], next_mask[e] = o, s, al, self.envs[e].mask.copy()
            self.obs, self.state, self.alive = np.stack(next_obs), np.stack(next_state), np.stack(next_alive)
            self.mask = np.stack(next_mask)
            self.first = torch.as_tensor(np.repeat(done, self.A).astype(np.float32), device=self.dev)

        last_v = self._peek_value(self.obs, self.state, update_first=True)
        data = {k: torch.stack(v) for k, v in buf.items()}  # [T,N,...]
        data["h_a0"], data["h_c0"] = h_a0, h_c0
        values = data["val"]
        next_values = torch.cat([values[1:], last_v[None]], dim=0)
        data["adv"], data["ret"] = compute_gae(data["rew"], values, next_values, data["term"], self.cfg.gamma, self.cfg.gae_lambda)
        return data

    @torch.no_grad()
    def _peek_value(self, obs: np.ndarray, state: np.ndarray, update_first: bool = False) -> torch.Tensor:
        """V(s) for the given inputs *without* changing the stored hidden state."""
        c_in = self._critic_in(obs, state)
        first = self.first if update_first else torch.zeros(self.N, device=self.dev)
        v, _ = self.model.critic(c_in[None], self.h_c, first[None])
        return v[0, :, 0]

    # ------------------------------------------------------------ update
    def _update(self, d: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.cfg
        T, N = d["act"].shape
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        n_updates = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(N, device=self.dev)
            for idx in torch.chunk(perm, cfg.n_minibatches):
                if len(idx) == 0:
                    continue
                logits, _ = self.model.actor(d["actor_in"][:, idx], d["h_a0"][idx], d["first"][:, idx])
                values, _ = self.model.critic(d["critic_in"][:, idx], d["h_c0"][idx], d["first"][:, idx])
                dist = Categorical(logits=self._mask_logits(logits, d["mask"][:, idx]))
                new_logp = dist.log_prob(d["act"][:, idx])
                entropy = dist.entropy()
                w = d["alive"][:, idx]
                wsum = w.sum().clamp(min=1.0)

                adv = d["adv"][:, idx]
                mean = (adv * w).sum() / wsum
                std = (((adv - mean) ** 2 * w).sum() / wsum).sqrt()
                adv = (adv - mean) / (std + 1e-8)

                ratio = (new_logp - d["logp"][:, idx]).exp()
                s1 = ratio * adv
                s2 = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv
                pi_loss = -(torch.min(s1, s2) * w).sum() / wsum
                v_loss = (((values[..., 0] - d["ret"][:, idx]) ** 2) * w).sum() / wsum * 0.5
                ent = (entropy * w).sum() / wsum
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["pi_loss"] += pi_loss.item()
                    stats["v_loss"] += v_loss.item()
                    stats["entropy"] += ent.item()
                    stats["approx_kl"] += (((ratio - 1) - (new_logp - d["logp"][:, idx])) * w).sum().item() / wsum.item()
                    stats["clip_frac"] += (((ratio - 1).abs() > cfg.clip).float() * w).sum().item() / wsum.item()
                n_updates += 1
        return {k: v / max(n_updates, 1) for k, v in stats.items()}

    # ------------------------------------------------------------ public API
    # =========== Train method ===========
    def learn(self, total_env_steps: int, log_every: int = 10, callback: Callable[[dict], None] | None = None) -> list[dict]:
        """Train to a total environment-step budget, rounded up to a complete rollout.

        History contains rolling means over the last 20 completed training episodes.
        Before an episode finishes, episode metrics are null rather than fabricated zeros.
        """
        if total_env_steps <= 0:
            raise ValueError("total_env_steps must be positive")
        # Rollout length / number of environments
        per_update = self.cfg.rollout_len * self.E
        n_updates = max(0, math.ceil((total_env_steps - self.env_steps) / per_update))
        target_updates = self.updates + n_updates
        for _ in range(n_updates):
            if self.cfg.anneal_lr:
                for g in self.opt.param_groups:
                    g["lr"] = self.cfg.lr * (1.0 - self.updates / target_updates)
            data = self._collect()
            stats = self._update(data)
            recent = self.finished_returns[-20:]
            self.updates += 1
            metrics = self.finished_metrics[-20:]
            keys = {key for episode in metrics for key in episode}
            episode_metrics = {
                key: float(np.mean([float(ep[key]) for ep in metrics if key in ep]))
                for key in sorted(keys)
                if all(isinstance(ep[key], (int, float, bool, np.number)) for ep in metrics if key in ep)
            }
            stats.update(
                update=self.updates,
                env_steps=self.env_steps,
                mean_return=float(np.mean(recent)) if recent else None,
                mean_ep_len=float(np.mean(self.finished_lengths[-20:])) if recent else None,
                episodes_completed=len(self.finished_returns),
                metrics_episodes=len(metrics),
                episode_metrics=episode_metrics,
                total_reward=episode_metrics.get("total_reward"),
            )
            self.history.append(stats)
            if callback:
                callback(stats)
            if log_every and self.updates % log_every == 0:
                mean_return = "n/a" if stats["mean_return"] is None else f"{stats['mean_return']:.2f}"
                mean_length = "n/a" if stats["mean_ep_len"] is None else f"{stats['mean_ep_len']:.0f}"
                print(
                    f"[{type(self).__name__}] upd {self.updates}/{target_updates}  steps {self.env_steps}  "
                    f"mean return {mean_return}  len {mean_length}  "
                    f"ent {stats['entropy']:.2f}  kl {stats['approx_kl']:.4f}"
                )
        return self.history

    def make_policy(self, deterministic: bool = False):
        """Return `act(obs[A,D], mask[A,n], new_episode=False) -> int array [A]` for running the trained policy.

        The closure keeps the GRU hidden state (when recurrent); pass new_episode=True on the first
        step of every episode. Actions are sampled from the masked policy (or argmax if deterministic).
        """
        h = {"a": torch.zeros(self.A, self.cfg.hidden, device=self.dev)}

        def act(obs: np.ndarray, mask: np.ndarray, new_episode: bool = False) -> np.ndarray:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=self.dev)
                if self.use_id:
                    x = torch.cat([x, torch.eye(self.A, device=self.dev)], dim=-1)
                first = torch.full((self.A,), 1.0 if new_episode else 0.0, device=self.dev)
                logits, h["a"] = self.model.actor(x[None], h["a"], first[None])
                logits = self._mask_logits(logits[0], torch.as_tensor(mask, dtype=torch.bool, device=self.dev))
                a = logits.argmax(-1) if deterministic else Categorical(logits=logits).sample()
                return a.cpu().numpy()

        return act

    def evaluate(self, n_episodes: int = 10, deterministic: bool = False, seed: int = 12345) -> dict[str, float]:
        """Run the current policy on fresh envs. Returns mean/std of the raw team return.

        Defaults to *sampling* from the policy (like during training). `deterministic=True`
        takes the argmax action, which can look much worse: a greedy policy with no
        randomness can get stuck oscillating between two cells in a gridworld.
        """
        env = ParallelEnvAdapter(self.env_fn(), **self.adapter_kwargs)
        rng = np.random.default_rng(seed)
        act = self.make_policy(deterministic)
        returns, lengths = [], []
        for _ in range(n_episodes):
            obs, state, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            new_episode, total, steps, done = True, 0.0, 0, False
            while not done:
                res = env.step(act(obs, env.mask, new_episode))
                obs, new_episode = res.obs, False
                total += float(res.reward.sum())
                steps += 1
                done = res.done
            returns.append(total)
            lengths.append(steps)
        return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)), "mean_length": float(np.mean(lengths))}

    def save(self, path: str) -> None:
        """Atomically save weights, optimizer, progress, metrics and reconstruction config."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        torch.save({
            "format_version": 2,
            "model": self.model.state_dict(), "config": asdict(self.cfg),
            "optimizer": self.opt.state_dict(),
            "env_steps": self.env_steps, "updates": self.updates,
            "history": self.history,
            "finished_returns": self.finished_returns,
            "finished_lengths": self.finished_lengths,
            "finished_metrics": self.finished_metrics,
            "rng": self.rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.dev.type == "cuda" else [],
            "run_config": self.run_config,
        }, temporary)
        temporary.replace(target)

    def load(self, path: str, resume: bool = False) -> None:
        """Load weights, or continue optimization with fresh episodes when resume=True.

        Environment snapshots and recurrent rollout states are not checkpointed.
        Resuming restarts episodes; it is not a bit-for-bit continuation mid-episode.
        """
        ckpt = torch.load(path, map_location=self.dev, weights_only=True)
        if resume and "optimizer" not in ckpt:
            raise ValueError("This legacy checkpoint has weights only; it can be evaluated but not resumed.")
        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError as exc:
            raise ValueError("Checkpoint architecture does not match this environment/configuration.") from exc
        self.run_config = ckpt.get("run_config", {})
        if resume:
            self.opt.load_state_dict(ckpt["optimizer"])
            self.env_steps = ckpt["env_steps"]
            self.updates = ckpt["updates"]
            self.history = ckpt["history"]
            self.finished_returns = ckpt["finished_returns"]
            self.finished_lengths = ckpt["finished_lengths"]
            self.finished_metrics = ckpt["finished_metrics"]
            self.rng.bit_generator.state = ckpt["rng"]
            torch.set_rng_state(ckpt["torch_rng"].cpu())
            if self.dev.type == "cuda" and ckpt.get("cuda_rng"):
                torch.cuda.set_rng_state_all([state.cpu() for state in ckpt["cuda_rng"]])
            self._reset_all()


class PPO(_PPOBase):
    """Single-agent PPO (Experiments 1 and 2: one plow)."""

    centralized_critic = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.A != 1:
            raise ValueError(f"PPO is single-agent but the env has {self.A} agents; use MAPPO.")


class MAPPO(_PPOBase):
    """Multi-Agent PPO: shared decentralised actor + centralised critic on the global state.

    Use `reward_transform=` to plug in reward shaping (e.g. local cooperation reward).
    """

    centralized_critic = True
