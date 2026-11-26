"""Single-file PPO: tanh-squashed Gaussian actor over CompactPolicy (Plan/01-theory/reinforcement-learning.md)."""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from hybrid_av.policy.io import save_policy
from hybrid_av.policy.network import CompactPolicy

_TANH_EPS = 1e-6


@dataclass(frozen=True, slots=True)
class PPOConfig:
    n_envs: int = 12
    n_steps: int = 512
    total_steps: int = 5_000_000
    lr: float = 3e-4
    lr_schedule: Literal["linear", "constant"] = "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    n_epochs: int = 10
    minibatch_size: int = 1024
    vf_coef: float = 0.5
    vf_clip: float = 10.0
    ent_coef: float = 0.0
    log_std_init: float = -1.0
    log_std_min: float = -3.0
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    reward_norm: bool = True
    critic_warmup_iters: int = 20
    checkpoint_every: int = 250_000
    seed: int = 0


class Critic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(24, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class ActorHead(nn.Module):
    def __init__(self, policy: CompactPolicy, cfg: PPOConfig) -> None:
        super().__init__()
        self.policy = policy
        self.log_std = nn.Parameter(torch.full((3,), cfg.log_std_init))
        self.log_std_min = cfg.log_std_min

    def dist(self, z: torch.Tensor) -> Normal:
        mean = self.policy.forward_with_activations(z)[1]["out_pre"]
        return Normal(mean, torch.exp(self.log_std.clamp(min=self.log_std_min)))

    @staticmethod
    def log_prob(d: Normal, u: torch.Tensor) -> torch.Tensor:
        return d.log_prob(u).sum(-1) - torch.log(1 - torch.tanh(u) ** 2 + _TANH_EPS).sum(-1)

    def sample(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the stored pre-tanh action u and its log-probability; the executed action is tanh(u)."""
        d = self.dist(z)
        u = d.sample()
        return u, self.log_prob(d, u)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones_terminated: np.ndarray,
    dones_truncated: np.ndarray,
    last_values: np.ndarray,
    truncated_values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    T = len(rewards)
    term, trunc = dones_terminated.astype(bool), dones_truncated.astype(bool)
    adv = np.zeros_like(rewards, dtype=np.float64)
    nxt_adv = np.zeros_like(last_values, dtype=np.float64)
    for t in reversed(range(T)):
        v_next = last_values if t == T - 1 else values[t + 1]
        v_next = np.where(trunc[t], truncated_values[t], v_next)
        v_next = np.where(term[t], 0.0, v_next)
        cut = term[t] | trunc[t]
        delta = rewards[t] + gamma * v_next - values[t]
        nxt_adv = delta + np.where(cut, 0.0, gamma * lam * nxt_adv)
        adv[t] = nxt_adv
    return adv, adv + values


def ppo_loss(
    head: ActorHead, critic: Critic, batch: dict[str, torch.Tensor], cfg: PPOConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    d = head.dist(batch["obs"])
    log_ratio = head.log_prob(d, batch["u"]) - batch["logp_old"]
    ratio = torch.exp(log_ratio)
    adv = batch["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    policy_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
    v = critic(batch["obs"])
    v_clipped = batch["v_old"] + torch.clamp(v - batch["v_old"], -cfg.vf_clip, cfg.vf_clip)
    value_loss = 0.5 * torch.max((v - batch["ret"]) ** 2, (v_clipped - batch["ret"]) ** 2).mean()
    entropy = d.entropy().sum(-1).mean()
    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
    with torch.no_grad():
        stats = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
            "clip_frac": ((ratio - 1).abs() > cfg.clip_eps).float().mean().item(),
        }
    return loss, stats


_LOG_COLS = "step,wall_s,ep_return_mean,ep_len_mean,completion_rate,collision_rate,approx_kl,clip_frac,entropy,value_loss"


def train_ppo(
    env_fns: list[Callable[[], gym.Env]],
    cfg: PPOConfig,
    out_dir: Path,
    init_actor: CompactPolicy | None = None,
) -> CompactPolicy:
    """Vectorised PPO. Rollout batch is n_steps * len(env_fns); cfg.n_envs is not used here."""
    torch.set_num_threads(1)  # thread-count-dependent reductions would break determinism
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    out_dir = Path(out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    policy = copy.deepcopy(init_actor) if init_actor is not None else CompactPolicy()
    head, critic = ActorHead(policy, cfg), Critic()
    opt = torch.optim.Adam(
        [{"params": head.parameters(), "name": "actor"}, {"params": critic.parameters(), "name": "critic"}],
        lr=cfg.lr,
    )
    n = len(env_fns)
    log_path = out_dir / "train_log.csv"
    log_path.write_text(_LOG_COLS + "\n")
    if cfg.total_steps > 0:
        envs_cls = gym.vector.AsyncVectorEnv if n > 1 else gym.vector.SyncVectorEnv
        envs = envs_cls(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        obs, _ = envs.reset(seed=[cfg.seed + i for i in range(n)])
        # per-env running discounted return and its running variance (Welford) for reward scaling
        disc, rn, rmean, rm2 = np.zeros(n), 0, 0.0, 0.0
        ep_ret, ep_len = np.zeros(n), np.zeros(n)
        # carried forward when no episode ends in an iteration
        prev = {"ep_return_mean": 0.0, "ep_len_mean": 0.0, "completion_rate": 0.0, "collision_rate": 0.0}
        step, it, t0, last_ckpt = 0, 0, time.time(), 0
        T = cfg.n_steps
        while step < cfg.total_steps:
            O, U, LP, V = (
                np.zeros((T, n, 24), np.float32),
                np.zeros((T, n, 3), np.float32),
                np.zeros((T, n)),
                np.zeros((T, n)),
            )
            R, TERM, TRUNC, TV = (np.zeros((T, n)) for _ in range(4))
            ends = []  # (return, length, completed, collided) of episodes finishing this iteration
            for t in range(T):
                o = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    u, logp = head.sample(o)
                    v = critic(o)
                obs, r, term, trunc, info = envs.step(torch.tanh(u).numpy())
                O[t], U[t], LP[t], V[t] = o.numpy(), u.numpy(), logp.numpy(), v.numpy()
                ep_ret += r
                ep_len += 1
                done = term | trunc
                for i in np.flatnonzero(done):
                    si = info["final_info"]["step_info"][i]
                    ends.append(
                        (ep_ret[i], ep_len[i], float(si.route_completed), float(si.collision != "none"))
                    )
                    ep_ret[i] = ep_len[i] = 0
                if trunc.any():
                    fo = torch.as_tensor(
                        np.stack(info["final_obs"][np.flatnonzero(trunc)]), dtype=torch.float32
                    )
                    with torch.no_grad():
                        TV[t, trunc] = critic(fo).numpy()
                if cfg.reward_norm:
                    disc = disc * cfg.gamma + r
                    for x in disc:
                        rn += 1
                        d = x - rmean
                        rmean += d / rn
                        rm2 += d * (x - rmean)
                    r = r / np.sqrt(rm2 / max(rn, 1) + 1e-8)
                    disc[done] = 0.0
                R[t], TERM[t], TRUNC[t] = r, term, trunc
            step += T * n
            it += 1
            with torch.no_grad():
                last_v = critic(torch.as_tensor(obs, dtype=torch.float32)).numpy()
            adv, ret = compute_gae(R, V, TERM, TRUNC, last_v, TV, cfg.gamma, cfg.gae_lambda)
            flat = lambda x: torch.as_tensor(x, dtype=torch.float32).reshape(T * n, *x.shape[2:])
            data = {
                "obs": flat(O),
                "u": flat(U),
                "logp_old": flat(LP),
                "v_old": flat(V),
                "adv": flat(adv),
                "ret": flat(ret),
            }
            lr = cfg.lr * (1 - (step - T * n) / cfg.total_steps) if cfg.lr_schedule == "linear" else cfg.lr
            for g in opt.param_groups:
                g["lr"] = lr
            warm = it <= cfg.critic_warmup_iters
            stats = []
            for _ in range(cfg.n_epochs):
                perm = torch.randperm(T * n)
                ep_stats = []
                for b in perm.split(cfg.minibatch_size):
                    loss, st = ppo_loss(head, critic, {k: v[b] for k, v in data.items()}, cfg)
                    opt.zero_grad()
                    loss.backward()
                    if warm:
                        for p in head.parameters():
                            p.grad = None
                    nn.utils.clip_grad_norm_(
                        [p for g in opt.param_groups for p in g["params"] if p.grad is not None],
                        cfg.max_grad_norm,
                    )
                    opt.step()
                    ep_stats.append(st)
                stats += ep_stats
                if np.mean([s["approx_kl"] for s in ep_stats]) > cfg.target_kl:
                    break
            if ends:
                e = np.mean(ends, axis=0)
                prev = {
                    "ep_return_mean": e[0],
                    "ep_len_mean": e[1],
                    "completion_rate": e[2],
                    "collision_rate": e[3],
                }
            m = {
                k: float(np.mean([s[k] for s in stats]))
                for k in ("approx_kl", "clip_frac", "entropy", "value_loss")
            }
            row = [
                step,
                time.time() - t0,
                *prev.values(),
                m["approx_kl"],
                m["clip_frac"],
                m["entropy"],
                m["value_loss"],
            ]
            with log_path.open("a") as f:
                f.write(",".join(str(x) for x in row) + "\n")
            if cfg.checkpoint_every and step // cfg.checkpoint_every > last_ckpt // cfg.checkpoint_every:
                save_policy(
                    policy, out_dir / "checkpoints" / f"actor_{step}.pt", {"step": step, "seed": cfg.seed}
                )
                torch.save(critic.state_dict(), out_dir / f"critic_{step}.pt")
            last_ckpt = step
        envs.close()
    save_policy(policy, out_dir / "actor_final.pt", {"step": cfg.total_steps, "seed": cfg.seed})
    return policy
