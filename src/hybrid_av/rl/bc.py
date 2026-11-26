"""Behaviour-cloning warm start (Plan/01-theory/reinforcement-learning.md section 5)."""

import logging
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

import cloudpickle
import numpy as np
import torch

from hybrid_av.features.spec import I_STOP_DIST, I_VX, denormalize
from hybrid_av.policy.network import CompactPolicy, unscale_action
from hybrid_av.rl.env import DrivingEnv
from hybrid_av.rl.expert import ActionSource

log = logging.getLogger(__name__)


def collect_expert_dataset(
    env_fns: list[Callable[[], DrivingEnv]],
    expert: ActionSource,
    n_steps: int,
    seed: int,
    noise_std: float = 0.1,
    policy: CompactPolicy | None = None,
    beta: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out the expert with tanh-space exploration noise on the executed action; labels stay clean.

    With `policy` (DAgger), each step executes the expert (plus noise) with probability `beta`, else the
    policy's deterministic tanh output. The coin is only drawn when beta < 1, so beta=1 equals plain collection.
    """
    E = len(env_fns)
    counts = [n_steps // E + (i < n_steps % E) for i in range(E)]
    sd = policy.state_dict() if policy is not None else None
    if E == 1:  # in-process: keeps unpicklable/patched envs working
        parts = [_collect_one(env_fns[0], expert, counts[0], seed, noise_std, sd, beta)]
    else:
        # cloudpickle ships lambdas; spawn context is safe with torch/macOS
        blobs = [
            cloudpickle.dumps((fn, expert, n, seed + i, noise_std, sd, beta))
            for i, (fn, n) in enumerate(zip(env_fns, counts))
        ]
        with ProcessPoolExecutor(E, mp_context=mp.get_context("spawn")) as ex:
            parts = list(ex.map(_worker, blobs))  # env-index order: deterministic
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _worker(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    torch.set_num_threads(1)
    return _collect_one(*cloudpickle.loads(blob))


def _collect_one(env_fn, expert, n, seed, noise_std, policy_state, beta):
    rng = np.random.default_rng(seed)
    env = env_fn()
    policy = None
    if policy_state is not None:
        policy = CompactPolicy()
        policy.load_state_dict(policy_state)
    zs, ys = [], []
    obs = env.reset(seed=seed)[0]
    for _ in range(n):
        label = unscale_action(expert(obs, env._frame))
        zs.append(obs), ys.append(label)
        if policy is not None and beta < 1.0 and rng.random() >= beta:
            with torch.no_grad():
                act = policy(torch.from_numpy(obs)[None])[0].numpy()
        else:
            act = np.clip(label + rng.normal(0.0, noise_std, 3), -1.0, 1.0)
        obs, _, term, trunc, _ = env.step(act.astype(np.float32))
        if term or trunc:
            obs = env.reset()[0]
    return np.asarray(zs, np.float32).reshape(-1, len(obs)), np.asarray(ys, np.float32).reshape(-1, 3)


def launch_mask(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Junction-launch samples: ego nearly stopped (vx < 1 m/s) within 10 m of a junction entry, expert alpha_v >= 0.5."""
    raw = denormalize(z)
    return (raw[:, I_VX] < 1.0) & (raw[:, I_STOP_DIST] <= 10.0) & (0.6 * (y[:, 0] + 1) >= 0.5)


def behaviour_clone(
    z: np.ndarray,
    y: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    seed: int = 0,
    launch_weight: float = 1.0,
) -> CompactPolicy:
    """MSE between pre-tanh output and atanh(clip(y, +-0.999)); Adam, batch 256, 10 % validation split.

    Training loss weights `launch_mask` samples by `launch_weight`; the validation loss is unweighted.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    zt = torch.from_numpy(z)
    target = torch.atanh(torch.from_numpy(np.clip(y, -0.999, 0.999)))
    w = torch.where(torch.from_numpy(launch_mask(z, y)), launch_weight, 1.0).float()
    perm = torch.randperm(len(zt), generator=gen)
    n_val = len(zt) // 10
    val, tr = perm[:n_val], perm[n_val:]
    pol = CompactPolicy()
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    for ep in range(epochs):
        total = 0.0
        for idx in tr[torch.randperm(len(tr), generator=gen)].split(256):
            err = ((pol.forward_with_activations(zt[idx])[1]["out_pre"] - target[idx]) ** 2).mean(1)
            loss = torch.mean(w[idx] * err)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        with torch.no_grad():
            v = torch.mean((pol.forward_with_activations(zt[val])[1]["out_pre"] - target[val]) ** 2).item()
        log.info("bc epoch %d train %.5f val %.5f", ep, total / len(tr), v)
    pol.val_loss = v  # read by dagger() for its per-round stats
    return pol


def dagger(
    env_fns: list[Callable[[], DrivingEnv]],
    expert: ActionSource,
    z0: np.ndarray,
    y0: np.ndarray,
    rounds: int,
    steps_per_round: int,
    seed: int,
    epochs: int = 50,
    launch_weight: float = 1.0,
) -> tuple[CompactPolicy, list[dict]]:
    """DAgger: round i collects with beta = 0.5**i under the current policy, then retrains BC on the aggregate."""
    pol = behaviour_clone(z0, y0, epochs=epochs, seed=seed, launch_weight=launch_weight)
    z, y, stats = z0, y0, []
    for i in range(1, rounds + 1):
        zi, yi = collect_expert_dataset(env_fns, expert, steps_per_round, seed + i, policy=pol, beta=0.5**i)
        z, y = np.concatenate([z, zi]), np.concatenate([y, yi])
        pol = behaviour_clone(z, y, epochs=epochs, seed=seed + i, launch_weight=launch_weight)
        stats.append({"round": i, "n_samples": len(z), "val_loss": pol.val_loss})
        log.info("dagger round %d: %s", i, stats[-1])
    return pol, stats
