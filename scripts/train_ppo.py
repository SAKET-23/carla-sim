"""Train the CompactPolicy: optional behaviour-cloning warm start from the expert, then vectorised PPO in LiteSim."""

import argparse
import dataclasses
import json
import time
from functools import partial
from pathlib import Path

from hybrid_av.config import load_dataclass, make_run_dir, seed_everything
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.policy.io import load_policy
from hybrid_av.rl.bc import behaviour_clone, collect_expert_dataset, dagger
from hybrid_av.rl.env import DrivingEnv, nominal_sampler
from hybrid_av.rl.expert import ExpertPolicy
from hybrid_av.rl.ppo import PPOConfig, train_ppo
from hybrid_av.rl.reward import RewardConfig
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def _litesim() -> LiteSim:
    return LiteSim(VehicleParams(), LongitudinalMap())


def make_env(reward: RewardConfig) -> DrivingEnv:
    """Nominal training env (seeds 0-9,999 only); picklable via functools.partial for AsyncVectorEnv."""
    stack = AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig())
    return DrivingEnv(_litesim, nominal_sampler, stack, reward)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--reward-config", type=Path, default=Path("configs/reward/default.yaml"))
    ap.add_argument("--bc-steps", type=int, default=300_000)
    ap.add_argument("--dagger-rounds", type=int, default=0)
    ap.add_argument(
        "--dagger-steps", type=int, default=100_000, help="expert-labelled steps per DAgger round"
    )
    ap.add_argument(
        "--launch-weight", type=float, default=1.0, help="BC/DAgger loss weight of junction-launch samples"
    )
    ap.add_argument("--envs", type=int, default=None, help="parallel envs (default: n_envs from the config)")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--init-actor", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    a = ap.parse_args()

    overrides = {"total_steps": a.total_steps} if a.total_steps is not None else None
    cfg = load_dataclass(PPOConfig, a.config, overrides)
    reward = load_dataclass(RewardConfig, a.reward_config)
    n_envs = a.envs or cfg.n_envs
    seed_everything(cfg.seed)
    d = make_run_dir(
        "train_ppo", {"ppo": cfg, "reward": reward, "args": vars(a) | {"envs": n_envs}}, root=a.out
    )
    env_fns = [partial(make_env, reward) for _ in range(n_envs)]

    init, bc_wall, dagger_wall, dagger_stats = None, 0.0, 0.0, []
    if a.init_actor is not None:
        init = load_policy(a.init_actor)[0]
    elif a.bc_steps > 0:
        t0 = time.time()
        z, y = collect_expert_dataset(env_fns, ExpertPolicy(), a.bc_steps, cfg.seed)
        init = behaviour_clone(z, y, seed=cfg.seed, launch_weight=a.launch_weight)
        bc_wall = time.time() - t0
        print(f"BC: {len(z)} samples in {bc_wall:.1f}s")
        if a.dagger_rounds > 0:
            t0 = time.time()
            init, dagger_stats = dagger(
                env_fns,
                ExpertPolicy(),
                z,
                y,
                a.dagger_rounds,
                a.dagger_steps,
                cfg.seed,
                launch_weight=a.launch_weight,
            )
            dagger_wall = time.time() - t0
            print(f"DAgger: {a.dagger_rounds} rounds in {dagger_wall:.1f}s, {dagger_stats[-1]}")
    t0 = time.time()
    train_ppo(env_fns, cfg, d, init_actor=init)
    ppo_wall = time.time() - t0
    print(f"PPO: {cfg.total_steps} steps in {ppo_wall:.1f}s")

    rows = (d / "train_log.csv").read_text().splitlines()
    final = dict(zip(rows[0].split(","), map(float, rows[-1].split(",")))) if len(rows) > 1 else {}
    metrics = {
        "bc_wall_s": bc_wall,
        "dagger_wall_s": dagger_wall,
        "dagger": dagger_stats,
        "ppo_wall_s": ppo_wall,
        "total_steps": int(final.get("step", cfg.total_steps)),
        "final_log_row": final,
        "ppo": dataclasses.asdict(cfg),
    }
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"run dir {d}")


if __name__ == "__main__":
    main()
