from dataclasses import asdict

import numpy as np
import pytest

from hybrid_av.config import load_dataclass
from hybrid_av.policy.io import policy_hash
from hybrid_av.policy.network import CompactPolicy
from hybrid_av.rl.ppo import PPOConfig, train_ppo
from tests.rl.helpers import _env


def _cfg(**kw):
    return PPOConfig(
        **{**asdict(load_dataclass(PPOConfig, "configs/ppo/litesim_nominal.yaml")), "total_steps": 1024, **kw}
    )


def test_smoke_outputs(tmp_path):
    pol = train_ppo([lambda: _env(overrides={"traffic": False})] * 2, _cfg(), tmp_path)
    assert isinstance(pol, CompactPolicy) and (tmp_path / "actor_final.pt").exists()
    log = (tmp_path / "train_log.csv").read_text().splitlines()
    assert log[0].startswith("step,wall_s,ep_return_mean")


def test_checkpoints_and_finite_log(tmp_path):
    train_ppo([lambda: _env(overrides={"traffic": False})] * 2, _cfg(checkpoint_every=512), tmp_path)
    assert (tmp_path / "checkpoints" / "actor_1024.pt").exists() and (tmp_path / "critic_1024.pt").exists()
    log = np.genfromtxt(tmp_path / "train_log.csv", delimiter=",", names=True)
    assert all(np.all(np.isfinite(np.atleast_1d(log[n]))) for n in log.dtype.names)


def test_init_actor_used(tmp_path):
    init = CompactPolicy()
    h0 = policy_hash(init)
    pol = train_ppo([lambda: _env()], _cfg(total_steps=0), tmp_path, init_actor=init)
    assert policy_hash(pol) == h0 and pol is not init


def test_training_deterministic(tmp_path):
    cfg = _cfg(seed=11, critic_warmup_iters=0)  # no warm-up, so the one update changes the actor
    a = train_ppo([lambda: _env()] * 2, cfg, tmp_path / "a")
    b = train_ppo([lambda: _env()] * 2, cfg, tmp_path / "b")
    untrained = train_ppo([lambda: _env()] * 2, _cfg(seed=11, total_steps=0), tmp_path / "c")
    assert policy_hash(a) == policy_hash(b) != policy_hash(untrained)


@pytest.mark.slow
def test_improves_over_random(tmp_path):
    train_ppo([lambda: _env(overrides={"traffic": False})] * 4, _cfg(total_steps=200_000), tmp_path)
    log = np.genfromtxt(tmp_path / "train_log.csv", delimiter=",", names=True)
    assert log["ep_return_mean"][-1] > log["ep_return_mean"][0]
