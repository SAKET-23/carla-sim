import numpy as np
import torch

from hybrid_av.policy.network import CompactPolicy
from hybrid_av.rl.ppo import ActorHead, Critic, PPOConfig, compute_gae, ppo_loss


def _batch(u, n=32):
    g = torch.Generator().manual_seed(0)
    return {
        "obs": torch.rand(n, 24, generator=g) * 2 - 1,
        "u": u,
        "logp_old": torch.zeros(n),
        "adv": torch.randn(n, generator=g),
        "ret": torch.randn(n, generator=g),
        "v_old": torch.zeros(n),
    }


def test_gae_terminal_no_bootstrap():
    r = np.array([[1.0], [1.0]])
    v = np.array([[0.5], [0.5]])
    _adv, ret = compute_gae(
        r, v, np.array([[0], [1]]), np.array([[0], [0]]), np.array([9.0]), np.zeros((2, 1)), 0.99, 0.95
    )
    assert abs(ret[1, 0] - 1.0) < 1e-9  # terminal: no value after it


def test_gae_bootstraps_truncation():
    r = np.array([[1.0]])
    v = np.array([[0.5]])
    _, ret = compute_gae(
        r, v, np.array([[0]]), np.array([[1]]), np.array([0.0]), np.array([[2.0]]), 0.99, 0.95
    )
    assert abs(ret[0, 0] - (1.0 + 0.99 * 2.0)) < 1e-9


def test_actor_head_keeps_policy_clean():
    head = ActorHead(CompactPolicy(), PPOConfig())
    assert list(head.policy.state_dict()) == [
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
    ]
    assert "log_std" in dict(head.named_parameters())


def test_ppo_no_nan_at_boundary():  # tanh(±20) == ±1 in float32: the correction term must stay finite
    head, critic = ActorHead(CompactPolicy(), PPOConfig()), Critic()
    batch = _batch(u=torch.tensor([[20.0, -20.0, 20.0]] * 32))
    loss, stats = ppo_loss(head, critic, batch, PPOConfig())
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in stats.values())
