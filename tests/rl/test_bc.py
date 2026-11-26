import numpy as np
import torch

from hybrid_av.eval.runner import run_episode
from hybrid_av.policy.network import CompactPolicy, TorchPolicySource
from hybrid_av.rl.bc import behaviour_clone, collect_expert_dataset, dagger
from hybrid_av.rl.expert import ExpertPolicy
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from tests.rl.helpers import _env, _stack


def test_dataset_shapes_and_range():
    Z, Y = collect_expert_dataset([lambda: _env()], ExpertPolicy(), 500, seed=0)
    assert Z.shape == (500, 24) and Y.shape == (500, 3) and np.all(np.abs(Y) < 1) and Z.dtype == np.float32


def test_bc_learns_and_drives():
    Z, Y = collect_expert_dataset([lambda: _env()], ExpertPolicy(), 5000, seed=0)
    pol = behaviour_clone(Z, Y, epochs=30, seed=0)
    with torch.no_grad():
        mse = torch.mean((pol(torch.from_numpy(Z)) - torch.from_numpy(Y)) ** 2).item()
    assert mse < 0.5 * float(np.mean((Y - Y.mean(0)) ** 2))  # beats predicting the mean by 2×
    res = run_episode(
        LiteSim(VehicleParams(), LongitudinalMap()),
        make_scenario("nominal", 50000, {"traffic": False}),
        _stack(),
        TorchPolicySource(pol),
    )
    assert res.route_completion == 1.0


def test_dagger_mixes_policy_actions():
    pol = CompactPolicy()
    plain = collect_expert_dataset([lambda: _env()], ExpertPolicy(), 30, seed=0)
    full = collect_expert_dataset([lambda: _env()], ExpertPolicy(), 30, seed=0, policy=pol, beta=1.0)
    assert np.array_equal(plain[0], full[0]) and np.array_equal(plain[1], full[1])
    # beta=0: executed actions are the policy's; labels stay the expert's
    env = _env()
    seen = []
    step = env.step
    env.step = lambda a: (seen.append(a), step(a))[1]
    Z, Y = collect_expert_dataset([lambda: env], ExpertPolicy(), 10, seed=0, policy=pol, beta=0.0)
    with torch.no_grad():
        want = pol(torch.from_numpy(Z)).numpy()
    assert np.allclose(np.array(seen), want, atol=1e-6)
    assert not np.allclose(Y, want)


def test_dagger_rounds_grow_dataset():
    Z, Y = collect_expert_dataset([lambda: _env()], ExpertPolicy(), 100, seed=0)
    pol, stats = dagger(
        [lambda: _env()], ExpertPolicy(), Z, Y, rounds=2, steps_per_round=200, seed=0, epochs=2
    )
    assert isinstance(pol, CompactPolicy)
    assert [s["n_samples"] for s in stats] == [300, 500] and [s["round"] for s in stats] == [1, 2]
    assert all("val_loss" in s for s in stats)


def test_collection_deterministic_parallel():
    fns = [lambda: _env(), lambda: _env()]
    a = collect_expert_dataset(fns, ExpertPolicy(), 40, seed=3)
    b = collect_expert_dataset(fns, ExpertPolicy(), 40, seed=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_collection_split_counts():
    Z, Y = collect_expert_dataset([lambda: _env(), lambda: _env()], ExpertPolicy(), 7, seed=0)
    assert len(Z) == len(Y) == 7


def test_launch_mask_selects_stopped_near_junction_with_launch_label():
    from hybrid_av.features.spec import I_STOP_DIST, I_VX, normalize
    from hybrid_av.rl.bc import launch_mask

    raw = np.zeros((4, 24))
    raw[:, I_VX] = [0.2, 0.2, 3.0, 0.2]
    raw[:, I_STOP_DIST] = [5.0, 25.0, 5.0, 5.0]
    y = np.array([[0.9, 0, 0], [0.9, 0, 0], [0.9, 0, 0], [-0.9, 0, 0]])
    assert launch_mask(normalize(raw), y).tolist() == [True, False, False, False]


def test_launch_weight_changes_fit():
    from hybrid_av.features.spec import I_STOP_DIST, I_VX, normalize

    rng = np.random.default_rng(0)
    raw = np.zeros((600, 24))
    raw[:, I_VX] = 0.2
    raw[:, I_STOP_DIST] = 5.0
    z = normalize(raw + rng.normal(0, 1e-3, raw.shape)).astype(np.float32)
    y = np.zeros((600, 3), np.float32)
    y[:, 0] = np.where(
        np.arange(600) < 60, 0.9, -0.9
    )  # 10 % launch labels inside an otherwise identical cluster
    z0 = torch.from_numpy(z[:1])

    def alpha(w):
        pol = behaviour_clone(z, y, epochs=40, seed=0, launch_weight=w)
        with torch.no_grad():
            return float(pol(z0)[0, 0])

    assert alpha(10.0) > alpha(1.0) + 0.1
