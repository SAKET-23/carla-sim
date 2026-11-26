import pytest
import torch

from hybrid_av.policy.io import save_policy
from hybrid_av.policy.network import CompactPolicy
from hybrid_av.weightspace.bundle import export_bundle, load_bundle
from tests.weightspace.test_bundle import fake_result


@pytest.fixture
def zero_policy():
    p = CompactPolicy()
    with torch.no_grad():
        for t in p.parameters():
            t.zero_()
    return p


@pytest.fixture
def tiny_policy():
    """One live path: z[7] -> h1[0] -> h2[0] -> out[0] (alpha_v channel), gains 1."""
    p = CompactPolicy()
    with torch.no_grad():
        for t in p.parameters():
            t.zero_()
        p.fc1.weight[0, 7] = 1.0
        p.fc2.weight[0, 0] = 1.0
        p.fc3.weight[0, 0] = 1.0
    return p


@pytest.fixture
def seeded_policy():
    torch.manual_seed(0)
    return CompactPolicy()


@pytest.fixture
def bundle_path(tmp_path, tiny_policy):
    """Wet-curve bundle from tiny_policy: two synthetic failures (seeds 50000, 50001), one nominal (seed 0)."""
    pp = tmp_path / "policy.pt"
    save_policy(tiny_policy, pp, {"name": "tiny"})
    return export_bundle(
        tiny_policy,
        [fake_result(50000, 40, fail_at=24), fake_result(50001, 30, fail_at=28)],
        [fake_result(0, 20, "nominal")],
        tmp_path,
        pp,
    )


@pytest.fixture
def bundle_fixture(bundle_path):
    return load_bundle(bundle_path)
