import torch

from hybrid_av.policy.network import PARAM_COUNT, CompactPolicy


def test_exact_architecture():
    p = CompactPolicy()
    assert sum(t.numel() for t in p.parameters()) == PARAM_COUNT == 5955
    assert list(p.state_dict()) == [
        "fc1.weight",
        "fc1.bias",
        "fc2.weight",
        "fc2.bias",
        "fc3.weight",
        "fc3.bias",
    ]
    assert p.fc1.weight.shape == (64, 24) and p.fc3.weight.shape == (3, 64)


def test_forward_range_and_dtype():
    y = CompactPolicy()(torch.randn(8, 24) * 10)
    assert y.shape == (8, 3) and y.dtype == torch.float32 and torch.all(y.abs() <= 1)


def test_activations_consistent():
    p = CompactPolicy()
    z = torch.randn(4, 24)
    out, a = p.forward_with_activations(z)
    assert set(a) == {"h1_pre", "h1", "h2_pre", "h2", "out_pre", "out"}
    assert (
        torch.allclose(a["h1"], torch.relu(a["h1_pre"]))
        and torch.allclose(out, p(z))
        and a["h2"].shape == (4, 64)
    )
