import numpy as np
import torch

from hybrid_av.policy.network import CompactPolicy, TorchPolicySource, scale_action


def test_source_matches_network():
    p = CompactPolicy()
    src = TorchPolicySource(p)
    z = np.random.default_rng(0).uniform(-1, 1, 24).astype(np.float32)
    expected = scale_action(p(torch.from_numpy(z)[None]).detach().numpy()[0])
    assert src(z, None) == expected and src(z, None) == expected
