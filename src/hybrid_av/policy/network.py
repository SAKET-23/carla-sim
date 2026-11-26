"""CompactPolicy: the 24-64-64-3 actor. See Plan/03-models/policy-mlp.md."""

import numpy as np
import torch
from torch import nn

from hybrid_av.types import PolicyAction

PARAM_COUNT = 5955


class CompactPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 3)
        # policy-mlp.md §3: small head, bias at the neutral action y = [2/3, 0, -0.995]
        with torch.no_grad():
            self.fc3.weight.mul_(0.01)
            self.fc3.bias.copy_(torch.atanh(torch.tensor([0.6667, 0.0, -0.995])))

    def forward_with_activations(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h1_pre = self.fc1(z)
        h1 = torch.relu(h1_pre)
        h2_pre = self.fc2(h1)
        h2 = torch.relu(h2_pre)
        out_pre = self.fc3(h2)
        out = torch.tanh(out_pre)
        acts = {"h1_pre": h1_pre, "h1": h1, "h2_pre": h2_pre, "h2": h2, "out_pre": out_pre, "out": out}
        return out, acts

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.forward_with_activations(z)[0]


_EPS = 1e-6


def scale_action(y: np.ndarray) -> PolicyAction:
    return PolicyAction(float(0.6 * (y[0] + 1)), float(3.5 * y[1]), float(0.5 * (y[2] + 1)))


def unscale_action(a: PolicyAction) -> np.ndarray:
    y = np.array([a.alpha_v / 0.6 - 1, a.delta_y / 3.5, 2 * a.brake - 1])
    return np.clip(y, -1 + _EPS, 1 - _EPS)


class TorchPolicySource:
    """ActionSource: deterministic forward pass."""

    def __init__(self, policy: CompactPolicy) -> None:
        self.policy = policy

    def __call__(self, z: np.ndarray, frame: object) -> PolicyAction:
        with torch.no_grad():
            y = self.policy(torch.as_tensor(z, dtype=torch.float32)[None])
        return scale_action(y.numpy()[0])
