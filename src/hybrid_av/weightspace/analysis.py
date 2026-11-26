"""Read-only analysis tools for the editor (interfaces.md section 12). Inputs are (B, 24) float32 states."""

from dataclasses import asdict

import numpy as np
import torch

from hybrid_av.policy.network import CompactPolicy
from hybrid_av.weightspace.apply import apply_edit_plan
from hybrid_av.weightspace.schema import EditPlan

_GAINS = {"alpha_v": 0.6, "delta_y": 3.5, "brake": 0.5}  # scale_action is affine per channel


def _acts(policy: CompactPolicy, Z: np.ndarray) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return policy.forward_with_activations(torch.as_tensor(Z, dtype=torch.float32))[1]


def activation_stats(policy: CompactPolicy, Z: np.ndarray) -> dict:
    acts = _acts(policy, Z)
    out = {}
    for layer in ("h1", "h2"):
        h = acts[layer]
        frac = (acts[f"{layer}_pre"] > 0).float().mean(0)
        out[layer] = {
            "active_frac": frac.tolist(),
            "mean": h.mean(0).tolist(),
            "max": h.max(0).values.tolist(),
            "dead": torch.nonzero(frac == 0).flatten().tolist(),
        }
    return out


def neuron_feature_attribution(policy: CompactPolicy, Z: np.ndarray, layer: str) -> np.ndarray:
    z = torch.as_tensor(Z, dtype=torch.float32).clone().requires_grad_(True)
    _, acts = policy.forward_with_activations(z)
    h, pre = acts[layer], acts[f"{layer}_pre"]
    n_active = (pre > 0).sum(0).clamp(min=1).detach().numpy()  # (64,)
    att = np.zeros((h.shape[1], z.shape[1]), np.float32)
    for j in range(h.shape[1]):
        (g,) = torch.autograd.grad(h[:, j].sum(), z, retain_graph=True)  # zero where unit j is inactive
        att[j] = (g * z).sum(0).detach().numpy() / n_active[j]
    return att


def output_attribution(policy: CompactPolicy, Z: np.ndarray, output: int) -> dict:
    h2 = _acts(policy, Z)["h2"]
    w = policy.fc3.weight[output].detach()
    return {"contribution": (w * h2).mean(0).tolist(), "bias": policy.fc3.bias[output].item()}


def counterfactual(policy: CompactPolicy, z: np.ndarray, feature: int, values: np.ndarray) -> np.ndarray:
    Z = np.tile(np.asarray(z, np.float32), (len(values), 1))
    Z[:, feature] = values
    return _acts(policy, Z)["out"].numpy()


def evaluate_plan_offline(
    policy: CompactPolicy, plan: EditPlan, Z_fail: np.ndarray, Z_nom: np.ndarray
) -> dict:
    edited, diff = apply_edit_plan(policy, plan)

    def shift(Z):  # edited - base tanh outputs, in physical units, (B, 3)
        d = (_acts(edited, Z)["out"] - _acts(policy, Z)["out"]).numpy()
        return d * np.array(list(_GAINS.values()))

    def named(v):
        return dict(zip(_GAINS, (float(x) for x in v)))

    sf, sn = shift(Z_fail), shift(Z_nom)
    empty = np.zeros((1, 3))
    sn = sn if len(sn) else empty
    return {
        "fail_mean_shift": named(sf.mean(0)),
        "nom_mean_abs_shift": named(np.abs(sn).mean(0)),
        "nom_max_abs_shift": named(np.abs(sn).max(0)),
        "diff": asdict(diff),
    }
