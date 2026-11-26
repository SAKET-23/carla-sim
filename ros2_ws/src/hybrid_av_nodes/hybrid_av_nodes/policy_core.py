"""Policy evaluation with NaN fallback, hot swap and the policy tick rule. Pure core: no rclpy."""

import logging
from pathlib import Path

import numpy as np

from hybrid_av.features.extractor import FeatureFrame
from hybrid_av.policy.io import load_policy, policy_hash
from hybrid_av.policy.network import TorchPolicySource
from hybrid_av.rl.expert import ConstantAction, ExpertPolicy
from hybrid_av.types import PolicyAction

log = logging.getLogger("policy_core")
_BRAKE = PolicyAction(alpha_v=0.0, delta_y=0.0, brake=1.0)


def _build(spec: str):
    if spec in ("constant", "expert"):
        return (ConstantAction() if spec == "constant" else ExpertPolicy()), spec
    policy, _ = load_policy(Path(spec))
    return TorchPolicySource(policy), policy_hash(policy)


class PolicyCore:
    def __init__(self, spec: str) -> None:
        self._source, self._hash = _build(spec)
        self.nan_count = 0

    def act(self, frame: FeatureFrame) -> tuple[PolicyAction, str]:
        a = self._source(frame.z, frame)
        if not np.isfinite([a.alpha_v, a.delta_y, a.brake]).all():
            self.nan_count += 1
            log.error("policy output is not finite; braking this tick (architecture.md section 8)")
            return _BRAKE, self._hash
        return a, self._hash

    def load(self, path: str) -> tuple[bool, str, str]:
        try:
            source, h = _build(path)
        except Exception as e:  # noqa: BLE001 - any load failure keeps the running policy
            return False, self._hash, f"cannot load {path}: {e}"
        self._source, self._hash = source, h
        return True, h, f"loaded {path}"


class PolicyTickGate:
    """True on policy ticks; the first stamp seen is tick 0 (same rule as mpc_node)."""

    def __init__(self, dt: float = 0.05, policy_period: int = 2) -> None:
        self.dt, self.period, self._t0 = dt, policy_period, None

    def __call__(self, stamp: float) -> bool:
        if self._t0 is None:
            self._t0 = stamp
        return round((stamp - self._t0) / self.dt) % self.period == 0
