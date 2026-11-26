import numpy as np
import torch
from hybrid_av_nodes import convert as C
from hybrid_av_nodes.policy_core import PolicyCore, PolicyTickGate
from msgns import feature_vector

from hybrid_av.features.extractor import FeatureFrame
from hybrid_av.policy.io import save_policy
from hybrid_av.policy.network import CompactPolicy


def frame():
    return FeatureFrame(raw=np.zeros(24), z=np.zeros(24, np.float32), route_s=0.0, lateral=0.0,
                        heading_error=0.0, mu_est=1.0, lead=None, stop_s=None,
                        dist_to_junction=np.inf, cross_dist_conflict=np.inf)  # fmt: skip


def test_constant_policy():
    a, h = PolicyCore("constant").act(frame())
    assert (a.alpha_v, a.delta_y, a.brake) == (1.0, 0.0, 0.0) and h == "constant"


def test_policy_core_nan_fallback(tmp_path):
    p = CompactPolicy()
    with torch.no_grad():
        p.fc3.bias[0] = float("nan")
    save_policy(p, tmp_path / "nan.pt", {})
    core = PolicyCore(str(tmp_path / "nan.pt"))
    a, _ = core.act(frame())
    assert (a.alpha_v, a.delta_y, a.brake) == (0.0, 0.0, 1.0) and core.nan_count == 1


def test_load_policy_bad_path_keeps_old(tmp_path):
    torch.manual_seed(0)
    save_policy(CompactPolicy(), tmp_path / "a.pt", {})
    core = PolicyCore(str(tmp_path / "a.pt"))
    _, h0 = core.act(frame())
    ok, _, msg = core.load(str(tmp_path / "missing.pt"))
    assert not ok and "missing.pt" in msg and core.act(frame())[1] == h0


def test_policy_tick_gate():
    g = PolicyTickGate(dt=0.05, policy_period=2)
    assert [g(12.30 + 0.05 * k) for k in range(5)] == [True, False, True, False, True]


def test_hot_swap(tmp_path):
    torch.manual_seed(0)
    save_policy(CompactPolicy(), tmp_path / "a.pt", {})
    torch.manual_seed(1)
    save_policy(CompactPolicy(), tmp_path / "b.pt", {})
    core = PolicyCore(str(tmp_path / "a.pt"))
    _, ha = core.act(frame())
    ok, hb, _ = core.load(str(tmp_path / "b.pt"))
    assert ok and hb != ha and core.act(frame())[1] == hb


def test_frame_from_feature_msg():
    m = feature_vector()
    m.raw, m.normalized = [float(i) for i in range(24)], [0.5] * 24
    m.mu_est, m.route_s = 0.8, 12.5
    f = C.frame_from_feature_msg(m)
    assert (f.lateral, f.heading_error, f.route_s) == (3.0, 4.0, 12.5)
    assert f.z.dtype == np.float32 and f.lead is None and f.stop_s is None
