import numpy as np

from hybrid_av.eval.metrics import steady_state_mask, steady_state_metrics
from hybrid_av.sim.base import Scenario

DT = 0.05
N = 400  # 20 s


def _sc(junctions=()):
    return Scenario("t", 0, None, None, (), 1.0, (), (), tuple(junctions), 60.0, 8.0)


def _trace(n=N):
    return {
        "t": np.arange(n) * DT,
        "route_s": np.linspace(0.0, 100.0, n),
        "brake_flag": np.zeros(n),
        "mpc_status": np.zeros(n, dtype=np.int8),
        "cross_track": np.full(n, 0.1),
        "heading_err": np.full(n, 0.01),
        "delta": np.sin(np.arange(n) * 0.1),
    }


def test_all_kept_after_warmup():
    m = steady_state_mask(_trace(), _sc(), DT)
    assert m.dtype == bool and not m[:100].any() and m[100:].all()


def test_warmup_only_first_5s():
    m = steady_state_mask(_trace(), _sc(), DT)
    assert m[99] == False and m[100] == True


def test_junction_window_with_5m_margin():
    tr = _trace()
    m = steady_state_mask(tr, _sc([(40.0, 50.0)]), DT)
    s = tr["route_s"]
    inside = (s >= 35.0) & (s <= 55.0)
    assert not m[inside].any()
    assert m[(~inside) & (tr["t"] >= 5.0)].all()


def test_brake_flag():
    tr = _trace()
    tr["brake_flag"][200:210] = 0.5
    tr["brake_flag"][300] = 0.49
    m = steady_state_mask(tr, _sc(), DT)
    assert not m[200:210].any() and m[300] and m[210]


def test_mpc_fallback():
    tr = _trace()
    tr["mpc_status"][150] = 1  # solved_inaccurate is not a fallback
    tr["mpc_status"][200] = 2
    tr["mpc_status"][250] = 3
    m = steady_state_mask(tr, _sc(), DT)
    assert m[150] and not m[200] and not m[250]


def test_uses_trace_time_key():
    tr = _trace()
    tr["t"] = tr["t"] + 10.0  # episode clock starts at 10 s: nothing is warm-up by t < 5
    assert steady_state_mask(tr, _sc(), DT).all()


def test_metrics_on_kept_steps():
    out = steady_state_metrics(_trace(), _sc(), DT)
    assert np.isclose(out["cross_track_rmse_ss"], 0.1)
    assert np.isclose(out["heading_rmse_deg_ss"], np.degrees(0.01))
    assert np.isclose(out["mask_fraction"], 0.25)
    assert out["steering_jerk_rms_ss"] > 0


def test_all_masked_gives_nan():
    tr = _trace()
    tr["brake_flag"][:] = 1.0
    out = steady_state_metrics(tr, _sc(), DT)
    assert out["mask_fraction"] == 1.0
    for k in ("cross_track_rmse_ss", "heading_rmse_deg_ss", "steering_jerk_rms_ss"):
        assert np.isnan(out[k])


def test_jerk_ss_ignores_masked_gap():
    """A steering jump inside a masked stretch must not show up as jerk."""
    import numpy as np

    from hybrid_av.eval.metrics import _segment_jerk_rms

    delta = np.r_[np.zeros(10), np.ones(5), np.ones(10)]  # jump only inside the masked middle
    keep = np.r_[np.ones(10, bool), np.zeros(5, bool), np.ones(10, bool)]
    assert _segment_jerk_rms(delta, keep, 0.05) == 0.0
    assert np.isnan(_segment_jerk_rms(delta, np.zeros(25, bool), 0.05))
