import numpy as np

from hybrid_av.features.spec import normalize
from hybrid_av.weightspace.analysis import evaluate_plan_offline
from hybrid_av.weightspace.editors.scripted import ScriptedEditor


class B:
    def __init__(self, p, scenario, Zw, Zn):
        self.policy, self.scenario, self.Z_window, self.Z_nom = p, scenario, Zw, Zn


def raw(n, **slots):
    r = np.zeros((n, 24))
    r[:, 0] = 12.0
    r[:, 6] = 1.0
    r[:, 20] = 10.0
    r[:, 21] = 13.9
    r[:, 22] = 3.5
    for k, v in slots.items():
        r[:, int(k[1:])] = v
    return normalize(r).astype(np.float32)


def test_wet_curve_ramp_on_surface_cue(seeded_policy):
    Zn = np.vstack([raw(20, s7=v) for v in (0.0, 0.15, 0.3)])  # nominal roughness 0–0.3
    b = B(seeded_policy, "wet_curve", raw(40, s7=0.8), Zn)
    plan = ScriptedEditor().propose(b, [])
    assert len(plan.edits) == 93 and "fc1.weight" in {e.tensor for e in plan.edits}
    assert any(e.index[1] == 7 for e in plan.edits if e.tensor == "fc1.weight" and e.value != 0)  # f* = 7
    r = evaluate_plan_offline(seeded_policy, plan, b.Z_window, b.Z_nom)
    assert r["fail_mean_shift"]["brake"] >= 0.5 and abs(r["fail_mean_shift"]["alpha_v"]) < 1e-3  # brake-routed
    assert r["nom_max_abs_shift"]["brake"] < 1e-5
    assert max(abs(e.value) for e in plan.edits) <= 3.0
    fc1 = {tuple(e.index): e.value for e in plan.edits if e.tensor == "fc1.weight" and e.value != 0}
    assert next(e.value for e in plan.edits if e.tensor == "fc3.weight" and e.index[0] == 2) == 3.0  # saturating
    assert {c for _, c in fc1} == {0, 7}  # gated on speed and roughness


def test_wet_curve_speed_threshold_lowers_on_failed_gate(seeded_policy):
    from hybrid_av.weightspace.editors.base import Attempt

    b = B(seeded_policy, "wet_curve", raw(40, s7=0.8), raw(60, s7=0.1))
    bias = lambda p: next(e.value for e in p.edits if e.tensor == "fc1.bias")
    p0 = ScriptedEditor().propose(b, [])
    miss = Attempt(1, p0, [], {"ood_success_rate": 0.15})
    p1 = ScriptedEditor().propose(b, [miss])
    p2 = ScriptedEditor().propose(b, [miss, Attempt(2, p1, [], {"ood_success_rate": 0.5})])
    assert bias(p1) > bias(p0) and bias(p2) > bias(p1)  # lower speed threshold -> larger bias


def test_merge_band_on_intersection_gap(seeded_policy):
    Zn = np.vstack([raw(30, s20=10.0), raw(30, s20=0.0)])  # nominal: no junction (hi) or zero gap
    Zw = np.vstack([raw(10, s20=v) for v in (1.0, 1.5, 2.0, 2.5)])  # failure window: gaps 1–2.5 s
    b = B(seeded_policy, "merge_deadlock", Zw, Zn)
    plan = ScriptedEditor(feature=20).propose(b, [])
    assert len(plan.edits) == 143
    r = evaluate_plan_offline(seeded_policy, plan, b.Z_window, b.Z_nom)
    assert r["nom_max_abs_shift"]["brake"] < 1e-5


def test_scripted_is_deterministic(seeded_policy):
    b = B(seeded_policy, "wet_curve", raw(40, s7=0.8), raw(60, s7=0.1))
    assert ScriptedEditor().propose(b, []) == ScriptedEditor().propose(b, [])


def test_make_editor_scripted():
    from hybrid_av.weightspace.editors import make_editor
    from hybrid_av.weightspace.schema import WeightSpaceConfig

    assert make_editor("scripted", WeightSpaceConfig()).name == "scripted"
