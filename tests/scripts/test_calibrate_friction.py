import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from hybrid_av.planning.route import RouteBuilder
from hybrid_av.vehicle.identify import G

_SPEC = importlib.util.spec_from_file_location(
    "calibrate_friction", Path(__file__).resolve().parents[2] / "scripts/calibrate_friction.py"
)
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)

DT, R = 0.05, 32.6


class FakeSim:
    """Kinematic ego on a straight + 80 deg left arc; it 'slides' (cross-track 2 m in the curve) above C*sqrt(k)."""

    dt = DT
    C = 16.5

    def __init__(self, off_road_above=None):
        self._route = RouteBuilder.concat(RouteBuilder.straight(40.0, 10.0), RouteBuilder.arc(R, 1.4, 10.0))
        self.off_road_above = off_road_above
        self.resets = 0

    def reset(self, sc):
        self.resets += 1
        self.v, self.k, self.s = sc.ego_start_speed, sc.friction_zones[0].mu, 0.0
        return self._obs()

    def route(self):
        return self._route

    def _obs(self):
        kap = float(np.interp(self.s, self._route.s, self._route.kappa))
        slid = abs(kap) > 1 / 40 and self.v > self.C * self.k**0.5
        return SimpleNamespace(ego=SimpleNamespace(vx=self.v, r=self.v * kap), s=self.s, lateral=2.0 * slid)

    def step(self, cmd):
        self.s += self.v * DT
        info = SimpleNamespace(
            collision="none",
            off_road=self.off_road_above is not None and self.v > self.off_road_above,
            route_s=self.s,
            route_completed=self.s >= self._route.length - 1.0,
        )
        return self._obs(), info


class FakeStack:
    def __init__(self, *a, **k):
        pass

    def reset(self, route):
        self.limit = float(route.speed_limit[0])  # the pass speed must be set on the route

    def features(self, obs):
        return SimpleNamespace(route_s=obs.s, lateral=obs.lateral)

    def act(self, obs, action, feats):
        return SimpleNamespace(command=None)


@pytest.fixture(autouse=True)
def _fake_stack(monkeypatch):
    monkeypatch.setattr(cal, "AutonomyStack", FakeStack)


def _patch(monkeypatch, sim):
    sim.close = lambda: None
    monkeypatch.setattr(cal, "CarlaSim", lambda cfg: sim)


def test_parse_curve_default():
    rc = cal.parse_curve(cal.CURVE, "Town03_Opt")
    assert rc == {
        "town": "Town03_Opt",
        "start": [-11.97, -204.81, 0.0, -178.6],
        "goal": [-84.97, -151.12, 0.0],
    }


def test_pass_holds_below_threshold_and_slides_above():
    sim, args = FakeSim(), (None, None)
    rc = cal.parse_curve(cal.CURVE, "T")
    peak, slid = cal.curve_pass(sim, rc, 1.0, 10.0, *args)
    assert not slid and abs(peak - 10.0**2 / R) < 0.2
    assert cal.curve_pass(sim, rc, 1.0, 17.0, *args)[1]


def test_off_road_counts_as_slide():
    sim = FakeSim(off_road_above=8.0)
    rc = cal.parse_curve(cal.CURVE, "T")
    assert cal.curve_pass(sim, rc, 1.0, 9.0, None, None)[1]


def test_run_returns_largest_held_peak_and_first_slide_speed():
    sim = FakeSim()
    a, first = cal.curve_run(sim, cal.parse_curve(cal.CURVE, "T"), 0.35, None, None)
    assert first == 10.0  # 16.5*sqrt(0.35) = 9.76: 9.5 held, 10 slides
    assert abs(a - 9.5**2 / R) < 0.2
    assert (
        sim.resets == 13
    )  # 4.0..10.0 m/s in 0.5 steps, one fresh reset each, sweep stops at the first slide


def test_main_writes_k_star_and_report(monkeypatch, tmp_path):
    _patch(monkeypatch, FakeSim())
    monkeypatch.setattr(cal, "SCALES", (1.0, 0.6, 0.35))
    out = tmp_path / "wet.yaml"
    cal.main(["--out", str(out), "--runs", str(tmp_path / "runs")])
    k = yaml.safe_load(out.read_text())[cal.KEY]
    assert 0.35 < k < 0.6
    rep = yaml.safe_load((tmp_path / "runs/friction.json").read_text())
    assert rep["k_star"] == pytest.approx(k, abs=1e-4) and len(rep["repeats"]) == 3
    assert rep["hold"]["held"] and rep["hold"]["slid"]
    assert abs(rep["hold"]["v_hold"] - (0.35 * G * R) ** 0.5) < 1e-6
    assert rep["hold"]["v_slide"] == 1.2 * rep["hold"]["v_hold"]


def test_main_fails_when_hold_speed_slides(monkeypatch, tmp_path):
    _patch(monkeypatch, FakeSim())
    monkeypatch.setattr(cal, "SCALES", (1.0, 0.6, 0.35))
    monkeypatch.setattr(cal, "CURVE_R", 45.0)  # sqrt(0.35 g R) = 12.4 m/s, above what the fake holds at k*
    with pytest.raises(SystemExit, match="held"):
        cal.main(["--out", str(tmp_path / "w.yaml"), "--runs", str(tmp_path / "r")])
