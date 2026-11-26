import dataclasses

import numpy as np

from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.base import ScriptedActor
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import ObjectClass
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def _stack():
    return AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig())


def test_run_episode_straight():
    sc = make_scenario("nominal", 50000, {"traffic": False})
    res = run_episode(LiteSim(VehicleParams(), LongitudinalMap()), sc, _stack(), ConstantAction())
    assert res.route_completion == 1.0 and res.driving_score == 100.0
    assert res.trace["z"].shape[1] == 24 and len(res.trace["t"]) == len(res.trace["delta"])
    ps = res.trace["policy_step"]
    assert ps[0] and not ps[1] and ps[2]
    assert np.isnan(res.trace["delta_meas"]).all() and "h1" not in res.trace


def test_episode_ends_on_collision(monkeypatch):
    sc = make_scenario("nominal", 50000, {"traffic": False})
    parked = ScriptedActor(
        99, ObjectClass.CAR, "parked", {"s": 60.0, "lateral": 0.0, "length": 4.5, "width": 1.9}
    )
    sc = dataclasses.replace(sc, actors=(parked,))
    st = _stack()
    monkeypatch.setattr(st, "_obstacles", lambda *a, **k: [])  # MPC does not see the car
    res = run_episode(LiteSim(VehicleParams(), LongitudinalMap()), sc, st, ConstantAction())
    assert res.infractions["collision_static"] == 1
    assert res.duration < sc.time_limit and res.route_completion < 1.0


class _ForcedInfo:
    """Wraps a Sim; step k >= 3 reports off_road, step 10 reports timeout."""

    def __init__(self, sim):
        self.sim, self.dt, self.n = sim, sim.dt, 0

    def __getattr__(self, name):
        return getattr(self.sim, name)

    def step(self, cmd):
        obs, info = self.sim.step(cmd)
        self.n += 1
        return obs, dataclasses.replace(info, off_road=self.n >= 3, timeout=self.n >= 10)


def test_off_road_step_has_infraction_code():
    from hybrid_av.eval.runner import INFRACTION_KEYS

    sc = make_scenario("nominal", 50000, {"traffic": False})
    sim = _ForcedInfo(LiteSim(VehicleParams(), LongitudinalMap()))
    res = run_episode(sim, sc, _stack(), ConstantAction())
    code = INFRACTION_KEYS.index("off_road_m") + 1
    assert list(res.trace["infraction"][:2]) == [0, 0] and (res.trace["infraction"][2:] == code).all()
    assert res.infractions["off_road_m"] > 0


def test_min_ttc_covers_cross_vehicle():
    from hybrid_av.features.spec import I_CROSS_VEH_TTC, I_VX

    sc = make_scenario("nominal", 50000, {"traffic": False})
    st = _stack()
    real = st.features

    def features(obs):
        f = real(obs)
        f.raw = f.raw.copy()
        f.raw[I_CROSS_VEH_TTC] = 2.5  # lead_ttc stays absent (10)
        f.cross_dist_conflict = 2.5 * max(f.raw[I_VX], 0.1)  # ego arrives with the vehicle: a real conflict
        return f

    st.features = features
    res = run_episode(LiteSim(VehicleParams(), LongitudinalMap()), sc, st, ConstantAction())
    assert res.min_ttc == 2.5


def test_ttc_min_ignores_passing_cross_vehicle():
    from hybrid_av.features.spec import I_CROSS_VEH_TTC

    sc = make_scenario("nominal", 50000, {"traffic": False})
    st = _stack()
    real = st.features

    def features(obs):
        f = real(obs)
        f.raw = f.raw.copy()
        f.raw[I_CROSS_VEH_TTC] = 0.01
        f.cross_dist_conflict = 1000.0  # the ego is far from the conflict point
        return f

    st.features = features
    res = run_episode(LiteSim(VehicleParams(), LongitudinalMap()), sc, st, ConstantAction())
    assert res.min_ttc == 10.0  # lead_ttc, absent
