import importlib.util
import math
from pathlib import Path

import yaml

_SPEC = importlib.util.spec_from_file_location(
    "pick_routes", Path(__file__).resolve().parents[2] / "scripts/pick_routes.py"
)
pick = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pick)


def _run(tmp_path, towns, n, extra=()):
    out = tmp_path / "r.yaml"
    rc = pick.main(
        ["--towns", towns, "--n", str(n), "--set", "train", "--seed", "0", "--out", str(out), *extra]
    )
    return rc, yaml.safe_load(out.read_text()) if out.exists() else None


def test_round_robin_spreads_over_towns_and_obeys_rules(fake_carla, tmp_path, capsys):
    rc, doc = _run(tmp_path, "Town01,Town02,Town03_Opt,Town05", 10)
    assert rc == 0 and doc["set"] == "train" and len(doc["routes"]) == 10
    counts = {
        t: sum(r["town"] == t for r in doc["routes"]) for t in ("Town01", "Town02", "Town03_Opt", "Town05")
    }
    assert min(counts.values()) >= 2
    for r in doc["routes"]:
        assert len(r["start"]) == 4 and len(r["goal"]) == 3
        assert 300 <= r["length_m"] <= 800
        assert math.dist(r["start"][:2], r["goal"][:2]) >= 50
    assert capsys.readouterr().out.count("Town0") >= 10  # one summary line per route


def test_same_seed_same_routes(fake_carla, tmp_path):
    a = _run(tmp_path, "Town01", 3)[1]
    assert a == _run(tmp_path, "Town01", 3)[1]


def test_lane_change_rejects_every_pair(fake_carla, tmp_path, capsys):
    from tests.fakes.carla_fake.agents.navigation.local_planner import RoadOption

    wp = lambda x, o: (fake_carla.Map("m").get_waypoint(fake_carla.Location(x, 0, 0)), o)
    fake_carla.planner_trace = [wp(0, RoadOption.LANEFOLLOW), wp(400, RoadOption.CHANGELANELEFT)]
    rc, doc = _run(tmp_path, "Town01,Town02", 2)
    assert rc != 0 and doc["routes"] == []
    err = capsys.readouterr().err
    assert "Town01" in err and "Town02" in err


def test_length_out_of_range_rejected(fake_carla, tmp_path):
    from tests.fakes.carla_fake.agents.navigation.local_planner import RoadOption

    mk = lambda x: (fake_carla.Map("m").get_waypoint(fake_carla.Location(x, 0, 0)), RoadOption.LANEFOLLOW)
    fake_carla.planner_trace = [mk(0), mk(900)]
    assert _run(tmp_path, "Town01", 1)[0] != 0


def test_unreachable_n_names_short_towns(fake_carla, tmp_path, capsys):
    rc, doc = _run(tmp_path, "Town01", 100000)
    assert rc != 0 and "Town01" in capsys.readouterr().err
    assert 0 < len(doc["routes"]) <= 400


def test_start_far_from_first_waypoint_rejected_like_the_simulator(fake_carla, tmp_path, capsys):
    from tests.fakes.carla_fake.agents.navigation.local_planner import RoadOption

    mk = lambda x: (fake_carla.Map("m").get_waypoint(fake_carla.Location(x, 500, 0)), RoadOption.LANEFOLLOW)
    fake_carla.planner_trace = [
        mk(0),
        mk(400),
    ]  # 400 m long, no lane change, but 480+ m from every spawn point
    rc, doc = _run(tmp_path, "Town01", 1)
    assert rc != 0 and doc["routes"] == []


def test_planner_built_at_the_simulators_resolution(fake_carla, tmp_path, monkeypatch):
    from hybrid_av.sim.carla_backend.scenario_setup import ROUTE_RESOLUTION
    from tests.fakes.carla_fake.agents.navigation import global_route_planner as grp

    seen = []
    init = grp.GlobalRoutePlanner.__init__
    monkeypatch.setattr(
        grp.GlobalRoutePlanner, "__init__", lambda self, w, res: (seen.append(res), init(self, w, res))[1]
    )
    _run(tmp_path, "Town01", 1)
    assert seen and set(seen) == {ROUTE_RESOLUTION}


def _light_at(monkeypatch, stop_s):
    from hybrid_av.sim.carla_backend.scenario_setup import LightOnRoute

    monkeypatch.setattr(pick, "route_lights", lambda world, wps: [LightOnRoute(object(), stop_s)])


def test_stop_line_near_start_rejected(fake_carla, tmp_path, monkeypatch):
    _light_at(monkeypatch, pick.MIN_FIRST_STOP_M - 1.0)
    rc, doc = _run(tmp_path, "Town01", 1)
    assert rc != 0 and doc["routes"] == []


def test_stop_line_beyond_min_first_stop_accepted(fake_carla, tmp_path, monkeypatch):
    _light_at(monkeypatch, pick.MIN_FIRST_STOP_M + 1.0)
    assert _run(tmp_path, "Town01", 1)[0] == 0
