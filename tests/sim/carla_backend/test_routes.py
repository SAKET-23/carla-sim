import pytest
import yaml

from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend.routes import carla_scenario, plain_scenario, straight_route


def test_carla_scenario_lookup_and_town_override(tmp_path):
    f = tmp_path / "routes.yaml"
    f.write_text(
        yaml.safe_dump({"nominal": {"town": "Town03_Opt", "start": [0, 0, 0, 0], "goal": [9, 0, 0]}})
    )
    sc = carla_scenario(make_scenario("nominal", 1), "nominal", 1, "Town05", f)
    assert sc.route is None and sc.carla_route["town"] == "Town05"
    with pytest.raises(LookupError, match="no CARLA route"):
        carla_scenario(make_scenario("wet_curve", 1), "wet_curve", 1, None, f)


def test_plain_and_straight_scenarios():
    rc = straight_route({"straight": [10.0, 20.0, 0.5, 90.0]}, "Town03_Opt", length=100.0)
    assert rc["goal"] == pytest.approx([10.0, 120.0, 0.5])  # yaw 90 deg: CARLA +y
    sc = plain_scenario(rc, 8.0)
    assert sc.actors == () and sc.controls == () and sc.ego_start_speed == 8.0 and sc.carla_route is rc


def test_route_sets_pick_seed_mod_size_and_overrides(tmp_path):
    def r(i):
        return {"town": "Town01", "start": [i, 0, 0, 0], "goal": [i + 400, 0, 0]}

    f = tmp_path / "routes.yaml"
    f.write_text(
        yaml.safe_dump(
            {"nominal": [r(i) for i in range(10)], "nominal_town10hd": [r(100 + i) for i in range(3)]}
        )
    )
    pick = lambda **kw: carla_scenario(
        make_scenario("nominal", 1), "nominal", kw.pop("seed", 13), None, f, **kw
    )
    assert pick().carla_route["start"][0] == 3
    assert pick(route_index=7).carla_route["start"][0] == 7
    assert pick(route_set="town10hd").carla_route["start"][0] == 101
    assert pick(route_set="town10hd", route_index=2).carla_route["start"][0] == 102
