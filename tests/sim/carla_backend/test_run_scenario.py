import importlib.util
import sys
from pathlib import Path

import yaml

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts/run_scenario.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_scenario", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_carla_backend_runs_ticks_and_measures(fake_carla, tmp_path, monkeypatch):
    mod = _load()
    routes = tmp_path / "routes.yaml"
    routes.write_text(
        yaml.safe_dump({"nominal": {"town": "Town03_Opt", "start": [0, 0, 0, 0], "goal": [200, 0, 0]}})
    )
    monkeypatch.setattr(mod, "ROUTES", routes)
    out = tmp_path / "out"
    argv = [
        "run_scenario.py",
        "--backend",
        "carla",
        "--town",
        "Town03_Opt",
        "--quality",
        "Low",
        "--ticks",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", [*argv, "--measure", "--mode", "A", "--out", str(out)])
    mod.main()
    rows = (out / "vram.csv").read_text().splitlines()
    assert rows[0].startswith("backend,town") and rows[1].startswith("carla,Town03_Opt,Low,A,5,")
    assert not fake_carla.last_world.get_settings().synchronous_mode  # the server is restored
    assert fake_carla.last_world.frame >= 15  # 10 settle ticks + 5 steps


def test_carla_backend_without_route_entry_falls_back_to_spawn_points(
    fake_carla, tmp_path, monkeypatch, caplog
):
    mod = _load()
    routes = tmp_path / "routes.yaml"
    routes.write_text("{}\n")
    monkeypatch.setattr(mod, "ROUTES", routes)
    monkeypatch.setattr(sys, "argv", ["run_scenario.py", "--backend", "carla", "--out", str(tmp_path)])
    mod.main()
    assert "no route in carla_routes.yaml" in caplog.text
