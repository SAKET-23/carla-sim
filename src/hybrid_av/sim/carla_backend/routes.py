"""CARLA start/goal lookup shared by scripts/run_scenario.py and the bridge node (no `carla` import)."""

import dataclasses
from pathlib import Path

import yaml


def carla_scenario(sc, family: str, seed: int, town: str | None, routes: Path, fallback: bool = False):
    """The LiteSim scenario with its route replaced by the CARLA start/goal from carla_routes.yaml (G2).

    fallback=True (scripts, bridge): a missing entry yields a marker route that CarlaSim resolves against the loaded
    town (scenario_setup.spawn_point_route), so G1 smoke runs before G2 fills the file."""
    table = yaml.safe_load(Path(routes).read_text()) or {}
    entry = table.get(f"{family}_{seed}") or table.get(family)
    if entry is None:
        if not fallback:
            raise LookupError(f"no CARLA route for {family} seed {seed}: add one to {routes}")
        # ECC-43: fallback route is resolved on the live map (spawn point 0 to a seed-chosen spawn point, 300-600 m)
        return dataclasses.replace(
            sc, route=None, carla_route={"town": town or "Town03_Opt", "spawn_fallback": True}
        )
    route = {k: entry[k] for k in ("town", "start", "goal")} | (
        {"cache": entry["cache"]} if "cache" in entry else {}
    )
    route["town"] = town or route["town"]
    return dataclasses.replace(sc, route=None, carla_route=route)


def plain_scenario(carla_route: dict, ego_start_speed: float = 0.0, friction_zones=()):
    """An empty-road scenario (no traffic, no controls) for calibration drives."""
    from hybrid_av.sim.base import Scenario

    return Scenario(
        "calibration", 0, None, carla_route, tuple(friction_zones), 1.0, (), (), (), 1e9, ego_start_speed
    )


def straight_route(points: dict, town: str, length: float = 190.0) -> dict:
    """carla_route along the fixture's `straight` start for `length` m (the fixture guarantees >= 200 m)."""
    import math

    x, y, z, yaw = points["straight"]
    return {
        "town": town,
        "start": [x, y, z, yaw],
        "goal": [x + length * math.cos(math.radians(yaw)), y + length * math.sin(math.radians(yaw)), z],
    }
