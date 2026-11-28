"""Search a CARLA town for the places the calibration and integration tests need.

find_map_features.py --town Town03_Opt [--fixture tests/integration/fixtures/town03_points.yaml]
Prints curves with R in [28, 35] m and unsignalised T-junctions with CARLA start/goal coordinates for
Scenario.carla_route. With --fixture it also writes the integration-test fixture: `straight` ([x, y, z, yaw], start
of a >= 200 m flat straight), `open_area` ([x, y, z], centre of a free disc of radius >= 35 m) and
`signalised_junction` ([x, y, z, yaw], >= 80 m before a stop line). Coordinates are CARLA's. Needs a CARLA server.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

from hybrid_av.planning.route import RouteBuilder, constant_radius_segments
from hybrid_av.sim.carla_backend import convert

STEP = 2.0  # m between lane waypoints
_OBSTACLES = (
    "Buildings",
    "Fences",
    "Poles",
    "Vegetation",
    "Walls",
    "Static",
    "TrafficSigns",
    "TrafficLight",
    "Bridge",
    "GuardRail",
)


def xyzyaw(wp) -> list[float]:
    t = wp.transform
    return [round(t.location.x, 2), round(t.location.y, 2), round(t.location.z, 2), round(t.rotation.yaw, 1)]


def lane_chains(wmap) -> list[list]:
    """Waypoints along every lane of the topology, entry to exit, every STEP metres."""
    chains = []
    for entry, exit_ in wmap.get_topology():
        chain, wp = [entry], entry
        for _ in range(2000):
            nxt = wp.next(STEP)
            if not nxt or wp.transform.location.distance(exit_.transform.location) < STEP:
                break
            wp = nxt[0]
            chain.append(wp)
        chain.append(exit_)
        chains.append(chain)
    return chains


def lib_xy(chain) -> np.ndarray:
    return np.array(
        [convert.loc_to_lib(w.transform.location.x, w.transform.location.y, 0.0)[:2] for w in chain]
    )


def curves(chains) -> list[dict]:
    out, seen = [], set()
    for chain in chains:
        if len(chain) < 12:
            continue
        route = RouteBuilder.from_waypoints(lib_xy(chain), 10.0, 3.5)
        for s0, s1, r in constant_radius_segments(route, 28.0, 35.0, min_length=20.0):
            key = (round(r, 1), tuple(xyzyaw(chain[0])))
            if key in seen:  # the opposite lane direction repeats the curve
                continue
            seen.add(key)
            out.append(
                {
                    "R": round(r, 1),
                    "arc_m": round(s1 - s0, 1),
                    "start": xyzyaw(chain[0]),
                    "goal": xyzyaw(chain[-1])[:3],
                }
            )
    return out


def t_junctions(world, wmap) -> list[dict]:
    import carla

    out, seen = [], set()
    for entry, _ in wmap.get_topology():
        junction = entry.get_junction() if entry.is_junction else None
        if junction is None or junction.id in seen:
            continue
        seen.add(junction.id)
        pairs = junction.get_waypoints(carla.LaneType.Driving)
        roads = {a.road_id for a, _ in pairs}
        if len(roads) == 3 and not world.get_traffic_lights_in_junction(junction.id):
            a, b = pairs[0]
            back = a.previous(40.0)
            out.append(
                {
                    "junction": junction.id,
                    "start": xyzyaw(back[0] if back else a),
                    "goal": xyzyaw(b.next(30.0)[0])[:3],
                }
            )
    return out


def straight(chains) -> list[float] | None:
    """Start of the longest flat straight, followed with wp.next across segment boundaries (>= 200 m, else None)."""
    best, best_len = None, 199.99
    for chain in chains:
        start = chain[0]
        t0, wp = start.transform, start
        while wp.transform.location.distance(t0.location) < 400.0:
            nxt = wp.next(STEP)
            if not nxt:
                break
            wp = nxt[0]
            t = wp.transform
            if abs((t.rotation.yaw - t0.rotation.yaw + 180) % 360 - 180) > 1.0:
                break
            if abs(t.location.z - t0.location.z) > 0.5:
                break
            run = t.location.distance(t0.location)
            if run > best_len:
                best, best_len = start, run
    return xyzyaw(best) if best else None


def open_area(world, wmap, clearance: float = 35.0) -> list[float] | None:
    """Grid point with the largest clearance (>= `clearance`) from every obstacle-labelled environment object."""
    import carla

    boxes = []
    for name in _OBSTACLES:
        label = getattr(carla.CityObjectLabel, name, None)
        # ECC-42: get_environment_objects and CityObjectLabel names are from the 0.9.16 docs, unchecked
        boxes += [o.bounding_box for o in world.get_environment_objects(label)] if label is not None else []
    centres = np.array([[b.location.x, b.location.y] for b in boxes])
    reach = np.array([max(b.extent.x, b.extent.y) for b in boxes])
    wps = wmap.generate_waypoints(20.0)
    xs = np.array([w.transform.location.x for w in wps])
    ys = np.array([w.transform.location.y for w in wps])
    best = None
    for x in np.arange(xs.min(), xs.max(), 10.0):
        for y in np.arange(ys.min(), ys.max(), 10.0):
            d = float(np.min(np.hypot(centres[:, 0] - x, centres[:, 1] - y) - reach))
            if d >= clearance and (best is None or d > best[0]):
                best = (d, [round(float(x), 1), round(float(y), 1)])
    if best is None:
        return None
    z = wmap.get_waypoint(carla.Location(*best[1], 0.0), project_to_road=True).transform.location.z
    return [
        *best[1],
        round(z, 2),
    ]  # shortcut: z is the nearest road's, flat ground is assumed (ECC-12 style check in G2)


def signalised_junction(world) -> list[float] | None:
    for light in world.get_actors().filter("traffic.traffic_light*"):
        for stop in light.get_stop_waypoints():
            back = stop.previous(90.0)
            if back and not back[0].is_junction:
                return xyzyaw(back[0])
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--town", required=True)
    ap.add_argument("--fixture", type=Path)
    a = ap.parse_args(argv)
    import carla  # noqa: F401  (fail early with a clear message when the wheel is missing)

    from hybrid_av.sim.carla_backend.client import CarlaSimConfig, connect

    _, world = connect(CarlaSimConfig(town=a.town))
    wmap = world.get_map()
    chains = lane_chains(wmap)
    print(f"curves with R in [28, 35] m ({a.town}):")
    for c in curves(chains):
        print(f"  {c}")
    print("unsignalised T-junctions:")
    for j in t_junctions(world, wmap):
        print(f"  {j}")
    if a.fixture:
        points = {
            "straight": straight(chains),
            "open_area": open_area(world, wmap),
            "signalised_junction": signalised_junction(world),
        }
        missing = [k for k, v in points.items() if v is None]
        if missing:
            raise SystemExit(f"no candidate found for {missing}")
        a.fixture.parent.mkdir(parents=True, exist_ok=True)
        a.fixture.write_text(yaml.safe_dump(points))
        print(f"wrote {a.fixture}")


if __name__ == "__main__":
    main()
