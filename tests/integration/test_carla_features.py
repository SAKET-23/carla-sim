"""IC-36: features computed from CarlaSim against truth recomputed from raw CARLA API calls (section H)."""

import math

import numpy as np
import pytest

from hybrid_av.features.spec import FEATURES
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend import convert
from hybrid_av.sim.carla_backend.routes import carla_scenario
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import TrafficLightState as Light
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from tests.integration.helpers import ROUTES, route_gap

pytestmark = pytest.mark.carla
_LIGHT = {"Red": Light.RED, "Yellow": Light.YELLOW, "Green": Light.GREEN}


@pytest.mark.slow
def test_features_match_independent_truth(make_sim):
    """IC-36 — see Plan/06-integration/integration-check.md; likely fix: ECC-25."""
    try:
        sc = carla_scenario(make_scenario("nominal", 0), "nominal", 0, None, ROUTES)
    except LookupError:
        pytest.skip("no CARLA route for nominal: fill carla_routes.yaml (G2)")
    sim = make_sim(sensors=None, town=sc.carla_route["town"])
    obs = sim.reset(sc)
    p = VehicleParams(**sim.steer_params)
    stack = AutonomyStack(p, LongitudinalMap(), MPCConfig(), ReferenceConfig())
    stack.reset(sim.route())
    errors = {k: [] for k in (0, 1, 2, 3, 4, 9, 10, 19, 21, 22, 23)}
    clipped = np.zeros(len(FEATURES))
    seen_min, seen_max = np.full(len(FEATURES), np.inf), np.full(len(FEATURES), -np.inf)
    steps = 0
    for _ in range(1200):  # 60 s
        feats = stack.features(obs)
        raw, ego = feats.raw, sim.ego
        tf, v, w = ego.get_transform(), ego.get_velocity(), ego.get_angular_velocity()
        psi = convert.yaw_to_lib(tf.rotation.yaw)
        vxw, vyw = v.x, -v.y
        errors[0].append(abs(raw[0] - (vxw * math.cos(psi) + vyw * math.sin(psi))) / 0.05)
        errors[1].append(abs(raw[1] - (-vxw * math.sin(psi) + vyw * math.cos(psi))) / 0.05)
        errors[2].append(abs(raw[2] - (-math.radians(w.z))) / 0.05)
        wp = sim._wmap.get_waypoint(tf.location)
        pw = convert.yaw_to_lib(wp.transform.rotation.yaw)
        ex, ey, _ = convert.loc_to_lib(tf.location.x, tf.location.y, 0.0)
        wx, wy, _ = convert.loc_to_lib(wp.transform.location.x, wp.transform.location.y, 0.0)
        # inside a junction get_waypoint can return an overlapping crossing lane, so the "truth" is undefined there
        # (pod G2: slot 4 held on 98.4 % of steps, the misses at junction crossings); compared outside junctions only
        if not wp.is_junction:
            errors[3].append(abs(raw[3] - (-math.sin(pw) * (ex - wx) + math.cos(pw) * (ey - wy))) / 0.1)
            errors[4].append(abs(raw[4] - (math.atan2(math.sin(psi - pw), math.cos(psi - pw)))) / 0.02)
        errors[22].append(abs(raw[22] - wp.lane_width) / 0.05)
        limit = ego.get_speed_limit()
        if limit:
            errors[21].append(abs(raw[21] - limit / 3.6) / 0.28)  # same value, up to km/h rounding
        light = ego.get_traffic_light()
        if light is not None:
            errors[19].append(float(raw[19] != float(_LIGHT.get(light.get_state().name, Light.NONE))))
        if raw[9] < 79.0 and abs(raw[5]) < 0.005:  # a lead on a straight stretch
            best = None
            for a in sim.world.get_actors():
                if a.id == ego.id or not a.type_id.startswith("vehicle."):
                    continue
                lx, ly, _ = convert.loc_to_lib(
                    a.get_transform().location.x, a.get_transform().location.y, 0.0
                )
                fwd = (lx - ex) * math.cos(psi) + (ly - ey) * math.sin(psi)
                lat = -(lx - ex) * math.sin(psi) + (ly - ey) * math.cos(psi)
                gap = fwd - a.bounding_box.extent.x - (p.a + 0.5)
                if fwd > 0 and abs(lat) < 1.5 and (best is None or gap < best[0]):
                    best = (gap, a)
            if best:
                lv = best[1].get_velocity()
                rel = (lv.x * math.cos(psi) + (-lv.y) * math.sin(psi)) - raw[0]
                errors[9].append(abs(raw[9] - best[0]) / 0.3)
                errors[10].append(abs(raw[10] - rel) / 0.2)
        if raw[19] in (float(Light.RED), float(Light.YELLOW)) and raw[23] < 60.0 and light is not None:
            # distance along the route to the stop line ahead of this light (the feature's definition)
            gaps = [route_gap(sim, tf.location, s.transform.location) for s in light.get_stop_waypoints()]
            gaps = [g for g in gaps if g >= 0.0]
            if gaps:
                errors[23].append(abs(raw[23] - min(gaps)) / 1.0)
        lo = np.array([f.lo for f in FEATURES])
        hi = np.array([f.hi for f in FEATURES])
        clipped += (raw < lo) | (raw > hi)
        seen_min, seen_max = np.minimum(seen_min, raw), np.maximum(seen_max, raw)
        steps += 1
        out = stack.act(obs, ConstantAction()(feats.z, feats), feats)
        obs, info = sim.step(out.command)
        if info.route_completed or info.collision != "none":
            break
    # errors are scaled by their tolerance: <= 1 passes; slot 19 is exact; 99 % of the samples must hold
    for slot, vals in errors.items():
        if vals:
            assert np.mean(np.array(vals) <= 1.0) >= 0.99, (
                f"slot {slot} ({FEATURES[slot].name}): {np.mean(np.array(vals) <= 1.0):.3f}"
            )
    saturated = {FEATURES[i].name: round(c / steps, 3) for i, c in enumerate(clipped) if c / steps > 0.01}
    print(f"slots that clip on > 1 % of steps: {saturated}")
    for i, f in enumerate(FEATURES):
        if f.name in saturated:
            print(f"  {f.name}: observed [{seen_min[i]:.3f}, {seen_max[i]:.3f}] vs range [{f.lo}, {f.hi}]")
    assert not saturated, "slots outside [lo, hi] on more than 1 % of steps"
    # shortcut: slot 16 (crossing conflict distance) and slot 23 outside light stops are not recomputed independently
