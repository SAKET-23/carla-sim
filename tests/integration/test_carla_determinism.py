"""IC-41: repeatability across two runs from a fresh reset() (integration-check.md section I)."""

import dataclasses

import numpy as np
import pytest

from hybrid_av.eval.runner import run_episode
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.expert import ConstantAction
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.carla_backend.routes import carla_scenario
from hybrid_av.stack import AutonomyStack
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from tests.integration.helpers import ROUTES

pytestmark = pytest.mark.carla


@pytest.mark.slow
def test_repeatability(make_sim):
    """IC-41 — see Plan/06-integration/integration-check.md; likely fix: ECC-27."""
    try:
        sc = carla_scenario(make_scenario("nominal", 0), "nominal", 0, None, ROUTES)
    except LookupError:
        pytest.skip("no CARLA route for nominal: fill carla_routes.yaml (G2)")
    sc = dataclasses.replace(sc, time_limit=60.0)  # the first 60 s
    runs = []
    for _ in range(2):
        sim = make_sim(sensors=None, town=sc.carla_route["town"], seed=0)
        stack = AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig())
        runs.append(run_episode(sim, sc, stack, ConstantAction()))
        sim.close()
    a, b = runs
    n = min(len(a.trace["x"]), len(b.trace["x"]))
    drift = float(np.max(np.hypot(a.trace["x"][:n] - b.trace["x"][:n], a.trace["y"][:n] - b.trace["y"][:n])))
    print(f"max ego position divergence over {n} steps: {drift:.3f} m")  # report the actual number
    assert a.infractions == b.infractions
    assert abs(a.route_completion - b.route_completion) <= 0.01
    assert drift < 0.5
