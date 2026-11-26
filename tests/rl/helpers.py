# tests/rl/helpers.py
import numpy as np

from hybrid_av.features.extractor import FeatureFrame
from hybrid_av.features.spec import FEATURE_NAMES, normalize
from hybrid_av.mpc.controller import MPCConfig, MPCResult
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.env import DrivingEnv
from hybrid_av.rl.reward import RewardConfig
from hybrid_av.scenarios import make_scenario
from hybrid_av.sim.base import StepInfo
from hybrid_av.sim.litesim.sim import LiteSim
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import Control, PolicyAction
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams

BASE = {
    "vx": 10.0,
    "friction_est": 1.0,
    "lead_dist": 80.0,
    "lead_ttc": 10.0,
    "ped_dist": 50.0,
    "cross_veh_dist": 80.0,
    "cross_veh_ttc": 10.0,
    "intersection_gap": 10.0,
    "speed_limit": 13.89,
    "lane_width": 3.5,
    "stop_dist": 100.0,
}


def _frame(route_s=0.0, lateral=0.0, dist_to_junction=np.inf, **raw):
    vals = {**BASE, **raw}
    r = np.array([vals.get(n, 0.0) for n in FEATURE_NAMES])
    return FeatureFrame(
        raw=r,
        z=normalize(r).astype(np.float32),
        route_s=route_s,
        lateral=lateral,
        heading_error=0.0,
        mu_est=vals["friction_est"],
        lead=None,
        stop_s=None,
        dist_to_junction=dist_to_junction,
        cross_dist_conflict=np.inf,
    )


A = PolicyAction(1.0, 0.0, 0.0)


def _info(**kw):
    d = {
        "collision": "none", "red_light_violation": False, "stop_sign_violation": False, "off_road": False,
        "route_s": 0.0, "route_completed": False, "route_deviation": False, "blocked": False,
        "timeout": False, "sim_time": 0.0,
    }  # fmt: skip
    d.update(kw)
    return StepInfo(**d)


def _mpc(**kw):
    d = {
        "control": Control(0.0, 0.0), "status": "solved", "solve_time": 0.001, "predicted": np.zeros((21, 6)),
        "slack_friction_max": 0.0, "slack_obstacle_max": 0.0, "cost": 0.0,
    }  # fmt: skip
    d.update(kw)
    return MPCResult(**d)


def _stack():
    return AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig())


def _env(family="nominal", overrides=None, sim_cls=LiteSim, edit=lambda sc: sc):
    return DrivingEnv(
        lambda: sim_cls(VehicleParams(), LongitudinalMap()),
        lambda rng: edit(make_scenario(family, int(rng.integers(0, 10000)), overrides)),
        _stack(),
        RewardConfig(),
    )
