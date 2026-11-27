import numpy as np
import pytest

from hybrid_av.rl.expert import (
    EXPERT_JUNCTION_WAIT_DIST,
    EXPERT_MIN_GAP,
    EXPERT_TTC_BRAKE,
    ConstantAction,
    ExpertPolicy,
)
from hybrid_av.types import PolicyAction
from tests.rl.helpers import _frame


def test_constant_action():
    assert ConstantAction()(np.zeros(24, np.float32), _frame()) == PolicyAction(1.0, 0.0, 0.0)


def test_expert_brakes_on_short_ttc():
    assert (
        ExpertPolicy()(
            None, _frame(lead_ttc=EXPERT_TTC_BRAKE - 0.5, lead_dist=8.0, lead_rel_speed=-4.0)
        ).brake
        >= 0.5
    )


def test_expert_brakes_for_ped_at_speed():
    assert ExpertPolicy()(None, _frame(vx=13.0, ped_dist=14.0)).brake == 1.0
    assert ExpertPolicy()(None, _frame(vx=5.0, ped_dist=14.0)).brake == 0.0


def test_expert_waits_then_goes_at_junction():  # waiting is via alpha_v (ppo-training.md §3 rule 4)
    near = {"cross_veh_dist": 30.0, "cross_veh_speed": 10.0, "cross_veh_ttc": 3.0}
    wait = ExpertPolicy()(
        None,
        _frame(
            vx=0.2, dist_to_junction=EXPERT_JUNCTION_WAIT_DIST, intersection_gap=EXPERT_MIN_GAP - 1.0, **near
        ),
    )
    go = ExpertPolicy()(
        None,
        _frame(
            vx=2.0, dist_to_junction=EXPERT_JUNCTION_WAIT_DIST, intersection_gap=EXPERT_MIN_GAP + 1.0, **near
        ),
    )
    assert wait.alpha_v == 0.0 and wait.brake < 0.5 and go.alpha_v == 1.0 and go.brake < 0.5


def test_expert_holds_brake_when_stopped_at_junction_with_cross_car_ahead_of_it():
    held = _frame(vx=0.0, dist_to_junction=EXPERT_JUNCTION_WAIT_DIST - 3.0, intersection_gap=-3.0)
    assert ExpertPolicy()(None, held).brake == 1.0
    assert (
        ExpertPolicy()(
            None, _frame(vx=0.0, dist_to_junction=EXPERT_JUNCTION_WAIT_DIST - 3.0, intersection_gap=0.5)
        ).brake
        == 0.0
    )
    assert (
        ExpertPolicy()(
            None, _frame(vx=3.0, dist_to_junction=EXPERT_JUNCTION_WAIT_DIST, intersection_gap=-3.0)
        ).brake
        == 0.0
    )


def test_expert_junction_stop_planned_at_junction_decel():
    from hybrid_av.rl.expert import EXPERT_JUNCTION_DECEL

    f = _frame(vx=10.0, dist_to_junction=40.0, intersection_gap=-1.0)
    cap = np.sqrt(2.0 * EXPERT_JUNCTION_DECEL * (40.0 - EXPERT_JUNCTION_WAIT_DIST)) / 13.89
    assert EXPERT_JUNCTION_DECEL == 1.0 and abs(ExpertPolicy()(None, f).alpha_v - cap) < 1e-9


@pytest.mark.slow
def test_expert_seed_50038_junction_no_collision():
    from hybrid_av.eval.runner import run_episode
    from hybrid_av.mpc.controller import MPCConfig
    from hybrid_av.planning.reference import ReferenceConfig
    from hybrid_av.scenarios import make_scenario
    from hybrid_av.sim.litesim.sim import LiteSim
    from hybrid_av.stack import AutonomyStack
    from hybrid_av.vehicle.actuation import LongitudinalMap
    from hybrid_av.vehicle.params import VehicleParams

    p, lm = VehicleParams(), LongitudinalMap()
    r = run_episode(
        LiteSim(p, lm),
        make_scenario("nominal", 50038),
        AutonomyStack(p, lm, MPCConfig(), ReferenceConfig()),
        ExpertPolicy(),
    )
    assert r.infractions["collision_vehicle"] == 0 and r.route_completion == 1.0


def test_expert_pedestrian_cone_is_route_relative():
    swerved = {"heading_error": np.radians(40.0), "ped_dist": 3.0, "ped_bearing": np.radians(-10.0)}
    assert ExpertPolicy()(None, _frame(**swerved)).brake < 0.5  # 30 deg off the route: roadside pedestrian
    ahead = {"heading_error": np.radians(40.0), "ped_dist": 3.0, "ped_bearing": np.radians(-40.0)}
    assert ExpertPolicy()(None, _frame(**ahead)).brake >= 0.5  # straight ahead along the route


def test_expert_waits_beyond_50m_at_speed():
    f = _frame(vx=12.7, dist_to_junction=80.0, intersection_gap=-1.0)
    assert ExpertPolicy()(None, f).alpha_v < 1.0


def test_expert_does_not_abort_after_point_of_no_return():
    # vx=5 needs 8.6 m to stop at 1.45 m/s²: at 7 m it cannot stop before the entry, so keep going
    committed = _frame(vx=5.0, dist_to_junction=7.0, intersection_gap=-1.0)
    assert ExpertPolicy()(None, committed).alpha_v == 1.0
    can_stop = _frame(vx=5.0, dist_to_junction=11.0, intersection_gap=-1.0)
    assert ExpertPolicy()(None, can_stop).alpha_v < 1.0


def test_expert_commit_hysteresis_while_moving():
    from hybrid_av.rl.expert import EXPERT_ABORT_GAP

    d = EXPERT_JUNCTION_WAIT_DIST + 6.0
    mid = EXPERT_ABORT_GAP + 0.5  # below the commit gap, above the abort gap
    assert (
        ExpertPolicy()(None, _frame(vx=0.0, dist_to_junction=d, intersection_gap=mid)).alpha_v < 1.0
    )  # stopped: wait
    assert (
        ExpertPolicy()(None, _frame(vx=1.5, dist_to_junction=d, intersection_gap=mid)).alpha_v == 1.0
    )  # moving: continue
    assert (
        ExpertPolicy()(
            None, _frame(vx=1.5, dist_to_junction=d, intersection_gap=EXPERT_ABORT_GAP - 0.5)
        ).alpha_v
        < 1.0
    )


def test_expert_commit_speed_is_below_hold_speed():
    d = EXPERT_JUNCTION_WAIT_DIST + 6.0  # a launch at 0.4 m/s with gap 0.5 is already committed
    assert ExpertPolicy()(None, _frame(vx=0.4, dist_to_junction=d, intersection_gap=0.5)).alpha_v == 1.0
    assert ExpertPolicy()(None, _frame(vx=0.2, dist_to_junction=d, intersection_gap=0.5)).alpha_v < 1.0
