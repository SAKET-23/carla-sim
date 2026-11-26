from hybrid_av.rl.reward import RewardConfig, compute_reward
from tests.rl.helpers import A, _frame, _info, _mpc

CFG = RewardConfig()


def test_total_is_sum():
    total, comp = compute_reward(_frame(route_s=0), _frame(route_s=1.0), _info(), _mpc(), A, A, CFG)
    assert abs(total - sum(comp.values())) < 1e-12


def test_progress_monotone():
    r1, _ = compute_reward(_frame(route_s=0), _frame(route_s=0.5), _info(), _mpc(), A, A, CFG)
    r2, _ = compute_reward(_frame(route_s=0), _frame(route_s=1.0), _info(), _mpc(), A, A, CFG)
    assert r2 > r1 > compute_reward(_frame(route_s=0), _frame(route_s=0), _info(), _mpc(), A, A, CFG)[0]


def test_collision_dominates():
    good, _ = compute_reward(
        _frame(route_s=0), _frame(route_s=1.4), _info(), _mpc(), A, A, CFG
    )  # best-case step
    bad, _ = compute_reward(
        _frame(route_s=0), _frame(route_s=1.4), _info(collision="vehicle"), _mpc(), A, A, CFG
    )
    assert bad < -10 * abs(good)


def test_infractions_and_slack_penalised():
    base, _ = compute_reward(_frame(route_s=0), _frame(route_s=1.0), _info(), _mpc(), A, A, CFG)
    for kw in ({"red_light_violation": True}, {"stop_sign_violation": True}, {"off_road": True}):
        assert (
            compute_reward(_frame(route_s=0), _frame(route_s=1.0), _info(**kw), _mpc(), A, A, CFG)[0] < base
        )
    assert (
        compute_reward(
            _frame(route_s=0), _frame(route_s=1.0), _info(), _mpc(slack_friction_max=0.5), A, A, CFG
        )[0]
        < base
    )


def test_waiting_at_red_not_penalised_as_blocked():
    _, comp = compute_reward(
        _frame(route_s=10, vx=0.0, traffic_light_state=3, stop_dist=1.0),
        _frame(route_s=10, vx=0.0, traffic_light_state=3, stop_dist=1.0),
        _info(),
        _mpc(),
        A,
        A,
        CFG,
    )
    assert comp.get("standstill", 0.0) == 0.0


def test_terms_hand_built():
    f0, f1 = _frame(route_s=0, vx=0.0), _frame(route_s=0, vx=0.0)
    _, c = compute_reward(f0, f1, _info(), _mpc(), A, A, CFG)
    assert c["standstill"] == CFG.w_standstill and c["progress"] == 0.0
    _, c = compute_reward(
        f0, _frame(vx=20.0, speed_limit=10.0), _info(), _mpc(status="fallback_brake"), A, A, CFG
    )
    assert c["overspeed"] == CFG.w_overspeed and c["fallback"] == CFG.w_fallback
    _, c = compute_reward(f0, f1, _info(collision="pedestrian"), _mpc(), A, A, CFG)
    assert c["collision"] + c["collision_ped_extra"] == -100.0 and c["complete"] == 0.0
    _, c = compute_reward(f0, f1, _info(route_completed=True, timeout=True), _mpc(), A, A, CFG)
    assert c["complete"] == CFG.r_complete and c["timeout"] == 0.0
