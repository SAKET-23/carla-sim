import numpy as np
from hybrid_av_nodes.mpc_core import MpcCore
from hybrid_av_nodes.tick_loop import CommandMailbox, StampMailbox

from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.planning.route import RouteBuilder
from hybrid_av.stack import AutonomyStack
from hybrid_av.types import (
    EgoState,
    Observation,
    PolicyAction,
    TrafficContext,
    TrafficLightState,
)
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams


def obs(t, x):
    ctx = TrafficContext(TrafficLightState.NONE, np.inf, np.inf, np.inf, 13.89, 3.5, 0.0, 0.1)
    return Observation(
        EgoState(t=t, x=x, y=0, psi=0, vx=10, vy=0, r=0), (), ctx, float("nan"), "ground_truth"
    )


def stack():
    return AutonomyStack(VehicleParams(), LongitudinalMap(), MPCConfig(), ReferenceConfig())


def core():
    c = MpcCore(stack())
    c.on_route(RouteBuilder.straight(300.0, 13.89))
    return c


def test_policy_ticks():
    c = core()
    assert [c.needs_action(k) for k in range(5)] == [True, False, True, False, True]


def test_matches_direct_stack_calls():
    c, s = core(), stack()
    s.reset(RouteBuilder.straight(300.0, 13.89))
    a = PolicyAction(1.0, 0.0, 0.0)
    for k in range(5):
        o = obs(k * 0.05, 10 + 0.5 * k)
        f1 = c.features(o)
        out1 = c.act(o, a, f1)
        out2 = s.act(o, a, s.features(o))
        assert out1.command == out2.command


def test_stamp_mailbox_generic_and_alias():
    assert CommandMailbox is StampMailbox
    mb = StampMailbox()
    mb.put(1.0, PolicyAction(1.0, 0.0, 0.0))
    assert mb.wait_for(1.0, 0.01) == PolicyAction(1.0, 0.0, 0.0) and mb.wait_for(2.0, 0.01) is None
