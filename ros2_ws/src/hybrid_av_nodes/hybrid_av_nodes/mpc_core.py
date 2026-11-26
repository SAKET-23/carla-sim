"""mpc_node's logic around the single AutonomyStack. Pure core: no rclpy."""

from hybrid_av.features.extractor import FeatureFrame
from hybrid_av.planning.route import Route
from hybrid_av.stack import AutonomyStack, StackOutput
from hybrid_av.types import Observation, PolicyAction


class MpcCore:
    def __init__(self, stack: AutonomyStack, policy_period: int = 2) -> None:
        self.stack, self.policy_period = stack, policy_period

    def on_route(self, route: Route) -> None:
        self.stack.reset(route)

    def features(self, obs: Observation) -> FeatureFrame:
        return self.stack.features(obs)

    def needs_action(self, tick: int) -> bool:
        return tick % self.policy_period == 0

    def act(self, obs: Observation, action: PolicyAction, frame: FeatureFrame) -> StackOutput:
        return self.stack.act(obs, action, frame)
