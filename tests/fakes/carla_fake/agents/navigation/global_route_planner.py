import math

from .local_planner import RoadOption


class GlobalRoutePlanner:
    """Straight line from origin to destination, one waypoint per sampling_resolution metres."""

    def __init__(self, wmap, sampling_resolution):
        self.wmap, self.step = wmap, sampling_resolution

    def trace_route(self, origin, destination):
        from tests.fakes import carla_fake

        if carla_fake.planner_trace is not None:
            return carla_fake.planner_trace
        n = max(1, int(math.dist((origin.x, origin.y), (destination.x, destination.y)) / self.step))
        return [
            (
                self.wmap.get_waypoint(
                    type(origin)(
                        origin.x + (destination.x - origin.x) * i / n,
                        origin.y + (destination.y - origin.y) * i / n,
                        origin.z,
                    )
                ),
                RoadOption.LANEFOLLOW,
            )
            for i in range(n + 1)
        ]
