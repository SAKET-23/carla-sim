"""A driving ego with Traffic Manager traffic and walkers, for dataset generation. `carla` is imported lazily."""

import random
from dataclasses import dataclass
from pathlib import Path

from hybrid_av.sim.carla_backend.sensors import SensorQueues, SensorSpec, load_rig, spawn_rig

WEATHER = ("ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon", "ClearSunset", "ClearNight")
_SENSOR_TIMEOUT = 5.0  # s, wall clock per frame (ECC-11)


@dataclass
class Collector:
    session: object
    ego: object
    queues: SensorQueues
    specs: list[SensorSpec]

    def tick(self):
        """Advance one tick; returns (snapshot, frame-matched sensor data of that snapshot)."""
        self.session.world.tick()
        snap = self.session.world.get_snapshot()
        return snap, self.queues.collect(snap.frame, _SENSOR_TIMEOUT)

    def set_weather(self, index: int) -> None:
        import carla

        # ECC-42: weather presets are class attributes of WeatherParameters (ClearNight needs 0.9.14+)
        self.session.world.set_weather(getattr(carla.WeatherParameters, WEATHER[index % len(WEATHER)]))


def open_collector(
    session, rig: Path, sensor_ids: list[str], seed: int, n_vehicles: int = 30, n_walkers: int = 40
) -> Collector:
    """Spawn an autopilot ego with the chosen sensors of the rig, `n_vehicles` TM vehicles and `n_walkers` walkers.
    Everything is tracked in `session.actors`, so the session's cleanup destroys it."""
    import carla

    world, tm = session.world, session.tm
    rng = random.Random(seed)
    bp_lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    rng.shuffle(spawns)

    ego = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), spawns.pop())
    session.actors.append(ego)
    ego.set_autopilot(True, tm.get_port())
    specs = [s for s in load_rig(rig) if s.id in sensor_ids]
    queues = SensorQueues([s.id for s in specs], multi=())
    spawn_rig(session, ego, specs, queues)

    vehicles = bp_lib.filter("vehicle.*")
    for transform in spawns[:n_vehicles]:
        vehicle = world.try_spawn_actor(rng.choice(vehicles), transform)
        if vehicle is not None:
            session.actors.append(vehicle)
            vehicle.set_autopilot(True, tm.get_port())

    walkers = bp_lib.filter("walker.pedestrian.*")
    controller_bp = bp_lib.find("controller.ai.walker")
    for _ in range(n_walkers):
        loc = (
            world.get_random_location_from_navigation()
        )  # ECC-42: WalkerAIController / navmesh; None when the navmesh has no point
        walker = world.try_spawn_actor(rng.choice(walkers), carla.Transform(loc)) if loc is not None else None
        if walker is None:
            continue
        session.actors.append(walker)
        controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
        session.actors.append(controller)
        controller.start()
        controller.go_to_location(world.get_random_location_from_navigation())
        controller.set_max_speed(1.4)
    return Collector(session, ego, queues, specs)
