"""CARLA connection, synchronous mode and guaranteed cleanup (simulation-backends.md §2). `carla` is imported lazily."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class CarlaSimConfig:
    host: str = "localhost"
    port: int = 2000
    tm_port: int = 8000
    town: str = "Town03_Opt"
    dt: float = 0.05  # fixed_delta_seconds
    substep_dt: float = 0.01  # max_substep_delta_time
    max_substeps: int = 10
    quality: Literal["Low", "Epic"] = "Low"  # informational; set at server launch
    ego_blueprint: str = "vehicle.tesla.model3"
    sensors: Path | None = None  # configs/sensors/carla_rig.yaml; None = no sensors (RL mode)
    seed: int = 0
    timeout: float = 20.0
    no_rendering: bool = False  # world.settings.no_rendering_mode; True for RL with no cameras
    tire_friction_scale: float | None = None  # calibrated k* (calibration.md §3); None = use zone.mu


@dataclass
class Session:
    client: object
    world: object
    tm: object
    actors: list = field(default_factory=list)  # every actor we spawn, destroyed on exit


def connect(cfg: CarlaSimConfig):
    import carla

    client = carla.Client(cfg.host, cfg.port)
    client.set_timeout(cfg.timeout)
    world = client.get_world()
    # ECC-02: map names look like "Carla/Maps/Town03_Opt"; load_world resets settings, so apply_sync runs after it
    if not world.get_map().name.endswith(cfg.town):
        world = client.load_world(cfg.town)
    return client, world


def apply_sync(world, tm, cfg: CarlaSimConfig):
    """Switch world and Traffic Manager to synchronous mode; returns the original settings."""
    if cfg.dt > cfg.substep_dt * cfg.max_substeps:
        raise ValueError(f"substep config cannot cover dt: {cfg.dt} > {cfg.substep_dt} * {cfg.max_substeps}")
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.dt
    settings.max_substep_delta_time = cfg.substep_dt
    settings.max_substeps = cfg.max_substeps
    settings.no_rendering_mode = cfg.no_rendering
    try:
        world.apply_settings(settings)
        # ECC-02: TM synchronous mode must follow apply_settings; one tick per world.tick()
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(cfg.seed)
    except BaseException:
        restore(world, tm, original)  # never leave the server frozen
        raise
    return original


def restore(world, tm, original) -> None:
    try:
        tm.set_synchronous_mode(False)
    finally:
        # forced asynchronous: a crashed earlier client may have left `original` synchronous (IC-03)
        original.synchronous_mode = False
        original.fixed_delta_seconds = None
        world.apply_settings(original)


def destroy_actors(client, actors) -> None:
    import carla

    # ECC-03: sensors (and walker AI controllers) must be stopped before destroy, or a late callback can segfault
    for sensor in (a for a in actors if isinstance(a, (carla.Sensor, carla.WalkerAIController))):
        sensor.stop()
    client.apply_batch_sync([carla.command.DestroyActor(a.id) for a in actors])


@contextmanager
def carla_session(cfg: CarlaSimConfig) -> Iterator[Session]:
    client, world = connect(cfg)
    tm = client.get_trafficmanager(cfg.tm_port)
    original = apply_sync(world, tm, cfg)
    s = Session(client, world, tm)
    try:
        yield s
    finally:  # also on exceptions and KeyboardInterrupt
        try:
            destroy_actors(client, s.actors)
        finally:
            restore(world, tm, original)
