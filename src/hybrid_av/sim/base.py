"""Backend-neutral scenario types and the Sim protocol (interfaces.md section 7)."""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from hybrid_av.planning.route import Route
from hybrid_av.types import ActuatorCommand, ObjectClass, Observation


@dataclass(frozen=True, slots=True)
class FrictionZone:
    s_start: float
    s_end: float
    mu: float  # along the ego route


@dataclass(frozen=True, slots=True)
class ScriptedActor:
    """Backend-neutral description of a non-ego actor."""

    id: int
    cls: ObjectClass
    kind: Literal["lead", "cross_stream", "pedestrian_crossing", "parked"]
    params: dict[str, float]  # documented per kind in scenarios.md


@dataclass(frozen=True, slots=True)
class TrafficControl:
    kind: Literal["light", "stop_sign"]
    s: float  # stop line position along ego route
    cycle: tuple[float, float, float] = (10.0, 3.0, 10.0)  # green, yellow, red (light only)
    phase_offset: float = 0.0


# Route holds arrays and actors hold dicts: no generated __eq__/__hash__ (identity semantics).
@dataclass(frozen=True, slots=True, eq=False)
class Scenario:
    name: str
    seed: int
    route: Route | None  # LiteSim: given; CARLA: None, built from carla_route
    carla_route: dict | None  # {"town", "start": [x,y,z,yaw], "goal": [x,y,z]} in CARLA coords
    friction_zones: tuple[FrictionZone, ...]
    base_mu: float
    actors: tuple[ScriptedActor, ...]
    controls: tuple[TrafficControl, ...]
    junction_s: tuple[tuple[float, float], ...]  # (entry_s, exit_s) along the route
    time_limit: float  # s
    ego_start_speed: float
    base_roughness: float = 0.0  # LAST: surface_roughness outside wet zones (state-vector.md slot 7)


@dataclass(frozen=True, slots=True)
class StepInfo:
    collision: Literal["none", "pedestrian", "vehicle", "static"]
    red_light_violation: bool
    stop_sign_violation: bool
    off_road: bool  # ego centre outside drivable area this step
    route_s: float
    route_completed: bool
    route_deviation: bool  # lateral distance from route > 30 m
    blocked: bool  # speed < 0.1 m/s for >= 180 s sim time without a valid reason
    timeout: bool
    sim_time: float


@runtime_checkable
class Sim(Protocol):
    dt: float

    def reset(self, scenario: Scenario) -> Observation: ...

    def step(self, cmd: ActuatorCommand) -> tuple[Observation, StepInfo]: ...

    def route(self) -> Route: ...

    def close(self) -> None: ...
