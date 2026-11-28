"""Physics-free fake of the `carla` 0.9.16 Python API: only what the backend uses (Phase 7 Task 2).

The conftest `fake_carla` fixture installs this package as `carla`. Test hooks: `reset`, `set_current_map`,
`load_world_calls`, `last_world`, `last_tm`, `FakeWorld.inject/drop_sensor`, `Vehicle.physics_apply_mode`.
"""

import copy
import fnmatch
import itertools
import math
from enum import Enum

from . import command  # noqa: F401  (carla.command)

load_world_calls = 0
last_world = None
last_tm = None
planner_trace = None  # test hook: what GlobalRoutePlanner.trace_route returns (default: a straight line)
_current_map = "Carla/Maps/Town10HD_Opt"
_tms: dict = {}
_ids = itertools.count(100)


class _Attrs:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class Vector3D(_Attrs):
    x = y = z = 0.0

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Location(Vector3D):
    def distance(self, other):
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


class Rotation(_Attrs):
    pitch = yaw = roll = 0.0

    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform(_Attrs):
    location = None
    rotation = None

    def __init__(self, location=None, rotation=None):
        self.location, self.rotation = location or Location(), rotation or Rotation()


class BoundingBox(_Attrs):
    location = None
    extent = None

    def __init__(self, location=None, extent=None):
        self.location, self.extent = location or Location(), extent or Vector3D(2.4, 1.0, 0.75)


class WorldSettings(_Attrs):
    synchronous_mode = False
    fixed_delta_seconds = None
    max_substep_delta_time = 0.01
    max_substeps = 10
    no_rendering_mode = False


class WorldSnapshot(_Attrs):
    frame = 0
    timestamp = None
    world = None

    def find(self, actor_id):  # the actor stands in for its ActorSnapshot (same get_* methods)
        return self.world.actors.get(actor_id)


class VehicleControl(_Attrs):
    throttle = steer = brake = 0.0

    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, **kw):
        self.throttle, self.steer, self.brake = throttle, steer, brake
        self.__dict__.update(kw)


class WheelPhysicsControl(_Attrs):
    tire_friction = 3.5
    max_steer_angle = 70.0
    position = None

    def __init__(self, tire_friction=3.5, max_steer_angle=70.0, position=None, **kw):
        self.tire_friction, self.max_steer_angle = tire_friction, max_steer_angle
        self.position = position or Vector3D()
        # the remaining 0.9.16 constructor fields (damping_rate, radius, ...) are carried, not interpreted
        self.__dict__.update(
            damping_rate=0.25, radius=30.0, max_brake_torque=1500.0, max_handbrake_torque=3000.0
        )
        self.__dict__.update(lat_stiff_max_load=2.0, lat_stiff_value=17.0, long_stiff_value=1000.0)
        self.__dict__.update(kw)


class VehiclePhysicsControl(_Attrs):
    mass = 1800.0
    center_of_mass = None
    wheels = None

    def __init__(self, mass=1800.0, center_of_mass=None, wheels=None):
        self.mass, self.center_of_mass = mass, center_of_mass or Vector3D(0.0, 0.0, -0.3)
        self.wheels = wheels or [WheelPhysicsControl() for _ in range(4)]


class WeatherParameters(_Attrs):
    pass


for _preset in ("ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon", "ClearSunset", "ClearNight"):
    setattr(WeatherParameters, _preset, WeatherParameters())


class LaneType(Enum):
    Driving = 2
    Sidewalk = 32
    Any = -2


class MapLayer(Enum):
    ParkedVehicles = 16
    All = 255


class TrafficLightState(Enum):
    Red = 0
    Yellow = 1
    Green = 2
    Off = 3
    Unknown = 4


class LidarMeasurement(_Attrs):
    raw_data = b""
    frame = 0
    channels = 64

    def get_point_count(self, channel=0):
        return len(self.raw_data) // 16


class RadarMeasurement(_Attrs):
    raw_data = b""
    frame = 0


class IMUMeasurement(_Attrs):
    accelerometer = gyroscope = None
    compass = 0.0
    frame = 0


class Image(_Attrs):
    raw_data = b""
    width = height = 0
    frame = 0


class CollisionEvent(_Attrs):
    other_actor = None
    frame = 0


class Actor:
    id = 0
    type_id = ""
    attributes = None
    bounding_box = None

    def __init__(self, world, type_id, transform=None):
        self.world, self.id, self.type_id = world, next(_ids), type_id
        self.attach_to = None
        self.attributes, self.bounding_box = {}, BoundingBox()
        self.transform = transform or Transform()
        self.velocity = self.angular_velocity = self.acceleration = None
        self.destroyed = False

    def get_transform(self):
        return self.transform

    def get_velocity(self):
        return self.velocity or Vector3D()

    def get_angular_velocity(self):
        return self.angular_velocity or Vector3D()

    def get_acceleration(self):
        return self.acceleration or Vector3D()

    def set_target_velocity(self, velocity):
        self.target_velocity = velocity

    def destroy(self):
        self.destroyed = True
        self.world.actors.pop(self.id, None)
        return True


class WalkerControl(_Attrs):
    direction = None
    speed = 0.0

    def __init__(self, direction=None, speed=0.0, jump=False):
        self.direction, self.speed, self.jump = direction or Vector3D(), speed, jump


class Walker(Actor):
    def apply_control(self, control):
        self.control = control


class Vehicle(Actor):
    physics_apply_mode = "ok"  # "ignore" simulates the wheel-physics bug (ECC-16)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.control, self.physics = VehicleControl(), VehiclePhysicsControl()

    @property
    def last_control(self):
        return self.control

    def apply_control(self, control):
        self.control = control

    def get_control(self):
        return self.control

    def get_physics_control(self):
        return copy.deepcopy(self.physics)

    def apply_physics_control(self, physics):
        if self.physics_apply_mode == "ok":
            self.physics = copy.deepcopy(physics)

    def get_traffic_light(self):
        return None

    def get_speed_limit(self):
        return 30.0  # km/h

    def set_autopilot(self, enabled=True, tm_port=8000):
        self.autopilot = enabled


class WalkerAIController(Actor):
    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def go_to_location(self, location):
        self.target = location

    def set_max_speed(self, speed):
        self.max_speed = speed


class Sensor(Actor):
    stopped = True
    stopped_before_destroy = None

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.callback = None

    def listen(self, callback):
        self.callback, self.stopped = callback, False

    def stop(self):
        self.callback, self.stopped = None, True

    def is_listening(self):
        return self.callback is not None

    def destroy(self):
        self.stopped_before_destroy = self.stopped
        return super().destroy()


class TrafficLight(Actor):
    state = TrafficLightState.Red

    def get_state(self):
        return self.state

    def set_state(self, state):
        self.state = state

    def freeze(self, frozen):
        self.frozen = frozen

    def set_green_time(self, t):
        self.green = t

    def set_yellow_time(self, t):
        self.yellow = t

    def set_red_time(self, t):
        self.red = t

    stop_waypoints = ()  # test hook: what get_stop_waypoints returns

    def get_stop_waypoints(self):
        return list(self.stop_waypoints)

    def get_elapsed_time(self):
        return 0.0


class Waypoint:
    transform = None
    lane_width = 3.5
    lane_type = LaneType.Driving
    road_id = lane_id = 0
    is_junction = False

    def __init__(self, transform=None, lane_width=3.5):
        self.transform, self.lane_width = transform or Transform(), lane_width

    def next(self, distance):
        return []

    def get_junction(self):
        return None


class Map:
    name = ""

    def __init__(self, name):
        self.name = name

    drivable = True  # test hook: False makes get_waypoint(project_to_road=False) return None (off the road)

    def get_waypoint(self, location, project_to_road=True, lane_type=LaneType.Driving):
        if not project_to_road and not self.drivable:
            return None
        return Waypoint(Transform(Location(location.x, location.y, location.z)))

    def get_spawn_points(self):
        return [Transform(Location(10.0 * i, 20.0, 0.0)) for i in range(60)]

    def get_topology(self):
        return []

    def generate_waypoints(self, distance):
        return []


class ActorBlueprint:
    id = ""

    def __init__(self, id):
        self.id, self.attributes = id, {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


_CATALOG = (
    "vehicle.audi.tt",
    "vehicle.tesla.model3",
    "vehicle.diamondback.century",
    "walker.pedestrian.0001",
    "walker.pedestrian.0002",
)


class BlueprintLibrary:
    def find(self, id):
        return ActorBlueprint(id)

    def filter(self, pattern):
        return [ActorBlueprint(i) for i in _CATALOG if fnmatch.fnmatch(i, pattern)]


class TrafficManager:
    synchronous = False
    seed = None

    def get_port(self):
        return 8000

    def set_synchronous_mode(self, on):
        self.synchronous = on

    def set_random_device_seed(self, seed):
        self.seed = seed

    def set_desired_speed(self, actor, speed):
        self.desired_speed = speed

    def ignore_lights_percentage(self, actor, pct):
        self.ignore_lights = pct

    def distance_to_leading_vehicle(self, actor, d):
        self.leading_distance = d

    def auto_lane_change(self, actor, on):
        self.lane_change = on

    def set_hybrid_physics_mode(self, on):
        self.hybrid = on


def _default_datum(type_id):
    kinds = {
        "sensor.camera": lambda: Image(raw_data=bytes(4 * 2 * 4), width=4, height=2),
        "sensor.lidar": LidarMeasurement,
        "sensor.other.radar": RadarMeasurement,
        "sensor.other.imu": lambda: IMUMeasurement(accelerometer=Vector3D(), gyroscope=Vector3D()),
    }  # the collision sensor only fires through collide()
    make = next((m for k, m in kinds.items() if type_id.startswith(k)), None)
    return make() if make else None


class ActorList(list):
    def filter(self, pattern):
        return ActorList(a for a in self if fnmatch.fnmatch(a.type_id, pattern))


class FakeWorld:
    def __init__(self):
        self.settings, self.frame, self.elapsed = WorldSettings(), 0, 0.0
        self.actors: dict[int, Actor] = {}
        self._map = Map(_current_map)
        self._pending: dict[int, object] = {}
        self._dropped: set[tuple[int, int]] = set()

    def get_settings(self):
        return copy.copy(self.settings)

    def apply_settings(self, settings):
        self.settings = copy.copy(settings)
        return self.frame

    def tick(self, seconds=None):
        self.frame += 1
        self.elapsed += self.settings.fixed_delta_seconds or 0.0
        for sensor in self.actors.values():  # every listening sensor delivers one datum per tick
            if isinstance(sensor, Sensor) and sensor.id not in self._pending and sensor.callback:
                datum = _default_datum(sensor.type_id)
                if datum is not None:
                    self._pending[sensor.id] = datum
        for sid, data in list(self._pending.items()):
            sensor = self.actors.get(sid)
            data.frame = vars(data).get("frame") or self.frame  # an explicit frame on the datum is kept
            if sensor is not None and sensor.callback and (sid, data.frame) not in self._dropped:
                sensor.callback(data)
        self._pending.clear()
        return self.frame

    def get_snapshot(self):
        return WorldSnapshot(frame=self.frame, timestamp=_Attrs(elapsed_seconds=self.elapsed), world=self)

    def get_random_location_from_navigation(self):
        return Location(next(_ids) % 50, 3.0, 0.0)

    def get_map(self):
        return self._map

    def get_blueprint_library(self):
        return BlueprintLibrary()

    def spawn_actor(self, blueprint, transform, attach_to=None):
        kind = (
            Vehicle
            if blueprint.id.startswith("vehicle.")
            else Sensor
            if blueprint.id.startswith("sensor.")
            else Walker
            if blueprint.id.startswith("walker.")
            else WalkerAIController
            if blueprint.id.startswith("controller.ai.walker")
            else TrafficLight
            if blueprint.id.startswith("traffic.traffic_light")
            else Actor
        )
        actor = kind(self, blueprint.id, transform)
        actor.attach_to, actor.attributes = attach_to, dict(blueprint.attributes)
        self.actors[actor.id] = actor
        return actor

    spawn_failures = (
        0  # test hook: the next n try_spawn_actor calls return None (collision at the spawn pose)
    )
    try_spawn_poses: list  # every try_spawn_actor transform, in order

    def try_spawn_actor(self, blueprint, transform, attach_to=None):
        self.__dict__.setdefault("try_spawn_poses", []).append(transform)
        if self.spawn_failures > 0:
            self.spawn_failures -= 1
            return None
        return self.spawn_actor(blueprint, transform, attach_to)

    def unload_map_layer(self, layer):
        self.__dict__.setdefault("unloaded_layers", []).append(layer)

    def get_actor(self, actor_id):
        return self.actors.get(actor_id)

    def get_actors(self):
        return ActorList(self.actors.values())

    def get_traffic_lights_in_junction(self, junction_id):
        return []

    def set_weather(self, weather):
        self.weather = weather

    # test hooks
    def inject(self, sensor_id, data):
        """Deliver `data` to the sensor's callback on the next tick (data.frame, if set, is kept)."""
        self._pending[sensor_id] = data

    def drop_sensor(self, sensor_id, frame):
        self._dropped.add((sensor_id, frame))


World = FakeWorld


class Client:
    def __init__(self, host="localhost", port=2000, worker_threads=0):
        self.host, self.port = host, port

    def set_timeout(self, seconds):
        self.timeout = seconds
        self.timeouts = [*getattr(self, "timeouts", []), seconds]

    def get_world(self):
        return last_world

    def load_world(self, map_name, reset_settings=True):
        global load_world_calls, _current_map
        load_world_calls += 1
        _current_map = f"Carla/Maps/{map_name}"
        last_world._map = Map(_current_map)
        if reset_settings:
            last_world.settings = WorldSettings()
        return last_world

    def get_trafficmanager(self, port=8000):
        global last_tm
        last_tm = _tms.setdefault(port, TrafficManager())
        return last_tm

    def get_server_version(self):
        return "0.9.16"

    def get_client_version(self):
        return "0.9.16"

    def apply_batch_sync(self, commands, due_tick_cue=False):
        for c in commands:
            actor = last_world.actors.get(c.actor_id)
            if actor is not None:
                actor.destroy()
        return [_Attrs(error=None) for _ in commands]


def set_current_map(name):
    global _current_map
    _current_map = name
    last_world._map = Map(name)


def collide(other_type="vehicle.audi.tt", other_id=42):
    """Test hook: the ego's collision sensor reports a contact on the next tick."""
    sensor = next(a for a in last_world.actors.values() if a.type_id == "sensor.other.collision")
    last_world.inject(sensor.id, CollisionEvent(other_actor=_Attrs(type_id=other_type, id=other_id)))


def __getattr__(name):
    if name == "ego":  # the first vehicle spawned in the current world
        return next(a for a in last_world.actors.values() if isinstance(a, Vehicle))
    raise AttributeError(name)


def last_or_new_world():
    return last_world


def spawn_vehicle(tire_friction=3.5):
    """Test hook: a vehicle whose four wheels have `tire_friction`."""
    vehicle = last_world.spawn_actor(ActorBlueprint("vehicle.tesla.model3"), Transform())
    for wheel in vehicle.physics.wheels:
        wheel.tire_friction = tire_friction
    return vehicle


def straight_trace(n, spacing=1.0, lane_width=3.5, lane_change_at=None):
    """Test hook: n waypoints along CARLA +x, as (Waypoint, RoadOption) pairs; one lane change if asked."""
    from .agents.navigation.local_planner import RoadOption

    return [
        (
            Waypoint(Transform(Location(i * spacing, 0.0, 0.0)), lane_width),
            RoadOption.CHANGELANELEFT if i == lane_change_at else RoadOption.LANEFOLLOW,
        )
        for i in range(n)
    ]


def reset():
    """A fresh server: new world in default (asynchronous) settings, no Traffic Managers."""
    global load_world_calls, last_world, last_tm, _current_map, planner_trace
    load_world_calls, last_tm, _current_map, planner_trace = 0, None, "Carla/Maps/Town10HD_Opt", None
    _tms.clear()
    last_world = FakeWorld()
