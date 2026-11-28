"""AST scan of the backend for the CARLA names it uses, and the documented list the fake must implement."""

import ast
from pathlib import Path

# Heuristic: an attribute read on a variable with one of these names is a method/attribute of that CARLA class.
# Real carla puts apply_control, listen, ... on Vehicle/Sensor, not Actor, so the distinction matters on the pod.
VAR_CLASS = {
    "client": "Client", "world": "World", "tm": "TrafficManager", "traffic_manager": "TrafficManager",
    "ego": "Vehicle", "vehicle": "Vehicle", "actor": "Actor", "sensor": "Sensor", "walker": "Walker",
    "wmap": "Map", "carla_map": "Map", "wp": "Waypoint", "waypoint": "Waypoint",
    "light": "TrafficLight", "traffic_light": "TrafficLight", "controller": "WalkerAIController",
    "snap": "WorldSnapshot", "snapshot": "WorldSnapshot",
    "bp_lib": "BlueprintLibrary", "blueprint_library": "BlueprintLibrary", "bp": "ActorBlueprint", "blueprint": "ActorBlueprint",
}  # fmt: skip


def _chain(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return (node.id, parts[::-1]) if isinstance(node, ast.Name) else (None, parts[::-1])


def used_names(paths: list[Path]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        tree = ast.parse(Path(p).read_text())
        inner = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for n in ast.walk(tree):
            if not isinstance(n, ast.Attribute) or not isinstance(n.ctx, ast.Load):
                continue
            root, parts = _chain(n)
            if root == "carla" and id(n) not in inner:
                out.add(".".join(parts))
            elif root != "carla":
                base = n.value
                name = (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr
                    if isinstance(base, ast.Attribute)
                    else None
                )
                if name in VAR_CLASS:
                    out.add(f"{VAR_CLASS[name]}.{n.attr}")
    return out


def _cls(cls, *names):
    return {f"{cls}.{n}" for n in names}


API_SURFACE: frozenset[str] = frozenset(
    {
        "Client",
        "World",
        "WorldSettings",
        "WorldSnapshot",
        "Map",
        "Waypoint",
        "LaneType",
        "Transform",
        "Location",
        "Rotation",
        "Vector3D",
        "BoundingBox",
        "Actor",
        "Vehicle",
        "Sensor",
        "Walker",
        "WalkerControl",
        "VehicleControl",
        "VehiclePhysicsControl",
        "WheelPhysicsControl",
        "TrafficManager",
        "TrafficLight",
        "TrafficLightState",
        "LidarMeasurement",
        "RadarMeasurement",
        "IMUMeasurement",
        "Image",
        "CollisionEvent",
        "WeatherParameters",
        "WalkerAIController",
        "BlueprintLibrary",
        "ActorBlueprint",
        "command",
        "command.DestroyActor",
        "MapLayer.ParkedVehicles",
        "LaneType.Driving",
        "LaneType.Sidewalk",
        "LaneType.Any",
        "TrafficLightState.Red",
        "TrafficLightState.Yellow",
        "TrafficLightState.Green",
        "TrafficLightState.Off",
        "TrafficLightState.Unknown",
    }
    | _cls(
        "Client",
        "set_timeout",
        "get_world",
        "load_world",
        "get_trafficmanager",
        "get_server_version",
        "get_client_version",
        "apply_batch_sync",
    )
    | _cls(
        "World",
        "get_settings",
        "apply_settings",
        "tick",
        "get_snapshot",
        "get_map",
        "get_blueprint_library",
        "spawn_actor",
        "try_spawn_actor",
        "get_actor",
        "get_actors",
        "get_random_location_from_navigation",
        "get_traffic_lights_in_junction",
        "set_weather",
        "unload_map_layer",
    )
    | _cls(
        "WorldSettings",
        "synchronous_mode",
        "fixed_delta_seconds",
        "max_substep_delta_time",
        "max_substeps",
        "no_rendering_mode",
    )
    | _cls("WorldSnapshot", "frame", "timestamp", "find")
    | _cls("Map", "name", "get_waypoint", "get_spawn_points", "get_topology", "generate_waypoints")
    | _cls(
        "Waypoint",
        "transform",
        "lane_width",
        "lane_type",
        "road_id",
        "lane_id",
        "is_junction",
        "next",
        "get_junction",
    )
    | _cls("BoundingBox", "extent", "location")
    | _cls(
        "Actor",
        "id",
        "type_id",
        "attributes",
        "bounding_box",
        "get_transform",
        "get_velocity",
        "get_angular_velocity",
        "get_acceleration",
        "destroy",
        "set_target_velocity",
    )
    | _cls(
        "Vehicle",
        "id",
        "type_id",
        "attributes",
        "bounding_box",
        "get_transform",
        "get_velocity",
        "get_angular_velocity",
        "get_acceleration",
        "destroy",
        "apply_control",
        "set_target_velocity",
        "get_control",
        "get_physics_control",
        "apply_physics_control",
        "get_traffic_light",
        "get_speed_limit",
        "set_autopilot",
    )
    | _cls("Walker", "id", "type_id", "get_transform", "destroy", "apply_control")
    | _cls(
        "Sensor", "id", "type_id", "attributes", "get_transform", "destroy", "listen", "stop", "is_listening"
    )
    | _cls("VehiclePhysicsControl", "mass", "center_of_mass", "wheels")
    | _cls("WheelPhysicsControl", "tire_friction", "max_steer_angle", "position")
    | _cls(
        "TrafficManager",
        "set_synchronous_mode",
        "set_random_device_seed",
        "set_desired_speed",
        "ignore_lights_percentage",
        "distance_to_leading_vehicle",
        "auto_lane_change",
        "set_hybrid_physics_mode",
        "get_port",
    )
    | _cls(
        "TrafficLight",
        "state",
        "get_state",
        "set_state",
        "freeze",
        "set_green_time",
        "set_yellow_time",
        "set_red_time",
        "get_stop_waypoints",
        "get_elapsed_time",
    )
    | _cls("LidarMeasurement", "raw_data", "frame", "channels", "get_point_count")
    | _cls("RadarMeasurement", "raw_data", "frame")
    | _cls("IMUMeasurement", "accelerometer", "gyroscope", "compass", "frame")
    | _cls("Image", "raw_data", "width", "height", "frame")
    | _cls("CollisionEvent", "other_actor", "frame")
    | _cls("BlueprintLibrary", "find", "filter")
    | _cls("WalkerAIController", "start", "stop", "go_to_location", "set_max_speed")
    | {
        f"WeatherParameters.{n}"
        for n in ("ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon", "ClearSunset", "ClearNight")
    }
    | _cls("ActorBlueprint", "id", "set_attribute")
)
