"""bridge_only.launch.py: carla_bridge with the full sensor rig, static TF and a fixed-command responder.
rig:=<yaml> picks another rig, rig:=none runs sensorless (no static TF); no_rendering:=true is the RL mode.

Used by smoke 05 and G1 (rates, VRAM); no autonomy nodes."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

ARGS = {  # quality: recorded only; rig "" = the repo carla_rig.yaml, "none" = no sensors
    "town": "Town03_Opt",
    "scenario": "nominal",
    "seed": "100000",
    "quality": "Low",
    "rig": "",
    "no_rendering": "false",
}


def _repo_rig() -> str:
    root = os.environ.get("HYBRID_AV_ROOT")
    if root:
        return str(Path(root) / "configs/sensors/carla_rig.yaml")
    for parent in Path(__file__).resolve().parents:  # symlink-install: this is the file in the source tree
        if (parent / "configs/sensors/carla_rig.yaml").exists():
            return str(parent / "configs/sensors/carla_rig.yaml")
    raise RuntimeError("configs/sensors/carla_rig.yaml not found: set HYBRID_AV_ROOT")


def _nodes(context):
    a = {k: context.launch_configurations[k] for k in ARGS}
    rig = a["rig"] or _repo_rig()
    sensors = "" if rig == "none" else rig
    bridge = Node(  # the tick owner publishes /clock: no use_sim_time
        package="hybrid_av_nodes", executable="carla_bridge", name="carla_bridge", output="screen",
        parameters=[{"town": a["town"], "scenario": a["scenario"], "seed": int(a["seed"]),
                     "quality": a["quality"], "sensors": sensors,
                     # explicit: the bridge otherwise derives it from the rig path, which breaks for rig:=none or rigs outside configs/
                     "routes": str(Path(_repo_rig()).parents[1] / "scenarios/carla_routes.yaml"),
                     "no_rendering": a["no_rendering"].lower() == "true"}],
    )  # fmt: skip
    static_tf = (
        []
        if sensors == ""
        else [  # load_rig needs a yaml path
            Node(
                package="hybrid_av_bringup",
                executable="static_tf",
                name="static_tf",
                output="screen",
                parameters=[{"use_sim_time": True, "rig": rig}],
            )
        ]
    )
    return static_tf + [
        Node(package="hybrid_av_nodes", executable="command_responder", name="command_responder",
             output="screen"),
        bridge,
        RegisterEventHandler(OnProcessExit(target_action=bridge, on_exit=[EmitEvent(event=Shutdown())])),
    ]  # fmt: skip


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
        + [OpaqueFunction(function=_nodes)]
    )
