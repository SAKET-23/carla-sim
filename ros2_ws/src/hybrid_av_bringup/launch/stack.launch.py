"""stack.launch.py: the hybrid_av ROS 2 graph (interfaces.md section 14) on LiteSim or CARLA."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

ARGS = {  # name -> default; empty string means "not set"
    "sim": "carla",
    "mode": "ground_truth",
    "policy": "constant",
    "scenario": "nominal",
    "seed": "100000",
    "seg_backend": "gt",
    "det_backend": "gt",
    "foxglove": "true",
    "kiss_icp": "",  # empty: true iff mode == perception
    "max_ticks": "0",
    "result_path": "",
    "rig": "",  # empty: configs/sensors/carla_rig.yaml of the repo ($HYBRID_AV_ROOT, else found from the source tree)
}


def _repo_rig() -> str:
    root = os.environ.get("HYBRID_AV_ROOT")
    if root:
        return str(Path(root) / "configs/sensors/carla_rig.yaml")
    here = Path(__file__).resolve()  # symlink-install: this is the file in the source tree
    for parent in here.parents:
        if (parent / "configs/sensors/carla_rig.yaml").exists():
            return str(parent / "configs/sensors/carla_rig.yaml")
    raise RuntimeError("configs/sensors/carla_rig.yaml not found: set HYBRID_AV_ROOT or pass rig:=<path>")


def _truthy(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "on")


def _nodes(context):
    a = {k: context.launch_configurations[k] for k in ARGS}
    if a["sim"] not in ("litesim", "carla"):
        raise RuntimeError(f"sim must be litesim or carla, got {a['sim']!r}")
    if a["mode"] not in ("ground_truth", "perception"):
        raise RuntimeError(f"mode must be ground_truth or perception, got {a['mode']!r}")
    perception = a["mode"] == "perception"
    if perception and a["sim"] == "litesim":
        raise RuntimeError("perception mode needs sim:=carla (LiteSim has no sensors)")
    rig = a["rig"] or _repo_rig()
    kiss = _truthy(a["kiss_icp"]) if a["kiss_icp"] else perception

    sim_time = {"use_sim_time": True}

    def node(package, executable, name=None, **params):
        return Node(
            package=package, executable=executable, name=name or executable, output="screen",
            parameters=[{**sim_time, **params}],
        )  # fmt: skip

    owner_params = {
        "scenario": a["scenario"], "seed": int(a["seed"]), "max_ticks": int(a["max_ticks"]),
        "result_path": a["result_path"],
    }  # fmt: skip
    owner = Node(  # the tick owner publishes /clock: no use_sim_time
        package="hybrid_av_nodes", executable="litesim_node" if a["sim"] == "litesim" else "carla_bridge",
        name="litesim_node" if a["sim"] == "litesim" else "carla_bridge", output="screen",
        parameters=[{**owner_params, "strict_peers": True, **({"sensors": rig} if a["sim"] == "carla" else {})}],
    )  # fmt: skip
    actions = [
        node("hybrid_av_bringup", "static_tf", rig=rig),
        node("hybrid_av_nodes", "policy_node", policy=a["policy"]),
        node("hybrid_av_nodes", "mpc_node", mode=a["mode"], peer_wait_s=30.0),
        node("hybrid_av_nodes", "localization_node", mode=a["mode"]),
    ]
    if perception:
        actions += [
            node("hybrid_av_nodes", "segmentation_node", backend=a["seg_backend"]),
            node("hybrid_av_nodes", "detection_node", backend=a["det_backend"], rig=rig),
            node("hybrid_av_nodes", "tracker_node", mode=a["mode"], rig=rig),
        ]
    if kiss:
        actions.append(
            Node(
                package="kiss_icp",
                executable="kiss_icp_node",
                name="kiss_icp_node",
                output="screen",
                parameters=[
                    os.path.join(get_package_share_directory("hybrid_av_bringup"), "config", "kiss_icp.yaml"),
                    sim_time,
                ],
                remappings=[("pointcloud_topic", "/carla/ego/lidar/points")],
            )
        )
    if _truthy(a["foxglove"]):
        actions += [
            node("foxglove_bridge", "foxglove_bridge", port=8765),
            Node(
                package="topic_tools",
                executable="throttle",
                name="camera_throttle",
                output="screen",
                arguments=["messages", "/carla/ego/camera/front/image", "5.0", "/viz/camera_throttled"],
                parameters=[sim_time],
            ),
        ]
    # the tick owner starts last so the peers exist; the launch ends when it exits
    return [*actions, owner,
            RegisterEventHandler(OnProcessExit(target_action=owner, on_exit=[EmitEvent(event=Shutdown())]))]  # fmt: skip


def generate_launch_description():
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
        + [OpaqueFunction(function=_nodes)]
    )
