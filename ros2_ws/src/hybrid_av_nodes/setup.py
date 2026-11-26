from setuptools import find_packages, setup

package_name = "hybrid_av_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="dev@example.com",
    description="ROS 2 nodes wrapping the hybrid_av library",
    license="MIT",
    tests_require=["pytest"],
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "litesim_node = hybrid_av_nodes.litesim_node:main",
            "carla_bridge = hybrid_av_nodes.carla_bridge:main",
            "command_responder = hybrid_av_nodes.command_responder:main",
            "policy_node = hybrid_av_nodes.policy_node:main",
            "mpc_node = hybrid_av_nodes.mpc_node:main",
            "localization_node = hybrid_av_nodes.localization_node:main",
            "segmentation_node = hybrid_av_nodes.segmentation_node:main",
            "detection_node = hybrid_av_nodes.detection_node:main",
            "tracker_node = hybrid_av_nodes.tracker_node:main",
        ]
    },
)
