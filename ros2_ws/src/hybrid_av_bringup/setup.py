from glob import glob

from setuptools import find_packages, setup

package_name = "hybrid_av_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="dev@example.com",
    description="Launch files, static TF and configs for the hybrid_av ROS 2 graph",
    license="MIT",
    tests_require=["pytest"],
    extras_require={"test": ["pytest"]},
    entry_points={"console_scripts": ["static_tf = hybrid_av_bringup.static_tf:main"]},
)
