"""Sensor rig spawn and frame-matched sensor queues (simulation-backends.md §2). `carla` is imported lazily."""

import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from hybrid_av.sim.carla_backend import convert


@dataclass(frozen=True)
class SensorSpec:
    id: str
    blueprint: str
    attributes: dict[str, str]
    x: float  # mount, library frame relative to base_link (m, rad)
    y: float
    z: float
    yaw: float
    pitch: float = 0.0


def load_rig(path) -> list[SensorSpec]:
    return [
        SensorSpec(
            s["id"],
            s["blueprint"],
            {k: str(v) for k, v in s["attributes"].items()},
            s["mount"]["x"],
            s["mount"]["y"],
            s["mount"]["z"],
            s["mount"]["yaw"],
            s["mount"].get("pitch", 0.0),
        )
        for s in yaml.safe_load(Path(path).read_text())["sensors"]
    ]


@dataclass
class SensorQueues:
    """One queue per sensor, matched on `frame`: a missing sensor yields None and a counter, never an old frame."""

    ids: Sequence[str]
    multi: Sequence[str] = ("imu",)  # ECC-07: 1 or 5 IMU samples per tick, all returned as a list
    missing: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        self._q: dict[str, list] = {i: [] for i in self.ids}
        self._cv = threading.Condition()

    def callback(self, sensor_id: str) -> Callable[[Any], None]:
        def put(data):
            with self._cv:
                self._q[sensor_id].append(data)
                self._cv.notify_all()

        return put

    def collect(self, frame: int, timeout: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        with self._cv:
            # ECC-11: camera data can arrive after world.tick() returns, so wait up to `timeout` for every sensor
            self._cv.wait_for(
                lambda: all(any(d.frame >= frame for d in q) for q in self._q.values()),
                max(deadline - time.monotonic(), 0.0),
            )
            out: dict[str, Any] = {}
            for i, q in self._q.items():
                take = [d for d in q if d.frame <= frame]
                self._q[i] = [d for d in q if d.frame > frame]  # older data is discarded, future data kept
                hit = [d for d in take if d.frame == frame]
                if i in self.multi:
                    out[i] = take or None
                else:
                    out[i] = hit[-1] if hit else None
                if not hit:
                    self.missing[i] += 1
            return out


def spawn_rig(session, ego, specs: Sequence[SensorSpec], queues: SensorQueues) -> list:
    import carla

    bp_lib = session.world.get_blueprint_library()
    sensors = []
    for spec in specs:
        bp = bp_lib.find(spec.blueprint)
        for key, value in spec.attributes.items():
            bp.set_attribute(key, value)
        loc, yaw = convert.mount_to_carla(spec.x, spec.y, spec.z, spec.yaw)
        # ECC-12: mounts are relative to the actor origin (what attach_to uses); pitch flips sign like yaw
        pitch = convert.lib_to_yaw(spec.pitch)
        transform = carla.Transform(carla.Location(*loc), carla.Rotation(pitch=pitch, yaw=yaw))
        sensor = session.world.spawn_actor(bp, transform, attach_to=ego)
        session.actors.append(sensor)  # tracked first so cleanup finds it even if listen() fails
        sensor.listen(queues.callback(spec.id))
        sensors.append(sensor)
    return sensors
