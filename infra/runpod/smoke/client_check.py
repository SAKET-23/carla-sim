"""Smoke 04 (runbook.md section 4): server and client are 0.9.16, a camera renders, actors and sync mode are cleaned up.

Loads Town03_Opt, spawns vehicle.tesla.model3 and a 1280x720 RGB camera, ticks 20 times in synchronous mode and saves
/workspace/logs/smoke04.png. Pass: mean pixel > 10 and standard deviation > 5.
"""

import queue
import sys

import numpy as np

EXPECTED = "0.9.16"
OUT = "/workspace/logs/smoke04.png"


def main() -> int:
    import carla

    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    server, wheel = client.get_server_version(), client.get_client_version()
    print(f"server {server}, client {wheel}")
    if server != EXPECTED or wheel != EXPECTED:  # ECC-01
        print(f"FAIL: both versions must be {EXPECTED}", file=sys.stderr)
        return 1
    world = client.load_world("Town03_Opt")  # ECC-02: settings are applied after load_world
    actors, last = [], None
    try:
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = True, 0.05
        world.apply_settings(settings)
        lib = world.get_blueprint_library()
        car = world.spawn_actor(lib.find("vehicle.tesla.model3"), world.get_map().get_spawn_points()[0])
        actors.append(car)
        bp = lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", "1280")
        bp.set_attribute("image_size_y", "720")
        cam = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.5, z=1.6)), attach_to=car)
        actors.append(cam)
        frames = queue.Queue()
        cam.listen(frames.put)
        for _ in range(20):
            world.tick()
            last = frames.get(timeout=10.0)
        last.save_to_disk(OUT)
        px = np.frombuffer(last.raw_data, dtype=np.uint8).reshape(last.height, last.width, 4)[..., :3]
        mean, std = float(px.mean()), float(px.std())
        print(f"last frame mean {mean:.1f}, std {std:.1f}, saved {OUT}")
        if mean <= 10 or std <= 5:
            print("FAIL: frame is black or flat", file=sys.stderr)
            return 1
        return 0
    finally:  # sensors stop before destroy (ECC-03); the server must not stay synchronous
        for a in reversed(actors):
            try:
                if hasattr(a, "stop"):
                    a.stop()
                a.destroy()
            except RuntimeError:
                pass
        settings = world.get_settings()
        settings.synchronous_mode, settings.fixed_delta_seconds = False, None
        world.apply_settings(settings)


if __name__ == "__main__":
    sys.exit(main())
