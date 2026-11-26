try:
    import rclpy  # noqa: F401
except ImportError:  # Mac without ROS: only the pure-core tests are collected
    collect_ignore_glob = ["test_*_node*.py"]
