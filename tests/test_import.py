import subprocess
import sys


def test_import_has_no_heavy_deps():
    code = "import sys, hybrid_av; print(','.join(m for m in ('carla','rclpy','tensorrt','anthropic') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_version():
    import hybrid_av

    assert hybrid_av.__version__ == "0.1.0"
