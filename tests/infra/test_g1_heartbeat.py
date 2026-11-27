import csv
import json
import time

from tests.infra.g1util import G1, need_bash4, run, stub

pytestmark = need_bash4

KEYS = {
    "ts", "pod_id", "host", "elapsed_s", "remaining_s", "current_step", "steps_done",
    "steps_failed", "cost_usd_so_far", "gpu", "ram", "disk", "flags",
}  # fmt: skip


def _go(tmp_path, deadline):
    b = tmp_path / "bin"
    stub(b, "nvidia-smi", "echo '37, 5120, 24576, 61, 180.5'")
    env = {
        "G1_DIR": str(tmp_path / "g1"),
        "POD_DEADLINE_EPOCH": str(deadline),
        "RUNPOD_POD_ID": "abc",
        "G1_ONCE": "1",
    }
    return run(G1 / "heartbeat.sh", [], env, b)


def test_status_has_all_keys(tmp_path):
    deadline = int(time.time()) + 30000
    state = tmp_path / "g1"
    state.mkdir()
    (state / "state.json").write_text(
        json.dumps(
            {"steps": {"a": {"status": "done"}, "b": {"status": "failed"}, "c": {"status": "running"}}}
        )
    )
    (state / "flags").mkdir()
    (state / "flags" / "warn_30min").touch()
    assert _go(tmp_path, deadline).returncode == 0
    s = json.loads((state / "status.json").read_text())
    assert KEYS <= set(s)
    assert abs(s["remaining_s"] - 30000) <= 2
    assert abs(s["elapsed_s"] - 6000) <= 2
    assert (s["current_step"], s["steps_done"], s["steps_failed"]) == ("c", 1, 1)
    assert s["gpu"]["util_pct"] == 37 and s["gpu"]["mem_used_mib"] == 5120
    assert s["flags"] == ["warn_30min"] and s["pod_id"] == "abc"


def test_metrics_header_written_once(tmp_path):
    deadline = int(time.time()) + 30000
    _go(tmp_path, deadline)
    _go(tmp_path, deadline)
    rows = list(csv.reader((tmp_path / "g1" / "metrics.csv").open()))
    assert rows[0][0] == "ts" and rows[0][-1] == "carla_running" and len(rows) == 3
    assert all(len(r) == len(rows[0]) for r in rows)


def test_status_written_atomically(tmp_path):
    _go(tmp_path, int(time.time()) + 100)
    assert (tmp_path / "g1" / "status.json").exists()
    assert not (tmp_path / "g1" / "status.json.tmp").exists()
