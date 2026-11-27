from tests.infra.g1util import G1, calls, need_bash4, run, stub

pytestmark = need_bash4
DEADLINE = 1_000_000


def _env(tmp_path, now, key=True):
    b = tmp_path / "bin"
    stub(b, "curl", "echo 200")
    stub(b, "runner", "")
    env = {
        "G1_DIR": str(tmp_path / "g1"),
        "POD_DEADLINE_EPOCH": str(DEADLINE),
        "RUNPOD_POD_ID": "abc",
        "G1_NOW_CMD": f"echo {now}",
        "G1_ONCE": "1",
        "G1_RUNNER": str(b / "runner"),
    }
    if key:
        env["RUNPOD_API_KEY"] = "rpa_TESTKEY"
    return b, env


def test_flags_written_at_thresholds(tmp_path):
    b, env = _env(tmp_path, DEADLINE - 1799)
    assert run(G1 / "deadline.sh", [], env, b).returncode == 0
    flags = tmp_path / "g1" / "flags"
    assert (flags / "warn_30min").exists() and not (flags / "final_checkpoint").exists()
    b, env = _env(tmp_path, DEADLINE - 899)
    assert run(G1 / "deadline.sh", [], env, b).returncode == 0
    assert (flags / "final_checkpoint").exists()
    assert calls(b, "runner") == ["--checkpoint"]
    assert not calls(b, "curl")


def test_deadline_stops_without_mac(tmp_path):
    b, env = _env(tmp_path, DEADLINE)
    assert run(G1 / "deadline.sh", [], env, b).returncode == 0
    (line,) = calls(b, "curl")
    assert "-X POST" in line and "https://rest.runpod.io/v1/pods/abc/stop" in line
    assert "Authorization: Bearer rpa_TESTKEY" in line


def test_deadline_without_key_flags_status(tmp_path):
    b, env = _env(tmp_path, DEADLINE, key=False)
    run(G1 / "deadline.sh", [], env, b)
    g1 = tmp_path / "g1"
    assert "NO POD KEY: Mac watcher is the only stop path" in (g1 / "logs/deadline.log").read_text()
    assert (g1 / "flags/no_pod_key").exists()
    assert not calls(b, "curl")


def test_missing_deadline_exits_2(tmp_path):
    b, env = _env(tmp_path, DEADLINE)
    del env["POD_DEADLINE_EPOCH"]
    assert run(G1 / "deadline.sh", [], env, b).returncode == 2
