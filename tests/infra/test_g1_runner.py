import fcntl
import json
import time

from tests.infra.g1util import G1, calls, need_bash4, run, stub

pytestmark = need_bash4
NOW = 1_000_000


def _setup(tmp_path, cmds=("true", "false", "sleep 0"), gating=("1", "1", "1"), budget=5, now=NOW):
    g1 = tmp_path / "g1"
    steps = tmp_path / "steps.tsv"
    # each command also appends its step name to a trace file so tests can see what ran
    rows = [f"s{i}\t{budget}\t{gating[i]}\techo s{i} >> {tmp_path}/trace; {c}" for i, c in enumerate(cmds)]
    steps.write_text("\n".join(rows) + "\n")
    env = {
        "G1_DIR": str(g1),
        "G1_STEPS": str(steps),
        "POD_DEADLINE_EPOCH": str(NOW + 36000),
        "G1_NOW_CMD": f"echo {now}",
        "RUNPOD_POD_ID": "abc",
    }
    return g1, env, tmp_path / "bin"


def _trace(tmp_path):
    f = tmp_path / "trace"
    return f.read_text().split() if f.exists() else []


def _state(g1):
    return json.loads((g1 / "state.json").read_text())


def _r(env, b, *args):
    return run(G1 / "run_g1.sh", list(args), env, b)


def test_done_steps_are_skipped(tmp_path):
    g1, env, b = _setup(tmp_path, ("true", "true", "true"))
    assert _r(env, b).returncode == 0
    assert _r(env, b).returncode == 0
    assert _trace(tmp_path) == ["s0", "s1", "s2"]
    st = _state(g1)
    assert st["deadline_epoch"] == NOW + 36000 and st["pod_id"] == "abc" and st["run_id"]
    assert st["steps"]["s0"]["status"] == "done" and st["steps"]["s0"]["exit_code"] == 0
    assert (g1 / "logs" / "s0.log").exists()


def test_running_step_is_rerun_after_restart(tmp_path):
    g1, env, b = _setup(tmp_path, ("true", "true", "true"))
    g1.mkdir()
    (g1 / "state.json").write_text(json.dumps({"steps": {"s0": {"status": "running"}}}))
    assert _r(env, b).returncode == 0
    assert _trace(tmp_path)[0] == "s0"
    assert _state(g1)["steps"]["s0"]["status"] == "done"


def test_gate_failure_stops_with_10(tmp_path):
    g1, env, b = _setup(tmp_path)
    assert _r(env, b).returncode == 10
    assert _trace(tmp_path) == ["s0", "s1"]
    assert _state(g1)["steps"]["s1"]["status"] == "failed"


def test_nongating_failure_continues(tmp_path):
    _, env, b = _setup(tmp_path, gating=("1", "0", "1"))
    assert _r(env, b).returncode == 0
    assert _trace(tmp_path) == ["s0", "s1", "s2"]


def test_step_deferred_when_budget_exceeds_remaining(tmp_path):
    # 24 min left, budget 5 + 20 > 24
    g1, env, b = _setup(tmp_path, ("true",), ("1",), now=NOW + 36000 - 24 * 60)
    assert _r(env, b).returncode == 11
    assert _trace(tmp_path) == []
    assert _state(g1)["steps"]["s0"]["status"] == "deferred_time"


def _hold_lock(g1):
    g1.mkdir(exist_ok=True)
    f = open(g1 / ".lock", "w")  # noqa: SIM115 (held by the caller)
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def test_second_instance_exits_3(tmp_path):
    g1, env, b = _setup(tmp_path, ("true",), ("1",))
    f = _hold_lock(g1)
    try:
        assert _r(env, b).returncode == 3
    finally:
        f.close()
    assert _trace(tmp_path) == []


def test_checkpoint_writes_copy(tmp_path):
    g1, env, b = _setup(tmp_path, ("true",), ("1",))
    _r(env, b)
    stub(b, "sync")
    assert _r(env, b, "--checkpoint").returncode == 0
    cp = json.loads((g1 / "checkpoint.json").read_text())
    assert cp["steps"] == _state(g1)["steps"] and cp["ts"]
    assert calls(b, "sync") == [""]


def test_checkpoint_works_while_locked(tmp_path):
    g1, env, b = _setup(tmp_path, ("true",), ("1",))
    f = _hold_lock(g1)
    try:
        assert _r(env, b, "--checkpoint").returncode == 0
        assert _r(env, b, "--status").returncode == 0
    finally:
        f.close()
    assert (g1 / "checkpoint.json").exists()


def test_from_and_only_and_status(tmp_path):
    _, env, b = _setup(tmp_path, ("true", "true", "true"))
    _r(env, b)
    (tmp_path / "trace").unlink()
    assert _r(env, b, "--from", "s1").returncode == 0
    assert _trace(tmp_path) == ["s1", "s2"]
    (tmp_path / "trace").unlink()
    assert _r(env, b, "--only", "s0").returncode == 0
    assert _trace(tmp_path) == ["s0"]
    out = _r(env, b, "--status").stdout
    assert "done=3" in out and out.count("\n") == 1


def test_timeout_counts_as_failed(tmp_path):
    _, env, b = _setup(tmp_path, ("sleep 5",), ("1",), budget=1)
    env["G1_BUDGET_UNIT"] = "s"  # test hook: budget_min counted in seconds
    t = time.time()
    assert _r(env, b, "--only", "s0").returncode == 10
    assert time.time() - t < 4
