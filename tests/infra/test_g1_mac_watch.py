import json
import re
import time

from tests.infra.g1util import G1, calls, run, stub

DEADLINE = 2_000_000
CURL = 'case "$*" in *"-X POST"*) echo 200;; *) echo \'{"status":"EXITED"}\';; esac\n'


def _iso(epoch):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def _setup(tmp_path, now, status=None, state=None, rsync_rc=0):
    """Stubs on PATH; the rsync stub copies the fixture dir into OUT (its last argument)."""
    b, fx = tmp_path / "bin", tmp_path / "fixture"
    fx.mkdir(parents=True)
    status = status or {}
    status = {
        "ts": _iso(now - 10), "current_step": "05_smoke04_client", "steps_done": 3, "steps_failed": 0,
        "gpu": {"util_pct": 37, "mem_used_mib": 5120}, "ram": {"used_mib": 1000},
        "cost_usd_so_far": 0.5, "flags": ["warn_30min"], **status,
    }  # fmt: skip
    (fx / "status.json").write_text(json.dumps(status))
    if state:
        (fx / "state.json").write_text(json.dumps(state))
    stub(b, "rsync", f'[ {rsync_rc} -ne 0 ] && exit {rsync_rc}; cp -R "{fx}/." "${{!#}}"')
    stub(b, "curl", CURL)
    stub(b, "security", "echo rpa_TESTKEY")
    stub(b, "ssh")
    out = tmp_path / "out"
    env = {"G1_NOW_CMD": f"echo {now}", "G1_ONCE": "1", "G1_CAFFEINATED": "1", "G1_POLL_S": "0"}
    return b, env, out


def _watch(b, env, out, *extra):
    args = ["--pod-id", "abc", "--ssh", "carla-pod", "--deadline", str(DEADLINE), "--out", str(out)]
    return run(G1 / "mac_watch.sh", [*args, "--interval", "0", *extra], env, b)


def _log(out):
    return (out / "watch.log").read_text()


def test_interval_line_format(tmp_path):
    now = DEADLINE - 36000 + 60
    b, env, out = _setup(tmp_path, now)
    assert _watch(b, env, out).returncode == 0
    line = _log(out).splitlines()[0]
    pat = (
        r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ step=05_smoke04_client done=3 failed=0 remaining=9:59 "
        r"gpu=37%/5120MiB ram=1000MiB cost=\$0\.5\d* flags=warn_30min$"
    )
    assert re.match(pat, line), line
    assert calls(b, "rsync") == [f"-az carla-pod:/workspace/g1/ {out}/"]
    assert not calls(b, "curl")


def test_alert_on_stale_status(tmp_path):
    now = DEADLINE - 36000
    b, env, out = _setup(tmp_path, now, status={"ts": _iso(now - 700)})
    _watch(b, env, out)
    assert "ALERT" in _log(out)
    b, env, out = _setup(tmp_path / "k", now, status={"flags": ["no_pod_key"]})
    _watch(b, env, out)
    assert "ALERT" in _log(out) and "no_pod_key" in _log(out)


def test_stop_at_deadline_minus_5min(tmp_path):
    b, env, out = _setup(tmp_path, DEADLINE - 300)
    assert _watch(b, env, out).returncode == 0
    posts = [c for c in calls(b, "curl") if "-X POST" in c]
    assert len(posts) == 1 and "https://api.runpod.io/v2/pods/abc/action" in posts[0] and "stop" in posts[0]
    assert "STOPPED" in _log(out)


def test_not_stopped_before_window(tmp_path):
    b, env, out = _setup(tmp_path, DEADLINE - 301)
    _watch(b, env, out)
    assert not calls(b, "curl")


def test_stop_happens_even_if_final_sync_fails(tmp_path):
    b, env, out = _setup(tmp_path, DEADLINE - 300, rsync_rc=255)
    assert _watch(b, env, out).returncode == 0
    assert any("-X POST" in c and "/action" in c for c in calls(b, "curl"))
    assert "STOPPED" in _log(out)


def test_stops_when_summary_done(tmp_path):
    state = {"steps": {"12_summary": {"status": "done"}}}
    b, env, out = _setup(tmp_path, DEADLINE - 36000 + 60, state=state)
    _watch(b, env, out)
    assert any("-X POST" in c and "/action" in c for c in calls(b, "curl"))


def test_dry_run_never_calls_stop(tmp_path):
    b, env, out = _setup(tmp_path, DEADLINE - 300)
    assert _watch(b, env, out, "--dry-run").returncode == 0
    assert not calls(b, "curl") and not calls(b, "security")
    assert "DRY-RUN: would stop abc" in _log(out)


def test_key_not_in_logs(tmp_path):
    b, env, out = _setup(tmp_path, DEADLINE - 300)
    r = _watch(b, env, out)
    assert "rpa_TESTKEY" not in r.stdout + r.stderr
    assert calls(b, "security")  # the key was read ...
    for f in out.rglob("*"):
        if f.is_file():
            assert "rpa_TESTKEY" not in f.read_text()  # ... and never written
