import json

from tests.infra.g1util import G1, calls, need_bash4, run, stub

pytestmark = need_bash4


def _smi(b, driver="580.65.06", name="NVIDIA GeForce RTX 4090"):
    stub(
        b,
        "nvidia-smi",
        f'case "$*" in *--query-gpu*) echo "{name}, {driver}";; '
        '*) echo "| NVIDIA-SMI | CUDA Version: 13.0 |";; esac',
    )


def _pre_env(tmp_path, **kw):
    env = {
        "G1_DIR": str(tmp_path / "g1"),
        "G1_WORKSPACE": str(tmp_path),
        "G1_MIN_FREE_GB": "0",  # test hook: the 40 GB floor is not testable on a CI disk
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "POD_DEADLINE_EPOCH": "1036000",
    }
    env.update(kw)
    return env


def test_preflight_fails_on_old_driver(tmp_path):
    b = tmp_path / "bin"
    _smi(b, driver="570.1")
    stub(b, "python", 'echo "2.14.1 12.8 1.0"')
    r = run(G1 / "preflight.sh", [], _pre_env(tmp_path), b)
    assert r.returncode == 1 and "driver" in r.stderr and "570" in r.stderr
    assert json.loads((tmp_path / "g1" / "hostinfo.json").read_text())["env_ok"] is False


def test_preflight_writes_hostinfo(tmp_path):
    b = tmp_path / "bin"
    _smi(b)
    stub(b, "python", 'echo "2.14.1 12.8 1.0"')
    r = run(G1 / "preflight.sh", [], _pre_env(tmp_path), b)
    assert r.returncode == 0, r.stderr
    h = json.loads((tmp_path / "g1" / "hostinfo.json").read_text())
    assert h["gpu_name"] == "NVIDIA GeForce RTX 4090" and h["driver_version"] == "580.65.06"
    assert h["cuda_version"] == "13.0" and h["torch_cuda_version"] == "12.8" and h["env_ok"] is True
    assert {"vcpu", "ram_gb", "workspace_free_gb"} <= set(h)


def test_preflight_cuda_fault_exits_20(tmp_path):
    b = tmp_path / "bin"
    _smi(b)
    stub(b, "python", 'echo "CUDA unknown error" >&2; exit 1')
    r = run(G1 / "preflight.sh", [], _pre_env(tmp_path), b)
    assert r.returncode == 20 and "HOST GPU FAULT" in r.stderr


def test_measure_replaces_existing_row(tmp_path):
    env = {
        "G1_DIR": str(tmp_path / "g1"),
        "G1_MEASURE_DRY": "1",
        "G1_REPO": str(tmp_path / "norepo"),
    }
    for _ in range(2):
        assert run(G1 / "measure.sh", [], env, tmp_path / "bin").returncode == 0
    rows = (tmp_path / "g1" / "vram.csv").read_text().splitlines()
    assert rows[0] == "town,quality,gpu_mem_peak_mib,ram_peak_mib,cpu_mean_pct,tick_hz_mean,ticks"
    keys = [r.split(",")[:2] for r in rows[1:]]
    assert len(keys) == 5 and len({tuple(k) for k in keys}) == 5
    assert ["Town10HD", "Epic"] in keys
    assert "deferred to G2 IC-40" in rows[-1]


def _summary(tmp_path, statuses):
    g1 = tmp_path / "g1"
    g1.mkdir(parents=True)
    steps = {n: {"status": s, "exit_code": 0 if s == "done" else 1} for n, s in statuses.items()}
    (g1 / "state.json").write_text(json.dumps({"run_id": "r", "deadline_epoch": 1036000, "steps": steps}))
    (g1 / "pins.txt").write_text("2.14.1 True NVIDIA GeForce RTX 4090\n")
    (g1 / "pytest_not_carla.xml").write_text(
        '<testsuites><testsuite tests="9" failures="1" errors="0" skipped="2"/></testsuites>'
    )
    env = {"G1_DIR": str(g1), "G1_NOW": "1010000"}
    r = run(G1 / "summary.py", [], env, tmp_path / "bin")
    assert r.returncode == 0, r.stderr
    return (g1 / "g1_summary.md").read_text()


def test_summary_go_verdict(tmp_path):
    ok = {
        "02_smoke01_gpu": "done",
        "03_smoke02_vulkan": "done",
        "04_smoke03_carla": "done",
        "05_smoke04_client": "done",
    }
    text = _summary(tmp_path, ok)
    assert "**GO**" in text and "NO-GO" not in text
    assert "2.78 h" in text  # 10000 s after the start
    text = _summary(tmp_path / "x", {**ok, "03_smoke02_vulkan": "failed"})
    assert "**NO-GO**" in text and "03_smoke02_vulkan" in text


def test_start_g1_idempotent(tmp_path):
    b = tmp_path / "bin"
    stub(b, "tmux", "exit 0")  # has-session succeeds
    assert run(G1 / "start_g1.sh", [], {}, b).returncode == 0
    assert calls(b, "tmux") == ["has-session -t g1"]
    b2 = tmp_path / "bin2"
    stub(b2, "tmux", '[ "$1" = has-session ] && exit 1; exit 0')
    assert run(G1 / "start_g1.sh", [], {}, b2).returncode == 0
    c = calls(b2, "tmux")
    assert (
        sum(x.startswith("new-session") for x in c) == 1 and sum(x.startswith("new-window") for x in c) == 2
    )
