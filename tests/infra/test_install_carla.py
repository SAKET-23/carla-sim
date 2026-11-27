import subprocess

from tests.infra.g1util import ROOT, calls, stub

SCRIPT = ROOT / "infra" / "runpod" / "install_carla.sh"


def test_fetch_keeps_content_length_digits_over_2gb(tmp_path):
    b = tmp_path / "bin"
    stub(b, "curl", "printf 'HTTP/1.1 200 OK\\r\\nContent-Length: 8346095504\\r\\n\\r\\n'")
    stub(b, "wget", "")
    # Source only fetch() (the script hardcodes /workspace); wget writes nothing, so the size check ends in die.
    cmd = f'die() {{ echo "$*" >&2; exit 1; }}; BASE=http://x; DL={tmp_path}; eval "$(sed -n \'/^fetch() {{/,/^}}/p\' {SCRIPT})"; fetch f.tar.gz'
    r = subprocess.run(
        ["bash", "-c", cmd], env={"PATH": f"{b}:/usr/bin:/bin"}, capture_output=True, text=True, check=False
    )
    assert "no Content-Length" not in r.stderr
    assert calls(b, "wget")


def test_tar_ignores_archive_ownership():
    tars = [ln for ln in SCRIPT.read_text().splitlines() if "| tar " in ln]
    assert len(tars) == 2
    assert all("--no-same-owner" in ln for ln in tars)


def test_chown_failure_is_tolerated(tmp_path):
    b = tmp_path / "bin"
    stub(b, "chown", "exit 1")
    cmd = f'set -euo pipefail; CARLA_ROOT=/x; eval "$(grep "^chown " {SCRIPT})"; echo survived'
    r = subprocess.run(
        ["bash", "-c", cmd], env={"PATH": f"{b}:/usr/bin:/bin"}, capture_output=True, text=True, check=False
    )
    assert "survived" in r.stdout
    assert "chown not supported" in r.stdout


def test_entrypoint_env_file_passes_g1_vars(tmp_path):
    ep = ROOT / "infra" / "runpod" / "entrypoint.sh"
    out = tmp_path / "env"
    cmd = (
        f"out={out}; : > $out; "
        f'eval "$(sed -n \'/for var in \\$(compgen/,/^  done/p\' {ep} | sed "s#/etc/carla-sim.env#$out#")"; cat $out'
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "POD_DEADLINE_EPOCH": "123",
        "G1_EXPECT_GPU": "A40",
        "SMOKE_MIN_VRAM_MIB": "1",
        "OTHER_VAR": "no",
    }
    r = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True, check=False)
    assert "POD_DEADLINE_EPOCH=123" in r.stdout
    assert "G1_EXPECT_GPU=A40" in r.stdout
    assert "SMOKE_MIN_VRAM_MIB=1" in r.stdout
    assert "OTHER_VAR" not in r.stdout


def test_smoke_vram_floor_overridable():
    text = (ROOT / "infra" / "runpod" / "smoke" / "01_gpu.sh").read_text()
    assert "${SMOKE_MIN_VRAM_MIB:-23552}" in text
    assert '-ge "$min"' in text
