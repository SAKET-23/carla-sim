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
