import json
import subprocess
import sys


def test_train_cli_tiny(tmp_path):
    subprocess.run([sys.executable, "scripts/train_ppo.py", "--config", "configs/ppo/litesim_nominal.yaml",
                    "--bc-steps", "200", "--envs", "1", "--total-steps", "512", "--out", str(tmp_path)], check=True)  # fmt: skip
    run = next(tmp_path.iterdir())
    m = json.loads((run / "metrics.json").read_text())
    assert (run / "actor_final.pt").exists() and m["total_steps"] >= 512 and m["ppo_wall_s"] > 0


def test_train_cli_dagger_tiny(tmp_path):
    subprocess.run([sys.executable, "scripts/train_ppo.py", "--config", "configs/ppo/litesim_nominal.yaml",
                    "--bc-steps", "200", "--dagger-rounds", "1", "--dagger-steps", "200", "--envs", "1",
                    "--total-steps", "0", "--out", str(tmp_path)], check=True)  # fmt: skip
    run = next(tmp_path.iterdir())
    m = json.loads((run / "metrics.json").read_text())
    assert (run / "actor_final.pt").exists() and m["dagger_wall_s"] > 0 and len(m["dagger"]) == 1
