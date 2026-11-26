import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pytest
import torch

from hybrid_av import config as config_mod
from hybrid_av.config import (
    config_hash,
    load_dataclass,
    load_run_config,
    make_run_dir,
    seed_everything,
)
from hybrid_av.mpc.controller import MPCConfig
from hybrid_av.planning.reference import ReferenceConfig
from hybrid_av.rl.ppo import PPOConfig
from hybrid_av.rl.reward import RewardConfig
from hybrid_av.tracking.tracker import TrackerConfig
from hybrid_av.vehicle.actuation import LongitudinalMap
from hybrid_av.vehicle.params import VehicleParams
from hybrid_av.weightspace.schema import WeightSpaceConfig


@dataclass(frozen=True, slots=True)
class _Sample:
    a: float = 1.0
    b: tuple[float, float] = (1.0, 2.0)
    name: str = "x"


FIX = "tests/fixtures/config"
# later tasks append (DataclassType, "configs/.../default.yaml")
REGISTRY: list[tuple[type, str]] = [
    (VehicleParams, "configs/vehicle/default.yaml"),
    (LongitudinalMap, "configs/vehicle/longitudinal_default.yaml"),
    (MPCConfig, "configs/mpc/default.yaml"),
    (ReferenceConfig, "configs/reference/default.yaml"),
    (RewardConfig, "configs/reward/default.yaml"),
    (PPOConfig, "configs/ppo/litesim_nominal.yaml"),
    (TrackerConfig, "configs/tracking/default.yaml"),
    (WeightSpaceConfig, "configs/weightspace/default.yaml"),
]


def test_load_and_override():
    s = load_dataclass(_Sample, f"{FIX}/sample.yaml", overrides={"a": 5.0})
    assert s == _Sample(a=5.0, b=(3.0, 4.0), name="x")


def test_list_becomes_tuple():
    s = load_dataclass(_Sample, f"{FIX}/sample.yaml")
    assert isinstance(s.b, tuple) and hash(s)


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="speeed"):
        load_dataclass(_Sample, f"{FIX}/bad_key.yaml")
    with pytest.raises(ValueError, match="nope"):
        load_dataclass(_Sample, f"{FIX}/sample.yaml", overrides={"nope": 1})


def test_seed_everything_reproducible():
    g1 = seed_everything(7)
    x1 = (g1.random(), torch.rand(1).item())
    g2 = seed_everything(7)
    x2 = (g2.random(), torch.rand(1).item())
    assert x1 == x2


def test_config_hash_stable():
    assert config_hash(_Sample()) == config_hash(asdict(_Sample()))
    assert config_hash(_Sample()) != config_hash(_Sample(a=1.5))


def test_load_run_config_nested():
    r = load_run_config(Path(f"{FIX}/run.yaml"))
    assert r["mpc_config"] == {"a": 2.0, "b": [3.0, 4.0]}
    assert r["seed"] == 3


@pytest.mark.parametrize(
    "cls,path",
    REGISTRY or [pytest.param(None, None, marks=pytest.mark.skip("registry empty"))],
)
def test_yaml_defaults_match_dataclasses(cls, path):
    assert load_dataclass(cls, path) == cls()


def test_make_run_dir(tmp_path):
    d = make_run_dir("unit", {"sample": _Sample()}, root=tmp_path)
    assert d.parent == tmp_path and d.name.endswith("_unit")
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["sample"]["a"] == 1.0 and cfg["hashes"]["sample"] == config_hash(_Sample())
    assert (d / "git.txt").exists() and "numpy" in (d / "env.txt").read_text()


def test_make_run_dir_collision(tmp_path, monkeypatch):
    class _Fixed(config_mod.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 1, 2, 3, 4, 5, tzinfo=tz)

    monkeypatch.setattr(config_mod, "datetime", _Fixed)
    d1 = make_run_dir("unit", {}, root=tmp_path)
    d2 = make_run_dir("unit", {}, root=tmp_path)
    assert d1.name == "20260102T030405Z_unit" and d2.name.endswith("_unit_2")


@dataclass(frozen=True, slots=True)
class _F:
    a: float = 1.0
    t: tuple[float, ...] = (1.0,)


@dataclass(frozen=True, slots=True)
class _P:
    p: Path | None = None
    x: float | None = None


def _load(cls, tmp_path, text):
    f = tmp_path / "c.yaml"
    f.write_text(text)
    return load_dataclass(cls, f)


def test_int_becomes_float(tmp_path):
    s = _load(_F, tmp_path, "a: 2\nt: [1, 2]\n")
    assert isinstance(s.a, float) and all(isinstance(v, float) for v in s.t)
    assert config_hash(s) == config_hash(_F(a=2.0, t=(1.0, 2.0)))


def test_optional_path(tmp_path):
    assert _load(_P, tmp_path, "p: some/dir\n").p == Path("some/dir")
    assert _load(_P, tmp_path, "p: null\n").p is None
    s = _load(_P, tmp_path, "x: 3\n")
    assert isinstance(s.x, float)
