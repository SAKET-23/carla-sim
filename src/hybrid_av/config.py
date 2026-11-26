"""YAML to dataclass loading, seeding and config hashing."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
import platform
import random
import subprocess
import sys
import types
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import numpy as np
import yaml

T = TypeVar("T")


def _coerce(value, tp):
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None:
            return None
        members = [a for a in typing.get_args(tp) if a is not type(None)]
        return _coerce(value, members[0]) if len(members) == 1 else value
    if origin is tuple and isinstance(value, list):
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            args = args[:1] * len(value)
        if len(args) != len(value):
            return tuple(value)
        return tuple(_coerce(v, a) for v, a in zip(value, args))
    if tp is Path and isinstance(value, str):
        return Path(value)
    if tp is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


def load_dataclass(cls: type[T], path: str | Path, overrides: dict | None = None) -> T:
    data = yaml.safe_load(Path(path).read_text()) or {}
    data = {**data, **(overrides or {})}
    hints = typing.get_type_hints(cls)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        raise ValueError(f"unknown keys {unknown} for {cls.__name__}")
    return cls(**{k: _coerce(v, hints[k]) for k, v in data.items()})


def load_run_config(path: Path) -> dict:
    path = Path(path)
    out = yaml.safe_load(path.read_text()) or {}
    for k, v in out.items():
        if k.endswith("_config") and isinstance(v, str) and v.endswith(".yaml"):
            p = Path(v)
            if not p.exists():
                p = path.parent / v
            out[k] = load_run_config(p)
    return out


def seed_everything(seed: int) -> np.random.Generator:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def _jsonable(obj):
    return dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj


def config_hash(obj) -> str:
    s = json.dumps(_jsonable(obj), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(s.encode()).hexdigest()


def _git_info() -> str:
    try:
        out = []
        for cmd in (["rev-parse", "HEAD"], ["diff", "--stat"]):
            r = subprocess.run(["git", *cmd], capture_output=True, text=True, check=False)
            if r.returncode != 0:
                return "no git"
            out.append(r.stdout.strip())
        return "\n".join(out) + "\n"
    except OSError:
        return "no git"


def _env_info() -> str:
    lines = [f"python {sys.version.split()[0]}", f"numpy {np.__version__}"]
    for name in ("torch", "osqp"):
        try:
            lines.append(f"{name} {importlib.import_module(name).__version__}")
        except ImportError:
            lines.append(f"{name} not installed")
    lines.append(platform.platform())
    return "\n".join(lines) + "\n"


def make_run_dir(name: str, configs: dict[str, object], root: Path = Path("runs")) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = Path(root) / f"{stamp}_{name}"
    d, n = base, 1
    while True:
        try:
            d.mkdir(parents=True)
            break
        except FileExistsError:
            n += 1
            d = base.with_name(f"{base.name}_{n}")
    cfg = {k: _jsonable(v) for k, v in configs.items()}
    cfg["hashes"] = {k: config_hash(v) for k, v in configs.items()}
    (d / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True, default=str))
    (d / "git.txt").write_text(_git_info())
    (d / "env.txt").write_text(_env_info())
    return d
