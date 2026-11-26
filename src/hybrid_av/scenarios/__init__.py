"""Seeded scenario builders; ranges live in configs/scenarios/*.yaml (Plan/02-system-design/scenarios.md)."""

from pathlib import Path
from typing import Literal

import yaml

from hybrid_av.sim.base import Scenario

# configs are located from the source checkout (parents[3]); an installed wheel will not find them
_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "scenarios"


def load_config(family: str, overrides: dict | None) -> dict:
    cfg = yaml.safe_load((_CONFIGS / f"{family}.yaml").read_text())
    unknown = sorted(set(overrides or {}) - set(cfg))
    if unknown:
        raise ValueError(f"unknown overrides {unknown} for scenario {family!r}")
    return {**cfg, **(overrides or {})}


def make_scenario(
    family: Literal["nominal", "wet_curve", "merge_deadlock"], seed: int, overrides: dict | None = None
) -> Scenario:
    from hybrid_av.scenarios import merge_deadlock, nominal, wet_curve

    build = {"nominal": nominal, "wet_curve": wet_curve, "merge_deadlock": merge_deadlock}[family].build
    return build(seed, load_config(family, overrides))
