# ruff: noqa: C408
import math

import pytest
from pydantic import ValidationError

from hybrid_av.weightspace.schema import TENSOR_SHAPES, EditPlan, WeightEdit, WeightSpaceConfig


def _e(**kw):
    base = dict(tensor="fc1.weight", index=[0, 7], op="add", value=0.5, rationale="r")
    base.update(kw)
    return WeightEdit(**base)


def test_valid_weight_and_bias_edits():
    assert _e().index == [0, 7]
    assert _e(tensor="fc3.bias", index=[2]).index == [2]


def test_unknown_tensor_rejected():
    with pytest.raises(ValidationError):
        _e(tensor="fc4.weight")


def test_weight_index_out_of_range_rejected():
    for t, i in [
        ("fc1.weight", [64, 0]),
        ("fc3.weight", [3, 0]),
        ("fc2.bias", [-1]),
        ("fc1.weight", [0, 24]),
    ]:
        with pytest.raises(ValidationError, match="out of range"):
            _e(tensor=t, index=i)


def test_index_arity_must_match_tensor():
    with pytest.raises(ValidationError):
        _e(tensor="fc1.weight", index=[3])
    with pytest.raises(ValidationError):
        _e(tensor="fc1.bias", index=[3, 0])


def test_non_finite_value_rejected():
    for bad in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValidationError):
            _e(value=bad)


def test_plan_bounds_and_duplicates_allowed():
    with pytest.raises(ValidationError):
        EditPlan(edits=[], hypothesis="h", expected_effect="e")
    p = EditPlan(edits=[_e(), _e(op="scale", value=2.0)], hypothesis="h", expected_effect="e")
    assert len(p.edits) == 2


def test_plan_json_roundtrip():
    p = EditPlan(edits=[_e()], hypothesis="h", expected_effect="e")
    assert EditPlan.model_validate_json(p.model_dump_json()) == p


def test_shapes_total_param_count():
    import numpy as np

    assert sum(int(np.prod(s)) for s in TENSOR_SHAPES.values()) == 5955


def test_config_pinned_values():
    c = WeightSpaceConfig()
    assert (c.max_edits, c.retention_threshold, c.ood_success_threshold, c.max_value_factor) == (
        208,
        0.95,
        0.8,
        3.0,
    )
    assert c.model == "claude-opus-5-5" and c.max_input_tokens_total == 3_000_000
    assert c.bundle_seeds == (50000, 50009) and c.gate_ood_seeds == (50010, 50049)


def test_yaml_matches_dataclass_defaults():
    from hybrid_av.config import load_dataclass

    assert load_dataclass(WeightSpaceConfig, "configs/weightspace/default.yaml") == WeightSpaceConfig()
