import dataclasses

import numpy as np
import pytest

from hybrid_av.types import Control, EgoState, ObjectClass, PolicyAction, TrafficLightState


def test_as_mpc_vector_order():
    e = EgoState(t=0.0, x=1.0, y=2.0, psi=0.3, vx=10.0, vy=0.1, r=0.05)
    np.testing.assert_array_equal(e.as_mpc_vector(), [10.0, 0.1, 0.05, 1.0, 2.0, 0.3])
    assert e.as_mpc_vector().dtype == np.float64


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Control(delta=0.0, ax=0.0).ax = 1.0


def test_enum_values():
    assert [s.value for s in TrafficLightState] == [0, 1, 2, 3]
    assert ObjectClass.PEDESTRIAN == 3 and ObjectClass.CYCLIST == 4


def test_policy_action_fields():
    assert [f.name for f in dataclasses.fields(PolicyAction)] == ["alpha_v", "delta_y", "brake"]
