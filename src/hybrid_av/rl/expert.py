"""Action sources; ExpertPolicy follows Plan/03-models/ppo-training.md section 3."""

from typing import Protocol

import numpy as np

from hybrid_av.features.extractor import FeatureFrame, junction_range
from hybrid_av.features.spec import (
    I_HEADING_ERROR,
    I_INTERSECTION_GAP,
    I_LEAD_DIST,
    I_LEAD_REL_SPEED,
    I_LEAD_TTC,
    I_PED_BEARING,
    I_PED_DIST,
    I_SPEED_LIMIT,
    I_VX,
)
from hybrid_av.types import PolicyAction

EXPERT_TTC_BRAKE = 1.5  # s
EXPERT_BRAKE_DECEL = 3.0  # m/s², planning decel for the pedestrian stop
EXPERT_BRAKE_STANDOFF = 5.0  # m, stopping distance margin (obstacle half-plane clearance)
EXPERT_MIN_GAP = 1.0  # s, junction commit threshold on intersection_gap
EXPERT_JUNCTION_WAIT_DIST = 6.0  # m before the junction entry where it waits
EXPERT_JUNCTION_DECEL = 1.0  # m/s², planning decel; the MPC realises ~1.45 under the 2.0 ramped reference
EXPERT_JUNCTION_STOP_DECEL = 1.45  # m/s², decel the MPC actually realises: decides whether the entry is still stoppable
EXPERT_ABORT_GAP = 0.0  # s, once moving the expert only aborts a launch when intersection_gap falls below this
EXPERT_HOLD_SPEED = 1.0  # m/s, brake hold applies below this speed
EXPERT_COMMIT_SPEED = 0.3  # m/s, hysteresis stopped/moving split: above this a launch is committed
_LEAD_RANGE, _GAP_ABSENT = 80.0, 10.0
_A_COMF = 2.0  # m/s², IDM acceleration and comfortable deceleration


class ActionSource(Protocol):
    def __call__(self, z: np.ndarray, frame: FeatureFrame) -> PolicyAction: ...


class ConstantAction:
    """The pure-MPC ablation."""

    def __call__(self, z: np.ndarray, frame: FeatureFrame) -> PolicyAction:
        return PolicyAction(1.0, 0.0, 0.0)


class ExpertPolicy:
    def __call__(self, z: np.ndarray, frame: FeatureFrame) -> PolicyAction:
        f = frame.raw
        vx, limit = f[I_VX], max(f[I_SPEED_LIMIT], 1e-3)
        alpha, brake = 1.0, 0.0
        if f[I_LEAD_DIST] < _LEAD_RANGE:  # IDM car following
            dv = -f[I_LEAD_REL_SPEED]
            s_star = 2.0 + 1.5 * vx + vx * dv / (2.0 * np.sqrt(_A_COMF * _A_COMF))
            a = _A_COMF * (1.0 - (vx / limit) ** 4 - (s_star / max(f[I_LEAD_DIST], 0.1)) ** 2)
            alpha = min(alpha, max(0.0, vx + a) / limit)
        ped_ahead = abs(f[I_PED_BEARING] + f[I_HEADING_ERROR])  # bearing relative to the route heading
        if f[I_LEAD_TTC] < EXPERT_TTC_BRAKE or (
            f[I_PED_DIST] < max(12.0, vx**2 / (2 * EXPERT_BRAKE_DECEL) + EXPERT_BRAKE_STANDOFF)
            and ped_ahead <= np.radians(15)
        ):
            brake = 1.0
        gap = f[I_INTERSECTION_GAP]
        min_gap = EXPERT_MIN_GAP if vx < EXPERT_COMMIT_SPEED else EXPERT_ABORT_GAP  # commit/abort hysteresis
        can_stop = vx**2 / (2.0 * EXPERT_JUNCTION_STOP_DECEL) < frame.dist_to_junction  # past this, no abort
        if can_stop and frame.dist_to_junction <= junction_range(f[I_VX]) and gap < _GAP_ABSENT and gap < min_gap:
            d = max(frame.dist_to_junction - EXPERT_JUNCTION_WAIT_DIST, 0.0)
            alpha = min(alpha, np.sqrt(2.0 * EXPERT_JUNCTION_DECEL * d) / limit)
            if vx < EXPERT_HOLD_SPEED and gap < 0.0:  # stopped with a cross car ahead of its clear time: the MPC would creep
                brake = 1.0
        if f[I_PED_DIST] < 30.0 and ped_ahead < np.radians(30):
            alpha = min(alpha, 0.5)
        return PolicyAction(float(alpha), 0.0, brake)
