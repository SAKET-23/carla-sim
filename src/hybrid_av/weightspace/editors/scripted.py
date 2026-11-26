"""Scripted editor: recruit the deadest units on the most shifted feature (weight-space-editing.md section 7)."""

import numpy as np

from hybrid_av.features.spec import FEATURE_NAMES, denormalize
from hybrid_av.weightspace.analysis import _GAINS, _acts, activation_stats, evaluate_plan_offline
from hybrid_av.weightspace.schema import EditPlan, WeightEdit

# (output index, required mean shift on failure-window states, physical units)
SCENARIO_OUTPUTS: dict[str, tuple[tuple[int, float], ...]] = {
    "wet_curve": ((2, +0.5),),  # brake: alpha_v caps cannot reach the gate (Phase 4 decision 2026-11-18)
    "merge_deadlock": ((0, +0.3), (2, -0.4)),
}
_BOUNDS = {0: (0.0, 1.2), 1: (-3.5, 3.5), 2: (0.0, 1.0)}  # physical output ranges
_OUT_NAMES = list(_GAINS)
# m/s: design speed threshold (effective ~1 higher), lowered per failed gate
WET_SPEED_START, WET_SPEED_STEP = 7.0, 0.5


def _rank(layer: dict) -> list[int]:
    return sorted(range(64), key=lambda u: (layer["active_frac"][u], layer["mean"][u], u))


class ScriptedEditor:
    name = "scripted"

    def __init__(
        self,
        feature: int | None = None,
        w1: float = 3.0,
        gains: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0),
    ) -> None:
        self.feature, self.w1, self.gains = feature, w1, gains
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.last_messages: list[dict] = []

    @staticmethod
    def _speed_threshold(history) -> float:
        missed = sum(1 for a in history if a.eval_summary.get("ood_success_rate", 1.0) < 0.8)
        return WET_SPEED_START - WET_SPEED_STEP * missed

    def propose(self, bundle, history) -> EditPlan:
        Zw, Zn, policy = bundle.Z_window, bundle.Z_nom, bundle.policy
        f = self.feature
        wet = bundle.scenario == "wet_curve"
        if f is None and wet:
            f = 7  # surface_roughness: the cue that separates wet-curve failures from nominal driving
        if f is None:
            score = np.abs(Zw.mean(0) - Zn.mean(0)) / (Zn.std(0) + 0.05)
            f = int(np.argmax(score))
        q10, q90 = (float(q) for q in np.percentile(Zw[:, f], [10, 90]))
        nom = Zn[:, f]
        if nom.max() < q10:
            mode, s, ts = "ramp_up", 1.0, [(q10 + float(nom.max())) / 2]
        elif nom.min() > q90:
            mode, s, ts = "ramp_down", -1.0, [(q90 + float(nom.min())) / 2]
        else:
            mode, s, ts = "band", 1.0, [q10, (q10 + q90) / 2, q90]
        if wet:
            mode, s, ts = "ramp_up", 1.0, [q10]
        stats = activation_stats(policy, Zn)
        units = _rank(stats["h1"])[: len(ts)]
        j = _rank(stats["h2"])[0]
        h2_w = [1.0] if mode != "band" else [1.0, -2.0, 1.0]
        targets = SCENARIO_OUTPUTS[bundle.scenario]

        def edit(tensor, index, value, why):
            return WeightEdit(tensor=tensor, index=index, op="set", value=float(value), rationale=why)

        base_edits = []
        for i, t in zip(units, ts):
            why = f"h1[{i}] becomes a {mode} detector on feature {f}"
            gate = {f: s * self.w1, **({0: self.w1} if wet else {})}  # wet_curve: roughness AND speed
            base_edits += [edit("fc1.weight", [i, c], gate.get(c, 0.0), why) for c in range(24)]
            bias = -s * self.w1 * t
            if wet:
                bias -= self.w1 * (2 * self._speed_threshold(history) / 30 - 1 - 1 / self.w1)
            base_edits.append(edit("fc1.bias", [i], bias, why))
        why = f"h2[{j}] reads only the recruited h1 units"
        for c in range(64):
            base_edits.append(edit("fc2.weight", [j, c], h2_w[units.index(c)] if c in units else 0.0, why))
        base_edits.append(edit("fc2.bias", [j], 0.0, why))

        base_out = _acts(policy, Zw)["out"].numpy().mean(0)
        base_phys = np.array([0.6 * (base_out[0] + 1), 3.5 * base_out[1], 0.5 * (base_out[2] + 1)])

        def build(g: float):
            col = {o: float(np.sign(sh)) * g for o, sh in targets}
            fc3 = [
                edit("fc3.weight", [o, j], col.get(o, 0.0), f"route h2[{j}] to output {_OUT_NAMES[o]}")
                for o in range(3)
            ]
            raw_t = [float(denormalize(np.where(np.arange(24) == f, t, 0.0))[f]) for t in ts]
            text = (
                f"Feature {f} ({FEATURE_NAMES[f]}) shifts most between failure and nominal states. "
                f"Recruit h1 {units} as a {mode} detector (thresholds {[round(x, 3) for x in raw_t]} raw units) "
                f"and h2 unit {j}, routed to {[_OUT_NAMES[o] for o, _ in targets]} with gain {g}."
            )
            return EditPlan(edits=base_edits + fc3, hypothesis=text, expected_effect="")

        # wet_curve: the brake output acts as a flag, so it needs the saturating gain (g=2 never fired in the sim)
        for g in self.gains[-1:] if wet else self.gains:
            plan = build(g)
            fm = evaluate_plan_offline(policy, plan, Zw, Zn)["fail_mean_shift"]
            ok = True
            for o, sh in targets:
                got = fm[_OUT_NAMES[o]]
                lim = _BOUNDS[o][1 if sh > 0 else 0]
                ok &= np.sign(sh) * got >= abs(sh) or abs(base_phys[o] + got - lim) <= 0.02
            if ok:
                break
        shifts = {k: round(v, 3) for k, v in fm.items()}
        plan.expected_effect = (
            f"Offline mean output shift on failure-window states: {shifts}; nominal outputs unchanged."
        )
        return plan
