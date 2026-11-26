import os

import numpy as np
import pytest

from hybrid_av.mpc.controller import BicycleMPC, MPCConfig
from hybrid_av.vehicle.params import VehicleParams
from tests.mpc.helpers import circle_reference, closed_loop

P, CFG = VehicleParams(), MPCConfig()


@pytest.mark.benchmark
def test_solve_rate(benchmark):
    m = BicycleMPC(P, CFG)
    out = benchmark.pedantic(
        lambda: closed_loop(
            m, lambda e: circle_reference(e, 40.0, 10.0), np.array([10.0, 0, 0, 0, 0, 0]), steps=200
        ),
        rounds=1,
        iterations=1,
    )
    t = np.asarray(out["solve_time"][5:])
    p50, p99 = float(np.median(t)), float(np.percentile(t, 99))
    print(f"MPC p50 {p50 * 1e3:.2f} ms, p99 {p99 * 1e3:.2f} ms ({1 / p50:.0f} Hz at p50)")
    if not os.environ.get("CI"):
        assert p99 < 0.020  # >= 50 Hz at p99 (metrics.md C4, Phase 1 Mac check)
