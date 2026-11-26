import numpy as np

from hybrid_av.mpc.controller import Reference
from hybrid_av.types import EgoState
from hybrid_av.vehicle.dynamics import rk4_step


def straight_reference(ego: EgoState, v: float, N: int = 20, dt: float = 0.05) -> Reference:
    """Line y = 0, starting one step ahead (the MPC solves from the delay-compensated state)."""
    x = ego.x + ego.vx * dt + v * np.arange(N + 1) * dt
    z = np.zeros(N + 1)
    return Reference(x=x, y=z, psi=z.copy(), v=np.full(N + 1, v), kappa=z.copy(), s=x.copy())


def circle_reference(ego: EgoState, R: float, v: float, N: int = 20, dt: float = 0.05) -> Reference:
    """CCW circle centred at (0, R), starting one step ahead."""
    th = np.arctan2(ego.x, R - ego.y) + (ego.vx * dt + v * np.arange(N + 1) * dt) / R
    psi = th + 2 * np.pi * np.round((ego.psi - th[0]) / (2 * np.pi))
    return Reference(
        x=R * np.sin(th),
        y=R - R * np.cos(th),
        psi=psi,
        v=np.full(N + 1, v),
        kappa=np.full(N + 1, 1.0 / R),
        s=R * th,
    )


def closed_loop(mpc, ref_fn, x0, steps, mu=1.0, dt=0.05):
    """RK4 plant with a one-step command delay (architecture.md section 5).

    The command solved from the observation at tick t is applied during tick t+1; tick 0 applies zero input.
    Returns x (steps+1, 6) including x0, u (steps, 2) the commands applied during each tick,
    status (steps,) and solve_time (steps,) in s.
    """
    x = np.asarray(x0, dtype=np.float64)
    xs, us, status, times = [x], [], [], []
    u_apply = np.zeros(2)
    for k in range(steps):
        ego = EgoState(k * dt, x[3], x[4], x[5], x[0], x[1], x[2])
        res = mpc.solve(ego, ref_fn(ego), mu=mu)
        us.append(u_apply)
        x = rk4_step(x, u_apply, mpc.params, dt)
        xs.append(x)
        u_apply = np.array([res.control.delta, res.control.ax])
        status.append(res.status)
        times.append(res.solve_time)
    return {"x": np.array(xs), "u": np.array(us), "status": np.array(status), "solve_time": np.array(times)}
