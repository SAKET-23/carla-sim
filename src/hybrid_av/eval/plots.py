"""One-PNG episode summary."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hybrid_av.eval.runner import EpisodeResult
from hybrid_av.planning.route import Route


def plot_episode(result: EpisodeResult, route: Route, out: Path) -> None:
    tr = result.trace
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(route.x, route.y, "k--", label="route")
    ax[0, 0].plot(tr["x"], tr["y"], label="ego")
    ax[0, 0].set(title="trajectory", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax[0, 1].plot(tr["t"], tr["vx"], label="vx")
    ax[0, 1].plot(tr["t"], tr["alpha_v"] * route.speed_limit[0], label="alpha_v * limit (route start)")
    ax[0, 1].set(title="speed", xlabel="t [s]", ylabel="m/s")
    ax[1, 0].plot(tr["t"], tr["delta"])
    ax[1, 0].set(title="commanded steering", xlabel="t [s]", ylabel="rad")
    ax[1, 1].plot(tr["t"], tr["mu_est"], label="mu_est")
    ax[1, 1].plot(tr["t"], tr["mu_true"], label="mu_true")
    ax[1, 1].set(title="friction", xlabel="t [s]")
    for a in ax.flat:
        if a.get_legend_handles_labels()[0]:
            a.legend()
    fig.suptitle(f"{result.scenario} seed {result.seed}: DS {result.driving_score:.1f}")
    fig.tight_layout()
    fig.savefig(out, dpi=100)
    plt.close(fig)
