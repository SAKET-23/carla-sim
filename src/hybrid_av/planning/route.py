"""Arc-length route and builders; the contract is Plan/02-system-design/interfaces.md section 4."""

from dataclasses import dataclass, fields

import numpy as np

from hybrid_av.geometry import rotate, unwrap_sequence

_SPACING = 0.5  # m, grid spacing of generated routes


@dataclass(frozen=True, slots=True)
class Projection:
    s: float  # arc length of closest point
    lateral: float  # signed lateral offset of the query point, left positive, m
    heading: float  # route heading at s (unwrapped)


@dataclass(frozen=True, slots=True, eq=False)
class Route:
    s: np.ndarray  # (M,) arc length, strictly increasing, starts at 0, spacing <= 1.0 m
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray  # unwrapped tangent heading
    kappa: np.ndarray  # signed curvature, 1/m (left positive)
    speed_limit: np.ndarray  # m/s
    lane_width: np.ndarray  # m

    @property
    def length(self) -> float:
        return float(self.s[-1])

    def project(self, x: float, y: float, s_hint: float | None = None) -> Projection:
        """Nearest point on the polyline; global (ties to smallest s) or within [s_hint-10, s_hint+30]."""
        lo, hi = 0, len(self.s) - 1  # segment index range [lo, hi)
        if s_hint is not None:
            lo = min(max(int(np.searchsorted(self.s, s_hint - 10.0, side="right")) - 1, 0), len(self.s) - 2)
            hi = min(max(int(np.searchsorted(self.s, s_hint + 30.0)), lo + 1), len(self.s) - 1)
        p0 = np.column_stack([self.x[lo:hi], self.y[lo:hi]])
        d = np.column_stack([self.x[lo + 1 : hi + 1], self.y[lo + 1 : hi + 1]]) - p0
        seg2 = np.maximum((d**2).sum(axis=1), 1e-18)
        off = np.array([x, y]) - p0
        t = np.clip((off * d).sum(axis=1) / seg2, 0.0, 1.0)
        dist2 = ((off - t[:, None] * d) ** 2).sum(axis=1)
        i = int(np.argmax(dist2 <= dist2.min() + 1e-9))  # first near-tie: lap-safe smallest s
        s = self.s[lo + i] + t[i] * (self.s[lo + i + 1] - self.s[lo + i])
        lateral = (d[i, 0] * off[i, 1] - d[i, 1] * off[i, 0]) / np.sqrt(seg2[i])
        return Projection(float(s), float(lateral), float(np.interp(s, self.s, self.psi)))

    def sample(self, s: np.ndarray) -> dict[str, np.ndarray]:
        s = np.asarray(s, dtype=np.float64)
        return {
            k: np.interp(s, self.s, getattr(self, k))
            for k in ("x", "y", "psi", "kappa", "speed_limit", "lane_width")
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Route):
            return NotImplemented
        return all(np.array_equal(getattr(self, f.name), getattr(other, f.name)) for f in fields(self))

    __hash__ = None  # type: ignore[assignment]


def _grid(length: float) -> np.ndarray:
    return np.linspace(0.0, length, int(np.ceil(length / _SPACING)) + 1)


def _const(s, x, y, psi, kappa, speed_limit, lane_width) -> Route:
    full = np.full_like
    return Route(s, x, y, psi, kappa, full(s, speed_limit), full(s, lane_width))


class RouteBuilder:
    @staticmethod
    def from_waypoints(
        xy: np.ndarray, speed_limit: float | np.ndarray, lane_width: float | np.ndarray, spacing: float = 0.5
    ) -> Route:
        xy = np.asarray(xy, dtype=np.float64)
        chord = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
        s = np.linspace(0.0, chord[-1], int(np.ceil(chord[-1] / spacing)) + 1)
        x, y = np.interp(s, chord, xy[:, 0]), np.interp(s, chord, xy[:, 1])
        psi = unwrap_sequence(np.arctan2(np.gradient(y, s), np.gradient(x, s)))
        k = np.pad(np.gradient(psi, s), 2, mode="edge")
        kappa = np.convolve(k, np.ones(5) / 5, mode="valid")
        lim, lw = (
            np.interp(s, chord, np.broadcast_to(v, chord.shape).astype(np.float64))
            for v in (speed_limit, lane_width)
        )
        return Route(s, x, y, psi, kappa, lim, lw)

    @staticmethod
    def straight(length: float, speed_limit: float, lane_width: float = 3.5) -> Route:
        s = _grid(length)
        return _const(
            s, s.copy(), np.zeros_like(s), np.zeros_like(s), np.zeros_like(s), speed_limit, lane_width
        )

    @staticmethod
    def arc(radius: float, angle: float, speed_limit: float, lane_width: float = 3.5) -> Route:
        s = _grid(abs(angle) * radius)
        sgn = np.sign(angle)
        a = s / radius  # unsigned swept angle
        return _const(
            s,
            radius * np.sin(a),
            sgn * radius * (1 - np.cos(a)),
            sgn * a,
            np.full_like(s, sgn / radius),
            speed_limit,
            lane_width,
        )

    @staticmethod
    def concat(*routes: Route) -> Route:
        out = routes[0]
        for r in routes[1:]:
            dpsi = out.psi[-1] - r.psi[0]
            xy = rotate(np.column_stack([r.x - r.x[0], r.y - r.y[0]]), dpsi)[1:] + [out.x[-1], out.y[-1]]
            out = Route(
                np.concatenate([out.s, out.s[-1] + r.s[1:]]),
                np.concatenate([out.x, xy[:, 0]]),
                np.concatenate([out.y, xy[:, 1]]),
                np.concatenate([out.psi, r.psi[1:] + dpsi]),
                np.concatenate([out.kappa, r.kappa[1:]]),
                np.concatenate([out.speed_limit, r.speed_limit[1:]]),
                np.concatenate([out.lane_width, r.lane_width[1:]]),
            )
        return out

    @staticmethod
    def circle_loop(radius: float, laps: float, speed_limit: float) -> Route:
        return RouteBuilder.arc(radius, laps * 2 * np.pi, speed_limit)


def constant_radius_segments(
    route: Route, r_min: float, r_max: float, min_length: float
) -> list[tuple[float, float, float]]:
    """Runs of the route with |1/kappa| in [r_min, r_max] and length >= min_length: (s_start, s_end, mean_radius)."""
    k = np.abs(route.kappa)
    inside = (k > 1e-9) & (1.0 / np.maximum(k, 1e-9) >= r_min) & (1.0 / np.maximum(k, 1e-9) <= r_max)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], inside.astype(int), [0]])))
    out = []
    for i, j in zip(edges[::2], edges[1::2], strict=True):  # [i, j) is one run
        if route.s[j - 1] - route.s[i] >= min_length:
            out.append((float(route.s[i]), float(route.s[j - 1]), float(np.mean(1.0 / k[i:j]))))
    return out


def place_route(route: Route, x0: float, y0: float, psi0: float) -> Route:
    """The route moved so that it starts at (x0, y0) with heading psi0 (it must start at the origin heading +x)."""
    xy = rotate(np.column_stack([route.x, route.y]), psi0) + [x0, y0]
    return Route(
        route.s, xy[:, 0], xy[:, 1], route.psi + psi0, route.kappa, route.speed_limit, route.lane_width
    )
