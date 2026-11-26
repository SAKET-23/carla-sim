"""Leaderboard-2.1-style driving score."""

PENALTIES_V21: dict[str, float] = {
    "collision_pedestrian": 1.0,
    "collision_vehicle": 0.7,
    "collision_static": 0.6,
    "red_light": 0.4,
    "stop_sign": 0.25,
    "timeout": 0.4,
}


def driving_score(route_completion: float, infractions: dict[str, int], route_length: float) -> float:
    r = max(0.0, route_completion - infractions.get("off_road_m", 0) / route_length)
    return 100.0 * r / (1.0 + sum(c * infractions.get(k, 0) for k, c in PENALTIES_V21.items()))
