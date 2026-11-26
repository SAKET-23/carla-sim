from hybrid_av.eval.driving_score import driving_score


def test_driving_score_v21():
    assert driving_score(1.0, {}, 400.0) == 100.0
    assert abs(driving_score(1.0, {"collision_vehicle": 1}, 400.0) - 100 / 1.7) < 1e-9  # 58.82
    assert abs(driving_score(0.5, {"red_light": 1, "stop_sign": 1}, 400.0) - 50 / 1.65) < 1e-9  # 30.30
    assert abs(driving_score(0.8, {"collision_pedestrian": 2}, 400.0) - 80 / 3) < 1e-9  # 26.67
    assert abs(driving_score(1.0, {"off_road_m": 12}, 400.0) - 97.0) < 1e-9  # R = 1 - 12/400
