from hybrid_av.vehicle.params import VehicleParams


def test_defaults_and_wheelbase():
    p = VehicleParams()
    assert (p.m, p.Iz, p.a, p.b, p.Cf, p.Cr, p.max_steer) == (
        1500.0,
        2250.0,
        1.2,
        1.6,
        19000.0,
        20000.0,
        0.70,
    )
    assert p.L == 2.8


def test_from_yaml_matches_defaults():
    assert VehicleParams.from_yaml("configs/vehicle/default.yaml") == VehicleParams()
