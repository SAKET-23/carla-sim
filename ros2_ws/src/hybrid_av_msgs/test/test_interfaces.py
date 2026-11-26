from rosidl_runtime_py import message_to_ordereddict


def test_feature_vector_has_fixed_arrays():
    from hybrid_av_msgs.msg import FeatureVector

    m = FeatureVector()
    assert len(m.raw) == 24 and len(m.normalized) == 24
    assert set(message_to_ordereddict(m)) == {"header", "raw", "normalized", "mu_est", "route_s", "source"}


def test_track_and_context_fields():
    from hybrid_av_msgs.msg import Track, TrafficContext

    assert len(Track().pos_cov) == 4
    assert set(message_to_ordereddict(TrafficContext())) == {
        "header", "light_state", "dist_to_stop_line", "dist_to_stop_sign",
        "dist_to_junction", "speed_limit", "lane_width", "road_grade", "surface_roughness",
    }  # fmt: skip


def test_load_policy_service():
    from hybrid_av_msgs.srv import LoadPolicy

    assert hasattr(LoadPolicy.Request(), "path")
    r = LoadPolicy.Response()
    assert {"success", "policy_hash", "message"} <= set(message_to_ordereddict(r))


def test_all_messages_import():
    import hybrid_av_msgs.msg as M

    for n in ["FeatureVector", "PolicyAction", "ReferenceTrajectory", "MpcDiagnostics", "VehicleCommand",
              "TrafficContext", "Track", "TrackArray", "RouteInfo"]:  # fmt: skip
        getattr(M, n)()
