import pytest

from baseline_lab.analytics.live import velocity
from baseline_lab.contracts import Observation


def point(t, x, metric=True, state="detected"):
    return Observation(
        t_s=t,
        frame=round(t * 30),
        state=state,
        xy_px=(x, 0),
        court_m=(x / 100, 0) if metric else None,
    )


def test_live_image_speed_uses_timestamps_and_units():
    points = [point(0, 0), point(0.033, 3.3), point(0.083, 8.3)]
    assert velocity(points, 2, False) == pytest.approx(100)
    assert velocity(points, 2, True) == pytest.approx(1)


def test_live_speed_does_not_bridge_missing_or_interpolated():
    for state in ["missing", "interpolated", "rejected"]:
        points = [point(0, 0), point(0.033, 3.3, state=state), point(0.083, 8.3)]
        assert velocity(points, 2, False) is None


def test_unavailable_geometry_and_impossible_player_velocity():
    assert velocity([point(0, 0, False), point(0.1, 100, False)], 1, True) is None
    assert velocity([point(0, 0), point(0.1, 2000)], 1, True) is None
    assert velocity([point(0, 0)], 0, False) is None


def test_image_speed_fit_suppresses_jitter_and_track_switch():
    points = [point(i * 0.03, 100 * i * 0.03 + (0.5 if i % 2 else -0.5)) for i in range(6)]
    assert velocity(points, 5, False) == pytest.approx(100, abs=5)
    points[-1].xy_px = (900, 500)
    assert velocity(points, 5, False) is None


def test_pose_association_keeps_box_and_ankle_coordinate_bias_separate():
    from baseline_lab.contracts import Calibration
    from baseline_lab.geometry.court import Court
    from baseline_lab.tracking.association import Players

    court = Court(Calibration(corners_px=[(210, 90), (430, 90), (550, 430), (90, 430)]))
    players = Players(court, 640, 480)
    x, y = court.image((4, 1))
    detection = {"bbox": [x - 8, y - 50, x + 8, y], "score": 0.9}
    players.update([detection], 0, 0)
    point = players.points["far"][-1]
    point.xy_px = (x, y - 12)
    point.court_m = court.project(point.xy_px)
    players.update([detection], 0.033, 1)
    assert players.points["far"][-1].state == "detected"


def test_player_velocity_robust_window_reduces_footpoint_jitter():
    points = [point(i * 0.03, 200 * i * 0.03 + (1.5 if i % 2 else -1.5)) for i in range(17)]
    assert velocity(points, 16, True) == pytest.approx(2, abs=0.1)
    points[-1].state = "missing"
    assert velocity(points, 16, True) is None
