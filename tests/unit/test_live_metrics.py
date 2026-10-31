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
