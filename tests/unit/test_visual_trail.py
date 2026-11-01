import pytest

from baseline_lab.contracts import Observation
from baseline_lab.rendering.trail import visual_trail


def point(t, xy=None):
    return Observation(t_s=t, frame=round(t * 30), state="detected" if xy else "missing", xy_px=xy)


def test_long_gap_is_continuous_display_without_mutating_evidence():
    points = [point(0, (10, 20)), point(0.1), point(0.5), point(1.2), point(2, (110, 80))]
    original = [p.model_dump() for p in points]
    trail = visual_trail(points, 640, 480)
    assert all(p.xy_px is not None for p in trail)
    assert [p.state for p in trail] == ["observed", "inferred", "inferred", "inferred", "observed"]
    assert trail[2].xy_px == pytest.approx((35, 35))
    assert [p.model_dump() for p in points] == original


def test_bridge_does_not_overshoot_even_with_contradicting_endpoint_velocity():
    points = [
        point(0, (300, 300)),
        point(0.03, (100, 100)),
        point(0.1),
        point(0.2),
        point(0.3, (150, 120)),
        point(0.33, (500, 400)),
    ]
    trail = visual_trail(points, 640, 480)
    for p in trail[2:4]:
        assert 100 <= p.xy_px[0] <= 150 and 100 <= p.xy_px[1] <= 120


def test_live_prediction_is_bounded_and_reconnects_when_observed_again():
    points = [point(0, (610, 100)), point(0.03, (625, 105)), point(0.2), point(4)]
    live = visual_trail(points, 640, 480, offline=False)
    assert live[-1].state == "inferred"
    assert live[-1].xy_px[0] == 639
    assert live[-1].xy_px[1] < 200
    points.append(point(4.1, (300, 150)))
    updated = visual_trail(points, 640, 480, offline=False)
    assert updated[-1].state == "observed"
    assert updated[-2].reason == "display_bridge_between_observations"
    assert 300 < updated[-2].xy_px[0] < 625


def test_leading_fill_requires_future_context_and_empty_video_has_no_anchor():
    points = [point(0), point(0.1, (10, 10)), point(0.13, (20, 20))]
    assert visual_trail(points, 640, 480)[0].state == "inferred"
    assert visual_trail(points, 640, 480, offline=False)[0].state == "unavailable"
    assert all(p.xy_px is None for p in visual_trail([point(0), point(1)], 640, 480))
