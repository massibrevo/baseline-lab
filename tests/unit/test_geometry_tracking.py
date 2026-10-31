import numpy as np
import pytest

from baseline_lab.contracts import Calibration, Observation
from baseline_lab.geometry.court import COURT, Court, letterbox, line_call
from baseline_lab.tracking.association import BallAssociation, Players, interpolate, summarize
from baseline_lab.video.stability import CameraStability


@pytest.fixture
def court():
    return Court(Calibration(corners_px=[(210, 90), (430, 90), (550, 430), (90, 430)]))


def test_homography_known_coordinates_inverse_and_degeneracy(court):
    for src, expected in zip(court.corners, COURT, strict=True):
        np.testing.assert_allclose(court.project(src), expected, atol=1e-5)
    point = (3.5, 14.2)
    np.testing.assert_allclose(court.project(court.image(point)), point, atol=1e-5)
    with pytest.raises(ValueError):
        Court(Calibration(corners_px=[(10, 10), (10, 10), (20, 20), (30, 30)]))
    with pytest.raises(ValueError):
        Court(Calibration(corners_px=[(430, 90), (210, 90), (90, 430), (550, 430)]))


def test_letterboxing():
    assert letterbox((640, 360), (1280, 720), (800, 800)) == (400, 400)
    assert letterbox((0, 0), (1280, 720), (800, 800)) == (0, 175)


@pytest.mark.parametrize(
    ("point", "uncertainty", "expected"),
    [
        ((2, 5), None, "uncertain"),
        ((0, 5), 0.25, "uncertain"),
        ((0.1, 5), 0.25, "uncertain"),
        ((-0.1, 5), 0.25, "uncertain"),
        ((-0.8, 5), 0.25, "estimated_out"),
        ((1, 5), 0.25, "estimated_in"),
    ],
)
def test_boundary_and_uncertainty(point, uncertainty, expected):
    assert line_call(point, uncertainty) == expected


def obs(t, xy=None, metric=None):
    return Observation(
        t_s=t, frame=round(t * 30), state="detected" if xy else "missing", xy_px=xy, court_m=metric
    )


def test_gap_is_bounded_and_missing_not_zero():
    points = [obs(0, (1, 2)), obs(0.03), obs(0.06, (7, 8)), obs(0.2), obs(0.4, (20, 30))]
    interpolate(points)
    assert points[1].state == "interpolated" and points[1].xy_px == (4, 5)
    assert points[3].state == "missing" and points[3].xy_px is None
    assert points[1].raw_xy_px is None


def test_impossible_ball_link_rejected():
    tracker = BallAssociation(1000)
    assert tracker.update([{"xy": (20, 20), "score": 0.8}], 0, 0).state == "missing"
    p = tracker.update([{"xy": (950, 950), "score": 0.9}], 0.03, 1)
    assert p.state == "missing"


def test_distance_excludes_gaps_jumps_and_missing_calibration():
    points = [
        obs(0, (1, 1), (1, 1)),
        obs(0.04, (2, 1), (1.2, 1)),
        obs(0.08),
        obs(0.5, (4, 1), (8, 1)),
        obs(0.54, (5, 1), (15, 1)),
    ]
    assert summarize("near", points, True).distance_m == pytest.approx(0.2)
    assert summarize("near", points, False).distance_m is None


def test_player_side_assignment_and_spectator_rejection(court):
    players = Players(court, 640, 480)

    def box(metric):
        x, y = court.image(metric)
        return {"bbox": [x - 10, y - 40, x + 10, y], "score": 0.9}

    players.update([box((4, 2)), box((4, 22)), box((40, 4))], 0, 0)
    assert players.points["near"][0].court_m[1] == pytest.approx(22)
    assert players.points["far"][0].court_m[1] == pytest.approx(2)


def test_camera_motion_detected():
    rng = np.random.default_rng(3)
    im = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    stability = CameraStability(im)
    assert stability.drift(im) < 0.01
    assert stability.drift(np.roll(im, 20, axis=1)) > 15


def test_ball_temporal_evidence_rejects_stationary_clutter_and_expires():
    tracker = BallAssociation(1000)
    positions = []
    for i in range(10):
        p = tracker.update(
            [{"xy": (50, 50), "score": 0.99}, {"xy": (200 + 12 * i, 200 + 4 * i), "score": 0.85}],
            i / 30,
            i,
        )
        positions.append(p)
    assert all(
        p.xy_px == (200 + 12 * i, 200 + 4 * i)
        for i, p in enumerate(positions)
        if p.state == "detected"
    )
    assert positions[-1].state == "detected"
    assert tracker.update([{"xy": (400, 300), "score": 0.9}], 2, 60).state == "missing"


def test_verified_camera_mapping_keeps_ground_coordinates_in_current_frame(court):
    import cv2

    rng = np.random.default_rng(12)
    image = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
    moved = cv2.warpAffine(image, np.array([[1, 0, 16], [0, 1, 7]], np.float32), (640, 480))
    mapping = CameraStability(image).mapping(moved)
    assert mapping is not None
    transform, drift, residual = mapping
    assert drift > 10 and residual < 2
    corners = cv2.perspectiveTransform(np.array([court.corners], np.float32), transform)[0]
    current = Court(Calibration(corners_px=corners.tolist()))
    expected = (3.0, 18.0)
    point = np.array([[court.image(expected)]], np.float32)
    shifted_point = cv2.perspectiveTransform(point, transform)[0, 0]
    assert current.project(tuple(shifted_point)) == pytest.approx(expected, abs=0.01)
