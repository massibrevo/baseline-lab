"""Physical consistency tests; synthetic recovery is not real-world validation."""

import numpy as np
import pytest

from baseline_lab.analytics.speed import (
    camera_intrinsics,
    camera_projection,
    fit_flight,
    research_speed,
)
from baseline_lab.contracts import Calibration, Metadata, Observation, Result, Track
from baseline_lab.geometry.court import COURT, Court


def fixture():
    intrinsic = np.array([[1200.0, 0, 640], [0, 1200.0, 360], [0, 0, 1]])
    center = np.array([4.0, 34.0, 7.0])
    forward = np.array([4.0, 10.0, 0.0]) - center
    forward /= np.linalg.norm(forward)
    right = np.cross([0.0, 0.0, 1.0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(right, forward)
    rotation = np.array([right, down, forward])
    projection = intrinsic @ np.column_stack([rotation, -rotation @ center])
    ground = np.column_stack([COURT, np.zeros(4), np.ones(4)]) @ projection.T
    corners = ground[:, :2] / ground[:, 2, None]
    court = Court(Calibration(corners_px=corners.tolist()))
    points = []
    for i, t in enumerate(np.arange(0, 0.61, 0.03)):
        position = (
            np.array([2.0, 5.0, 1.0]) + np.array([1.0, 22.0, 6.0]) * t - [0, 0, 4.905 * t * t]
        )
        pixel = projection @ np.r_[position, 1]
        points.append(
            Observation(t_s=float(t), frame=i, state="detected", xy_px=tuple(pixel[:2] / pixel[2]))
        )
    return intrinsic, projection, court, points


def test_metric_camera_and_flight_recover_synthetic_physics():
    k, projection, court, points = fixture()
    recovered = camera_intrinsics(court, 1280, 720)
    assert recovered[0, 0] == pytest.approx(k[0, 0], rel=0.001)
    _, center, error = camera_projection(court, recovered)
    assert center[2] == pytest.approx(7, abs=0.01)
    assert error < 0.01
    model = fit_flight(points, {p.frame: projection for p in points})
    expected = np.linalg.norm([1, 22, 6 - 9.81 * 0.3]) * 3.6
    assert model is not None and model["speed"] == pytest.approx(expected, abs=0.01)
    assert model["sensitivity"] >= model["speed"] * 0.25


def test_unsupported_short_flight_and_gaps_are_not_filled():
    _, projection, _, points = fixture()
    assert fit_flight(points[:5], {p.frame: projection for p in points}) is None
    assert fit_flight(points[:6] + points[13:], {p.frame: projection for p in points}) is None
    points[7].xy_px = (50, 50)
    assert fit_flight(points, {p.frame: projection for p in points}) is None


def test_recording_timing_confirmation_is_required():
    _, _, court, points = fixture()
    meta = Metadata(
        width=1280,
        height=720,
        duration_s=0.6,
        codec="fixture",
        frames=len(points),
        nominal_fps=None,
        variable_frame_rate=True,
        rotation_deg=0,
        timestamps_s=[p.t_s for p in points],
        original_pts=list(range(len(points))),
        time_base="3/100",
    )
    result = Result(
        metadata=meta,
        tracks=[Track(id="ball", observations=points, coverage=1)],
        events=[],
        calibration=court.details(),
        warnings=[],
        provenance={},
        elapsed_s=0,
    )
    samples, details = research_speed(result, None, False)
    assert all(
        p.estimated_kmh is None and p.reason == "normal_1x_timing_not_confirmed" for p in samples
    )
    assert details["validated_with_radar"] is False
