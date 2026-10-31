import pytest
from pydantic import ValidationError

from baseline_lab.analytics.quality import quality
from baseline_lab.contracts import Event, ShotEvidence


def contact(x=1.0, y=1.0, **kwargs):
    return Event(
        id="review", t_s=1, frame=30, xy_px=(200, 200), status="manual", court_m=(x, y), **kwargs
    )


def test_placement_index_is_partial_and_symmetric_at_either_end():
    a, b = quality(contact()), quality(contact(7.23, 22.77))
    assert a.score == b.score
    assert a.status == "partial" and a.available_factors == 2
    assert a.evidence_coverage == 0.45
    assert a.factors["speed"].score is None
    assert a.error_attribution == "undetermined"


def test_measured_factors_complete_index_but_do_not_classify_errors():
    e = contact(
        shot_evidence=ShotEvidence(
            speed_kmh=140,
            spin_rpm=3000,
            bounce_angle_deg=45,
            source="radar and high-speed camera fixture",
        )
    )
    q = quality(e)
    assert q.status == "complete" and q.available_factors == 5
    assert q.evidence_coverage == 1
    assert 0 <= q.score <= 10
    assert q.error_attribution == "undetermined"
    e.status = "candidate"
    assert quality(e).score is None
    e.status = "rejected"
    assert quality(e).score is None


def test_scoring_bounds_and_measurement_validation():
    q = quality(contact(-1, -2))
    assert q.score == 0 and q.factors["depth"].value == -2
    for value in [-1, 301, float("nan"), float("inf")]:
        with pytest.raises(ValidationError):
            ShotEvidence(speed_kmh=value, source="radar")


def test_stroke_estimate_requires_pose_evidence_and_stays_labeled():
    from baseline_lab.analytics.shots import shot_signals
    from baseline_lab.contracts import Keypoint, Metadata, Observation, Result, Track

    meta = Metadata(
        width=640,
        height=480,
        duration_s=1,
        codec="fixture",
        frames=1,
        nominal_fps=30,
        variable_frame_rate=False,
        rotation_deg=0,
        timestamps_s=[0],
        original_pts=[0],
        time_base="1/30",
    )
    ball = Observation(t_s=0, frame=0, state="detected", xy_px=(100, 70))
    person = Observation(t_s=0, frame=0, state="detected", bbox_px=(30, 20, 120, 160))
    missing = Observation(t_s=0, frame=0, state="missing")
    result = Result(
        metadata=meta,
        tracks=[
            Track(id="ball", observations=[ball], coverage=1),
            Track(id="near", observations=[person], coverage=1),
            Track(id="far", observations=[missing], coverage=0),
        ],
        events=[],
        calibration=None,
        warnings=[],
        provenance={},
        elapsed_s=0,
    )
    assert shot_signals(result) == []
    person.keypoints = [
        Keypoint(
            name=name, xy_px=point, state="detected", visibility=None, presence=None, score=0.8
        )
        for name, point in {
            "right_wrist": (100, 70),
            "right_shoulder": (80, 50),
            "left_shoulder": (50, 50),
            "right_hip": (80, 90),
            "left_hip": (50, 90),
        }.items()
    ]
    signal = shot_signals(result)[0]
    assert signal.kind == "unknown" and signal.status == "estimated"
    assert signal.image_speed_px_s is None
    assert "unclassified" in signal.method
