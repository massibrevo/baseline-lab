import inspect

import numpy as np
import pytest

from baseline_lab.contracts import Keypoint, Metadata, Observation
from baseline_lab.vision.strokes import (
    STROKE_FILE,
    LiveStrokeRecognizer,
    decode_strokes,
    recognize_strokes,
    wing_type,
)


def metadata(duration=6):
    times = np.arange(0, duration + 0.001, 0.04).tolist()
    return Metadata(
        width=1280,
        height=720,
        duration_s=duration,
        codec="fixture",
        frames=len(times),
        nominal_fps=25,
        variable_frame_rate=False,
        rotation_deg=0,
        timestamps_s=times,
        original_pts=list(range(len(times))),
        time_base="1/25",
    )


def attributes(fh=0.95, bh=0.02):
    row = np.zeros(29, np.float32)
    row[0], row[7], row[8], row[9], row[10] = 0.95, 0.95, fh, bh, 0.9
    return row


def test_attribute_decoder_abstains_on_ambiguous_wing_and_distinguishes_serve():
    assert wing_type(attributes())[0] == "forehand"
    assert wing_type(attributes(0.03, 0.94))[0] == "backhand"
    assert wing_type(attributes(0.52, 0.49))[0] == "unknown"
    row = attributes()
    row[5], row[7] = 0.98, 0.01
    assert wing_type(row)[0] == "serve"


def test_predictions_follow_model_peaks_not_manual_timestamp_guides():
    meta = metadata()
    times = np.arange(0, 6, 0.08)
    coarse = np.tile([1.0, 0.0], (len(times), 1))
    fine = np.tile(attributes(), (len(times), 1))
    for index, wing in [(14, attributes()), (53, attributes(0.02, 0.97))]:
        coarse[index] = [0.05, 0.95]
        fine[index] = wing
    shots = decode_strokes(times, coarse, fine, meta, {})
    assert [(s.t_s, s.kind) for s in shots] == [(1.12, "forehand"), (4.24, "backhand")]
    assert all(s.status == "estimated" and s.original_prediction is None for s in shots)
    assert "annotations" not in inspect.signature(recognize_strokes).parameters
    assert "result" not in inspect.signature(recognize_strokes).parameters


def test_boundary_motion_has_a_type_but_never_claims_verified_contact():
    meta = metadata(2)
    observations = []
    for i, t in enumerate(meta.timestamps_s):
        points = {
            "left_hip": (90, 200),
            "right_hip": (110, 200),
            "left_shoulder": (90, 130),
            "right_shoulder": (110, 130),
            "left_wrist": (80, 150),
            "right_wrist": (120 + max(0, t - 1.55) * 500, 150),
        }
        observations.append(
            Observation(
                t_s=t,
                frame=i,
                state="detected",
                keypoints=[
                    Keypoint(name=k, xy_px=v, state="detected", visibility=0.9, presence=0.9)
                    for k, v in points.items()
                ],
            )
        )
    times = np.arange(0, 2.001, 0.08)
    fine = np.tile(attributes(0.02, 0.96), (len(times), 1))
    # Neural identity is weak here. The observed moving skeleton supplies identity.
    fine[:, :2] = [0.56, 0.4]
    shots = decode_strokes(
        times, np.tile([1.0, 0.0], (len(times), 1)), fine, meta, {"near": observations}
    )
    assert shots and shots[-1].kind == "backhand"
    assert shots[-1].recognition_basis == "clip_boundary_swing_not_verified_contact"
    assert shots[-1].model_scores["event"] == 0


def test_sparse_camera_gaps_do_not_trigger_motion_inference(tmp_path, monkeypatch):
    (tmp_path / STROKE_FILE).touch()

    def forbidden(*args):
        pytest.fail("Sparse camera frames must not be stretched into a swing")

    monkeypatch.setattr("baseline_lab.vision.strokes.StrokeRecognizer", forbidden)
    engine = LiveStrokeRecognizer(tmp_path)
    frames = [(t, np.zeros((224, 224, 3), np.uint8)) for t in [0, 0.03, 2, 2.03]]
    meta = metadata(2.04)
    shots, info = engine.update(frames, meta, {}, lambda *_: None)
    assert shots == [] and info["status"] == "collecting"
    assert info["motion_context_coverage"] < 0.3
