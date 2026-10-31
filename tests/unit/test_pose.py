import numpy as np
import pytest


def test_pose_crop_inverse_preserves_model_offsets_and_aspect_ratio():
    from baseline_lab.vision.pose import pose_crop

    for box in [(100, 50, 200, 250), (30, 100, 300, 200)]:
        matrix, w, h, left, top = pose_crop(box)
        center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1])
        np.testing.assert_allclose(matrix @ center, [96, 128], atol=1e-4)
        assert w / h == pytest.approx(0.75)
        assert w >= (box[2] - box[0]) * 1.25
        assert h >= (box[3] - box[1]) * 1.25
        point = np.array([35, 80])
        mapped = matrix @ np.r_[point, 1]
        recovered = mapped / [192, 256] * [w, h] + [left, top]
        np.testing.assert_allclose(recovered, point, atol=1e-4)


def test_checked_flow_measures_translation_and_rejects_blank_images():
    import cv2

    from baseline_lab.contracts import Observation
    from baseline_lab.vision.people import flow_people

    rng = np.random.default_rng(8)
    first = rng.integers(0, 256, (180, 240, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(first, np.array([[1, 0, 6], [0, 1, 4]], np.float32), (240, 180))
    p = Observation(t_s=0, frame=0, state="detected", bbox_px=(50, 40, 130, 150), score=0.9)
    detections = flow_people(first, shifted, [p])
    assert len(detections) == 1
    assert detections[0]["bbox"] == pytest.approx([56, 44, 136, 154], abs=0.5)
    blank = np.zeros_like(first)
    assert flow_people(blank, blank, [p]) == []


def test_joint_smoothing_preserves_motion_and_resets_long_gaps():
    from baseline_lab.tracking.smoothing import OneEuro

    smoother = OneEuro()
    values = [smoother.update([100 + (-2 if i % 2 else 2), 50], i / 30)[0] for i in range(20)]
    assert np.std(values[5:]) < 1
    assert smoother.update([300, 50], 2)[0] == 300
