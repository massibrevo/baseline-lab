import numpy as np
import pytest

from baseline_lab.settings import ROOT
from baseline_lab.vision.strokes import STROKE_FILE, StrokeRecognizer, thumbnail


@pytest.mark.model
def test_cached_neural_stroke_model_is_mirror_invariant():
    import cv2

    directory = ROOT / "data/models"
    if not (directory / STROKE_FILE).exists():
        pytest.skip("Explicit setup-models required; model test never downloads")
    model = StrokeRecognizer(directory)
    capture = cv2.VideoCapture(str(ROOT / "examples/commons-stabilized.mp4"))
    images = []
    while len(images) < 48:
        ok, image = capture.read()
        if not ok:
            break
        images.append(thumbnail(image))
    capture.release()
    batch = np.stack(images)
    a, b = model.predict(batch)
    mirrored_a, mirrored_b = model.predict(batch[:, :, ::-1].copy())
    assert a.shape == (48, 2) and b.shape == (48, 29)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert a == pytest.approx(mirrored_a, abs=2e-5)
    assert b == pytest.approx(mirrored_b, abs=2e-5)
