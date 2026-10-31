from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from baseline_lab.settings import Settings


@pytest.fixture
def config(tmp_path):
    return Settings(data_dir=tmp_path / "data", player_model="hog")


def make_clip(path: Path, variable: bool = False, cut: bool = False, rotation: int = 0):
    with av.open(str(path), "w") as out:
        stream = out.add_stream("libx264", rate=30)
        stream.width, stream.height = 640, 480
        stream.pix_fmt = "yuv420p"
        if rotation:
            stream.set_display_rotation(rotation)
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = Fraction(1, 90000)
        rng = np.random.default_rng(4)
        # Texture allows the camera stability check to verify this synthetic fixture.
        background = rng.integers(0, 15, (480, 640, 3), dtype=np.uint8) + [50, 105, 45]
        background = background.astype(np.uint8)
        for i in range(160):
            image = background.copy()
            cv2.polylines(
                image,
                [np.array([[210, 90], [430, 90], [550, 430], [90, 430]])],
                True,
                (230, 230, 230),
                2,
            )
            cv2.circle(
                image,
                (100 + (i * 3) % 440, 200 + int(abs(np.sin(i / 15)) * 140)),
                4,
                (30, 240, 225),
                -1,
            )
            if cut and i > 80:
                image = 255 - image
            f = av.VideoFrame.from_ndarray(image, format="bgr24")
            # 33.3/50ms intervals: no frame-index/FPS shortcuts allowed.
            f.pts = i * 3000 + (i // 5) * 1500 if variable else i * 3000
            f.time_base = Fraction(1, 90000)
            for p in stream.encode(f):
                out.mux(p)
        for p in stream.encode():
            out.mux(p)
    return path


@pytest.fixture
def clip(tmp_path):
    return make_clip(tmp_path / "fixture.mp4")
