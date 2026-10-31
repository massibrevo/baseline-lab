"""Timestamp-preserving recording of processed camera samples, not unsampled frames."""

from fractions import Fraction
from pathlib import Path

import av
import numpy as np


class CameraRecording:
    def __init__(self, path: Path, width: int, height: int):
        self.container = av.open(str(path), "w")
        self.stream = self.container.add_stream("libx264", rate=30)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = Fraction(1, 90000)
        self.stream.codec_context.time_base = Fraction(1, 90000)
        self.stream.options = {"crf": "24", "preset": "fast", "threads": "2"}

    def write(self, image: np.ndarray, t_s: float):
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts = round(t_s * 90000)
        frame.time_base = Fraction(1, 90000)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self):
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()
