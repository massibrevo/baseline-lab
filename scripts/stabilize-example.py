"""Offline preparation of attributed example footage; not a detector benchmark.

Track static feature correspondences into first-frame pixels, warp every frame and
preserve PTS. All output pixels come from the licensed real recording. Black border
pixels are honest empty padding. Private uploads never run this preparation implicitly.
"""

import argparse
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("target", type=Path)
args = parser.parse_args()
with av.open(str(args.source)) as source, av.open(str(args.target), "w") as out:
    stream = out.add_stream("libx264", rate=30)
    stream.width, stream.height = 1280, 720
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 90000)
    stream.codec_context.time_base = Fraction(1, 90000)
    stream.options = {"crf": "23", "preset": "fast", "threads": "2"}
    orb = cv2.ORB_create(nfeatures=2500)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    reference = None
    for i, f in enumerate(source.decode(video=0)):
        if i % 2:
            continue
        image = f.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, desc = orb.detectAndCompute(gray, None)
        if reference is None:
            reference = (kp, desc)
        pairs = matcher.knnMatch(desc, reference[1], k=2)
        good = [m for m, n in pairs if m.distance < 0.65 * n.distance]
        if len(good) < 20:
            raise ValueError("stabilization correspondence failure")
        a = np.float32([kp[m.queryIdx].pt for m in good])
        b = np.float32([reference[0][m.trainIdx].pt for m in good])
        h, mask = cv2.findHomography(a, b, cv2.RANSAC, 2)
        if h is None:
            raise ValueError("stabilization failure")
        image = cv2.warpPerspective(image, h, (1280, 720))
        v = av.VideoFrame.from_ndarray(image, format="bgr24")
        v.pts = round(float(f.time) * 90000)
        v.time_base = Fraction(1, 90000)
        for p in stream.encode(v):
            out.mux(p)
    for p in stream.encode():
        out.mux(p)
