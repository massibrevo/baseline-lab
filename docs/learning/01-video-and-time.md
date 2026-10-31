# 1. Frames, codecs, timestamps and aliasing

A codec describes compressed pictures; a container stores packets, timestamps and streams. A nominal 30 fps is a summary, not a promise that every interval is 1/30 second. This pipeline stores integer PTS and its rational time base: t = (PTS − PTS₀) × time_base. A frame may be displayed after other packets were decoded (B-frames); use presentation order.

Rotation is display geometry, not merely a width/height swap. Decode the display matrix, rotate pixels once, and keep calibration and rendering in that normalized coordinate system. H.264/yuv420p MP4 is the browser export baseline; the encoder codec time base must also preserve variable frame intervals. The VFR integration test caught quantization when only the stream time base was set.

Sampling 60 fps at 30 fps halves compute and can lose a bounce between samples. Aliasing means a missing temporal event cannot be reconstructed by drawing a smooth trail. The example preparation explicitly records this reduction; private input keeps its PTS. Inspect `video/media.py` and the VFR roundtrip tests.

Run from the repository root:

```sh
uv run python scripts/learning.py 1
```
