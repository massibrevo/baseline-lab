# 2. Detection, association and tiny objects

Detection answers “what plausible object is visible in this image?” Tracking answers “which observation belongs to which trajectory over time?” A detector score is a learned ranking, not a calibrated probability. Tiny tennis balls have few pixels and motion blur; resizing a 1280-pixel image to 320 pixels can erase a four-pixel ball.

The person adapter uses overlapping tiles at larger widths, restores offsets to original display pixels, and suppresses duplicate boxes. This improves apparent scale but adds CPU work and tile-edge failures. Person regions plus ground-plane plausibility reduce spectators; they cannot guarantee semantic athlete identity.

The ball adapter uses positive frame change, HSV color, static-white exclusion, circularity and local contrast, then a motion gate. It is an experimental baseline with false positives and low validated localization. Raw detections, chosen observations, bounded interpolation and missing states are separate. A model like TrackNet/WASB uses multiple frames to exploit temporal evidence but needs verified weights/data rights and an independent evaluation. The runnable example rejects an impossible observation instead of creating a long link.

Run from the repository root:

```sh
uv run python scripts/learning.py 2
```
