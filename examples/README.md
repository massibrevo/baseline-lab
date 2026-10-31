# Reproducible public example

Oleg Yunakov's own-work [Carlos Alcaraz vs. Wu Yibing (2026 US Open) - 01.webm](https://commons.wikimedia.org/wiki/File:Carlos_Alcaraz_vs._Wu_Yibing_(2026_US_Open)_-_01.webm), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), downloaded 2026-10-31. The source page identifies 4 September 2026 as recording date. Full notice in THIRD_PARTY_NOTICES.md.

- `commons-singles.mp4`: source's full 12.5-second sequence at 1280×720, 60 fps, H.264, audio removed.
- `commons-stabilized.mp4`: same real sequence sampled at 30 fps and feature-homography stabilized into first-frame pixels with honest black padding. The preparation script does not synthesize tennis content.
- `commons-calibration.json`: approximate **singles** outside-painted corners in the stabilized first-frame display coordinate system. Four-point fit accuracy is not independently established.
- `visual-labels.json`: provisional visual annotations made by Codex during the build, not independently verified human ground truth. Early frames and later frames are separated temporally but share the same venue/recording. Ambiguous ball locations are null and excluded, not negative examples. This tiny study does not validate private-baseline performance.

Reproduce preparation from the attributed source:

```sh
# Download the exact Wikimedia original URL listed in THIRD_PARTY_NOTICES.md.
# Keep it outside Git. Example Python invocation after source download:
uv run python - <<'PY'
import imageio_ffmpeg, subprocess
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-i','data/original.webm',
 '-vf','scale=1280:720','-an','-c:v','libx264','-crf','23','-threads','2',
 '-fps_mode','vfr','-movflags','+faststart','examples/commons-singles.mp4'],check=True)
PY
uv run python scripts/stabilize-example.py examples/commons-singles.mp4 examples/commons-stabilized.mp4
uv run baseline-lab analyze examples/commons-stabilized.mp4 data/example-result \
  --calibration examples/commons-calibration.json
uv run python scripts/evaluate.py data/example-result/result.json
uv run python scripts/prepare-demo.py examples/commons-stabilized.mp4 data/example-result
```

The final command is deliberately explicit: it copies a chosen clip/result into public static demo assets. Never run it on a private session. CLI private analysis outputs stay ignored under `data/` by default.
