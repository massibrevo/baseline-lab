"""Recompute derived measurements from retained observations with explicit provenance."""

import argparse
import shutil
import time
from pathlib import Path

from baseline_lab.analytics.live import live_metrics
from baseline_lab.analytics.quality import update_quality
from baseline_lab.analytics.shots import shot_signals
from baseline_lab.contracts import Result
from baseline_lab.rendering.export import annotated, csvs
from baseline_lab.rendering.trail import visual_trail
from baseline_lab.settings import settings
from baseline_lab.vision.strokes import recognize_strokes

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("analysis", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
start = time.monotonic()
args.output.mkdir(parents=True, exist_ok=True)
r = Result.model_validate_json((args.analysis / "result.json").read_text())
r.schema_version = "1.4"
r.display_ball_trail = visual_trail(r.tracks[0].observations, r.metadata.width, r.metadata.height)
r.live_metrics = live_metrics(r)
update_quality(r)
r.shots = sorted(
    shot_signals(r) + [s for s in r.shots if s.status == "reviewed"], key=lambda s: s.t_s
)
r.automatic_shots, r.stroke_analysis = recognize_strokes(
    args.source,
    r.metadata,
    {t.id: t.observations for t in r.tracks if t.id != "ball"},
    settings().data_dir / "models",
    lambda *_: None,
)
r.provenance["derived_metrics"] = (
    "live-playback-v2; robust player m/s and integrated path; research ball km/h separate; real PTS"
)
for name in ["review.mp4", "raw-detections.json"]:
    if (args.analysis / name).exists():
        shutil.copyfile(args.analysis / name, args.output / name)
annotated(args.source, args.output / "annotated.mp4", r, lambda stage, done, total: None)
r.provenance["derived_metrics_elapsed_s"] = round(time.monotonic() - start, 3)
(args.output / "result.json").write_text(r.model_dump_json())
csvs(r, args.output)
print({"frames": len(r.live_metrics), "postprocess_s": r.provenance["derived_metrics_elapsed_s"]})
