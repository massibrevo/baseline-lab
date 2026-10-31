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

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("analysis", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
start = time.monotonic()
args.output.mkdir(parents=True, exist_ok=True)
r = Result.model_validate_json((args.analysis / "result.json").read_text())
r.schema_version = "1.3"
r.live_metrics = live_metrics(r)
update_quality(r)
r.shots = shot_signals(r)
r.provenance["derived_metrics"] = (
    "live-playback-v1; px/s ball image motion; court-plane player m/s; real PTS"
)
for name in ["review.mp4", "raw-detections.json"]:
    if (args.analysis / name).exists():
        shutil.copyfile(args.analysis / name, args.output / name)
annotated(args.source, args.output / "annotated.mp4", r, lambda stage, done, total: None)
r.provenance["derived_metrics_elapsed_s"] = round(time.monotonic() - start, 3)
(args.output / "result.json").write_text(r.model_dump_json())
csvs(r, args.output)
print({"frames": len(r.live_metrics), "postprocess_s": r.provenance["derived_metrics_elapsed_s"]})
