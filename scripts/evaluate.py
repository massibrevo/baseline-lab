"""Small, explicitly provisional point-label check. Never turn coverage into accuracy."""

import argparse
import json
from pathlib import Path

import numpy as np

from baseline_lab.contracts import Result

parser = argparse.ArgumentParser()
parser.add_argument("result", type=Path)
parser.add_argument("--labels", type=Path, default=Path("examples/visual-labels.json"))
parser.add_argument("--output", type=Path)
args = parser.parse_args()
result = Result.model_validate_json(args.result.read_text())
labels = json.loads(args.labels.read_text())
report = {
    "annotation_method": labels["annotation_method"],
    "pipeline_elapsed_s": result.elapsed_s,
    "frames": result.metadata.frames,
    "duration_s": result.metadata.duration_s,
    "detection_coverage_not_accuracy": {t.id: t.coverage for t in result.tracks},
    "candidate_events": len(result.events),
    "bounce_precision": None,
    "court_independent_error_px": None,
    "ball_speed_m_s": None,
    "splits": {},
}
for split in ["tuning_frames", "validation_frames"]:
    report["splits"][split] = {}
    for track in result.tracks:
        distances = []
        missed = 0
        labeled = [label for label in labels[split] if label.get(track.id) is not None]
        tolerance = labels["ball_tolerance_px" if track.id == "ball" else "player_tolerance_px"]
        for label in labeled:
            p = track.observations[label["frame"]]
            if p.state != "detected" or p.xy_px is None:
                missed += 1
            else:
                distances.append(float(np.linalg.norm(np.array(p.xy_px) - label[track.id])))
        report["splits"][split][track.id] = {
            "n_labeled": len(labeled),
            "missing": missed,
            "hits_within_tolerance": sum(d <= tolerance for d in distances),
            "tolerance_px": tolerance,
            "median_error_when_observed_px": float(np.median(distances)) if distances else None,
            "errors_px": distances,
        }
text = json.dumps(report, indent=2)
if args.output:
    args.output.write_text(text + "\n")
print(text)
