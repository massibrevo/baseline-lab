"""Publish only explicitly selected rights-cleared pipeline output to static demo."""

import argparse
import json
import shutil
from pathlib import Path

from baseline_lab.contracts import Result

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("analysis", type=Path)
args = parser.parse_args()
result = Result.model_validate_json((args.analysis / "result.json").read_text())
# Explicit invocation only: never copy arbitrary private sessions automatically.
result.provenance["example_media"] = (
    "Oleg Yunakov, Carlos Alcaraz vs Wu Yibing 2026 US Open 01; CC BY-SA 4.0"
)
result.provenance["example_preparation"] = (
    "stabilized to first-frame pixels; 60→30 fps; 1280x720; audio omitted"
)
result.warnings.append(
    "Example is an oblique elevated-view stress study, not a private-baseline accuracy benchmark"
)
evaluation = Path("docs/evaluation-provisional.json")
if evaluation.exists():
    result.provenance["provisional_visual_check"] = json.loads(evaluation.read_text())["splits"][
        "validation_frames"
    ]
out = Path("apps/web/public/demo")
out.mkdir(parents=True, exist_ok=True)
(out / "result.json").write_text(result.model_dump_json())
shutil.copyfile(args.source, out / "clip.mp4")
shutil.copyfile(args.analysis / "annotated.mp4", out / "annotated.mp4")
print(
    json.dumps(
        {
            "example_frames": result.metadata.frames,
            "elapsed_s": result.elapsed_s,
            "coverage": {t.id: t.coverage for t in result.tracks},
            "events": len(result.events),
        }
    )
)
