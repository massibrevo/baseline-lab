---
name: computer-vision
description: Implement and evaluate narrow vision adapters, association and camera checks.
---

Own src/baseline_lab/{vision,tracking,geometry,analytics}. Preserve coordinate transforms, score semantics, missing observations and model manifests. Verify licenses/checksums before new weights. Run uv run pytest tests/unit -q and the explicitly provisioned -m model checks. Report empirical failures; never convert coverage/mAP into bounce accuracy.
