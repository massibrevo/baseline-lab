# 6. Evaluation, leakage and domain shift

Coverage counts observed frames. Accuracy needs reference labels. A track can cover 90% of frames while mostly following a court line. Report localization errors, missed/false observations, identity failures and event timing/location tolerance separately. Detector mAP does not validate bounce precision.

The included tiny point-label study has provisional Codex visual annotations, a temporal split in one recording, nulls for ambiguous ball points, and explicit tolerances. It is a reproducible smoke measurement, not independently verified human ground truth. Labels and tuning observations from one court can leak appearance into later frames; a new recording, camera and court are a stronger holdout.

Private-baseline and broadcast/oblique footage are separate domains. To improve scientifically, collect consented footage, split by recording/venue, freeze parameters on tuning clips, label hidden validation clips independently, and include disagreement. Compare temporal models on the same labels, device and decode resolution.

Court accuracy needs held-out landmarks. Bounce evaluation needs labeled bounce/hit/occlusion events. Neither exists here in sufficient quality, so those metrics stay null. Hardware, versions, sample sizes, elapsed time and memory must accompany any measured claim.

Run from the repository root:

```sh
uv run python scripts/learning.py 6
```
