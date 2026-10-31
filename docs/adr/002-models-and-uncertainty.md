# ADR 002 — Explainable baseline, explicit uncertainty and separate model rights

Status: accepted as an experimental baseline, 2026-10-31.

CourtSide detects rackets, balls and court-zone boxes, but lacks person and court-keypoint classes. Its MIT card does not settle Ultralytics dependency or underlying training-data terms. WASB offers useful temporal ideas but independently verified checkpoint/data redistribution rights were not established. Neither is bundled.

Use an explicit official SSDlite checkpoint download (weights_only, full SHA256, fixed COCO label=1) and local color/motion ball adapter. At 1280 width, overlapping tiles improve small person scale at the cost of CPU latency and duplicate suppression. HOG provides an offline fast-test fallback, not a credible replacement for distant athletes. The source code remains MIT, media CC BY-SA, dependency and pretrained rights separate.

A four-point manual homography is inspectable and tested mathematically. Its fit residual cannot establish accuracy. All first-release line calls stay uncertain; airborne ball speed stays null. Candidate bounce extrema additionally need temporal support and reject proximity to a person, but still require review. Temporal models, independent labels and held-out private-camera evidence are prerequisites to stronger product claims.
