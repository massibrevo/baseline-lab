# Executed verification — 31 October 2026

This is a working experimental local vertical slice and hosted-example frontend, not a completed broad real-world accuracy validation. Statuses below distinguish execution from configuration. The exact reproducible commands are in README.md.

## Local checks

| Check | Status and evidence |
|---|---|
| Locked Python environment and frontend dependencies | Executed; uv.lock and package-lock.json; Python 3.12.14, Node 22.12.0 |
| Ruff lint / formatter | Passed on src, scripts and tests |
| mypy | Passed across 16 source modules |
| Pydantic-generated frontend types / production TypeScript build | Passed |
| pytest | **22 passed**, including the explicit real-checkpoint/real-image model test; one Starlette/httpx deprecation warning |
| Geometry | Known correspondences, inverse, degenerate/order rejection, letterbox offsets, uncertainty around lines passed |
| Timestamps | Real-PTS MP4/VFR decode and annotated H.264 roundtrip passed; original integer PTS retained |
| Rotation | H.264 MOV with a real 90-degree display matrix passed; normalized display dimensions and PTS verified |
| Association | Bounded gaps, impossible ball link, player side assignment/spectator-region rejection, distance excluding gaps/jumps passed |
| Durable lifecycle | Actual inference child through upload→submit→worker→result→CSV/MP4→correction→deletion passed; idempotent submissions, atomic claim, restart state recovery, active cancellation and timeout cleanup passed |
| Private access / invalid media | Two browser capabilities, denied cross-session reads/source/deletion, CSRF, invalid extension/data, worker-unavailable and physical deletion passed |
| Real model | Official SSDlite COCO_V1 checkpoint downloaded, SHA256 verified, weights_only loaded, real inference passed |
| Real clip | Entire 12.5 s / 375-frame stabilized attributed singles clip processed, annotated and browser-review H.264 files generated |
| Docker | Compose configuration validated; container build/run **not executed** because the installed Docker client has no running daemon |
| Remote private inference | **Not provisioned/not tested**; hosted Site is demo-only, no private HTTPS Python service |
| Independent human labels / private baseline-camera validation | **Not available**; provisional visual labels are explicitly not an independent human benchmark |
| Peak memory | **Not measured**; `/usr/bin/time -l` resource details were blocked by an OS sysctl restriction; no estimate substituted |

The offline suite executes a real child worker and codecs with the built-in HOG fallback; it is not merely a mocked queue/provider test. Separate real-model inference runs after explicit setup. The source uses a known HOG-capable OpenCV version after an executed API-compatibility failure with the initially resolved OpenCV 5 package. Current pins are authoritative.

## Measured perception limits

Machine: **Apple M3 Pro, 12 cores (6 performance/6 efficiency), 18 GB RAM, arm64 macOS 27.2**. CPU inference uses 2 Torch threads. Full example pipeline: **84.614 s** internal elapsed; **85.66 s** external wall measurement, including process startup. Hardware model and memory were read from filtered system_profiler output, without device identifiers. These timings are one run, not an SLA.

[Machine-readable provisional evaluation](evaluation-provisional.json): one real source recording, early 3-frame tuning sample and later 6-frame visual-check sample. Only four later ball locations were sufficiently visible for provisional point annotation. These labels were made visually by Codex during implementation, not independently verified by a human. The later segment was observed during development, so this is a smoke study, not a sealed holdout.

- Later ball localization: **2/4** labeled points within 15 px; one missing and one observed false association (about 74 px error). Observed-point errors `[74.20, 0.50, 0.00]` must be read with that missing point; the median alone would hide the failure.
- Near player: **6/6** later provisional footpoints within 35 px; median observed error about **9.1 px**. This does not establish tracking identity or metric-distance accuracy.
- Far player: **1/6** later labeled frames observed/in tolerance; **5/6 missing**. Distant athlete detection is a major limitation.
- Full-clip observation coverage: ball-like **69.1%**, near **93.9%**, far **13.3%**. Coverage is not localization accuracy; false detections count toward observed coverage.
- **8 candidate contacts**, all unconfirmed. Bounce timing/location precision is **null**, because no independently labeled bounce set exists. Candidate extrema are not verified bounces.
- Independent court error, calibrated uncertainty, line-call accuracy, 3D ball speed and spin are **unavailable**. A four-point residual is not an independent court-error estimate. All line calls stay uncertain and 3D speed stays null.

The example is a stabilized oblique elevated camera study. The unstabilized source shows substantial camera drift; using its initial homography through the rally would be wrong. The pipeline invalidates metric geometry for detectable/unverifiable drift, and strong camera cuts fail with a recalibration message. These are conservative heuristics, not full-match shot segmentation. No private-camera or broadcast-general accuracy result is implied.

## Browser checks

The local app was opened and exercised in the Codex in-app browser. Tested: real public example loading and playback controls, PTS frame stepping, overlay controls, selectable contact review, full metadata upload, four-corner calibration and real worker progress. Desktop and narrow responsive layouts were visually inspected. Screenshot is in `workspace.png`; source-media attribution applies. Additional final browser/publication outcomes are recorded below when completed.

## Known limits and next verification

No independent human bounce/court ground truth, private-baseline validation set, automatic court-keypoint model, metric uncertainty calibration, hosted authentication, paid inference endpoint, audio export, doubles or full-match segmentation is claimed. The complete local workflow is implemented and tested within bounded resources; perceptual quality is exploratory. Obtain consented diverse baseline-camera clips and independently verified labels before judging a temporal ball-model improvement or enabling stronger line/speed claims.

Final local browser result: the uploaded public clip completed through the actual durable worker, reported warnings, opened its generated review MP4, sought a candidate to 1.400 s, stepped to 1.433 s at 30 fps, saved a rejection with the original prediction intact, and downloaded versioned JSON. Browser video decoded at 1280×720 with readyState=4 and no media error; the saved screenshot shows paused review at 6.500 s. The direct local service was reachable and served range requests. Temporary browser-test data is local only and excluded from source/static publication.
