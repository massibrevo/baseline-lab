# Executed verification — 1 November 2026

This is an experimental local analysis app and a precomputed hosted example. It has not passed independent broad tennis-perception or physical-speed validation.

## Current checks

Seven frontend behavior tests also pass, including automatic/reviewed separation and continuous/observed-only/point trails. New display-only trajectory migrations preserve raw tracks, reviewed events and speed evidence. The full current public example run completed with automatic stroke recognition and playable annotated exports.

- Ruff lint/format, mypy across 32 application modules, Pydantic-generated TypeScript and production frontend build pass.
- 51 offline unit/integration/end-to-end checks pass. They include an actual isolated child worker, codecs, durable jobs, cancellation/timeouts, ownership/CSRF controls, event and stroke review, VFR/rotation, bounded 4K import, proxy timestamps, camera mapping, association, pose transforms, smoothing, metric gaps, Shot Quality and synthetic flight consistency.
- Three explicit real-checkpoint checks pass: cached SSD/RTMPose inference mirrored stroke-model equivalence, and actual API/worker camera transport using licensed tennis frames and WASB. One Starlette/httpx deprecation warning remains.
- An initial real-camera transport run found an idle-timeout bug: slow first-burst inference counted as time waiting for new input. The idle clock now restarts after processing; the actual model transport test then passed. Physical webcam permission/lens capture remains unverified in the in-app browser.
- The reported high-resolution MOV uploads through the browser/API, creates a playable bounded analysis copy, completes analysis and reopens by private deep link. Every decoded PTS survives the preview at the 90 kHz encoder/muxer time base. Private footage, geometry, telemetry, identifiers and screenshots stay outside Git/Site artifacts.
- Actual private playback shows both 17-joint skeletons, a smooth continuous cyan ball trail, km/h research estimates, player speed/path metrics and the adjacent shot/court panel. PTS stepping updates values; unsupported values remain missing. Playback advances with readyState 4, no media error and no console errors. This browser does not expose frame-drop counters; zero dropped frames is not asserted.
- The UI now has a measurement dock immediately below the video, evidence/freshness badges, direct bounce/stroke review, and an evidence inspector. The unused external-sensor speed card was removed; unsupported values have explicit actions instead of empty dashes. Bounce review no longer silently changes playback to 0.25×. Duplicate frame renders and repeated canvas resizing were removed. Side panels update at a bounded rate. One Euro joint smoothing retains raw coordinates, and robust player-velocity fits use a 0.5 s observed window with bounded temporal smoothing. Player path now integrates those filtered speeds over adjacent observed links; raw footpoint path is retained in provenance. A stationary-jitter regression test prevents raw ankle noise from inflating the displayed distance.

## Public example measurement

Machine: Apple M3 Pro, 12 cores, 18 GB RAM, arm64 macOS 27.2. Native MPS is used for Torch inference when available; OpenCV pose uses two CPU threads. One complete final 375-frame / 12.5-second licensed example run took **105.749 s**, including model initialization, inference and exports. Recorded stages: ball_heatmaps_s 8.119 s, tracking_and_pose_s 83.488 s, annotated_export_s 5.298 s. Automatic stroke analysis took 2.566 s including its decoding/model setup. Stages omit setup/proxy time; their sum is not the whole runtime. These are individual observations, not a latency SLA or real-time guarantee.

Full observed coverage ball/near/far: **73.1% / 97.6% / 75.2%**. Coverage is not accuracy. Later provisional visual check: ball **3/4** within 15 px with one missing; near **6/6** within 35 px; far **5/6** with one missing. Observed ball errors **[2.11, 0.15, 1.78] px**. The development labels were seen during iteration, are not independently human-verified, and are not a sealed holdout. See evaluation-provisional.json. No bounce precision, independent court error or radar-speed accuracy result exists.

CourtSide was separately tested locally with safe weight loading at 640 and 1280 inputs, on nine public frames. It returned no ball candidates on the seven visible-ball annotations in either run. It was not adopted. See courtside-trial.md; the small negative result does not establish general model quality.

## Measurement limits

Validated `ball_speed_m_s` remains null. Schema 1.4 separates experimental 3D fits, assumptions, unsupported intervals and sensitivity ranges. Normal 1× action timing must be confirmed for uploads. The UI can hold a historical supported-flight value with age, never label it as a new observation. This is not reliable continuous speed-gun output. Spin, independently measured bounce angle and forced/unforced attribution are unavailable. Shot Quality is a transparent partial heuristic until the required independent factors exist.

Automatic stroke recognition now uses the pretrained F3ED temporal event and attribute heads with mirrored-view averaging and ambiguity checks. Human annotations remain separate and are excluded from inference. Clip-boundary swings are distinct from verified impacts. See stroke-recognition.md. Corrections preserve original predictions. Reviewers can add a timestamped stroke even when automatic impact detection missed it, and use jump buttons to verify the label. No broad stroke-classification accuracy or tenfold improvement is asserted. Camera bursts are sampled/latest-only and show their matching analyzed frame and latency, not an old overlay on a newer preview. Failed registration leaves the affected frame without ground geometry. All line calls remain uncertain.

## Exposure and publication

The running API was verified listening only at **127.0.0.1:8000**. A pattern scan of 118 Git-history text blobs and current tracked text found no matching GitHub/HF/OpenAI tokens, AWS access keys, private keys or credential URLs. No private data/runtime directory or .env is tracked. This is a limited check, not a penetration test or guarantee of no vulnerabilities. Dependencies and source remain reviewable. Never expose the local browser-capability API directly to the internet.

Public source: https://github.com/massibrevo/baseline-lab. CI runs lint, types, offline tests, generated-contract consistency and frontend build; real-model tests are opt-in. The Site retains its owner-private audience and contains only the licensed precomputed example/frontend. Private hosted Python inference is not deployed. Docker configuration exists but was not built/run because no Docker daemon was available. Peak memory was not measured. No paid providers or external video-inference requests were used.

The broadcast-style visual update uses a single cyan quadratic path with a soft halo and tapered tail in canvas, SVG, camera and video exports. Inference remains identifiable in status/data and observed-only mode. It changes no detector, observations or measurement calculations. Existing geometry, source-separation and lifecycle checks were rerun; real-model inference was not repeated for this rendering-only change.
