# Critical review: Badminton AI Analytics

Reviewed 4 November 2026 at upstream revision `038b27b11b5fbe6b56eb6a5f32dcf21f9399d198`.

Source: [Muhammad Rahman Shahid / Badminton-AI-Analytics](https://github.com/muhammadrahman-hehe/Badminton-AI-Analytics/tree/038b27b11b5fbe6b56eb6a5f32dcf21f9399d198). Inspected README, repository tree, player detection/cleaning, hit logic, rally logic and dashboard source. The inspected tree has no license, custom model weights, training code, held-out evaluation or tests. We inspected source only; we did not execute upstream scripts or copy their code/assets. The additions below are original TypeScript implementations of general dashboard ideas, retaining our architecture and data contracts.

## What transfers

| Component | Upstream behavior | Baseline Lab decision |
| --- | --- | --- |
| Movement heatmap | Adds one count for every frame at each tracked footpoint | Adapt the idea with **real PTS duration** in 1 m court cells. Require adjacent detected footpoints, available geometry at both endpoints, at most 100 ms separation and at most 12 m/s displacement. This makes supported dwell comparable across frame rates and excludes tracking gaps. Labels show supported seconds and fraction of elapsed time, not model accuracy. |
| Player paths | Short recent paths and cumulative distance bars | Add selectable last-3-second paths and per-player cards beside our video. Preserve gaps. Reuse existing robust speed-integrated distance instead of introducing a second noisy distance calculation. |
| Hit and landing map | Draws hitter footpoints and estimated landing markers | Add keyboard/click-seekable **stroke footpoints** and **reviewed bounces**. Do not map an airborne ball onto the floor. Automatic impact candidates remain separate from reviewed strokes; clip-boundary swings are not counted as contacts. |
| Synchronized dashboard | Baked into an OpenCV output video | Adapt to interactive TypeScript controls: movement / time heatmap / stroke positions and both / near / far player. Seeking backwards removes future marks, counts and dwell. No inference rerun or new dependency needed. |

## What does not establish an improvement

- **TrackNetV3 + InpaintNet:** upstream references the separate `qaz812345/TrackNetV3` badminton pipeline. It does not provide a tennis comparison or its own trained model. Baseline Lab already uses a tennis-specific temporal WASB checkpoint. Shuttlecock-trained weights and badminton inpainting are not evidence of better tennis-ball localization. Do not change the selected model or feed inpainted positions into speed/depth without a separate held-out tennis evaluation.
- **YOLOv8n-pose + BoT-SORT:** a potentially useful detector/tracker alternative, but there is no comparative tennis result or runtime benchmark against our SSDlite + RTMPose + court association. Adopting it would add Ultralytics/model licensing and deployment work without an established benefit. Existing ankle-based footpoint projection and court-half filtering already cover its useful geometric idea.
- **Player cleaning:** upstream interpolates up to 15 frames and applies a centered rolling median without retaining separate observation states. Its per-frame nearest-to-half-center selection can change identities. We retain our temporal association and missing-state contracts.
- **Hit/serve rules:** hardcoded pixel/frame thresholds, long interpolated shuttle gaps, enforced hitter alternation, and both-players-still serve rules are tuned to badminton. These are not a replacement for our learned tennis event/attribute model and would mis-handle practice footage, occlusion, frame-rate changes and incomplete clips.
- **Winner/score rules:** a stationary shuttle or last-seen point is used as a landing and next server overrides the winner. Tennis balls bounce and tennis service does not follow badminton's winner-serves-next rule. No automatic scoring, line-call certainty or forced-error attribution is imported.
- **Time alignment:** frame/fps calculations and frame-count realignment discard real variable-rate timing. Keep our original PTS path.

## Demonstrable gains and limits

The new court view adds occupancy, recent movement, per-player support/count/path summaries and seekable stroke positions that were absent from the old dot-only map. Unit tests establish timestamp weighting, exclusion rules, backwards seeking, label-source separation and footpoint-only plotting. This is a dashboard and evidence-handling improvement, **not a measured improvement in CV model accuracy**. No new model was installed, private video was not uploaded, and no paid API was enabled. Existing media exports, raw tracks, experimental speed and reviewed depth remain their original evidence.
