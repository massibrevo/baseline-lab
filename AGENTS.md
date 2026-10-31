# Baseline Lab contributor instructions

## Architecture and ownership

`src/baseline_lab` owns domain contracts, media, model adapters, court geometry, association, candidate events, rendering, private storage, CLI and FastAPI. `jobs/worker.py` exclusively owns durable worker orchestration; inference runs in an isolated child, never HTTP handlers. SQLite claims use BEGIN IMMEDIATE; an OS lock enforces one worker per data directory. `apps/web` owns the responsive static TypeScript review desk. Its Pydantic-derived `contracts.ts` is generated with `scripts/generate-types.py`. `examples` holds only attributed public footage, known calibration and provisional labels. `docs` records evidence and learning. `data` is private ignored runtime state. These are review responsibilities, not exclusive file ownership among humans.

## Commands and completion

Use Python 3.12.14 and uv.lock; `uv sync --frozen`; `uv run baseline-lab setup-models`; `npm ci --prefix apps/web`; `npm run build --prefix apps/web`; start `uv run baseline-lab serve` and `uv run baseline-lab worker` in separate terminals. Check `uv run ruff check src scripts tests`, `uv run ruff format --check src scripts tests`, `uv run mypy src`, `uv run pytest -q -m 'not model'`, regenerate types and build frontend. Model tests are explicit and separate. Completion requires a real supported workflow or a documented concrete blocker, playable exports, visible errors, and an updated executed/skipped/blocked verification report. Do not claim Docker, CI, hosted uploads or real-model accuracy from configuration alone.

## Coordinate and measurement rules

Display pixels mean rotation-normalized, bounded analysis pixels. Metadata retains source display dimensions and per-axis resize ratios; never confuse source/model/analysis coordinates. Model-input pixels are temporary; tile offsets must be inverted. Image normalization is x/width,y/height. Court meters have origin at far-left outside-painted singles corner, x right (8.23 m), y toward near baseline (23.77 m). Use real PTS relative to the first frame, preserve original integer PTS/time_base, never frame/nominal-FPS for timing. Browser overlay aspect ratio follows display geometry.

Homography is for ground-plane points and approximate athlete footpoints. Never ground-project an airborne ball as true position/speed. Raw/detected/interpolated/missing states remain distinguishable. Interpolation is at most 3 frames and 100 ms. Do not add gaps or >12 m/s links to player distance. Validated ball 3D speed/spin is unavailable. A separate opt-in experimental speed fit may expose assumptions, sensitivity and missing support; never overwrite the validated scalar or use it as independent sensor evidence. All first-release line calls stay uncertain because independent error bounds do not exist. Scores are not calibrated probabilities. Coverage is not accuracy. Preserve original event predictions when editing. Camera cuts fail. Verified camera registration transforms calibration per frame; unverifiable mappings leave that frame without meters. Never retain a stale reference homography as current geometry.

## Licensing and privacy

Original code MIT; example media/derivatives CC BY-SA 4.0. Never assume a Hub card licenses dependencies or datasets. Do not add unverified weights or run arbitrary remote checkpoint code. Check SHA256 and use weights_only loading. No paid provider calls. No secret, private upload, result directory, cache or weight binary in Git/Sites. Every private read/mutation/file must check ownership. Restrict local origin/hosts; never convert the development API into hosted private processing without proper authentication and independent cross-user tests.

## Working policy

Prefer small cohesive modules and honest failure over decorative controls. Test meaningful scientific/lifecycle behavior, then avoid unnecessary repetitions. Keep docs and model manifests aligned with actual execution. Agent profiles in `.github/agents` describe actionable duties only; do not fabricate delegation or reports. No runtime multi-agent AI is needed. Do not spawn subagents unless directly requested by the user or an applicable skill explicitly directs delegation.

## Live camera path

`api/camera.py` transports bounded JPEG batches; `jobs/camera.py` runs in the same isolated single-worker slot as uploads. `video/camera_recording.py` preserves sampled capture timing. `apps/web/src/camera.ts` owns browser permission/capture/review cleanup. Never overlay delayed camera analysis on a newer live frame; show the matching analyzed image with latency. Use only manually reviewed contacts for shot depth and use verified current-frame camera mappings and invalidate unsupported geometry. Test `tests/e2e/test_camera.py`; distinguish protocol replay from physical hardware verification.
