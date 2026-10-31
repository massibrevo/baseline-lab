# 7. Serving, jobs and CPU/GPU deployment

An HTTP response lifetime is not a durable job lifetime. A request-scoped background task can disappear on restart or duplicate inference on retries. SQLite stores states and a per-session idempotency key. BEGIN IMMEDIATE serializes the claim; an OS lock enforces one worker; one child isolates expensive inference and can be terminated on timeout.

Per-stage frame counts are meaningful progress. Model loading and final transcoding have unknown work counts and use indeterminate progress. Cancelled/failed temporary output is never published as completed artifacts. Restart recovery requeues interrupted jobs and cleans partial/tombstoned directories.

CPU inference is simple, reproducible and bounded but tiling multiplies work. GPU speedups require actual benchmarking, memory measurements and hardware availability, not a model-card headline. Hosted Spaces have plan/compute/storage constraints; a Hub repository is not a free inference endpoint. Sites provides the browser review layer; the native pipeline remains separate.

Scale to PostgreSQL, private object storage and a durable external queue only after one-host limits matter. Hosted private processing additionally needs real authentication, ownership on every artifact, secure transport and retention policies. The runnable example demonstrates idempotent jobs and a single atomic claim without network services.

Run from the repository root:

```sh
uv run python scripts/learning.py 7
```
