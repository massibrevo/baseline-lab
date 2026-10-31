---
name: backend
description: Maintain private API contracts, durable jobs, storage and exports.
---

Own api/jobs/storage/video interfaces. Keep inference out of HTTP handlers; preserve atomic claims, cancellation, restart cleanup and owner checks on every artifact. Run uv run pytest tests/integration tests/e2e -q -m "not model" plus mypy and Ruff. Verify active cancellation and playable timestamp-preserving exports.
