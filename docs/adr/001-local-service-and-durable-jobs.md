# ADR 001 — Separate native inference; use SQLite and one isolated child

Status: accepted for bounded local release, 2026-10-31.

Sites runs Workers/static JavaScript, not a long-lived native Torch/FFmpeg worker. The frontend must remain a portable static review desk. FastAPI plus a durable SQLite job table and filesystem artifacts provide an inspectable local deployment without paid services. Atomic BEGIN IMMEDIATE claims and an exclusive worker file lock prevent double processing. An inference child adds timeout/cancellation isolation; per-job partial directories prevent incomplete artifacts from looking successful.

Tradeoff: one host/worker, polling and local disk, no multi-machine scaling. CLI and HTTP share one pipeline. PostgreSQL/object storage/external queue is the next scaling boundary, not a dependency for this release. The hosted demo cannot accept private uploads until a real authenticated HTTPS backend is integrated and tested.
