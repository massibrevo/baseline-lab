# Contributing

Read AGENTS.md, docs/verification.md and MODEL_MANIFEST.yaml. Install with the README's locked startup path. Keep production logic in `src/baseline_lab`; notebooks explain it rather than duplicate it. Run Ruff, mypy, offline pytest, generated-types consistency and the frontend build before proposing a change. Add tests for geometry, timestamp, evidence, privacy or lifecycle changes; avoid tests that merely repeat implementation constants.

A perception change needs labeled evidence, clip rights, a fixed split, thresholds/device/version provenance, sample sizes, failures and elapsed time. Never describe coverage as accuracy or a model score as probability. Do not silently interpolate long gaps, identify spectators as known athletes, infer airborne ground position, or replace nulls with plausible values.

Keep uploads, results, caches, secrets and weights out of Git. Public examples require independently checked media rights and attribution. New model adapters need explicit file/revision/checksum/class mapping/preprocessing and a licensing review separate from code licensing. Changes affecting private hosting need ownership, TLS, retention and real end-to-end evidence before exposing the workflow.
