# Hosting separation and remaining integration

Checked **31 October 2026**. No paid infrastructure or provider requests were enabled.

The installed Sites runtime reference describes Cloudflare Workers (128 MB/isolate, HTTP external connections, no raw TCP) and static builds. This application uses a static TypeScript/Vite build there. Native PyTorch/OpenCV/PyAV and the long-lived Python worker run separately. A browser on a hosted origin cannot assume access to a user's localhost. The deployed Site therefore offers a working public precomputed study, browser-local review edits and exports, and explains how to run private processing locally. It does not claim hosted uploads work.

Hugging Face integrations are distinct:

1. **Hub download:** retrieve a pinned file, verify its hash and run locally. CourtSide Hub metadata at the inspected revision lists `model.pt`; no Inference Provider was mapped. Its card does not imply a free API.
2. **Inference Providers:** a separately supported provider/model endpoint, credentials, billing and limits. None is configured or called. [Provider pricing](https://huggingface.co/docs/inference-providers/pricing) is subject to change; do not assume credits mean unlimited compute.
3. **Docker Space:** deploy your own service on hosted compute. [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) currently states static Spaces are free, while creating Gradio/Docker compute Spaces requires an eligible paid plan, with limited free ZeroGPU Gradio exceptions. [Hardware specifications/prices](https://huggingface.co/docs/hub/spaces-gpus) and [storage documentation](https://huggingface.co/docs/hub/spaces-storage) are the authoritative changing references. Free/low-cost hardware can sleep or cold-start, CPU is not GPU, storage may be ephemeral, and replication/concurrency affect SQLite correctness. No fixed price or free-GPU entitlement is promised here.

Dated price snapshot (2026-10-31): the [hardware table](https://huggingface.co/docs/hub/spaces-gpus) lists CPU Basic as 2 vCPU / 16 GB RAM / 50 GB disk with no hourly compute charge but a paid-plan creation requirement; CPU Upgrade is $0.03/hour and T4-small $0.40/hour. [Provider credits](https://huggingface.co/docs/inference-providers/pricing) are none for free accounts and $2/month for PRO (organization terms differ). Space local disk is ephemeral; current docs recommend attached Storage Buckets for persistence. These are reference prices, not enabled services or an account-specific entitlement.

## Exact work before real hosted private processing

1. Select an HTTPS Python container host and inspect its current plan, CPU/RAM limits, storage persistence, backups, sleep/cold-start behavior, egress limits, concurrency and price. Obtain explicit user credentials/configuration. No such backend is provisioned in this release.
2. Replace local browser capability issuance with authenticated user identity, using a validated auth gateway/session exchange. Check ownership on every read/write/download. Keep secrets only server-side. Configure secure cookies, exact allowed origins, trusted proxy/host policy and CSRF handling; the current local-only host guard deliberately blocks arbitrary remote hosts.
3. Provide one private persistent volume and exactly one worker, or migrate to PostgreSQL/object storage/durable queue before scaling. Do not place user uploads in a public Space static tree, logs or example directory. Configure documented retention/deletion/backups.
4. Implement the Sites server gateway/token exchange against that service; use authenticated fetch and expiring owned artifact access where needed. Do not bake a backend API key into the browser or publish a user-supplied URL proxy.
5. Execute real HTTPS upload → job → result → event edit → range playback → export → cancellation → deletion tests across two accounts, including forbidden cross-account access and quota errors. Only then enable hosted upload controls and describe the setup as connected.

A development-only tunnel must have an access gate, short lifetime, restricted audience and exact origin configuration. This project does not create one implicitly. Docker Compose is the reference packaging, not proof of a deployed remote service. Production scale is documented in architecture/system.md; no Kubernetes/Redis/service mesh is necessary for this local release.
