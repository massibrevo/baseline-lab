# CourtSide local comparison — 31 October 2026

The user requested a free trial if useful. The published `Davidsv/CourtSide-Computer-Vision-v1` checkpoint was downloaded to an isolated work directory, not the application dependency tree. Revision `2c66f41b89ec4614f652f52c1c9593d0c00a5fad`; size 5,488,986 bytes; SHA256 `be98152dc5ddcf5552d2b71ae6480d2c6bcf360a1f2369e21b4dc66da442dfcb`, matched against the Hub LFS metadata. Ultralytics 8.4.195 loaded it with its explicit safe-loading mode (`weights_only=True`, known-class allowlist). No unsafe fallback, API inference, private-media upload or paid call was used.

The actual class list is racket, tennis_ball and eight court/net zones. It has no person joints, stroke classification or metric-speed output. The screenshot on the model card cannot establish that this checkpoint alone produces the entire illustrated pipeline. The card reports MIT, while the Ultralytics runtime has separate AGPL terms; neither overrides dataset/base-model rights. The application does not add or redistribute either runtime or checkpoint.

A small development comparison used the nine existing provisionally labeled frames from the attributed public example. Seven contain a visible ball annotation. These are not independent human ground truth or a representative benchmark.

| Input / threshold | Visible-ball candidate locations within 15 px | Ball candidates returned | Racket-positive frames | CPU inference call for nine images |
|---|---:|---:|---:|---:|
| 640 px / 0.15 | 0/7 | 0 | 5/9 | 1.168 s |
| 1280 px / 0.10 | 0/7 | 0 | 8/9 | 1.960 s |

Timings exclude initialization and video decoding, and are not application latency. Candidate recall is an upper bound before tracking, not tracking accuracy. This local result gives no evidence that replacing the temporal tennis detector with CourtSide would help this clip domain. It is therefore **not adopted**. No claim is made that the model fails on all tennis videos. Racket detection could be investigated separately with suitable annotations; it would not by itself establish stroke type or contact timing.

Primary source: [model card and class list](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1).
