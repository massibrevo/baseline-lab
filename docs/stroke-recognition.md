# Automatic stroke recognition

The local worker uses the released **F3ED tennis event localizer and multilabel classifier**, described by Zhaoyu Liu et al. in [F3Set, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c8e2cd11daf2d0a8c68663d504a841d-Abstract-Conference.html). This is a learned temporal video model, with an original local inference adapter. It replaces unclassified proximity candidates as the default stroke readout. The selected checkpoint is 22.65 MB, separately downloaded and SHA256 verified. Runtime inference stays on the machine.

The recognition function receives source video, metadata and observed player skeletons. Its interface excludes human stroke annotations, reviewed contacts and calibration. **Automatic** and **Human reviewed** are separate UI sources and separate JSON arrays (`automatic_shots` and `shots`). Reviewing a prediction creates a human record and preserves the original neural prediction. A saved timestamp cannot become an automatic classification rule. The recognition weights are unchanged by reviews.

Frames are sampled using real presentation times at 12.5 Hz, resized with preserved aspect ratio and center cropped/padded to the training geometry. RegNet features, temporal shifts and a bidirectional GRU supply event and stroke attributes over overlapping 96-frame windows. Scores from original and horizontally flipped views are averaged. Swapping the input orientation therefore swaps the two passes, leaving the average unchanged. This is an invariance property, **not proof that either prediction is correct**. The optional published contextual head is disabled because player handedness is not independently verified.

A neural impact candidate must pass an event threshold. Forehand/backhand labels also need a strong attribute score and separation between the two wings; conflicts remain **uncertain**. Serve, overhead and volley use their own attribute checks. A visible swing cut off by the clip boundary may receive a stroke label from body motion and the attribute head, while explicitly retaining `clip_boundary_swing_not_verified_contact`. It never becomes a verified racket contact. The displayed swing window spans preparation and follow-through; the candidate timestamp is not a ground-truth contact annotation.

Camera analysis uses a bounded causal image history. It waits for adequate contiguous motion context. Skipped capture intervals are excluded, and sparse bursts cannot be stretched into a recognized swing. Processing latency and matching analyzed-frame timestamps remain visible. Physical webcam testing is still outstanding.

## Validation and limits

Automated checks cover neural-score decoding, ambiguity, arbitrary model event times, clip-boundary motion, sparse camera gaps, immutable automatic predictions after human correction, and UI source separation. A real-checkpoint test on the licensed public example verifies finite outputs and mirrored-input equivalence within `2e-5`. File recognition and the complete pipeline are exercised separately. These checks establish implementation behavior, not a broad accuracy percentage.

The model was trained on tennis broadcasts. Oblique views, partial bodies, tiny players, occlusions, camera cuts and other domains can fail. There is no measured tenfold accuracy gain or guarantee for every video. Scores are uncalibrated neural outputs. Predicted stroke attributes do not supply physical ball speed, spin, bounce depth, line calls or forced-error judgments.

General Qwen vision models (4B and 9B), an older tennis LRCNN, and small classifiers trained on a public single-venue pose/image dataset were evaluated and rejected for transfer or mirror errors. The latter dataset is [Wang et al., Mendeley Data V1, CC BY 4.0](https://data.mendeley.com/datasets/nv3rpsxhhk/1); training used only that public dataset. Those experimental models, training packages and datasets are not part of the app, repository or Site. No private clip or user correction was used to train a classifier.

See [model manifest](../MODEL_MANIFEST.yaml), [notices](../THIRD_PARTY_NOTICES.md) and [verification](verification.md).
