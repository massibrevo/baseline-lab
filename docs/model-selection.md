# Model selection and provenance

Research checked 2026-10-31. Selection prioritizes a reproducible local CPU path, transparent failures, and separable licenses. Source inspection and downloads are distinguished from model-quality evaluation; executed evaluation belongs in the verification report.

## First-release baseline

Use manual court landmarks, a narrow ball-color/motion baseline, and a separate person detector. Color/motion detection has no downloaded learned ball weights and is expected to miss blurred, occluded, low-contrast, and tiny balls. Yellow clothing, signage, and court markings can create false candidates. Its score is a heuristic ranking, not a probability of ball presence. Missing observations are valid outputs. A temporal learned model is a later measured experiment, not an advertised capability of this baseline.

For people, the recommended learned option is Torchvision SSDlite320 MobileNetV3 Large COCO V1. OpenCV's built-in HOG people detector is a download-free fallback, but its 64×128 detector window and upright-person assumptions make distant tennis players challenging. Both need court-region filtering and temporal association; neither identifies the players by name. [OpenCV HOG API](https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html).

| Candidate | Verified evidence | Decision |
| --- | --- | --- |
| Torchvision SSDlite320 MobileNetV3 Large, `COCO_V1` | Official fixed weight URL and SHA256 verified by downloading in memory; person class is label 1. Torchvision code is BSD-3-Clause; model/dataset terms remain separate | Small optional learned person detector, tested independently from ball logic |
| OpenCV HOG `getDefaultPeopleDetector` | Built-in linear SVM coefficients for 64×128 windows; no Hub or provider dependency | Offline fallback with explicit coverage limitations |
| HSV/motion ball candidates | Original application logic; no checkpoint or training-data download | Honest minimal baseline; requires real-video coverage and false-positive evaluation |
| Davidsv CourtSide v1 | Actual `model.pt`, revision, LFS digest, card classes, and example code inspected | Excluded from default dependencies; licensing and accuracy need a separate experiment |
| NTT WASB-SBDT | MIT implementation and a model-zoo tennis checkpoint link exist | Strong temporal research candidate; checkpoint terms, download, checksum, and local runtime not verified here |
| alenzenx TrackNetV3 | MIT implementation; linked checkpoint and three-frame prediction workflow; documentation targets badminton | Domain mismatch and unverified checkpoint terms prevent default inclusion |

## Torchvision person model

The pinned [v0.24.1 SSDlite source](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/detection/ssdlite.py) declares `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1` and its fixed official URL:

- File: `ssdlite320_mobilenet_v3_large_coco-a79551df.pth`.
- URL: <https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth>.
- Observed bytes: `14069355`.
- Full SHA256 measured from the downloaded bytes: `a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`.
- Class filter: COCO label `1` = `person`; do not remap all output boxes to people.
- Preprocessing: BGR → RGB, CHW float tensor in `[0,1]`; the architecture's internal transform resizes to 320×320 and maps boxes back to the input tensor's dimensions. Preserve any application resize separately. [Official transforms](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/transforms/_presets.py), [model API](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html).
- Load as a checked state dictionary with `torch.load(..., map_location="cpu", weights_only=True)` into `weights=None, weights_backbone=None`. This avoids executing remote repository code and avoids an implicit second backbone download. Call `eval()` and use inference mode.
- Cache outside Git; verify the complete checksum before loading. After setup, inference must need no network.

[Torchvision's BSD-3-Clause license](https://github.com/pytorch/vision/blob/v0.24.1/LICENSE) governs its code. Its [official model documentation](https://docs.pytorch.org/vision/stable/models.html#general-information-on-pre-trained-weights) separately warns that pretrained models may inherit terms from training datasets; it does not grant a blanket BSD license to every weight. See [COCO's terms](https://cocodataset.org/#termsofuse). This project downloads weights from the official source for the educational reference workflow and does not commit or relicense them. A commercial deployment must determine whether its intended use satisfies those separate terms.

OpenCV 4.5+ is [Apache-2.0](https://opencv.org/license/); the [HOG implementation](https://github.com/opencv/opencv/blob/4.12.0/modules/objdetect/src/hog.cpp) also carries retained Intel/Willow Garage BSD-style notices. Preserve dependency notices when distributing binaries. No HOG coefficient file is copied into this repository.

## CourtSide candidate: what actually exists

[Hub model](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1), inspected revision [`2c66f41b89ec4614f652f52c1c9593d0c00a5fad`](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1/tree/2c66f41b89ec4614f652f52c1c9593d0c00a5fad).

The tree contains `.gitattributes`, `README.md`, `config.txt`, `example.py`, `example_detection.png`, `model.pt`, and `requirements.txt`. `model.pt` is 5,488,986 bytes; the Hub tree's LFS SHA256 is `be98152dc5ddcf5552d2b71ae6480d2c6bcf360a1f2369e21b4dc66da442dfcb`. That digest is server metadata, not a locally executed checkpoint verification. The example constructs `ultralytics.YOLO("model.pt")`; requirements are broad lower bounds, not a reproducible lockfile. [Pinned files](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1/tree/2c66f41b89ec4614f652f52c1c9593d0c00a5fad).

The [card](https://huggingface.co/Davidsv/CourtSide-Computer-Vision-v1/blob/2c66f41b89ec4614f652f52c1c9593d0c00a5fad/README.md) states YOLOv11n, 640×640 input, and these classes:

| ID | Class |
| --- | --- |
| 0 | racket |
| 1 | tennis_ball |
| 2 | bottom-dead-zone |
| 3 | court |
| 4 | left-doubles-alley |
| 5 | left-service-box |
| 6 | net |
| 7 | right-doubles-alley |
| 8 | right-service-box |
| 9 | top-dead-zone |

These are the published card classes; the checkpoint class map was not loaded and independently checked. There is no person class. Zone boxes do not provide court-line intersections. The card names Roboflow ball/racket/court datasets, but does not establish held-out performance on this application's private-camera domain. Its reported metrics are author claims, not Baseline Lab measurements.

The card declares MIT; the inspected file tree contains no standalone license text. The [Ultralytics source license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) is AGPL-3.0, with separately offered enterprise terms. The card's metadata does not override the dependency's terms or settle training-data and base-weight rights. Baseline Lab does not install Ultralytics, download this checkpoint, or claim those issues are resolved.

The Hub metadata API returned `inferenceProviderMapping: {}` on the research date. This supplies no evidence of a supported Inference Providers endpoint; no real hosted request was executed. A model listing and example Python script are not an unlimited free API.

## Temporal candidates

[WASB-SBDT](https://github.com/nttcom/WASB-SBDT/tree/923462cacdeb3353b84ddebdedb3f4b7a8553b0f), revision `923462cacdeb3353b84ddebdedb3f4b7a8553b0f`, supplies evaluation code under [MIT](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/LICENSE.md) and a [model zoo](https://github.com/nttcom/WASB-SBDT/blob/923462cacdeb3353b84ddebdedb3f4b7a8553b0f/MODEL_ZOO.md) with tennis-specific WASB and TrackNetV2 download links. The README reports a Python 3.8/CUDA 11.3 environment. Code license verified; separate checkpoint/data license, file digest, download reliability, CPU compatibility, and performance in this app remain unverified. Do not invent those values in the model manifest. An adapter may be added after those gates pass and a held-out labeled clip shows an improvement.

[TrackNetV3](https://github.com/alenzenx/TrackNetV3/tree/d1d96f39d275970d3597a45926f4f46e6ee85e2c), revision `d1d96f39d275970d3597a45926f4f46e6ee85e2c`, has [MIT code](https://github.com/alenzenx/TrackNetV3/blob/d1d96f39d275970d3597a45926f4f46e6ee85e2c/LICENSE) and a README-linked training checkpoint. Its examples and data conversion target badminton; its headline accuracy cannot be transferred to tennis footage. No checkpoint was downloaded or redistributed here. A temporal architecture alone does not validate bounce detection.

## Three distinct Hugging Face paths

1. **Hub download:** fetch a revision-pinned, checksum-checked file and run locally. No paid inference request is implied.
2. **Inference Providers:** use only a model/provider combination that is actually supported, after testing its real endpoint and current quota/pricing. Disabled in this release.
3. **Docker Space:** deploy the complete Python application on a container host, with explicit authentication, owner checks, persistence, resource limits, and restricted CORS. This is application hosting, not a property of a Hub model listing.

No candidate research above enables billing, paid API calls, or automatic provider fallback. Hosted demo playback must remain distinct from a verified hosted private-processing service.
