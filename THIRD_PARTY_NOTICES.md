# Separate rights and notices

Original application code: MIT. WASB HRNet architecture is vendored under its MIT license, with Microsoft/NTT attribution and the upstream revision retained in the file. See licenses/wasb-MIT.txt. No code was copied from the reviewed Tennis-Analysis-System (Apache-2.0). Its study is attributed in docs/reference-review.md; its media/model weights are not bundled.

## Example media

Oleg Yunakov, “Carlos Alcaraz vs. Wu Yibing (2026 US Open) - 01.webm”, own work, 4 September 2026, [Commons source](https://commons.wikimedia.org/wiki/File:Carlos_Alcaraz_vs._Wu_Yibing_(2026_US_Open)_-_01.webm), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Original: https://upload.wikimedia.org/wikipedia/commons/5/56/Carlos_Alcaraz_vs._Wu_Yibing_%282026_US_Open%29_-_01.webm . Source attribution is not endorsement.

Changes: 3840×2160 to 1280×720, silent H.264 transcode, 60→30 fps in stabilized derivative, planar feature-based stabilization with visible black padding; annotated derivative adds court/player/ball overlays. `examples/commons-*.mp4`, `apps/web/public/demo/*.mp4` and screenshots containing this footage are distributed as CC BY-SA 4.0, **not MIT**. Retain this notice and license link when redistributing adaptations. Result annotations/provisional point labels are provided for reproducible study; underlying media rights remain unchanged.

## Dependencies and pretrained models

- [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE): BSD-style code and its bundled third-party notices.
- [torchvision](https://github.com/pytorch/vision/blob/main/LICENSE): BSD-3-Clause code. [Official pretrained-model notice](https://docs.pytorch.org/vision/stable/models.html): dataset terms may apply. The SSDlite checkpoint is downloaded separately from the official PyTorch model host. COCO image rights vary; no blanket permissive training-image or weight grant is asserted. Check current terms before commercial redistribution/use.
- [OpenCV](https://github.com/opencv/opencv/blob/4.12.0/LICENSE): Apache-2.0 distribution with third-party notices; HOG default people detector is supplied by that package. Wheel notices are installed with the dependency.
- [PyAV](https://github.com/PyAV-Org/PyAV/blob/main/LICENSE): BSD-3-Clause. Linked FFmpeg build licenses remain separate.
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg/blob/main/LICENSE): BSD-2-Clause wrapper; its bundled FFmpeg includes codec libraries with separate LGPL/GPL terms. The executed binary reports its build configuration with `ffmpeg -L`. No binary is committed to this repository. Container redistribution must include the wheel/library notices and comply with the actual codec build.
- FastAPI/Starlette, Pydantic, uvicorn, NumPy, Vite, TypeScript and their transitive dependencies retain their installed license files. Python and frontend lockfiles define the exact distributions. This notice does not replace those licenses.

CourtSide was reviewed but is not installed/downloaded by the app. Its model card says MIT; Ultralytics dependency/model terms (AGPL-3.0 or enterprise) and training-data rights require separate review. WASB code is MIT but separately verified checkpoint/data terms were unavailable. Neither is included by implication.

Fonts: platform sans-serif/monospace/Georgia fallbacks; no downloaded font files. Original SVG court/wordmark uses original geometry, no third-party icon pack.


Current research models: RTMPose-m graph from OpenMMLab / Tau-J's maintainer mirror; code Apache-2.0, dataset/checkpoint terms separate. WASB tennis weights from the author's published model zoo; the software repository is MIT, while no additional blanket grant over training video rights is asserted. Checkpoints are fetched separately with checksums and are not in Git/Sites. The retired MediaPipe evaluation graph was from the Apache-2.0 OpenCV Zoo directory and is not required by the current profile.

## F3ED stroke recognition

[F3Set / F3ED](https://github.com/F3Set/F3Set), Zhaoyu Liu, Kan Jiang, Murong Ma, Zhe Hou, Yun Lin and Jin Song Dong, ICLR 2025, supplies the separately downloaded tennis research checkpoint. Our inference architecture implementation uses timm RegNet (Apache-2.0) and standard PyTorch layers; the temporal shift operation follows MIT HAN Lab's TSM, whose [MIT notice](licenses/tsm-MIT.txt) is retained. Upstream training scripts are not bundled. The checkpoint is loaded only as checksum-verified weights with `weights_only=True`. Its training-video rights and the author's research release are separate from this project's MIT code license; no blanket permissive license over the weights or dataset is asserted. Neither checkpoint nor training footage is redistributed in Git or Sites. Contextual refinement requiring verified handedness is disabled. See [model details](docs/stroke-recognition.md).
