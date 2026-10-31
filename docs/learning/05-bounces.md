# 5. Bounces, uncertainty and inference

A vertical trajectory reversal can represent a bounce, a hit, camera movement, an occlusion or a false association. The candidate generator requires seven observed frames, temporal continuity and a reversal, and suppresses contact near a detected person. That is additional evidence, not proof. Predicted contacts stay on the image; only human-reviewed ground contact gets a court-plane location.

A useful line rule needs a signed distance d from a painted outer boundary and a justified uncertainty band ε including ball radius/contact extent, calibration error and localization error. If d>ε, estimated_in; if d<−ε, estimated_out; otherwise uncertain. In this release ε is not measured, so every event remains uncertain. It never turns a tiny four-point residual into centimeter precision.

Human corrections keep the original prediction, edit time/location/status, and export both provenance and current review. Model score is not a bounce probability. Ball speed is null: 2D image velocity does not reveal 3D motion or spin from one unconstrained view. The runnable example illustrates how a line is included in the uncertainty band.

Run from the repository root:

```sh
uv run python scripts/learning.py 5
```
