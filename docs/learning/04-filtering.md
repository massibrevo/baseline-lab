# 4. Motion prediction, occlusion and filtering

A constant-velocity Kalman state might be x=[u,v,du/dt,dv/dt]ᵀ, predicted by x⁻=Fx and P⁻=FPFᵀ+Q. A measurement z updates it with K=P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹. Larger measurement uncertainty R reduces trust in a new observation; larger process uncertainty Q admits more acceleration.

The release uses an intentionally simpler last-two-observation velocity prediction and bounded association gate, not a full Kalman implementation. Approximate player footpoints have a light exponential smoother. The runnable scalar Kalman example is educational and is not silently described as the production tracker.

Filtering reduces jitter but introduces lag. A single wrong link can propagate through an otherwise attractive smooth track. Detect identity switches and impossible speeds before summing distance. D=Σ||pᵢ−pᵢ₋₁|| only over contiguous plausible observed ground-plane pairs. Missing intervals and jumps are excluded, so this sum is a partial observed path, not a complete distance.

Interpolation is a visualization hypothesis, limited here to at most three missing frames and 100 ms. It does not improve measured detection coverage. Dash interpolated trail segments and retain their state in exports.

Run from the repository root:

```sh
uv run python scripts/learning.py 4
```
