# 3. Perspective, homography and coordinate systems

Four non-collinear ground correspondences determine a planar homography: λ [u,v,1]ᵀ = H [x,y,1]ᵀ. The implementation fits image → court meters; its inverse renders court → display. The origin is the outside-painted far-left singles corner, x increases right to 8.23 m, y increases toward the near baseline to 23.77 m. Court lines count as inside.

Original pixels, crop/model pixels, normalized image coordinates and court meters are different units. A crop at (x₀,y₀) returns image coordinates by adding that offset. A letterbox uses scale s=min(W/w,H/h) plus centered padding; resize without accounting for padding misaligns overlays.

A four-point fit has almost zero residual at its four defining points even if those points are wrong. Use extra independently labeled landmarks to evaluate error and reject degenerate/ill-conditioned fits. Camera motion invalidates that mapping.

An airborne ball does not lie on the court plane. Applying H to its image creates a ray/plane intersection, not its actual ground location or 3D speed. Only supported ground contacts and approximate grounded footpoints may populate the minimap.

Run from the repository root:

```sh
uv run python scripts/learning.py 3
```
