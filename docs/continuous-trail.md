# Continuous display trail

The default video and adjacent shot view show a continuous trajectory, including tracking gaps. Cyan denotes observed positions; pale cyan denotes the existing bounded short interpolation; **solid amber denotes artificial display fill**. The same distinction appears in camera overlays and annotated MP4 exports. Controls switch between continuous fill, the supported track and observation points, with 2-second, 5-second or full-rally history.

`display_ball_trail` is separate from `tracks`. Short display gaps use a shape-preserving Hermite bridge, with bounded endpoint tangents; long gaps use a conservative bridge between known endpoints. A missing tail uses damped velocity and remains within image bounds until another detection reconnects it. Recorded footage can backfill before its first anchor; causal camera processing cannot. A video with no ball anchor stays unavailable.

These positions are visual guesses. They never feed ball-speed fits, contact inference, shot depth, player path, stroke recognition or reported observation coverage. The original rule of at most three frames/100 ms for measured-track interpolation remains intact. `visual-trail.csv` labels every sample with its display state, reason and `visualization_only_not_measurement` purpose. JSON schema 1.4 contains both representations. Old results receive a display-only frontend fallback without modifying their observations.

Tests cover long gaps, endpoint overshoot, bounded tail prediction, reconnection, leading context, unavailable clips, unchanged observations, point mode and measured-only rendering. Real result migrations assert that tracks, reviewed contacts and speed evidence remain unchanged.
