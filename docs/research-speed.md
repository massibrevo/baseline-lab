# Experimental ball km/h

The UI uses km/h for the optional research estimate. This is a constrained 3D flight fit, **not a validated speed gun**. `ball_speed_m_s` remains null; timestamped `ball_speed_estimates` and `speed_analysis` are separate schema 1.3 fields. Experiments do not feed the independently measured speed factor in Shot Quality.

Enable only when the original action plays at normal 1× speed, without slow motion, pauses, seeking or retiming. The upload dialog asks for that confirmation; CLI uses `--real-time-playback`. Live camera timestamps use the capture clock. Real presentation timestamps are retained throughout.

Method: refine nearby painted court-line pixels around manually selected singles boundaries; estimate a pinhole camera from the metric court plane with centered principal point and square pixels; register the court per frame; fit observed ball rays to a constant-gravity 3D trajectory. A nominal 67 mm ball diameter can provide a noisy depth constraint when a local blob is measurable. This does **not** project an airborne ball onto the ground plane and call the result speed.

Assumptions include negligible distortion and omitted aerodynamic drag and Magnus force. Screen recording geometry, camera motion, compression, blur, occlusion, tracking errors, mixed flights and poorly conditioned camera views can invalidate the fit. Minimum observed support, gap, reprojection, conditioning, court-volume and sensitivity checks reject unsupported windows. The displayed range includes a 25% minimum sensitivity allowance; it is **not a calibrated confidence interval**, and the true error can be larger. Synthetic recovery tests prove mathematical consistency only. No radar or independently measured speed validation has been performed.

Offline analysis can use future observations within the fitted flight interval. Camera analysis uses only an ending causal window. Unsupported samples stay null. The playback side card may retain the last supported flight, explicitly labeled with its age; that held value is historical, not a fresh continuous measurement. The raw per-frame CSV preserves every unsupported interval.

The current model cannot promise fresh speed estimates throughout arbitrary footage. Accurate continuous physical speed requires stronger calibrated reconstruction and independent validation, potentially additional cameras or a radar reference.

References: [known-size ball localization research](https://arxiv.org/abs/2204.00003), [Where Is The Ball? 3D reconstruction research](https://where-is-the-ball.github.io/). These motivate constraints, not claims that their released trained models or reported accuracy are implemented here.
