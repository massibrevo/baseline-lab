# Shot Quality: an experimental pressure index

The app calculates a transparent 0–10 groundstroke pressure index after a reviewed ground contact. It is not a validated tennis skill rating or a forced/unforced-error classifier. The profile favors deep, wide, fast, high-spin, steep-bouncing groundstrokes; it does not fairly rank every serve, slice or drop shot.

| Factor | Weight | Input | Normalization to 0–10 |
|---|---:|---|---|
| Speed | 30% | Independently measured km/h, entered with source | clamp((km/h − 40) / 14, 0, 10) |
| Spin magnitude | 15% | Independently measured rpm, entered with source | clamp(rpm / 400, 0, 10) |
| Depth | 25% | Reviewed contact, current-frame court mapping | 10 × (1 − distance inside receiving baseline / 11.885), clamped |
| Width | 20% | Reviewed contact, current-frame court mapping | 10 × distance from center / 4.115, clamped |
| Outgoing vertical bounce angle | 10% | Independently measured degrees above court | clamp(degrees / 6, 0, 10) |

Placement components are zero when the estimated contact lies outside the singles rectangle. Every line classification remains uncertain because independent court/contact error bounds are unavailable. Width and depth are approximate meters. The depth convention is signed inside the receiving baseline; a negative value means beyond that baseline.

The score is the weighted mean over **available** factors. At least two factors and a reviewed calibrated contact are required. An automatic placement-only result is labeled **partial, 2/5 factors, 45% weighted evidence**. The missing factors are shown explicitly, not assigned zero or a typical pro value. The five-factor result requires measured inputs and a source; it is still an experimental heuristic.

Image-plane ball velocity in px/s is never substituted for physical km/h. An image trajectory's angle is never substituted for vertical bounce angle. Ordinary distant video does not supply reliable ball surface rotation. The ITF's [high-speed research apparatus](https://www.itftennis.com/media/2283/rackets-and-strings-research.pdf) and [single-camera validation study](https://shura.shu.ac.uk/2355/) illustrate the measurement/validation required for spin.

Results include component values, units, weights, scores and sources, and `error_attribution: undetermined`. Incoming-shot pressure alone cannot decide whether the receiver's error was forced: receiver position, reach, recovery time, racket contact and execution need additional evidence. No trained or independently validated error classifier is supplied.

The panel updates with reviewed contacts during playback and in live camera analysis. Corrections preserve the original event prediction. Rejecting a contact clears its derived placement and quality. JSON and `shot-quality.csv` preserve the breakdown; later edits do not rewrite the immutable annotated-video snapshot.
