# CD210 vs Nevada — Same-Scene Weave Comparison

Date: 2026-06-29
Tool: `model_replay_sim/` (same-scene model-replay simulator; analysis-only)
Method: replay each driving-model bundle on **identical recorded camera frames**, compare the 0.10–0.35 Hz lateral "weave" band of the model's commanded `desiredCurvature`. Every model gated on a same-model fidelity anchor before its number is trusted.

## Bottom line (UPDATED 2026-06-30 — balanced 8-scene firm-up)

**Effectively a wash, with a weak non-significant lean toward Nevada (~7% less weave by median).** A balanced 8-scene comparison (4 CD210-native + 4 Nevada-native, every scene native-leg-validated) gives median Nevada/CD210 = **0.931** but mean **1.006**, with Nevada weaving less on **6/8** roads — sign test **p=0.14**, bootstrap 95% CI **[0.920, 1.018]** (includes 1.0). NOT statistically significant. The earlier 3-scene "~10%, suggestive" result was optimistic; more data regressed it toward a wash.

**Key robustness win:** the CD210-native and Nevada-native scene groups AGREE (medians 0.909 vs 0.931) → no native-side bias; the lean is real in direction but small and inconsistent. **Big caveat:** large road-to-road variation, incl. route_c0 where Nevada weaves **57% MORE** (real — CD210 native-leg validates there at corr 0.998).

### Balanced 8-scene table
| scene | native | Nev/CD210 | native-leg corr |
|---|---|---|---|
| route_b5 | CD210 | 0.853 | 0.993 |
| route_c4 | Nevada | 0.856 | 0.998 |
| route_bb | CD210 | 0.877 | 0.994 |
| route_c5 | Nevada | 0.928 | 0.990 |
| route_c6 | Nevada | 0.934 | 0.999 |
| route_c3 | CD210 | 0.942 | 0.978 |
| route_c7 | Nevada | 1.084 | 0.999 |
| route_c0 | CD210 | 1.573 | 0.998 |
| **median** | | **0.931** | |

### Original 3-scene result (superseded — kept for the mechanism analysis below)

The mechanism analysis (smoothing/lookahead ruled out → it's the plan) was done on the original 3 scenes (median 0.900). That mechanism reasoning still stands for the scenes where a difference exists; the magnitude/significance is superseded by the 8-scene result above.

| scene | CD210 weave | Nevada weave | Nev/CD210 | native-leg check |
|---|---|---|---|---|
| route_b5 (CD210 native) | 0.001266 | 0.001080 | 0.853 | CD210 replay = 1.008× logged ✓ |
| route_c5 (Nevada native) | 0.000285 | 0.000264 | 0.928 | Nevada replay = 1.073× logged ✓ |
| route_7f (neutral) | 0.000258 | 0.000232 | 0.900 | — |

## Why this is trustworthy (unlike the OPM7 attempt)

- **Both models are fidelity-anchor-validated:** CD210 on route_b5 (corr 0.993, band_ratio 1.008), Nevada on route_c5 (corr 0.990, band_ratio 1.060). Each replay reproduces that model's own logged output on a route it actually drove.
- **Genuine paired same-scene comparison:** within each scene the two models see byte-identical frames (`frameId`) and speed (`vEgo`); only the curvature differs.
- **Native legs reproduce logged ground truth** (1.008×, 1.073×).

## The reduction is the model's PLAN — not smoothing, not lookahead

Two config differences between the bundles were ruled out as the cause:

1. **Nevada's LAT_SMOOTH = 0.1** (CD210 = 0): this EMA has a ~1.6 Hz cutoff and attenuates only **0.2–2.3% in the 0.10–0.35 Hz weave band**. EMA-inverting Nevada (removing the smoothing entirely) changes the ratio by <1%.
2. **Nevada's longer lookahead** (lat_action_t ≈ 0.50 s vs CD210 ≈ 0.40 s, a consequence of its LAT_SMOOTH): worst-case in-band attenuation ≤ 1.8%; removing even a generous 4% leaves the median < 0.94. **Decisive refutation:** the frequency-resolved Nevada/CD210 amplitude ratio is *non-monotonic and exceeds 1.0 in sub-bands* (e.g. 1.15 at 0.10–0.175 Hz on route_7f) — physically impossible for any low-pass filter. So the difference is genuine plan **shape**, not a filtering artifact.

⇒ Nevada's path prediction inherently commands a calmer trajectory on the same scene. This is **not** a knob (like smoothing) you could simply copy onto CD210.

## Statistical strength — honest read

**Direction:** unanimous and robust. All 3 scenes < 1.0, under 4 band-edge choices (medians 0.87–0.92), both temporal halves, edge-trimming, and an independent Welch-PSD estimator (median 0.870); all 36 scene-level ratios stayed below 1.0.

**Magnitude significance:** borderline and method-sensitive.
- Paired moving-block bootstrap (correct, given identical frames): **P(median < 1) = 0.99**, CI [0.838, 0.977].
- Unpaired bootstrap: P = 0.81, CI touches 1.0.
- **n = 3 scenes is the binding limit** — a cross-scene sign test caps at one-sided p = 0.125 for 3/3 same-sign. Within-scene 5 s blocks are autocorrelated, so per-block counts overstate significance.
- The effect **leans on route_b5** (0.853, 9/10 blocks); **route_c5 — the Nevada-native leg — is a near-wash** (0.928, 5/10 blocks).

Verified by 4 independent adversarial agents (robustness, shift-null, from-scratch re-derivation, lookahead confound) + synthesis. All point estimates reproduced bug-free; the only non-reproducing number was the exact bootstrap P-value (method-sensitive).

## Recommended next steps (in priority order)

1. **More scenes (highest value).** n = 3 is the only real limit. Run the same anchor-validated CD210-vs-Nevada replay on ~6–10 more corridors spanning the speed range, **prioritizing Nevada-native scenes** (we have routes c4/c6/c7 still on the device). With ~10 scenes a cross-scene test can clear p < 0.05 without relying on autocorrelated within-scene blocks.
2. **Capture raw plans** during replay → recompute CD210's curvature at Nevada's lat_action_t on the *same* plan: the one clean counterfactual that fully closes the lookahead question (currently bounded + spectrally argued, not directly swapped).
3. **On-road A/B** is the ultimate confirmer but lower-yield immediately — a ~10% effect is easily swamped by road/speed/lead noise without many matched passes.

## Provenance / reproducibility

- Bundles (pinned): CD210 `55f66e22…`, Nevada `3193eac5…` (commaai/openpilot), both 2-model, img_buffer_length 5.
- Pipeline: `model_replay_sim/{config,alignment,context,warp,infer,parse,metrics,anchor,compare}.py`. Comparison CLI: `python -m model_replay_sim.run --compare --scenes route_b5,route_c5,route_7f --bundles CD210,Nevada`.
- Nevada video (route_c5 segs 4–6) pulled read-only from the device 2026-06-29.
- Cached series + scripts under the session scratchpad; result JSON at `retrospective_lateral/results/model_replay/`.

## OPM7 (out of scope for this report)
A preliminary CD210-vs-OPM7 run suggested OPM7 weaves ~1.23× *more*, but OPM7 could not be cleanly anchored (confirmed-OPM7 routes have rotated off the device; route_7f corr 0.925 is an unconfirmed build). That comparison is **not trustworthy** and awaits a fresh OPM7 drive.
