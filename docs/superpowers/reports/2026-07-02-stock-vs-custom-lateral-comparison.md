# Stock sunnypilot dev vs. custom fork — objective lateral comparison (2026-07-02)

**Data.** 6 stock drives pulled from device, all genuinely stock **sunnypilot dev v2026.002.000**
(`github.com/sunnypilot/openpilot@3f8e959`, steerRatio 16.8, not the fork). Matchable corridor =
**Powder Springs↔Hiram** (stock `00000004`+`00000005`, 38 seg vs custom Nevada `c5`+`c7`).
The Marietta drive (`00000002`) is **not matchable** — the custom incident drives (ce/cf) stayed
south around Smyrna and were ~90% manual takeovers (only 2 shared GPS cells), so all conclusions
below are Hiram-corridor. Method: GPS-cell(~150 m)+speed-bin matched, model-independent
straight/curve classification from GPS heading, band-limited 0.10–0.35 Hz weave, robust stats
(median/MAD/percentile, paired Wilcoxon), per `feedback_lateral_ab_metrics`. Independently
verified by 5 parallel agents (weave via a different signal, centering re-extracted from rlogs,
curve, EPS, and a dedicated confound auditor).

## Calibrated conclusions (by confidence)

### ✅ SOLID: stock weaves ~2× less than custom at HIGHWAY speed (22–29 m/s)
- band|steerDeg| (independent of the yaw metric that found it): stock 0.116° vs custom 0.194° =
  **40% less**, Wilcoxon **p=0.0022**, 74% of 34 matched cells stock-calmer.
- Corroborated by band|yawRate| (57% less, p=0.0007) and aLat-proxy (60% less, p=0.0003).
- **Survives every stress test:** all 4 leave-one-drive-out pairs agree; survives *tight*
  speed-matching (residual −0.61 m/s, i.e. stock slightly *slower* yet still calmer, ratio
  2.1×, 100% of 16 tight cells); steerRatio (16.8 vs 17.2) explicitly neutralized (~2.4% vs a
  40–60% gap). This is the clean headline.

### ⚠️ CONFOUNDED: the mid-speed (15–22 m/s) weave gap is mostly a speed artifact
- Pooled it looks strong (58% less) but stock is **2.2 m/s faster in that bin** (faster = less
  weave, which flatters stock); equalizing speed halves it and the per-cell win-rate collapses
  to 50%. It also fails leave-one-drive-out (stock_05 vs c5 *reverses*). **Not established.**
- 8–15 m/s: **UNSUPPORTED** (6 cells, null). ⇒ *At surface-street speeds — where your weave
  complaint was worst — we CANNOT show stock is better.* The clean stock advantage is highway-only.

### ❌ UNSUPPORTED/CONFOUNDED: "custom centers tighter"
- Custom's |offset| is smaller (0.08–0.10 vs stock 0.14–0.24 m) and that measurement is robust,
  BUT it's a one-sided **DC bias**: stock's comma model sits +0.15 m LEFT of *its own* detected
  lane-center and commands ~0 correction (the model *targets* left of its midpoint). Re-extraction
  confirmed both models detect **identical lane widths** (Δ≤2 cm) — ruling out a scale artifact —
  but a pure lateral translation/calibration bias is **unidentifiable from logs** (no
  model-independent lane-position reference; GPS is meters-coarse). So this is a model
  lane-center-*definition* difference, not demonstrably better physical centering. (Note: our
  fork's PI *is* a lane-offset loop, so custom does actively center to its model's midpoint —
  which is ~unbiased — but that still can't be separated from the comma model's left bias.)

### 🟡 SUGGESTIVE (medium): stock tracks CURVE commands more faithfully; custom understeers
- In curves, stock achieves ~1.0× its commanded curvature; custom **under-turns its own command
  by ~6–9%** (ach/cmd 0.91–0.94) and lags curve entry ~100 ms more (300 vs 200 ms). Survives
  steerRatio normalization and severity-control. BUT same-location matched curve cells are sparse
  (1–3), so it's a pooled/severity-controlled effect, not location-matched — medium confidence.
  (Consistent with the fork's long-noted curve-understeer/apex behavior.)

### ✅ SOLID: EPS command-tracking is build-INDEPENDENT
- Wheel-follows-command is the same on both builds: cmd→achieved lag ~0.25–0.37 s, gain ~1.2–1.3,
  weave-band coherence 0.95–0.98, indistinguishable stock vs custom. ⇒ the weave difference lives
  in **what is commanded** (model/controller path), not in delivery — consistent with the
  long-standing "weave is the model, not the actuation." (This is weave-band small-command
  behavior on straights; it does not directly re-test the incident's specific transient.)

## The load-bearing caveat
Even the SOLID highway-weave result is a **whole-build** comparison: model bundle (comma vs
Nevada) + PI + steerRatio all differ at once. So it means **"the stock build weaves less,"** NOT
"our PI is the whole cause" and NOT "the old model is the cause" — those cannot be separated from
this data (you'd need same-model, e.g. stock-model-with/without-PI, which only an on-device A/B or
a control-only process_replay could isolate). The custom side here is **old Nevada** (c5/c7), not
the migration build (ce/cf, unmatchable) — so this is stock-dev vs old-custom, not vs the incident
build.

## Bottom line (objective, pre-subjective)
- Stock is **materially calmer on the highway** (~2× less weave) — real and robust.
- Stock does **not** demonstrably fix the **low-speed** weave (the historically worst case) — unproven.
- Centering is a **wash / model-frame artifact** — neither shown physically better; stock's comma
  model carries a +0.15 m left bias.
- Stock **tracks curves more faithfully**; custom slightly **understeers** curves (medium conf).
- The wheel delivers commands **identically** on both — weave is upstream in the commanded path.
