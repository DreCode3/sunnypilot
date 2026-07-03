# 2026-06-28 Nevada-vs-CD210 Matched-Corridor Weave Test

Subject: 2021 Ford Explorer ST sunnypilot — testing the hypothesis **"the Nevada driving model weaves less than CD210"** using newly gathered on-road data.

Method: `superpowers:systematic-debugging` (hypothesis testing with new evidence). This is the first **matched-corridor, speed-matched, PI-held-constant** cross-model weave test — the gold-standard test that the [2026-06-28 fresh full-analysis review](2026-06-28-retrospective-lateral-fresh-full-analysis-review.md) named as the open gate before any model-change decision.

Constraints honored: pulling logs is read-only (tar-over-ssh, the established method); no remote update/install/reboot performed; no vehicle-control code touched; `opendbc_repo/`/`panda/`/Ford car code verified clean; new caches and raw rlogs are gitignored/unstaged; analysis-only.

---

## 1. Executive Summary

**The hypothesis is NOT confirmed.** At matched corridor + matched speed, holding the PI controller constant (golden on both sides), the Nevada model does **not** weave measurably less than CD210 — every weave metric is a statistical wash (Nevada/CD210 ≈ 0.96–1.04, Wilcoxon p = 0.4–0.97, Nevada-better in ~50% of matched strata).

The large apparent advantage seen in pooled/earlier data is a **confound**, decomposed cleanly here:

| Control level | steering | path (yaw/v) | model_y20 | desired | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| (a) none — pooled (route-confounded) | 0.77 | 0.69 | 0.74 | 0.69 | 244 vs 501 |
| (b) speed-matched only (location pooled) | 0.89 | 0.84 | 0.89 | 0.89 | per-bin |
| (c) **cell + speed matched (strict)** | **0.96** | **0.98** | **0.97** | **1.04** | 35 strata |

Pooled, Nevada looks ~25–31% quieter — **but Nevada was driven ~8 mph faster (60.9 vs 53.1 mph)** and weave scales strongly with speed (Spearman ≈ −0.80). Controlling speed removes about half the gap; controlling location too removes the rest. This is the same failure mode the project has hit before (the b8 golden-PI "−20% win" that was a 7 mph speed offset).

**Honest power caveat:** with only 4 Nevada routes, the looser cell-only match (58 cells, more power) shows a *non-significant* ~7–12% lean toward Nevada (p ≈ 0.26–0.36). A **large** Nevada benefit is ruled out; a **small (~10%)** one cannot be excluded from this data.

**Bottom line:** the earlier `model_era_weave.csv` "Nevada ≈ −27%" result was route + speed confounded. A model swap (CD210→Nevada) is still **not** a validated weave fix. The decisive test remains an **interleaved same-day A/B on one fixed corridor**.

---

## 2. Data (pulled 2026-06-28 from device 192.168.98.237, home WiFi)

Device currently runs **Nevada** (`ModelManager_ActiveBundle` → `{"internalName":"NM","displayName":"Nevada Model (September 07, 2025)"}`). New routes since the prior cache (route_c3):

| Route | Date | Segments | Driving time | Model (recovered) | PI config |
| --- | --- | ---: | ---: | --- | --- |
| route_c4 | 2026-06-26 | 32 | ~31.8 min | Nevada (NM) | golden |
| route_c5 | 2026-06-27 | 13 | ~12.8 min | Nevada (NM) | golden |
| route_c6 | 2026-06-27 | 30 | ~29.6 min | Nevada (NM) | golden |
| route_c7 | 2026-06-27 | 36 | ~35.4 min | Nevada (NM) | golden |

All four: commit `5e0785b9` (dirty), steerRatio 17.2, lc_kp 0.0005 (golden). Model label recovered per-route from `ModelManager_ActiveBundle` in each boot rlog's `initData.params` (via `retrospective_lateral/code/model_labels.py`) — **verified all four are Nevada**, not an A/B mix.

**CD210 comparison set** (existing cache, golden PI, model isolated): route_b8, be, bf, c1, c2, c3. Robustness set adds the CD210 weak/unknown-PI routes b1–b7, c0.

**Geographic overlap (enables the matched test):** the new Nevada routes share **290 distinct 80 m GPS cells** with CD210 routes. Top overlaps: route_be 143, b1 122, b8 121, b5 115, c3 110, b2 102, b4 96, c1 74.

---

## 3. Method

Analysis-only script (`scratchpad/matched_model_weave.py`), reusing the `retrospective_lateral` signal utilities. No driving-code change.

- **Eligibility (per sample):** lateral-engaged, not overriding/blinker/lane-change, lead-gated (headway ≥ 1.6 s with buffer), speed 10–70 mph, **straight/gentle** (low-passed yaw/v curvature below threshold), 2 s engage erosion.
- **Metrics (weave band 0.10–0.35 Hz, duration-weighted RMS over eligible samples):**
  - `steer` = steering-angle band-RMS (deg) — **model-independent felt wheel motion, PRIMARY**
  - `pathc` = (yawRate/vEgo) band-RMS (×10⁻⁴) — model-independent actual path weave
  - `model` = model_y20 band-RMS (m) — the model's predicted-path weave (the mechanism)
  - `desire` = desiredCurvature band-RMS (×10⁻⁴) — planner output
- **Strata:** per (route, 80 m GPS cell, 5 mph speed bin) with ≥ 40 eligible samples (≥ 2 s).
- **Matching:** a (cell, speed-bin) stratum present in ≥ 1 Nevada route **and** ≥ 1 CD210 route; Nevada and CD210 each aggregated by median across routes, then compared paired.
- **Stats:** median per-stratum Nevada/CD210 ratio, sign split, Wilcoxon signed-rank. Primary = golden-PI-only (model isolated); robustness = all CD210 (PI varies — defensible because prior work found PI is not a weave lever).

---

## 4. Results

### 4.1 The confound decomposition (headline)

See the §1 table. Pooled 0.69–0.77 → speed-matched 0.84–0.89 → cell+speed-matched 0.96–1.04. Pooled median speeds: **Nevada 60.9 mph vs CD210 53.1 mph (+7.8 mph)**.

### 4.2 Strict matched (cell + speed-bin), golden PI both sides — PRIMARY

35 matched strata, median speed gap Nevada−CD210 = **−1.0 mph**:

| Metric | Nevada med | CD210 med | Nevada/CD210 | Nevada<CD210 | Wilcoxon p |
| --- | ---: | ---: | ---: | ---: | ---: |
| steer (deg) | 0.679 | 0.643 | 0.957 | 19/35 (54%) | 0.88 |
| pathc (×10⁻⁴) | 1.407 | 1.328 | 0.978 | 19/35 (54%) | 0.59 |
| model_y20 (m) | 0.0229 | 0.0257 | 0.970 | 18/35 (51%) | 0.97 |
| desired (×10⁻⁴) | 1.173 | 1.232 | 1.041 | 17/35 (49%) | 0.40 |

→ **Wash.** No metric is significant; sign split ~50/50.

### 4.3 Robustness — all CD210 (incl. weak/unknown PI)

42 matched strata, speed gap −0.5 mph: steer 0.995 (p 0.54), pathc 1.014 (p 0.46), model_y20 **1.171** (p 0.50), desired 1.100 (p 0.10). → Wash, with Nevada slightly *worse* on the model/desired channels.

### 4.4 Power probe — cell-only match (pool speed bins, |Δmph| ≤ 6)

58 matched cells, speed gap +1.0 mph: steer 0.928 (60% Nevada<CD210, p 0.36), pathc 0.885 (60%, p 0.26), model_y20 0.923 (59%, p 0.31), desired 0.897 (57%, p 0.27). → A consistent but **non-significant** ~7–12% lean toward Nevada.

### 4.5 Per-speed-bin (golden, location-pooled) — illustrates the noise

| Speed bin (mph) | n Nev | n CD | steer | pathc | model | desired |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 5 | 17 | 0.38 | 0.30 | 0.44 | 0.62 |
| 25 | 3 | 17 | 0.97 | 0.99 | 1.02 | 1.15 |
| 30 | 4 | 16 | 0.93 | 1.02 | 0.78 | 0.77 |
| 35 | 8 | 28 | 1.02 | 0.95 | 1.15 | 1.01 |
| 40 | 8 | 29 | 1.18 | 1.23 | 0.96 | 1.28 |
| 45 | 19 | 67 | 0.71 | 0.72 | 0.80 | 0.62 |
| 50 | 29 | 87 | 0.77 | 0.72 | 0.88 | 0.97 |
| 55 | 27 | 95 | 1.16 | 1.15 | 1.23 | 1.03 |
| 60 | 95 | 104 | 0.87 | 0.81 | 0.80 | 0.88 |
| 65 | 44 | 17 | 1.07 | 0.94 | 1.13 | 0.91 |

Direction flips bin-to-bin (45/50 favor Nevada; 40/55 favor CD210); the apparent pooled advantage rides on the thin, low-speed, likely-different-road 20 mph bin and the speed mismatch. No robust per-speed signal.

---

## 5. Verdict (ranked)

| Rank | Conclusion | Confidence | Basis |
| ---: | --- | --- | --- |
| 1 | A **large** Nevada weave advantage over CD210 does **not** exist at matched corridor+speed. | **High** | Strict match washes to 0.96–1.04, p 0.4–0.97, ~50% sign split, across 4 metrics + 2 comparison sets. |
| 2 | The pooled "Nevada ≈ −27%" is a **speed (+8 mph) + route confound**, not a model effect. | **High** | Controlled decomposition 0.69→0.89→1.0; pooled speed gap +7.8 mph; weave Spearman(speed) ≈ −0.80. |
| 3 | A **small (~10%)** genuine Nevada benefit cannot be excluded. | Open | Cell-only match shows a consistent non-significant ~7–12% lean (p 0.26–0.36); only 4 Nevada routes = underpowered. |
| 4 | A model swap (CD210→Nevada) is **not** a validated weave fix. | Medium-high | Follows from 1–3; consistent with the cross-era open-loop finding that the model path-prediction artifact persists across CD210/Nevada/OPM7. |

---

## 6. Confounders & Limits

- **Power:** 4 Nevada routes (~110 min), 35 strict-matched strata — enough to exclude a large effect, not a small one.
- **Different days, not interleaved:** Nevada Jun 26–27 vs CD210 Jun 13–23. Location-matching controls *where*, not *when* (traffic/weather/road-surface drift across days adds noise).
- **Engaged real-drive, not same-scene replay:** this matches roads, not exact camera scenes. A true same-scene cross-model replay (re-running both models on identical logged frames) would be the cleanest model A/B but needs model re-inference (out of current scope).
- **Speed coupling is the dominant confound** and must always be controlled; the straight/gentle gate + (cell,speed-bin) matching handle it here.
- The earlier `model_era_weave.csv` aggregates by model across all routes (route + speed confounded) and should be read as descriptive only — this report supersedes its cross-model comparison.

---

## 7. Next Steps

1. **Today's Powder Springs → Norcross Nevada drive (separate report):** adds Nevada coverage on roads that overlap the CD210 corridors → more matched strata → more power to detect a small effect. To maximize value: hold a steady, consistent speed on straight/gentle stretches (keeps data speed-clean), favor roads also in the CD210 set, and note subjective weave. Still a different-day, all-Nevada drive, so it sharpens but does not settle the small-effect question. I can fold the new route(s) into this matched comparison once uploaded.
2. **Decisive test (settles a small effect):** an **interleaved same-day A/B on one fixed corridor** — drive a segment on Nevada, switch the model to CD210, redrive the same segment, repeat ≥ 5×, at a held target speed, no lead. This removes day/route/speed confounds together. Requires a mid-drive model switch (param + reboot) done manually by the driver; any device action follows the `AGENTS.md` offroad-safe workflow.
3. **Do not change driving code** on the basis of this result.

---

## 8. Commands Run & Verification (read-only)

```text
# Pull (read-only, tar-over-ssh; single ControlMaster connection)
ssh-add --apple-load-keychain
ssh ...comma@192.168.98.237 'tar cf - 000000c{4,5,6,7}--*/rlog.zst' | tar xf -   # 111 segs, 1.3 GB
# Active model on device:
ModelManager_ActiveBundle -> {"internalName":"NM","displayName":"Nevada Model (September 07, 2025)"}

# Extract to v5 cache (analysis pipeline) + per-route model recovery
.venv311/bin/python -m retrospective_lateral.code.extract --route route_c4 ... route_c7   # 137-channel v5 caches
model_labels.parse_active_bundle on each boot rlog -> route_c4..c7 all "NM" (Nevada)
PI per route (sidecars): route_c4..c7 = golden

# Matched analysis (scratchpad/matched_model_weave.py)
per-(route,cell,speedbin) strata: 1167 across 14 routes
PRIMARY golden-matched 35 strata: steer 0.957/pathc 0.978/model 0.970/desire 1.041 (p 0.88/0.59/0.97/0.40)
all-CD210 42 strata: 0.995/1.014/1.171/1.100 (p 0.54/0.46/0.50/0.10)
cell-only 58 cells: 0.928/0.885/0.923/0.897 (p 0.36/0.26/0.31/0.27)
pooled: 0.771/0.686/0.742/0.689 ; speed Nevada 60.9 vs CD210 53.1 mph
GPS overlap new-Nevada vs CD210: 290 distinct shared 80 m cells

# Cleanliness
git status --short opendbc_repo panda            -> empty ✓
git -C opendbc_repo status opendbc/car/ford ...   -> empty (driving code untouched) ✓
git status --short retrospective_lateral/results explorer_st_logs/route_c4..c7 -> empty (gitignored) ✓
manifest.json restored to 147 + c4..c7 = 151 routes (the --route extract had trimmed it)
```

Verification note: model labels, PI configs, GPS overlap, and all §4 statistics were derived this session with pandas/scipy over the v5 caches; raw logs pulled read-only; no `retrospective_lateral/results/` file is staged; `opendbc_repo/`, `panda/`, and the Ford car code were not modified.
