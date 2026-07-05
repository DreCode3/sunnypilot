# 2026-06-25 Weave Is Not Road-Locked — Hypothesis B Refuted (Tuesday golden-PI repeat corridors)

Subject: 2021 Ford Explorer ST sunnypilot, 10–70 mph straight/gentle-section weave.

This is a focused follow-up to the [model/path-vs-road discrimination](2026-06-24-retrospective-lateral-model-vs-road-discrimination.md) report, which built the discriminator but found the road (B) test **under-powered** (few repeat-traversed corridors) and recommended a follow-up run seeded from repeat corridors. A fresh on-device drive provided exactly that.

Analysis-only. No vehicle-control code, `opendbc_repo/`, or `panda/` was modified. Device interaction was a read-only log pull (offroad). All generated outputs are gitignored under `retrospective_lateral/results/`.

## Executive Summary

A 2026-06-23 (Tuesday) to-and-from drive — **3 routes, 90 segments, ~1.08 GB rlogs, all on the current golden-PI / proven config** — was pulled and extracted. It retraces the Powder Springs↔Marietta corridors that **68 prior corpus drives** also cover same-direction (sharing up to 1,500+ fine cells). That powered the road (B) test, and a built-in **positive control** makes the result trustworthy:

| Signal (68 same-direction Tuesday×corpus corridors) | Median per-cell corr | p90 | frac ≥ 0.5 |
| --- | ---: | ---: | ---: |
| **Road shape** (low-passed path curvature) — *positive control* | **0.761** | 0.961 | 0.59 |
| **Weave** (0.10–0.35 Hz lane curvature) | **0.095** | 0.307 | 0.03 |

On the same corridors where the **road geometry reproduces across passes at 0.76–0.99**, the **weave reproduces at ~0.0–0.3**. The slow weave does **not** recur at the same physical places.

**Conclusion: hypothesis B (the weave is a real roadway feature the model faithfully follows) is REFUTED.** The weave is vehicle-generated, not road-triggered. Combined with the prior finding that the controller is not the source and the weave's fixed ~0.20 Hz, speed-independent spectral peak, the remaining candidates are **(A) a model/path-planner artifact** or **(C) a fixed-period loop limit-cycle**.

**Recommendation unchanged for controller tuning: do not change driving code yet.** But the search space is now materially smaller and the correction direction is firmly **upstream (model/path or loop-timing), not PI/controller tuning and not avoidable by routing.**

## What Was Done

1. **Pulled Tuesday's drive (read-only, offroad):** routes `000000c1--81d43a8186` (48 seg), `000000c2--5ae85eed96` (14 seg), `000000c3--d6e652baec` (28 seg) → local `route_c1/c2/c3` via tar-over-ssh (rlog.zst only; cameras skipped). Single SSH ControlMaster connection per AGENTS.md.
2. **Extracted** to `retrolat-v5` caches. All three: **pi_set=golden, config_confidence=proven, lc_kp=0.0005**, GPS finite 0.97–0.98, speeds to 76 mph. This is the first substantial golden-PI weave coverage (the prior corpus had 38 golden weave rows, 0 golden low-speed).
3. **Measured corridor overlap and direction structure.** The drive is an out-and-back: c1 (outbound) overlaps c2+c3 (return) **100% opposite-direction** (383 shared cells), so Tuesday's *internal* repeats can't feed a signed-correlation test. The power comes from **Tuesday × prior corpus: 68 route-pairs share ≥4 same-direction cells**, top pairs sharing 200–1,500 cells.
4. **Diagnosed and fixed a resolution bug in the B test.** At the discriminator's default 80 m GPS cell, all 68 pairs read non-reproducible (median corr 0.17) — but 80 m ≈ the weave's ~110 m wavelength at highway speed, so cell-averaging washed the signal out. A cell-size sweep confirmed correlation rises as cells shrink (e.g. c3×route_a8: 0.29 → 0.59 from 80 m → 10 m). This also explains why the original full run found so few B classifications.
5. **Built a refined corridor-level B test** (`retrospective_lateral/code/corridor_repro.py`, +7 tests): fine ~15 m cells, direction-aware (sign-flip for opposite-direction passes), parallelized across cores (`ProcessPoolExecutor`; 148 routes profiled in 2.4 s real / 19.7 s user on the M4 Max).
6. **Ran the positive control** (`scratchpad/posctrl.py`, spawn-safe parallel): per-cell reproducibility of the model-independent road shape (low-passed `yawRate/v`) vs the weave band, on the same corridors. The road shape reproduces (median 0.76); the weave does not (median 0.095) — proving the alignment/metric works and the weave's non-reproduction is real, not a methodology artifact. Speed-matching (30–45 mph) did not rescue the weave correlation (0.25–0.41).

## Updated Root-Cause Picture

| Hypothesis | Status | Basis |
| --- | --- | --- |
| Controller (PI / EMA / rate-limit / final command) is the source | **Refuted** | Prior review: attenuation ladder + golden natural experiment + lead/speed robustness |
| **B — real road feature the model follows** | **Refuted (new)** | Weave not road-locked: road shape reproduces 0.76, weave 0.095, on the same golden-PI repeat corridors, with a validated positive control |
| **A — model/path-planner artifact** | **Live candidate** | Upstream-of-controller origin established; not road-locked |
| **C — fixed-period loop limit-cycle** | **Live candidate (leaning)** | Fixed ~0.20 Hz peak independent of speed |

## Caveats

- The refined B test found 2/68 corridors marginally reproducible (route_ab×c1 0.60 over 26 cells, route_aa×c1 0.60 over 75 cells) — consistent with chance near threshold over 68 comparisons; they do not rescue B.
- This is weave-specific (10–70 mph). Low-speed swing (a separate, collapsed-model-horizon mechanism) was not retested; the prior run already found 0 low-speed B.
- The fixed ~0.20 Hz peak is partly a band-detection artifact; it leans C but is not decisive on its own — A vs C is the open question.
- A 0.20 Hz (≈5 s) period is slow for a fast steering-loop resonance, which hints the responsible element (if C) is a slow one (PI integrator, perception/path-planning dynamics, or a localization-lag path) rather than `steerActuatorDelay`.

## Artifacts / Commands

- New module: `retrospective_lateral/code/corridor_repro.py` (+ `tests/test_corridor_repro.py`), commit `f05f3af1cb`. Full suite: 124 passed.
- Generated (gitignored): `results/reports/corridor_repro_pairs.csv`, `corridor_repro_summary.csv`; Tuesday caches `results/cache/route_c{1,2,3}.npz`.
- Key runs: `python -m retrospective_lateral.code.corridor_repro --focus route_c1 --focus route_c2 --focus route_c3`; spawn-safe positive control in scratchpad.
- Hygiene verified: `opendbc_repo`/`panda` untouched; `results/` and `explorer_st_logs/route_c*` gitignored; the user's uncommitted `config.py`/`extract.py` work left untouched.

## Next Step — A vs C Discrimination (offline)

Both remaining candidates are upstream of the controller, but imply different fixes (model/path change vs loop-timing element), so the next analysis discriminates them:

1. **Engaged vs disengaged model-weave (primary discriminator):** in straight/gentle, speed-matched windows, compare the weave-band amplitude of the model's own output (`model_y20`, orientation-rate curvature, model-minus-lane-center, `desiredCurvature`) when `latActive=1` (closed loop) vs `latActive=0` (model running, not actuating → open loop). If the model path still weaves at ~0.2 Hz when disengaged → **A** (intrinsic model wobble). If the model is calm disengaged and only weaves engaged → **C** (closed-loop resonance).
2. **Integrator-phase check:** is the PI integrator (`cx1_integral` / `LaneBiasIntegral`) oscillating at ~0.2 Hz coherent with the weave (→ slow-integrator C) or calm while `desiredCurvature` weaves upstream (→ A)?
3. Restrict to golden-PI (Tuesday + the golden corpus rows) for current-config relevance; parallelize across cores.

This is analysis-only. A model-swap A/B on a controlled corridor (PI held at golden) remains the first worthwhile *on-road* test once A vs C is narrowed.
