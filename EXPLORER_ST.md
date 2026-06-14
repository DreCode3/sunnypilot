# 2021 Ford Explorer ST — SunnyPilot lateral-control project

Front door for this fork. It's a personal sunnypilot/openpilot fork tuned for a **2021 Ford Explorer ST** (CAN/Q3
harness, NOT CANFD) on a **Comma 4**, focused almost entirely on **lateral control** (a custom PI lane-centering
controller + curvature pipeline) and a long, data-driven investigation of a slow steering "weave." Generic openpilot
build/run/setup lives in [`docs/`](docs/) (upstream `CONTRIBUTING.md`, `DEVELOPMENT.md`, `SAFETY.md`); this file
covers everything specific to THIS project.

> Branch: `2021_explorer_st-mici` (sunnypilot) + `2021_explorer_st-mici` (opendbc submodule, where the lateral code lives).
> The customized car code is in **`opendbc_repo/opendbc/car/ford/`**, NOT this repo's tree.

---

## 1. Current state (TL;DR, 2026-06-14)
- **The symptom under investigation:** a small slow left-right wheel/path **weave** on straights & gentle curves
  (multi-second period, small amplitude, uncomfortable, visible to other vehicles). NOT centering offset, NOT
  high-frequency hunt.
- **What's settled:**
  - Weave is overwhelmingly a **LOW-SPEED phenomenon** — `Spearman(weave, speed) ≈ −0.83`. Speed is the dominant
    factor in every analysis and must always be controlled.
  - **No learned parameter is a validated weave lever** (steerRatio / angleOffset / calibration / the PI integrator)
    — see `learned_param_studies/RESULTS.md` (3 rounds of adversarial QA).
  - The **"weave grows over time / since a learned-param reset" hypothesis is REFUTED** by device reset ground-truth
    — see `longitudinal_weave/README.md`. The apparent trend was an artifact of inferred resets that never happened.
  - The **right-curve apex-cutting / lane-crossing** is a **driving-MODEL/path-planning** behavior, not controller
    gains — see memory `finding_directional_apex_cutting` (path/model is the lever, not PI/smooth_tau/steerRatio).
  - **golden-PI vs weak-PI centering** difference is **undecidable** from observational data (confounded by
    speed/route/single-drive) — see `explorer_st_logs/CHECKPOINT_2026-06-13_golden_PI_b8.md`.
- **Device now:** CD210 driving model + GOLDEN PI (lc_kp 0.0005, speed-interp int cap) + smooth_tau (0.12,0.04);
  `DisableUpdates=1`. iter3 smooth_tau (0.25,0.12) was tested and reverted (no benefit).
- **Repo reconciled to the device (2026-06-14):** the committed tree now MATCHES the live car — `carcontroller.py`
  has the golden PI (lc_kp 0.0005, speed-interp int_cap, decay 0.995) **plus a live-toggle harness** (reads
  `/data/lc_pi_config` every ~50 frames, defaults golden; lets you A/B weak↔golden on-device with no recompile), and
  `values.py:40` smooth_tau = (0.12,0.04). ⚠️ Tuning still happens via on-device edits under `DisableUpdates=1` (§3),
  so the repo CAN diverge again — after any on-device tuning session, re-reconcile (pull the device `ford/` files,
  `diff`, commit). The toggle is harmless off-device (the path won't exist → golden); strip it if you want pure-golden source.
- **The one clean open test:** a deliberate **reset-and-monitor** drive (clear `LiveParametersV2`/`CalibrationParams`,
  repeat one fixed corridor for ~a week) to settle the drift question with controlled reset timing.

## 2. Architecture (lateral)
- Ford is **curvature-controlled** (1/m), not torque — EPAS does curvature→angle. So `MAX_LATERAL_ACCEL` / ISO
  lateral-accel checks do NOT apply (CAN Q3). Three safety layers: app code → panda (`ford.h`) → EPAS.
- Pipeline: modelV2 `desiredCurvature` → blend (predicted+desired) → EMA (`smooth_tau`) → custom **PI lane-centering
  controller** (P=lc_kp·offset, I gated to straights/non-override, clamped to int_cap, warm-started from
  `LaneBiasIntegral`) → rate limits → CAN (curvature + curvature_rate, **negated** before send).
- **The full, authoritative customization guide is [`explorer_st_logs/customizations.md`](explorer_st_logs/customizations.md)**
  (1159 lines: files modified, the CAN sign convention, the lateral pipeline line-by-line, the PI controller, the
  BluePilot longitudinal work). CAN signal details: [`explorer_st_logs/ford_can_reference.md`](explorer_st_logs/ford_can_reference.md).
- Customized files: `opendbc_repo/opendbc/car/ford/{carcontroller.py, values.py, fordcan.py, interface.py}`,
  `opendbc_repo/opendbc/safety/modes/ford.h`, and `selfdrive/modeld/modeld.py` (`LAT_SMOOTH_SECONDS=0.1` — a model-side
  EMA on `desiredCurvature`; currently equals the upstream default after commaai/openpilot#36987, so re-check it stays
  0.1 on any modeld sync rather than assuming).

## 3. Device dev loop (the hard-won rules — read before touching the car)
- **SSH: single ControlMaster connection only.** Parallel SSH crashes the device sshd. Run `ssh-add
  --apple-load-keychain` once per session (the key has a passphrase). Home `192.168.98.237`.
- **Reboot ONLY via** `echo -n "1" > /data/params/d/DoReboot` — never `sudo reboot` (causes EPAS alerts).
- **`DisableUpdates=1` BEFORE any on-device code edit** — the updater does `git reset --hard` against origin and will
  silently revert local edits on the next fetch (this is why "iter3 never deployed" for weeks). Params (`/data/params`)
  and downloaded models (`/data/media`) live outside the git repo, so model swaps survive.
- On-device rapid loop: edit `values.py` (interpreted → just reboot) → `DoReboot` → drive → repeat. Editing `ford.h`
  needs a **panda reflash** (auto on boot when the signature changes).
- Pulling logs (macOS openrsync can't do `--files-from`): **tar-over-ssh** —
  `ssh comma@HOST 'cd /data/media/0/realdata && tar cf - <route>--*/rlog*' | tar xf - -C <dir>/`.
- Register any new param key in `common/params_keys.h` BEFORE writing it (unregistered writes crash the process;
  reads silently return None). Our custom keys: `FordPath4Enabled, LaneBiasIntegral, disable_BP_long_UI,
  disable_downhill_comp_UI`.

## 4. Analysis tooling
Local analysis runs on a MacBook Pro M4 Max (16 cores) in the **`.venv311`** environment (Python 3.11). **Setup:**
`python3.11 -m venv .venv311 && .venv311/bin/pip install -r requirements-analysis.txt` — then run scripts from the
repo root so `openpilot.tools.lib.logreader` resolves. (The main app's `pyproject.toml` targets Python 3.12, separate
from this 3.11 analysis venv.) **Parallelize** compute-heavy work across cores (`concurrent.futures.ProcessPoolExecutor`;
macOS spawn-safe = top-level worker fn + picklable args).
- [`learned_param_studies/`](learned_param_studies/RESULTS.md) — does any learned param correlate with weave/centering?
  (Verdict: no lever.) Calibrated stats: Freedman-Lane permutation + BH-FDR; `qa_calibration.py` Monte-Carlos the FPR.
- [`longitudinal_weave/`](longitudinal_weave/README.md) — does weave grow over time/since reset? (Verdict: refuted.)
  Parallel extractor + the only-calibrated trend test (`blockperm` on time-since-reset).
- [`controlled_test_analysis/`](controlled_test_analysis/ANALYSIS_PLAN.md) — pre-registered, bias-resistant analysis
  for a future controlled PI/model A/B drive (+ [`DRIVE_PROTOCOL.md`](controlled_test_analysis/DRIVE_PROTOCOL.md)).
- [`cleanroom_lateral_analysis/`](cleanroom_lateral_analysis/README.md) — a blinded package for an independent model.
- `explorer_st_logs/*.py` — ~90 drive-analysis scripts (e.g. `analyze_drive_v6.py`, `pipeline_simulator.py` the
  production-faithful replay, `discriminator_run.py` the parallel curve-analysis). Raw rlogs + caches are gitignored.

## 5. Conventions / culture (why this project is the way it is)
- **Metric discipline:** centering = mean offset / slow drift (NEVER std or detrended band-RMS — they discard it);
  oscillation/weave = matched-speed band-RMS; always SPEED-match + LOCATION-pair; never `aLat` as primary (f²-blind);
  never pooled mean-of-variance (outlier-dominated). A comparison has flipped under proper control many times.
- **Verify everything, adversarially.** Cooperative AND adversarial agents both make confident errors. Every
  analysis script + result gets independently re-derived and its estimator's false-positive rate Monte-Carlo'd
  before "significant" is trusted (the hard lesson: naive permutation of OLS residuals is ~2× anti-conservative →
  use Freedman-Lane). Pre-register decision rules; correct for multiple comparisons.
- Suggestive signals collapse under rigor: assume any exciting result is a speed/route/confound artifact until a
  calibrated, confounded-controlled test survives.

## 6. Git — push ONLY to your fork (you are not contributing upstream yet)
This clone is locked to push **only to your fork** (`github.com/DreCode3/*`); pushes to `upstream` (sunnypilot /
sunnyhaibin), `commaai`, or `bluepilot` are **blocked**. Three layers enforce it, applied per repo (the main repo +
each initialized submodule) by [`setup-push-guard.sh`](setup-push-guard.sh):
1. `remote.pushDefault = origin` → a bare `git push` always goes to your fork.
2. every non-DreCode3 remote's **push URL** → `DISABLED://…`, so `git push upstream` fails immediately.
3. a `pre-push` **guard hook** rejects any push to a non-`DreCode3` URL (so even a remote you add later is caught),
   and preserves Git LFS.

**Day-to-day:** just `git push` (→ your fork). The `opendbc` submodule pushes separately — push it first, then the
parent: `git -C opendbc_repo push && git push`.
**⚠️ Fresh clone / new machine:** these settings live in each repo's local `.git` (config + hooks) and are NOT
committed, so re-apply them once with `./setup-push-guard.sh` (also re-run it after `git submodule update --init` to
guard newly-initialized submodules). `git push --no-verify` skips the hook, but the disabled upstream URLs still block
the known upstreams.

## 7. Documentation map
- **This file** — front door / current state / dev loop.
- **Architecture & customizations:** [`explorer_st_logs/customizations.md`](explorer_st_logs/customizations.md) (the
  big guide) · [`explorer_st_logs/ford_can_reference.md`](explorer_st_logs/ford_can_reference.md) ·
  [`explorer_st_logs/discriminator_design.md`](explorer_st_logs/discriminator_design.md) ·
  [`explorer_st_logs/bluepilot_601_diffs.md`](explorer_st_logs/bluepilot_601_diffs.md) ·
  [`explorer_st_logs/longitudinal_customizations.md`](explorer_st_logs/longitudinal_customizations.md) (longitudinal/ACC).
- **Investigations (verdicts):** `learned_param_studies/RESULTS.md` · `longitudinal_weave/README.md` ·
  `explorer_st_logs/CHECKPOINT_2026-06-13_golden_PI_b8.md` · `explorer_st_logs/CHECKPOINT_2026-06-03_iter3_deployed.md`.
- **Protocols / deploys:** `controlled_test_analysis/{ANALYSIS_PLAN,DRIVE_PROTOCOL}.md` ·
  `explorer_st_logs/DEPLOY_golden_PI_on_CD210.md`.
- **Upstream / updates:** track `upstream/master` (NOT dev) + opendbc `sunnypilot/master`; surgical cherry-picks only.
- **Generic openpilot:** [`docs/`](docs/) (build, dev, safety, integration).

> Note: the assistant's private working memory (investigation log + conventions) is the most detailed source but is
> not in this repo; this file is the committable distillation of its durable parts.
