#!/usr/bin/env python3
"""
Did effective lateral SMOOTHING (tau) or lateral DELAY (action_t) change between builds?

Per route (c7, c5 = OLD build; ce, cf = NEW build; all Nevada model):
 1. liveDelay medians (lateralDelay, lateralDelayEstimate, validBlocks) + initData params
    (LagdToggle, LagdValueCache, ModelManager_ActiveBundle overrides/generation).
 2. Reconstruct curv_from_plan per modelV2 msg:
      psi = interp(action_t, T_IDXS, orientation.z)
      curv = 2*psi/(v*action_t) - orientationRate.z[0]/v   (v clipped >= 1, matches
      curv_from_psis in selfdrive/controls/lib/drive_helpers.py)
 3. Grid-fit tau in [0, 0.05 .. 1.0] minimizing RMS between logged
    modelV2.action.desiredCurvature and recursive smooth(curv_from_plan, tau) at dt=0.05,
    scored on moving frames (v > 5 m/s).
 4. Joint 2D grid over (action_t, tau) -> effective action_t independent of manifest.
 5. Cross-correlation peak lag (frames) between logged desired and curv_from_plan.

Sign convention note: everything here is compared in the SAME native frame
(modelV2 orientation.z / orientationRate.z / action.desiredCurvature), so no
cross-frame sign assumptions are made.
"""
import json
import os
import sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/Users/dregilley/Documents/GitHub/sunnypilot")
from openpilot.tools.lib.logreader import LogReader
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

T_IDXS = np.array(ModelConstants.T_IDXS)
DT = 0.05
MIN_SPEED = 1.0
MOVING_V = 5.0

BASE = "/Users/dregilley/Documents/GitHub/sunnypilot/explorer_st_logs"
ROUTES = {
  "c7": ("OLD", "route_c7"),
  "c5": ("OLD", "route_c5"),
  "ce": ("NEW", "route_ce"),
  "cf": ("NEW", "route_cf"),
}


def seg_key(d):
  return int(d.rsplit("--", 1)[1])


def load_segment(path):
  """Extract per-segment arrays. Returns dict."""
  out = {
    "t": [], "desired": [], "ori_z": [], "orate_z0": [], "v_ego": [],
    "ld_lateralDelay": [], "ld_estimate": [], "ld_validBlocks": [],
    "params": None,
  }
  cur_v = np.nan
  try:
    lr = LogReader(path)
  except Exception as e:
    return {"error": f"{path}: open fail {e}", **out}
  it = iter(lr)
  while True:
    try:
      msg = next(it)
    except StopIteration:
      break
    except Exception:
      continue  # corrupted events mid-iteration: skip
    try:
      w = msg.which()
    except Exception:
      continue
    try:
      if w == "carState":
        cur_v = msg.carState.vEgo
      elif w == "modelV2":
        m = msg.modelV2
        oz = np.array(m.orientation.z, dtype=np.float64)
        if oz.size != len(T_IDXS):
          continue
        out["t"].append(msg.logMonoTime * 1e-9)
        out["desired"].append(m.action.desiredCurvature)
        out["ori_z"].append(oz)
        out["orate_z0"].append(m.orientationRate.z[0])
        out["v_ego"].append(cur_v)
      elif w == "liveDelay":
        ld = msg.liveDelay
        out["ld_lateralDelay"].append(ld.lateralDelay)
        out["ld_estimate"].append(ld.lateralDelayEstimate)
        out["ld_validBlocks"].append(ld.validBlocks)
      elif w == "initData" and out["params"] is None:
        p = {}
        for e in msg.initData.params.entries:
          k = e.key
          if k in ("LagdToggle", "LagdValueCache", "ModelManager_ActiveBundle"):
            p[k] = bytes(e.value).decode("utf-8", errors="replace")
        out["params"] = p
    except Exception:
      continue
  for k in ("t", "desired", "orate_z0", "v_ego", "ld_lateralDelay", "ld_estimate", "ld_validBlocks"):
    out[k] = np.array(out[k], dtype=np.float64)
  out["ori_z"] = np.array(out["ori_z"], dtype=np.float64) if out["ori_z"] else np.zeros((0, len(T_IDXS)))
  return out


def curv_from_plan(ori_z, orate_z0, v_ego, action_t):
  """Vectorized: psi = interp(action_t, T_IDXS, ori_z[i]); curv = 2*psi/(v*action_t) - orate_z0/v."""
  j = np.searchsorted(T_IDXS, action_t)
  j = min(max(j, 1), len(T_IDXS) - 1)
  t0, t1 = T_IDXS[j - 1], T_IDXS[j]
  w = (action_t - t0) / (t1 - t0)
  psi = (1 - w) * ori_z[:, j - 1] + w * ori_z[:, j]
  v = np.clip(v_ego, MIN_SPEED, np.inf)
  return 2.0 * psi / (v * action_t) - orate_z0 / v


def simulate_smooth(plan_curv, desired, v_ego, t, tau):
  """Recursive smoother at dt=0.05 per contiguous chunk (gap > 0.25s resets, seeded from log).
  Emulates modeld hold at v<=0.3."""
  alpha = 1.0 - np.exp(-DT / tau) if tau > 0 else 1.0
  n = len(plan_curv)
  pred = np.empty(n)
  prev = desired[0]
  prev_t = t[0] - DT
  for i in range(n):
    if t[i] - prev_t > 0.25:
      prev = desired[i - 1] if i > 0 else desired[0]
      # chunk reset: seed with logged value at reset point
      prev = desired[i]
      pred[i] = desired[i]
      prev_t = t[i]
      continue
    if v_ego[i] <= 0.3:
      pred[i] = prev
    else:
      pred[i] = alpha * plan_curv[i] + (1 - alpha) * prev
    prev = pred[i]
    prev_t = t[i]
  return pred


def fit_route(name, kind, route_dir):
  d = os.path.join(BASE, route_dir)
  segs = sorted([s for s in os.listdir(d) if "--" in s], key=seg_key)
  paths = [os.path.join(d, s, "rlog.zst") for s in segs]
  paths = [p for p in paths if os.path.exists(p)]

  with ProcessPoolExecutor(max_workers=12) as ex:
    results = list(ex.map(load_segment, paths))

  # concatenate in segment order
  t = np.concatenate([r["t"] for r in results])
  desired = np.concatenate([r["desired"] for r in results])
  ori_z = np.concatenate([r["ori_z"] for r in results])
  orate_z0 = np.concatenate([r["orate_z0"] for r in results])
  v_ego = np.concatenate([r["v_ego"] for r in results])
  ld = np.concatenate([r["ld_lateralDelay"] for r in results])
  ld_est = np.concatenate([r["ld_estimate"] for r in results])
  ld_vb = np.concatenate([r["ld_validBlocks"] for r in results])

  params = next((r["params"] for r in results if r["params"]), {})
  lagd_toggle = params.get("LagdToggle", "?")
  lagd_cache = float(params.get("LagdValueCache", "nan"))
  bundle = {}
  try:
    bundle = json.loads(params.get("ModelManager_ActiveBundle", "{}"))
  except Exception:
    pass
  overrides = {o.get("key"): o.get("value") for o in bundle.get("overrides", [])} if isinstance(bundle.get("overrides"), list) else bundle.get("overrides", {})
  lat_override = overrides.get("lat", None)
  gen = bundle.get("generation", None)

  # NaN v_ego (no carState yet at start): drop those rows entirely
  ok = np.isfinite(v_ego)
  t, desired, ori_z, orate_z0, v_ego = t[ok], desired[ok], ori_z[ok], orate_z0[ok], v_ego[ok]

  med_ld = float(np.median(ld))
  action_t_task = med_ld + DT  # task formula
  lat_ov_f = float(lat_override) if lat_override not in (None, "") else 0.0
  eff_lagd = lagd_cache if lagd_toggle == "1" else med_ld
  action_t_code = eff_lagd + lat_ov_f + DT  # code-true: (get_lat_delay + LAT_SMOOTH_SECONDS) + DT_MDL

  moving = v_ego > MOVING_V

  def rms_for(action_t, tau):
    pc = curv_from_plan(ori_z, orate_z0, v_ego, action_t)
    pred = simulate_smooth(pc, desired, v_ego, t, tau)
    e = pred[moving] - desired[moving]
    return float(np.sqrt(np.mean(e * e))), pc

  taus = np.round(np.arange(0.0, 1.0001, 0.05), 3)

  # 1D tau fit at action_t_task
  rms_task = [rms_for(action_t_task, tau)[0] for tau in taus]
  best_i = int(np.argmin(rms_task))

  # 1D tau fit at action_t_code
  rms_code = [rms_for(action_t_code, tau)[0] for tau in taus]
  best_j = int(np.argmin(rms_code))

  # 2D joint fit (finer action_t grid)
  ats = np.round(np.arange(0.10, 1.2001, 0.05), 3)
  best2 = (None, None, np.inf)
  for at in ats:
    pc = curv_from_plan(ori_z, orate_z0, v_ego, at)
    for tau in taus:
      pred = simulate_smooth(pc, desired, v_ego, t, tau)
      e = pred[moving] - desired[moving]
      r = float(np.sqrt(np.mean(e * e)))
      if r < best2[2]:
        best2 = (float(at), float(tau), r)

  # refine action_t around best on 0.01 grid
  at0 = best2[0]
  for at in np.round(np.arange(max(0.05, at0 - 0.06), at0 + 0.0601, 0.01), 3):
    pc = curv_from_plan(ori_z, orate_z0, v_ego, at)
    for tau in np.round(np.arange(max(0.0, best2[1] - 0.05), best2[1] + 0.0501, 0.01), 3):
      pred = simulate_smooth(pc, desired, v_ego, t, tau)
      e = pred[moving] - desired[moving]
      r = float(np.sqrt(np.mean(e * e)))
      if r < best2[2]:
        best2 = (float(at), float(tau), r)

  # one-step alpha regression (sanity): d[i] = a*p[i] + (1-a)*d[i-1]
  pc_task = curv_from_plan(ori_z, orate_z0, v_ego, action_t_task)
  dd = desired[1:] - desired[:-1]
  pp = pc_task[1:] - desired[:-1]
  m2 = moving[1:] & moving[:-1] & (np.diff(t) < 0.25)
  a_hat = float(np.sum(dd[m2] * pp[m2]) / np.sum(pp[m2] * pp[m2]))
  tau_hat = float(-DT / np.log(1 - a_hat)) if 0 < a_hat < 1 else (0.0 if a_hat >= 1 else np.nan)

  # cross-correlation peak lag (frames), moving frames, demeaned, at action_t_task
  x = desired.copy()
  y = pc_task.copy()
  x[~moving] = np.nan
  y[~moving] = np.nan
  # use only the largest contiguous run logic: simpler — fill nan with 0 after demean of moving part
  xm, ym = np.nanmean(x), np.nanmean(y)
  x = np.where(np.isnan(x), 0.0, x - xm)
  y = np.where(np.isnan(y), 0.0, y - ym)
  lags = np.arange(-40, 41)
  cc = np.array([np.sum(x[max(0, k):len(x) + min(0, k)] * y[max(0, -k):len(y) - max(0, k)]) /
                 (np.sqrt(np.sum(x**2) * np.sum(y**2)) + 1e-12) for k in lags])
  peak_lag = int(lags[np.argmax(cc)])
  peak_cc = float(np.max(cc))

  return {
    "route": name, "kind": kind,
    "n_segs": len(paths), "n_modelV2": int(len(t)), "n_moving": int(moving.sum()),
    "lagd_toggle": lagd_toggle, "lagd_cache": lagd_cache,
    "bundle_name": bundle.get("displayName"), "bundle_index": bundle.get("index"),
    "generation": gen, "overrides": overrides,
    "ld_median": med_ld, "ld_est_median": float(np.median(ld_est)),
    "ld_vb_median": float(np.median(ld_vb)), "n_liveDelay": int(len(ld)),
    "ld_min": float(np.min(ld)), "ld_max": float(np.max(ld)),
    "action_t_task": action_t_task, "action_t_code": action_t_code,
    "tau_grid": taus.tolist(),
    "rms_task": rms_task, "best_tau_task": float(taus[best_i]), "best_rms_task": rms_task[best_i],
    "rms_code": rms_code, "best_tau_code": float(taus[best_j]), "best_rms_code": rms_code[best_j],
    "joint_action_t": best2[0], "joint_tau": best2[1], "joint_rms": best2[2],
    "alpha_hat": a_hat, "tau_hat_onestep": tau_hat,
    "xcorr_peak_lag_frames": peak_lag, "xcorr_peak": peak_cc,
    "desired_rms_moving": float(np.sqrt(np.mean(desired[moving]**2))),
  }


def main():
  out = {}
  for name, (kind, rd) in ROUTES.items():
    print(f"--- loading/fitting route_{name} ({kind}) ...", flush=True)
    out[name] = fit_route(name, kind, rd)
    r = out[name]
    print(json.dumps({k: v for k, v in r.items() if k not in ("rms_task", "rms_code", "tau_grid")}, indent=1))
  cache = "/private/tmp/claude-501/-Users-dregilley-Documents-GitHub-sunnypilot/29f90cf8-9f2a-4abc-94c6-33e5771ceae1/scratchpad/lat_smoothing_fit_results.json"
  os.makedirs(os.path.dirname(cache), exist_ok=True)
  with open(cache, "w") as f:
    json.dump(out, f, indent=1)
  print("\nsaved:", cache)

  # summary table
  print("\n===== SUMMARY =====")
  hdr = f"{'route':6}{'build':6}{'ld_med':>8}{'lagdCache':>10}{'lat_ov':>8}{'at_task':>9}{'at_code':>9}" \
        f"{'tau@task':>9}{'rms@task':>10}{'tau@code':>9}{'rms@code':>10}{'jointAt':>9}{'jointTau':>9}{'jointRMS':>10}{'xlag':>6}{'xcc':>7}"
  print(hdr)
  for name, r in out.items():
    print(f"{name:6}{r['kind']:6}{r['ld_median']:8.3f}{r['lagd_cache']:10.3f}{str(r['overrides'].get('lat')):>8}"
          f"{r['action_t_task']:9.3f}{r['action_t_code']:9.3f}"
          f"{r['best_tau_task']:9.2f}{r['best_rms_task']:10.2e}{r['best_tau_code']:9.2f}{r['best_rms_code']:10.2e}"
          f"{r['joint_action_t']:9.2f}{r['joint_tau']:9.2f}{r['joint_rms']:10.2e}{r['xcorr_peak_lag_frames']:6d}{r['xcorr_peak']:7.3f}")


if __name__ == "__main__":
  main()
