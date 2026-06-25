"""Offline model/path-vs-road discrimination for the Explorer ST lateral symptoms.

Analysis-only. Reads existing route NPZ caches and the symptom catalog; writes only
to retrospective_lateral/results/. Imports nothing from the vehicle-control stack.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from retrospective_lateral.code import config as C
from retrospective_lateral.code.signal_utils import (
    contiguous_regions,
    dilate_flags,
    erode_true,
    filter_continuous,
    gps_cells,
    heading_bin_deg,
    rms_masked,
    spectral_peak_hz,
)


def offset_to_curvature(offset_m, lookahead_m: float):
    """Convert a lateral offset at a forward lookahead to an implied path curvature.

    For a small-curvature arc through the origin, y(L) ~= kappa * L^2 / 2, so
    kappa ~= 2*y / L^2. Lets model path / lane center / road edge offsets be compared
    on the same 1/m axis as desired_curvature and yawRate/v.
    """
    L = float(lookahead_m)
    if L <= 0:
        raise ValueError("lookahead_m must be > 0")
    return 2.0 * np.asarray(offset_m, dtype=float) / (L * L)


def gps_course_deg(lat_deg, lon_deg):
    """Course over ground in degrees [0, 360), 0 = North, clockwise, from GPS displacement.

    Independent of the vision model and of EPAS. NaN where it cannot be computed.
    Approximate: treats finite samples as evenly spaced; intended as a coarse,
    model-independent heading reference, not a precision signal.
    """
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    out = np.full(lat.shape, np.nan)
    ok = np.isfinite(lat) & np.isfinite(lon)
    if ok.sum() < 2:
        return out
    lat0 = float(np.nanmedian(lat[ok]))
    east_m = lon * C.M_PER_DEG_LON_AT_EQUATOR * math.cos(math.radians(lat0))
    north_m = lat * C.M_PER_DEG_LAT
    dx = np.gradient(east_m)
    dy = np.gradient(north_m)
    course = np.degrees(np.arctan2(dx, dy)) % 360.0
    # np.gradient poisons the valid samples bordering each NaN gap, so explicitly
    # invalidate the gap samples and their immediate neighbors.
    course[dilate_flags(~ok, 1)] = np.nan
    return course


def path_curvature_from_rate(rate, v_mps, min_speed_mps: float = C.DISCRIM_MIN_SPEED_MPS):
    """Realized path curvature = angular rate / speed (1/m). NaN below the speed guard.

    Use with CAN yawRate or calibrated yawRate to get a vision-model-independent path.
    """
    rate = np.asarray(rate, dtype=float)
    v = np.asarray(v_mps, dtype=float)
    v = np.broadcast_to(v, rate.shape)
    out = np.full(rate.shape, np.nan)
    ok = np.isfinite(rate) & np.isfinite(v) & (v >= float(min_speed_mps))
    out[ok] = rate[ok] / v[ok]
    return out


def gps_path_curvature(lat_deg, lon_deg, v_mps, fs_hz: float = C.FS_HZ,
                       min_speed_mps: float = C.DISCRIM_MIN_SPEED_MPS):
    """Heading-rate curvature from GPS course (1/m). Fully model- and EPAS-independent.

    Coarse/noisy; used only as a third independent corroborator, not a primary metric.
    Because np.gradient runs on the gap-compressed valid samples, the rate can be
    slightly inflated at GPS-gap boundaries (mitigated by the dilation in gps_course_deg).
    """
    course = gps_course_deg(lat_deg, lon_deg)
    rate = np.full(course.shape, np.nan)
    ok = np.isfinite(course)
    if ok.sum() >= 3:
        unwrapped = np.unwrap(np.radians(course[ok]))
        rate[ok] = np.gradient(unwrapped) * float(fs_hz)
    return path_curvature_from_rate(rate, v_mps, min_speed_mps=min_speed_mps)


def xcorr_best(a, b, fs_hz: float, max_lag_s: float = C.DISCRIM_MAX_LAG_S):
    """Best Pearson correlation over integer lags on the largest shared finite run.

    Returns (corr, lag_s). lag_s > 0 means `a` leads `b`. (nan, nan) if undecidable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    min_len = max(12, int(fs_hz * 3))
    regions = contiguous_regions(np.isfinite(a) & np.isfinite(b), min_len=min_len)
    if not regions:
        return (math.nan, math.nan)
    s, e = max(regions, key=lambda r: r[1] - r[0])
    aa = a[s:e]
    bb = b[s:e]
    n = len(aa)
    if np.std(aa) == 0 or np.std(bb) == 0:
        return (math.nan, math.nan)
    max_lag = int(round(max_lag_s * fs_hz))
    best_corr = -2.0
    best_lag = 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = aa[-lag:], bb[:n + lag]
        elif lag > 0:
            x, y = aa[:n - lag], bb[lag:]
        else:
            x, y = aa, bb
        if len(x) < min_len or np.std(x) == 0 or np.std(y) == 0:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best_corr:
            best_corr = c
            best_lag = lag
    if best_corr < -1.5:
        return (math.nan, math.nan)
    return (best_corr, best_lag / fs_hz)


@dataclass(frozen=True)
class DiscriminationWindow:
    symptom: str
    route_id: str
    start_s: float
    end_s: float
    peak_s: float
    speed_mph_median: float
    lookahead_m: float
    gps_cell: float
    heading_bin: int
    speed_bin: int
    road_curv_level_1pm: float
    model_curv_rms_1pm: float
    lane_curv_rms_1pm: float
    roadedge_curv_rms_1pm: float
    desired_rms_1pm: float
    cp_final_rms_1pm: float
    can_yaw_curv_rms_1pm: float
    cal_yaw_curv_rms_1pm: float
    gps_curv_rms_1pm: float
    steering_band_rms_deg: float
    model_vs_lane_corr: float
    model_vs_lane_lag_s: float
    lane_vs_independent_corr: float
    lane_vs_independent_lag_s: float
    model_minus_lane_residual_rms_1pm: float
    model_residual_over_lane: float
    spectral_peak_hz: float
    clean_fraction: float
    straight_clean: int  # 0/1 flag (int for DataFrame/CSV compatibility)
    tentative_label: str


def _band_for(symptom: str):
    return C.LOW_SPEED_INSPECT_BAND_HZ if symptom == "low_speed_wheel_swing" else C.DEFAULT_WEAVE_BAND_HZ


def _clean_mask(arrays) -> np.ndarray:
    n = len(arrays["t"])
    fs = C.FS_HZ

    def chan(name, default):
        return np.asarray(arrays.get(name, np.full(n, default)), dtype=float)

    lat_active = chan("lat_active", 0.0) > 0.5
    pressed = chan("steering_pressed", 0.0) > 0.5
    blink = chan("blinker", 0.0) > 0.5
    lane_change = chan("lane_change_state", 0.0) > 0.5
    headway = chan("lead_time_headway_s", np.nan)
    near_lead = np.isfinite(headway) & (headway < C.LEAD_HEADWAY_S)
    bad = dilate_flags(pressed | blink | lane_change | near_lead, int(round(C.OVERRIDE_BUFFER_S * fs)))
    clean = lat_active & ~bad
    return erode_true(clean, int(round(C.ENGAGE_ERODE_S * fs)))


def _roadedge_center(arrays, lk: str):
    left = arrays.get(f"road_edge_left_{lk}")
    right = arrays.get(f"road_edge_right_{lk}")
    if left is None or right is None:
        return np.full(len(arrays["t"]), np.nan)
    return 0.5 * (np.asarray(left, dtype=float) + np.asarray(right, dtype=float))


def _rms_band(sig, win_clean, fs, band):
    return rms_masked(filter_continuous(sig, fs, band=band), win_clean)


def discriminate_window(arrays, symptom: str, route_id: str, start_s: float, end_s: float,
                        peak_s: float, *, lookahead_m: float = C.DISCRIM_LOOKAHEAD_M,
                        band=None) -> DiscriminationWindow:
    fs = C.FS_HZ
    t = np.asarray(arrays["t"], dtype=float)
    lk = f"y{int(lookahead_m)}"
    if band is None:
        band = _band_for(symptom)

    win = (t >= start_s) & (t <= end_s)
    clean = _clean_mask(arrays)
    win_clean = win & clean
    clean_fraction = float(np.mean(clean[win])) if win.any() else math.nan

    def _finite_median(x, mask):
        # Median over finite values under `mask`; nan for empty/all-NaN. Avoids the
        # RuntimeWarning('All-NaN slice') that np.nanmedian emits on degenerate windows.
        vals = np.asarray(x, dtype=float)[mask]
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if len(vals) else math.nan

    v = np.asarray(arrays["v_ego"], dtype=float)
    speed_med = _finite_median(v, win)
    speed_mph = speed_med * C.MPS_TO_MPH if np.isfinite(speed_med) else math.nan

    model_curv = offset_to_curvature(arrays[f"model_{lk}"], lookahead_m)
    lane_curv = offset_to_curvature(arrays[f"lane_center_{lk}"], lookahead_m)
    roadedge_curv = offset_to_curvature(_roadedge_center(arrays, lk), lookahead_m)
    desired = np.asarray(arrays["desired_curvature"], dtype=float)
    cp_final = np.asarray(arrays.get("cp_final_command", np.full(len(t), np.nan)), dtype=float)
    can_yaw_curv = path_curvature_from_rate(arrays["yaw_rate"], v)
    cal_yaw_curv = path_curvature_from_rate(arrays.get("yaw_rate_calibrated", np.full(len(t), np.nan)), v)
    gps_curv = gps_path_curvature(arrays["lat"], arrays["lon"], v)
    steering = np.asarray(arrays["steering_angle_deg"], dtype=float)

    model_b = filter_continuous(model_curv, fs, band=band)
    lane_b = filter_continuous(lane_curv, fs, band=band)
    indep_src = cal_yaw_curv if np.isfinite(cal_yaw_curv[win]).sum() >= int(fs * 3) else can_yaw_curv
    indep_b = filter_continuous(indep_src, fs, band=band)

    mvl_corr, mvl_lag = xcorr_best(np.where(win_clean, model_b, np.nan), np.where(win_clean, lane_b, np.nan), fs)
    lvi_corr, lvi_lag = xcorr_best(np.where(win_clean, lane_b, np.nan), np.where(win_clean, indep_b, np.nan), fs)

    lane_rms = _rms_band(lane_curv, win_clean, fs, band)
    model_rms = _rms_band(model_curv, win_clean, fs, band)
    residual = model_b - lane_b
    residual_rms = rms_masked(residual, win_clean)
    residual_ratio = residual_rms / lane_rms if (np.isfinite(lane_rms) and lane_rms > 0) else math.nan

    road_lp = filter_continuous(can_yaw_curv, fs, lowpass_hz=C.ROAD_LP_HZ)
    road_vals = np.abs(road_lp[win_clean])
    road_vals = road_vals[np.isfinite(road_vals)]
    road_level = float(np.median(road_vals)) if len(road_vals) else math.nan

    lane_prob_l = arrays.get("lane_prob_left", np.full(len(t), np.nan))
    lane_prob_r = arrays.get("lane_prob_right", np.full(len(t), np.nan))
    med_l = _finite_median(lane_prob_l, win)
    med_r = _finite_median(lane_prob_r, win)
    lane_ok = np.isfinite(med_l) and med_l >= 0.5 and np.isfinite(med_r) and med_r >= 0.5
    # int (0/1) so the dataclass serializes cleanly to a DataFrame/CSV column.
    straight_clean = int(
        np.isfinite(road_level) and road_level < C.ROAD_CURV_ABS_MAX_1PM
        and bool(lane_ok) and np.isfinite(clean_fraction) and clean_fraction >= 0.8
    )

    coh = C.DISCRIM_ROAD_COHERENCE_MIN
    if not (np.isfinite(lane_rms) and np.isfinite(model_rms)) or (lane_rms == 0.0 and model_rms == 0.0):
        label = "insufficient"
    elif np.isfinite(residual_ratio) and residual_ratio >= C.DISCRIM_ARTIFACT_RESIDUAL_RATIO:
        label = "artifact_like"
    elif np.isfinite(mvl_corr) and abs(mvl_corr) >= coh and np.isfinite(lvi_corr) and abs(lvi_corr) >= coh:
        label = "road_like"
    else:
        label = "ambiguous"

    course = gps_course_deg(arrays["lat"], arrays["lon"])
    heading_bin = int(np.nanmedian(heading_bin_deg(course[win], C.HEADING_BIN_DEG))) if win.any() else -1
    cell_vals = gps_cells(arrays["lat"], arrays["lon"], C.GPS_CELL_M)[win]
    cell_vals = cell_vals[np.isfinite(cell_vals)]
    gps_cell = float(np.median(cell_vals)) if len(cell_vals) else math.nan
    speed_bin = int(speed_mph // C.SPEED_BIN_MPH) if np.isfinite(speed_mph) else -1

    return DiscriminationWindow(
        symptom=symptom, route_id=route_id, start_s=float(start_s), end_s=float(end_s),
        peak_s=float(peak_s), speed_mph_median=speed_mph, lookahead_m=float(lookahead_m),
        gps_cell=gps_cell, heading_bin=heading_bin, speed_bin=speed_bin,
        road_curv_level_1pm=road_level,
        model_curv_rms_1pm=model_rms, lane_curv_rms_1pm=lane_rms,
        roadedge_curv_rms_1pm=_rms_band(roadedge_curv, win_clean, fs, band),
        desired_rms_1pm=_rms_band(desired, win_clean, fs, band),
        cp_final_rms_1pm=_rms_band(cp_final, win_clean, fs, band),
        can_yaw_curv_rms_1pm=_rms_band(can_yaw_curv, win_clean, fs, band),
        cal_yaw_curv_rms_1pm=_rms_band(cal_yaw_curv, win_clean, fs, band),
        gps_curv_rms_1pm=_rms_band(gps_curv, win_clean, fs, band),
        steering_band_rms_deg=_rms_band(steering, win_clean, fs, band),
        model_vs_lane_corr=mvl_corr, model_vs_lane_lag_s=mvl_lag,
        lane_vs_independent_corr=lvi_corr, lane_vs_independent_lag_s=lvi_lag,
        model_minus_lane_residual_rms_1pm=residual_rms, model_residual_over_lane=residual_ratio,
        spectral_peak_hz=spectral_peak_hz(np.where(win_clean, lane_b, np.nan), fs, band),
        clean_fraction=clean_fraction, straight_clean=straight_clean, tentative_label=label,
    )


def spatial_curvature_profile(arrays, start_s: float, end_s: float, band,
                              *, lookahead_m: float = C.DISCRIM_LOOKAHEAD_M,
                              cell_m: float = C.GPS_CELL_M) -> dict:
    """Mean band-filtered lane/model curvature per GPS cell over a window.

    Keying the wobble to physical GPS cells (not time) lets two passes over the same
    road be compared. A wobble that reproduces by cell across passes is a road feature.
    """
    fs = C.FS_HZ
    t = np.asarray(arrays["t"], dtype=float)
    lk = f"y{int(lookahead_m)}"
    win = (t >= start_s) & (t <= end_s)
    lane_b = filter_continuous(offset_to_curvature(arrays[f"lane_center_{lk}"], lookahead_m), fs, band=band)
    model_b = filter_continuous(offset_to_curvature(arrays[f"model_{lk}"], lookahead_m), fs, band=band)
    cells = gps_cells(arrays["lat"], arrays["lon"], cell_m)
    out: dict[float, dict] = {}
    idx = np.flatnonzero(win)
    for i in idx:
        c = cells[i]
        if not np.isfinite(c):
            continue
        bucket = out.setdefault(float(c), {"_lane": [], "_model": []})
        if np.isfinite(lane_b[i]):
            bucket["_lane"].append(float(lane_b[i]))
        if np.isfinite(model_b[i]):
            bucket["_model"].append(float(model_b[i]))
    profile: dict[float, dict] = {}
    for c, bucket in out.items():
        if not bucket["_lane"]:
            continue
        profile[c] = {
            "lane": float(np.mean(bucket["_lane"])),
            "model": float(np.mean(bucket["_model"])) if bucket["_model"] else math.nan,
            "n": len(bucket["_lane"]),
        }
    return profile


def cross_pass_reproducibility(profiles: list, key: str = "lane",
                               min_shared_cells: int = C.DISCRIM_REPRO_MIN_SHARED_CELLS) -> dict:
    """Median pairwise Pearson correlation of per-cell profiles across passes.

    profiles: list of {gps_cell: {"lane":..., "model":..., "n":...}}, one per pass.
    """
    corrs = []
    shared_counts = []
    for pa, pb in combinations(profiles, 2):
        shared = sorted(set(pa) & set(pb))
        shared = [c for c in shared if np.isfinite(pa[c].get(key, np.nan)) and np.isfinite(pb[c].get(key, np.nan))]
        if len(shared) < min_shared_cells:
            continue
        va = np.array([pa[c][key] for c in shared])
        vb = np.array([pb[c][key] for c in shared])
        if np.std(va) == 0 or np.std(vb) == 0:
            continue
        corrs.append(float(np.corrcoef(va, vb)[0, 1]))
        shared_counts.append(len(shared))
    if not corrs:
        return {"n_passes": len(profiles), "n_pairs": 0, "n_shared_cells": 0,
                "median_pairwise_corr": math.nan, "reproducible": False}
    median_corr = float(np.median(corrs))
    return {
        "n_passes": len(profiles), "n_pairs": len(corrs),
        "n_shared_cells": int(np.median(shared_counts)),
        "median_pairwise_corr": median_corr,
        "reproducible": bool(median_corr >= C.DISCRIM_REPRO_FRACTION_ROAD),
    }


def frequency_speed_slope(records: list) -> dict:
    """OLS slope of spectral_peak_hz on speed_mph_median across windows.

    |slope| below the flat threshold => the weave frequency is speed-independent,
    which is consistent with a fixed time-constant loop limit-cycle, not a road wavelength.
    """
    pts = [(float(r["speed_mph_median"]), float(r["spectral_peak_hz"])) for r in records
           if np.isfinite(r.get("speed_mph_median", np.nan)) and np.isfinite(r.get("spectral_peak_hz", np.nan))]
    if len(pts) < 5:
        return {"n": len(pts), "slope_hz_per_mph": math.nan, "flat": False}
    speeds = np.array([p[0] for p in pts])
    peaks = np.array([p[1] for p in pts])
    if np.std(speeds) == 0:
        return {"n": len(pts), "slope_hz_per_mph": math.nan, "flat": False}
    slope = float(np.polyfit(speeds, peaks, 1)[0])
    return {"n": len(pts), "slope_hz_per_mph": slope,
            "flat": bool(abs(slope) < C.DISCRIM_FREQ_FLAT_HZ_PER_MPH)}


def classify_source(record: dict, repro_fraction: float, freq_flat: bool) -> tuple[str, str]:
    """Combine the three tests into a single A/B/C/ambiguous source label.

    A/B/C are taxonomy codes (A=model_artifact, B=road_feature, C=loop_limit_cycle),
    NOT priority ranks. Priority (evaluation order): insufficient -> road (reproducible)
    -> artifact (model residual) -> loop (fixed-frequency on a straight clean road) -> ambiguous.
    """
    lane_rms = record.get("lane_curv_rms_1pm", math.nan)
    model_rms = record.get("model_curv_rms_1pm", math.nan)
    if not (np.isfinite(lane_rms) and np.isfinite(model_rms)):
        return ("insufficient_evidence", "lane/model curvature unavailable")

    residual_ratio = record.get("model_residual_over_lane", math.nan)
    mvl = abs(record.get("model_vs_lane_corr", 0.0) or 0.0)
    lvi = abs(record.get("lane_vs_independent_corr", 0.0) or 0.0)
    coh = C.DISCRIM_ROAD_COHERENCE_MIN
    moving_together = mvl >= coh and lvi >= coh

    if np.isfinite(repro_fraction) and repro_fraction >= C.DISCRIM_REPRO_FRACTION_ROAD and moving_together:
        return ("road_feature_B", f"reproducible by location (corr={repro_fraction:.2f}); planned/perceived/realized coherent")
    if np.isfinite(residual_ratio) and residual_ratio >= C.DISCRIM_ARTIFACT_RESIDUAL_RATIO and not moving_together:
        return ("model_artifact_A", f"model adds motion beyond lane/road (residual/lane={residual_ratio:.2f})")
    if freq_flat and int(record.get("straight_clean", 0)) == 1:
        return ("loop_limit_cycle_C", "fixed-frequency oscillation on a straight, clean, lead-free road")
    return ("ambiguous", "no single test decisive")
