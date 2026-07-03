#!/usr/bin/env python3
"""NEGATIVE CONTROL for the toolkit port (criteria pre-registered in ACCEPTANCE.md §C).

Two layers:
1) Observed 04-vs-05: stock_hiram_04 vs stock_hiram_05 through the full matched pipeline
   (same build, same corridor, opposite directions) — expect null.
2) Visit-permutation null: within each (GPS cell, speed bin), pool both drives' per-visit
   medians and randomly reassign them to two pseudo-groups (preserving group sizes), then
   recompute the paired stats. N shuffles give the null band the observed stats must sit in.
   This preserves the matching structure exactly, so it calibrates what "null" looks like
   for THIS corridor/coverage instead of assuming win-rate==50%.

RUN: .venv311/bin/python stock_lateral_toolkit/validation/self_split.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_compare as AC

NSHUFFLE = 1000   # QA 2026-07-03: 200 left [2.5,97.5] band edges ~2x seed-wobbly; 1000 stabilizes
SEED = 11


def paired_stats(sm, cm, si):
    """Median-of-paired-medians delta + fraction of cells where group A is lower."""
    keys = [k for k in sm if k[1] == si and k in cm]
    if len(keys) < 4:
        return None
    pairs = np.array([(np.median(sm[k]), np.median(cm[k])) for k in keys])
    delta = float(np.median(pairs[:, 0]) - np.median(pairs[:, 1]))
    win_a = float(np.mean(pairs[:, 0] - pairs[:, 1] < 0))
    return delta, win_a, len(keys)


def shuffle_groups(sm, cm, rng):
    """Within each key, pool visit-medians from both groups and reassign preserving sizes."""
    sa, sb = {}, {}
    for k in set(sm) | set(cm):
        pool = list(sm.get(k, [])) + list(cm.get(k, []))
        na = len(sm.get(k, []))
        rng.shuffle(pool)
        if na:
            sa[k] = pool[:na]
        if len(pool) > na:
            sb[k] = pool[na:]
    return sa, sb


def run_metric(grids_a, grids_b, mask_fn, value_fn, label):
    sm = AC.cellmetric(grids_a, mask_fn, value_fn)
    cm = AC.cellmetric(grids_b, mask_fn, value_fn)
    print(f"\n  {label}")
    print(f"   speedbin | n_cells | observed delta (04-05) | win04 | null delta [2.5,97.5] | null win [2.5,97.5] | verdict")
    rng = np.random.default_rng(SEED)
    verdicts = []
    for si, (lo, hi) in enumerate(AC.SPEED_BINS):
        obs = paired_stats(sm, cm, si)
        if obs is None:
            print(f"   {lo:>2}-{hi:<2} m/s |   <4    | (too few matched cells — bin not gated)")
            continue
        delta, win, n = obs
        nd, nw = [], []
        for _ in range(NSHUFFLE):
            pa, pb = shuffle_groups(sm, cm, rng)
            st = paired_stats(pa, pb, si)
            if st is not None:
                nd.append(st[0]); nw.append(st[1])
        dlo, dhi = np.percentile(nd, [2.5, 97.5])
        wlo, whi = np.percentile(nw, [2.5, 97.5])
        in_null = (dlo <= delta <= dhi) and (wlo <= win <= whi)
        win_band = 0.30 <= win <= 0.70
        verdict = "PASS" if (in_null and win_band) else "FAIL"
        verdicts.append(verdict)
        print(f"   {lo:>2}-{hi:<2} m/s |  {n:>5}  | {delta:+22.5f} | {win*100:4.0f}% | "
              f"[{dlo:+.5f},{dhi:+.5f}] | [{wlo*100:3.0f}%,{whi*100:3.0f}%] | {verdict}")
    return verdicts


def main():
    grids_a = AC.collect(AC.load(["stock_hiram_04"]))
    grids_b = AC.collect(AC.load(["stock_hiram_05"]))

    # discipline: report per-group speed medians first (barely-overlapping speeds ⇒ invalid)
    for name, gg in (("stock_hiram_04", grids_a), ("stock_hiram_05", grids_b)):
        v = np.concatenate([d["vEgo"][d["active"]] for d in gg if d["active"].any()])
        print(f"{name}: active speed median {np.median(v):.1f} m/s, IQR [{np.percentile(v,25):.1f}, {np.percentile(v,75):.1f}]")

    Astr = lambda d: d["active"] & d["straight"]
    Agood = lambda d: d["active"] & (d["innerProb"] > 0.6)

    all_v = []
    all_v += run_metric(grids_a, grids_b, Astr, lambda d: np.abs(d["bp_yaw"]),
                        "PRIMARY (gates acceptance): band|yawRate| 0.10-0.35Hz, straights (rad/s)")
    # reported, not gating (04/05 are opposite directions; crown/direction can be real):
    run_metric(grids_a, grids_b, Agood, lambda d: np.abs(d["offset"]),
               "reported only: |lane offset| good-perception (m)")

    fails = [v for v in all_v if v == "FAIL"]
    print(f"\nNEGATIVE CONTROL {'PASS' if not fails else 'FAIL'} "
          f"({len(all_v) - len(fails)}/{len(all_v)} gated bins pass; primary metric only)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
