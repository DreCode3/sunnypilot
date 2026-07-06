#!/usr/bin/env python3
"""F3 post-drive road check: selfdrived CPU budget + ACTUAL on-device F3 events.

The desk-side F3 gates (f3_replay_check / f3_stock_census) replay the monitor over
logged modelV2; this script instead reads what the DEPLOYED build really did:

  CPU  — selfdrived CPU% from procLog (delta cpuUser+cpuSystem over wall time,
         percent of one core, the test_onroad convention). GATE: median must stay
         within +3.0 %-points of the pre-F3 baseline (18.5 % median, QA packet
         2026-07-04 §CPU). card/modeld medians reported for context.
  F3   — onroadEventsSP census: aolLowLaneConfidence episodes + active seconds
         (the shipped visual alert), and aolLaneDeparture event count, which MUST
         be 0 (departure ships in SHADOW mode — cloudlog only, never an event).
  LOG  — logMessage lines mentioning the AOL safeguard (shadow-departure
         fire/active/clear telemetry), counted per route.

RUN:  cd /Users/dregilley/Documents/GitHub/sunnypilot && \
      PYTHONPATH=$PWD:$PWD/opendbc_repo \
      .venv311/bin/python retrospective_lateral/incident_2026_07_01/scripts/f3_cpu_check.py \
      <route_dir_prefix> [...]   # e.g. explorer_st_logs/shakedown_2026-07-06/0000000b--4eab43a852
Exit status: 0 = CPU gate PASS and zero departure events on every route, else 1.
"""
import glob
import json
import os
import re
import sys

ROOT = "/Users/dregilley/Documents/GitHub/sunnypilot"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/opendbc_repo")

from openpilot.tools.lib.logreader import LogReader  # noqa: E402

CPU_BASELINE_MEDIAN = 18.5   # pre-F3 selfdrived median, QA packet 2026-07-04
CPU_GATE_DELTA = 3.0         # allowed median regression in %-points
# procLog .name is a 15-char comm (modeld/dmonitoringmodeld collide) -> match on
# full cmdline; processes are launched by module path (setproctitle).
WATCH_PROCS = ("selfdrived", "card", "modeld")
CMDLINE_MATCH = {
    "selfdrived": "selfdrive.selfdrived.selfdrived",
    "card": "selfdrive.car.card",
    "modeld": "selfdrive.modeld.modeld",
}
AOL_LOWCONF = "aolLowLaneConfidence"
AOL_DEPART = "aolLaneDeparture"


def _seg_num(path):
    m = re.search(r"--(\d+)/rlog", path)
    return int(m.group(1)) if m else 0


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def scan_route(prefix):
    segs = sorted(glob.glob(f"{prefix}--*/rlog.zst"), key=_seg_num) or \
           sorted(glob.glob(f"{prefix}/*/rlog.zst"), key=_seg_num)
    assert segs, f"no rlogs under {prefix}"

    cpu_samples = {p: [] for p in WATCH_PROCS}   # %-of-one-core between samples
    prev = {}                                    # pid -> (mono_t, cpu_seconds)
    lowconf_secs = 0.0
    lowconf_eps = 0
    depart_events = 0
    aol_log_lines = 0
    last_names, last_t, ep_open = set(), None, False

    for seg in segs:
        for msg in LogReader(seg):
            w = msg.which()
            if w == "procLog":
                t = msg.logMonoTime / 1e9
                for p in msg.procLog.procs:
                    cmd = " ".join(p.cmdline)
                    name = next((k for k, pat in CMDLINE_MATCH.items() if pat in cmd), None)
                    if name is None:
                        continue
                    total = float(p.cpuUser) + float(p.cpuSystem)
                    key = (name, p.pid)
                    if key in prev:
                        t0, c0 = prev[key]
                        if t > t0:
                            cpu_samples[name].append((total - c0) / (t - t0) * 100.0)
                    prev[key] = (t, total)
            elif w == "onroadEventsSP":
                t = msg.logMonoTime / 1e9
                names = {str(e.name) for e in msg.onroadEventsSP.events}
                if AOL_DEPART in names:
                    depart_events += 1
                if AOL_LOWCONF in names:
                    if not ep_open:
                        lowconf_eps += 1
                        ep_open = True
                    if last_t is not None and AOL_LOWCONF in last_names:
                        lowconf_secs += min(t - last_t, 2.0)
                else:
                    ep_open = False
                last_names, last_t = names, t
            elif w == "logMessage":
                if "aol" in msg.logMessage.lower():
                    aol_log_lines += 1

    return {
        "route": os.path.basename(prefix),
        "segs": len(segs),
        "cpu_median": {p: round(_median(v), 2) for p, v in cpu_samples.items()},
        "cpu_p95": {p: (round(sorted(v)[int(len(v) * 0.95)], 2) if v else None)
                    for p, v in cpu_samples.items()},
        "lowconf_episodes": lowconf_eps,
        "lowconf_alert_seconds": round(lowconf_secs, 1),
        "departure_EVENTS(must be 0)": depart_events,
        "aol_logMessage_lines": aol_log_lines,
    }


def main():
    assert len(sys.argv) > 1, __doc__
    fail = False
    for prefix in sys.argv[1:]:
        r = scan_route(prefix.rstrip("/"))
        sd = r["cpu_median"].get("selfdrived", float("nan"))
        cpu_ok = sd == sd and sd <= CPU_BASELINE_MEDIAN + CPU_GATE_DELTA
        dep_ok = r["departure_EVENTS(must be 0)"] == 0
        r["CPU_GATE"] = f"{'PASS' if cpu_ok else 'FAIL'} (selfdrived median {sd} vs baseline {CPU_BASELINE_MEDIAN} +{CPU_GATE_DELTA})"
        r["DEPARTURE_SHADOW_GATE"] = "PASS" if dep_ok else "FAIL (departure surfaced as an event!)"
        fail |= not (cpu_ok and dep_ok)
        print(json.dumps(r, indent=2))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
