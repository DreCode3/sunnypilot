#!/usr/bin/env python3
"""Install a LIVE PI-config toggle into the Ford carcontroller (run ON the device).
Switches the lane-centering PI between WEAK (lc_kp 0.0001, int_cap fixed 0.30, off-gate decay 0.98) and
GOLDEN (lc_kp 0.0005, int_cap interp 0.30->1.00, decay 0.995) based on the plain file /data/lc_pi_config
(contents "weak" or "golden"), re-read ~every 50 PI frames. Plain file => no Params-key registration (no
UnknownKeyName crash). Defaults to GOLDEN on any error. Idempotent, backs up, syntax-checks; ABORTS (no
write) if any target string isn't found exactly once, so it can never corrupt the live controller.

Deploy:  scp this file to the device, then:  python3 cc_toggle_patch.py
Switch:  echo -n weak  > /data/lc_pi_config     (takes effect within ~2-3 s, no reboot, after first install reboot)
         echo -n golden > /data/lc_pi_config
"""
import ast, os, shutil, sys

F = '/data/openpilot/opendbc_repo/opendbc/car/ford/carcontroller.py'
s = open(F).read()
if '_pi_cfg' in s:
    print('TOGGLE ALREADY INSTALLED — nothing to do.'); sys.exit(0)

# Each (old -> new). First line of each `new` inherits the file's existing leading indent at the match site;
# subsequent lines carry explicit indentation. Targets are the live GOLDEN config sites.
REPS = [
    # (1) __init__: add toggle state right after the lc_kp default
    ('self.lc_kp = 0.0005  # GOLDEN restore (Apr-8) - CD210 weave test',
     'self.lc_kp = 0.0005  # default golden; live-toggled via /data/lc_pi_config\n'
     '    self._pi_cfg = "golden"\n'
     '    self._pi_cfg_ctr = 0'),
    # (2) refresh config + set lc_kp immediately before the P-term
    ('pi_p = self.lc_kp * lane_offset',
     'self._pi_cfg_ctr += 1\n'
     '            if self._pi_cfg_ctr % 50 == 0:\n'
     '              try:\n'
     '                _c = open("/data/lc_pi_config").read().strip().lower()\n'
     '                self._pi_cfg = _c if _c in ("weak", "golden") else "golden"\n'
     '              except Exception:\n'
     '                self._pi_cfg = "golden"\n'
     '            self.lc_kp = 0.0001 if self._pi_cfg == "weak" else 0.0005\n'
     '            pi_p = self.lc_kp * lane_offset'),
    # (3) int_cap conditional
    ('int_cap = float(np.interp(CS.out.vEgoRaw, [20., 30.], [0.3, 1.0]))',
     'int_cap = 0.30 if self._pi_cfg == "weak" else float(np.interp(CS.out.vEgoRaw, [20., 30.], [0.3, 1.0]))'),
    # (4) off-gate decay conditional
    ('self.lane_centering_integral *= 0.995  # GOLDEN restore',
     'self.lane_centering_integral *= (0.98 if self._pi_cfg == "weak" else 0.995)  # live-toggled'),
]

for old, new in REPS:
    n = s.count(old)
    if n != 1:
        print(f'ABORT (no write): target {"not found" if n == 0 else f"found {n}x"}: {old[:70]!r}')
        print('  -> the live file differs from the assumed GOLDEN sites; pull it and adjust REPS.')
        sys.exit(1)

bak = F + '.bak_pre_toggle'
if not os.path.exists(bak):
    shutil.copy(F, bak)
for old, new in REPS:
    s = s.replace(old, new)
try:
    ast.parse(s)
except SyntaxError as e:
    print('ABORT (no write): patched file fails syntax check:', e); sys.exit(2)
open(F, 'w').write(s)
# default the toggle file to golden so behavior is unchanged until you switch it
if not os.path.exists('/data/lc_pi_config'):
    open('/data/lc_pi_config', 'w').write('golden')
print('TOGGLE INSTALLED OK. Backup:', bak)
print('  switch with:  echo -n weak > /data/lc_pi_config   (or golden)')
print('  reboot ONCE now to load the new carcontroller, then switching is live (no reboot).')
