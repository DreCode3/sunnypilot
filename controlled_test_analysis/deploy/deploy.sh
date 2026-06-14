#!/usr/bin/env bash
# Deploy the live PI toggle and set a config. Run from REPO ROOT when the device is reachable.
# usage: deploy.sh <home|work|vpn> <weak|golden>     (default config: golden)
set -e
case "$1" in
  home) HOST=192.168.98.237; SOCK=comma-home;;
  work) HOST=10.20.10.66;    SOCK=comma-work;;
  vpn)  HOST=10.10.7.236;    SOCK=comma-vpn;;
  *) echo "usage: deploy.sh <home|work|vpn> <weak|golden>"; exit 1;;
esac
CFG="${2:-golden}"
SSHO="-o ControlMaster=auto -o ControlPath=$HOME/.ssh/sockets/$SOCK -o ControlPersist=600 -o ConnectTimeout=20"
ssh-add --apple-load-keychain >/dev/null 2>&1

# 1) push + run the safe patch (aborts if live strings differ -> no corruption)
cat controlled_test_analysis/deploy/cc_toggle_patch.py | ssh $SSHO comma@$HOST 'cat > /data/cc_toggle_patch.py && python3 /data/cc_toggle_patch.py'

# 2) set the config file + zero the integrator + reboot ONCE to load the new carcontroller code
ssh $SSHO comma@$HOST "echo -n '$CFG' > /data/lc_pi_config && echo -n '0.0' > /data/params/d/LaneBiasIntegral && echo -n '1' > /data/params/d/DoReboot"
echo "Deployed toggle, set config=$CFG, rebooting. Wait ~90 s, then verify:"
echo "  ssh $SSHO comma@$HOST 'grep -c _pi_cfg /data/openpilot/opendbc_repo/opendbc/car/ford/carcontroller.py; echo cfg=\$(cat /data/lc_pi_config)'"

# --- after the FIRST install reboot, switching is live (no reboot), just zero the integrator each pass: ---
#   ssh ... "echo -n weak   > /data/lc_pi_config && echo -n 0.0 > /data/params/d/LaneBiasIntegral"
#   ssh ... "echo -n golden > /data/lc_pi_config && echo -n 0.0 > /data/params/d/LaneBiasIntegral"
