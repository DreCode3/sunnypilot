#!/usr/bin/env python3
"""
Ford Explorer ST — CAN Decoder Analysis v1

Decodes raw CAN frames from rlogs using the Ford DBC file and cross-references
them with openpilot's internal signals (carState, carOutput, carControl).

Analysis sections:
  1. IPMA lateral control commands (LateralMotionControl, 0x3D3)
  2. PSCM/EPAS feedback (EPAS_INFO 0x82, SteeringPinion_Data 0x7E)
  3. Vehicle dynamics sensors (speed, yaw, lateral accel)
  4. CAN message timing and frame delivery rates
  5. Cross-validation: CAN vs openpilot internal signals
  6. PSCM steering status (Lane_Assist_Data3_FD1, 0x3CC)

Requires: cantools (pip install cantools)
"""

import sys
import os
import glob
import struct
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openpilot.tools.lib.logreader import LogReader

try:
    import cantools
    HAS_CANTOOLS = True
except ImportError:
    HAS_CANTOOLS = False
    print("WARNING: cantools not installed. Install with: pip install cantools")
    print("         CAN decoding will be disabled; only openpilot-level signals shown.\n")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DBC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'opendbc_repo', 'opendbc', 'dbc', 'ford_lincoln_base_pt.dbc')

# CAN arbitration IDs (standard 11-bit, decimal)
# Explorer ST is CAN (Q3), uses LateralMotionControl (not LateralMotionControl2)
CAN_ID_LATERAL_MOTION_CTRL  = 979   # 0x3D3 — IPMA -> PSCM, lateral commands
CAN_ID_LATERAL_MOTION_CTRL2 = 982   # 0x3D6 — IPMA -> PSCM, CAN FD variant
CAN_ID_EPAS_INFO            = 130   # 0x82  — PSCM -> bus, EPAS torque/status
CAN_ID_STEERING_PINION      = 126   # 0x7E  — PSCM -> bus, steering angle
CAN_ID_STEERING_PINION_ALT  = 133   # 0x85  — PSCM -> bus, alt steering angle
CAN_ID_YAW_DATA             = 145   # 0x91  — GWM, yaw + roll rate
CAN_ID_BRAKE_SYS_FEATURES   = 1045  # 0x415 — ABS/ESC, vehicle speed
CAN_ID_BRAKE_SN_DATA3       = 119   # 0x77  — ABS/ESC, lat/long accel
CAN_ID_LANE_ASSIST_DATA3    = 972   # 0x3CC — PSCM -> IPMA, lat ctl status

TRACKED_CAN_IDS = {
    CAN_ID_LATERAL_MOTION_CTRL,
    CAN_ID_LATERAL_MOTION_CTRL2,
    CAN_ID_EPAS_INFO,
    CAN_ID_STEERING_PINION,
    CAN_ID_YAW_DATA,
    CAN_ID_BRAKE_SYS_FEATURES,
    CAN_ID_BRAKE_SN_DATA3,
    CAN_ID_LANE_ASSIST_DATA3,
}

# Expected CAN frame rates (Hz)
EXPECTED_RATES = {
    CAN_ID_LATERAL_MOTION_CTRL:  20,
    CAN_ID_LATERAL_MOTION_CTRL2: 20,
    CAN_ID_EPAS_INFO:            100,
    CAN_ID_STEERING_PINION:      100,
    CAN_ID_YAW_DATA:             100,
    CAN_ID_BRAKE_SYS_FEATURES:   100,
    CAN_ID_BRAKE_SN_DATA3:       100,
    CAN_ID_LANE_ASSIST_DATA3:    10,
}

CAN_ID_NAMES = {
    CAN_ID_LATERAL_MOTION_CTRL:  'LateralMotionControl',
    CAN_ID_LATERAL_MOTION_CTRL2: 'LateralMotionControl2',
    CAN_ID_EPAS_INFO:            'EPAS_INFO',
    CAN_ID_STEERING_PINION:      'SteeringPinion_Data',
    CAN_ID_YAW_DATA:             'Yaw_Data_FD1',
    CAN_ID_BRAKE_SYS_FEATURES:   'BrakeSysFeatures',
    CAN_ID_BRAKE_SN_DATA3:       'BrakeSnData_3',
    CAN_ID_LANE_ASSIST_DATA3:    'Lane_Assist_Data3_FD1',
}

# Ford Explorer ST
WHEELBASE = 3.025
STEER_RATIO = 15.0


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_route(route_dir):
    """Load all rlog segments from a route directory."""
    # Format 1: rlog_0.zst, rlog_1.zst, ...
    files = sorted(glob.glob(os.path.join(route_dir, 'rlog_*.zst')),
                   key=lambda f: int(os.path.basename(f).split('_')[1].split('.')[0]))

    # Format 2: per-segment dirs {route}--{n}/rlog.zst
    if not files:
        route_name = os.path.basename(route_dir)
        parent = os.path.dirname(route_dir)
        seg_dirs = sorted(glob.glob(os.path.join(parent, route_name + '--*')),
                          key=lambda d: int(d.rsplit('--', 1)[-1]))
        for sd in seg_dirs:
            rlog = os.path.join(sd, 'rlog.zst')
            qlog = os.path.join(sd, 'qlog.zst')
            if os.path.exists(rlog):
                files.append(rlog)
            elif os.path.exists(qlog):
                files.append(qlog)

    route_name = os.path.basename(route_dir)
    print(f"  Loading {len(files)} segments from {route_name}...")
    all_msgs = []
    for i, f in enumerate(files):
        try:
            lr = LogReader(f)
            all_msgs.extend(list(lr))
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(files)} segments...")
        except Exception as e:
            print(f"    skip seg {i}: {e}")
    print(f"    {len(all_msgs):,} messages loaded")
    return all_msgs


# ─────────────────────────────────────────────────────────────────────────────
# CAN FRAME EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_can_frames(msgs):
    """
    Extract raw CAN frames from rlog messages.

    rlogs contain 'can' (received from bus) and 'sendcan' (sent to bus) events.
    Each event has a list of CanData: {address, busTime, dat, src}.

    For IPMA commands (LateralMotionControl), we look in 'sendcan' (bus 0, sent by openpilot).
    For sensor data (EPAS, SAS, Yaw, Speed), we look in 'can' (bus 0, received from car).
    """
    # {can_id: [(timestamp_sec, raw_bytes), ...]}
    rx_frames = defaultdict(list)  # received from car
    tx_frames = defaultdict(list)  # sent by openpilot

    for msg in msgs:
        t = msg.logMonoTime / 1e9
        w = msg.which()

        if w == 'can':
            for frame in msg.can:
                addr = frame.address
                if addr in TRACKED_CAN_IDS:
                    rx_frames[addr].append((t, bytes(frame.dat), frame.src))

        elif w == 'sendcan':
            for frame in msg.sendcan:
                addr = frame.address
                if addr in TRACKED_CAN_IDS:
                    tx_frames[addr].append((t, bytes(frame.dat), frame.src))

    return rx_frames, tx_frames


def extract_openpilot_signals(msgs):
    """Extract openpilot-level signals for cross-validation."""
    data = defaultdict(list)

    for msg in msgs:
        t = msg.logMonoTime / 1e9
        w = msg.which()

        if w == 'carState':
            cs = msg.carState
            data['cs_t'].append(t)
            data['cs_v_ego'].append(cs.vEgoRaw)
            data['cs_steering_angle'].append(cs.steeringAngleDeg)
            data['cs_steering_torque'].append(cs.steeringTorque)
            data['cs_steering_pressed'].append(cs.steeringPressed)
            data['cs_yaw_rate'].append(cs.yawRate)
            data['cs_steer_fault_temp'].append(cs.steerFaultTemporary)
            data['cs_steer_fault_perm'].append(cs.steerFaultPermanent)

        elif w == 'carControl':
            cc = msg.carControl
            data['cc_t'].append(t)
            data['cc_lat_active'].append(cc.latActive)
            data['cc_curvature'].append(cc.actuators.curvature)

        elif w == 'carOutput':
            co = msg.carOutput
            data['co_t'].append(t)
            data['co_curvature'].append(co.actuatorsOutput.curvature)

    # Convert to numpy
    for k in list(data.keys()):
        data[k] = np.array(data[k])

    return data


# ─────────────────────────────────────────────────────────────────────────────
# CAN DECODING
# ─────────────────────────────────────────────────────────────────────────────

def load_dbc():
    """Load and return the Ford DBC database."""
    if not HAS_CANTOOLS:
        return None
    if not os.path.exists(DBC_PATH):
        print(f"WARNING: DBC file not found at {DBC_PATH}")
        return None
    db = cantools.db.load_file(DBC_PATH, strict=False)
    return db


def decode_frames(db, frames_dict, msg_name, can_id):
    """
    Decode a list of (timestamp, raw_bytes, src) tuples using the DBC.
    Returns: list of (timestamp, {signal_name: value}) dicts.
    """
    if db is None:
        return []

    try:
        msg_def = db.get_message_by_name(msg_name)
    except KeyError:
        try:
            msg_def = db.get_message_by_frame_id(can_id)
        except KeyError:
            print(f"  WARNING: Cannot find message {msg_name} (0x{can_id:03X}) in DBC")
            return []

    decoded = []
    decode_errors = 0
    for t, dat, src in frames_dict.get(can_id, []):
        try:
            signals = msg_def.decode(dat, decode_choices=False)
            decoded.append((t, signals))
        except Exception:
            decode_errors += 1

    if decode_errors > 0:
        total = len(frames_dict.get(can_id, []))
        print(f"  WARNING: {decode_errors}/{total} decode errors for {msg_name}")

    return decoded


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS SECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_timing(frames_dict, label="RX"):
    """Analyze CAN frame timing and delivery rates."""
    print(f"\n{'='*80}")
    print(f"  CAN FRAME TIMING ANALYSIS ({label})")
    print(f"{'='*80}")

    for can_id in sorted(frames_dict.keys()):
        name = CAN_ID_NAMES.get(can_id, f"0x{can_id:03X}")
        frames = frames_dict[can_id]
        if len(frames) < 2:
            print(f"\n  {name} (0x{can_id:03X}): {len(frames)} frames — insufficient data")
            continue

        timestamps = np.array([f[0] for f in frames])
        dt = np.diff(timestamps)
        dt = dt[dt > 0]  # filter zero-time duplicates
        if len(dt) == 0:
            continue

        duration = timestamps[-1] - timestamps[0]
        actual_rate = len(frames) / duration if duration > 0 else 0
        expected = EXPECTED_RATES.get(can_id, 0)

        # Bus source distribution
        srcs = [f[2] for f in frames]
        src_counts = defaultdict(int)
        for s in srcs:
            src_counts[s] += 1

        print(f"\n  {name} (0x{can_id:03X}):")
        print(f"    Total frames:    {len(frames):,}")
        print(f"    Duration:        {duration:.1f}s")
        print(f"    Actual rate:     {actual_rate:.1f} Hz (expected {expected} Hz)")
        print(f"    dt mean:         {np.mean(dt)*1000:.2f} ms")
        print(f"    dt median:       {np.median(dt)*1000:.2f} ms")
        print(f"    dt std:          {np.std(dt)*1000:.2f} ms")
        print(f"    dt min:          {np.min(dt)*1000:.2f} ms")
        print(f"    dt max:          {np.max(dt)*1000:.2f} ms")
        print(f"    Bus sources:     {dict(src_counts)}")

        # Gap analysis: frames with dt > 2x expected period
        if expected > 0:
            gap_threshold = 2.0 / expected
            gaps = dt[dt > gap_threshold]
            if len(gaps) > 0:
                print(f"    Gaps (>{gap_threshold*1000:.0f}ms): {len(gaps)} "
                      f"(max {np.max(gaps)*1000:.1f}ms, mean {np.mean(gaps)*1000:.1f}ms)")
            else:
                print(f"    Gaps (>{gap_threshold*1000:.0f}ms): none")

        # Delivery rate over 1-second windows
        if duration > 2:
            window_start = timestamps[0]
            window_rates = []
            while window_start + 1.0 <= timestamps[-1]:
                count = np.sum((timestamps >= window_start) & (timestamps < window_start + 1.0))
                window_rates.append(count)
                window_start += 1.0
            if window_rates:
                wr = np.array(window_rates)
                print(f"    1s window rate:  min={np.min(wr)}, max={np.max(wr)}, "
                      f"mean={np.mean(wr):.1f}, std={np.std(wr):.1f}")


def analyze_lateral_commands(decoded_frames, op_data):
    """Analyze IPMA lateral control commands and compare with openpilot."""
    print(f"\n{'='*80}")
    print(f"  IPMA LATERAL CONTROL COMMANDS")
    print(f"{'='*80}")

    if not decoded_frames:
        print("  No decoded LateralMotionControl frames available.")
        return

    times = np.array([f[0] for f in decoded_frames])

    # Extract signals — handle both CAN and CAN FD message variants
    sample = decoded_frames[0][1]
    is_canfd = 'LatCtl_D2_Rq' in sample

    if is_canfd:
        mode_key = 'LatCtl_D2_Rq'
    else:
        mode_key = 'LatCtl_D_Rq'

    curvatures = np.array([f[1].get('LatCtlCurv_No_Actl', 0) for f in decoded_frames])
    if 'LatCtlCurv_NoRate_Actl' in sample:
        curv_rates = np.array([f[1].get('LatCtlCurv_NoRate_Actl', 0) for f in decoded_frames])
    else:
        curv_rates = np.array([f[1].get('LatCtlCrv_NoRate2_Actl', 0) for f in decoded_frames])
    path_angles = np.array([f[1].get('LatCtlPath_An_Actl', 0) for f in decoded_frames])
    path_offsets = np.array([f[1].get('LatCtlPathOffst_L_Actl', 0) for f in decoded_frames])
    modes = np.array([f[1].get(mode_key, 0) for f in decoded_frames])
    ramp_types = np.array([f[1].get('LatCtlRampType_D_Rq', 0) for f in decoded_frames])
    precisions = np.array([f[1].get('LatCtlPrecision_D_Rq', 0) for f in decoded_frames])

    # Note: carcontroller.py NEGATES curvature before sending to CAN
    # So CAN curvature = -openpilot_curvature
    # We negate back to match openpilot sign convention for comparison
    can_curv_op_sign = -curvatures

    # Mode distribution
    mode_names = {0: 'None', 1: 'ContinuousPathFollowing', 2: 'InterventionLeft',
                  3: 'InterventionRight'} if not is_canfd else \
                 {0: 'None', 1: 'PathFollowingLimited', 2: 'PathFollowingExtended',
                  3: 'SafeRampOut'}
    print(f"\n  Message variant: {'CAN FD (LateralMotionControl2)' if is_canfd else 'CAN (LateralMotionControl)'}")
    print(f"  Total frames:    {len(decoded_frames):,}")
    print(f"  Duration:        {times[-1] - times[0]:.1f}s")

    print(f"\n  Mode distribution:")
    for mode_val in sorted(np.unique(modes)):
        count = np.sum(modes == mode_val)
        pct = count / len(modes) * 100
        name = mode_names.get(int(mode_val), f'Unknown({int(mode_val)})')
        print(f"    {int(mode_val)} ({name}): {count:,} ({pct:.1f}%)")

    print(f"\n  Ramp type distribution:")
    ramp_names = {0: 'Slow', 1: 'Medium', 2: 'Fast', 3: 'Immediate'}
    for rv in sorted(np.unique(ramp_types)):
        count = np.sum(ramp_types == rv)
        pct = count / len(ramp_types) * 100
        print(f"    {int(rv)} ({ramp_names.get(int(rv), '?')}): {count:,} ({pct:.1f}%)")

    print(f"\n  Precision type distribution:")
    prec_names = {0: 'Comfortable', 1: 'Precise', 2: 'NotUsed', 3: 'NotUsed'}
    for pv in sorted(np.unique(precisions)):
        count = np.sum(precisions == pv)
        pct = count / len(precisions) * 100
        print(f"    {int(pv)} ({prec_names.get(int(pv), '?')}): {count:,} ({pct:.1f}%)")

    # Active frames only (mode != 0)
    active = modes > 0
    n_active = np.sum(active)
    print(f"\n  Active frames:   {n_active:,} ({n_active/len(modes)*100:.1f}%)")

    if n_active > 0:
        ac = curvatures[active]
        print(f"\n  CAN curvature (as-sent, raw DBC values, active only):")
        print(f"    mean:   {np.mean(ac):.6f} 1/m")
        print(f"    std:    {np.std(ac):.6f} 1/m")
        print(f"    min:    {np.min(ac):.6f} 1/m")
        print(f"    max:    {np.max(ac):.6f} 1/m")
        print(f"    |mean|: {np.mean(np.abs(ac)):.6f} 1/m")

        acr = curv_rates[active]
        print(f"\n  CAN curvature rate (active only):")
        print(f"    mean:   {np.mean(acr):.9f} 1/m^2")
        print(f"    std:    {np.std(acr):.9f} 1/m^2")
        print(f"    min:    {np.min(acr):.9f} 1/m^2")
        print(f"    max:    {np.max(acr):.9f} 1/m^2")
        print(f"    nonzero: {np.sum(np.abs(acr) > 1e-9):,} ({np.sum(np.abs(acr) > 1e-9)/len(acr)*100:.1f}%)")

        ap = path_angles[active]
        print(f"\n  CAN path angle (active only):")
        print(f"    mean:   {np.mean(ap):.6f} rad")
        print(f"    std:    {np.std(ap):.6f} rad")
        print(f"    all-zero: {np.all(np.abs(ap) < 1e-6)}")

        ao = path_offsets[active]
        print(f"\n  CAN path offset (active only):")
        print(f"    mean:   {np.mean(ao):.4f} m")
        print(f"    std:    {np.std(ao):.4f} m")
        print(f"    all-zero: {np.all(np.abs(ao) < 1e-4)}")

    # Cross-validate: CAN curvature vs carOutput.actuatorsOutput.curvature
    if len(op_data.get('co_t', [])) > 0 and n_active > 0:
        print(f"\n  --- CAN vs carOutput curvature cross-validation ---")
        # Interpolate carOutput curvature to CAN timestamps (active frames only)
        active_times = times[active]
        active_can_curv = can_curv_op_sign[active]

        co_interp = np.interp(active_times, op_data['co_t'], op_data['co_curvature'])

        diff = active_can_curv - co_interp
        print(f"    Samples compared: {len(diff):,}")
        print(f"    CAN curv (OP sign) mean:    {np.mean(active_can_curv):.6f}")
        print(f"    carOutput curv mean:         {np.mean(co_interp):.6f}")
        print(f"    Difference (CAN - carOutput):")
        print(f"      mean:   {np.mean(diff):.6f}")
        print(f"      std:    {np.std(diff):.6f}")
        print(f"      max|diff|: {np.max(np.abs(diff)):.6f}")
        print(f"      RMSE:   {np.sqrt(np.mean(diff**2)):.6f}")

        # Check for exact match (within DBC quantization: 2E-005 = 0.00002)
        exact = np.sum(np.abs(diff) < 0.000025)
        close = np.sum(np.abs(diff) < 0.0001)
        print(f"      Exact match (<0.000025): {exact:,} ({exact/len(diff)*100:.1f}%)")
        print(f"      Close match (<0.0001):   {close:,} ({close/len(diff)*100:.1f}%)")


def analyze_epas_feedback(decoded_epas, decoded_pinion, op_data):
    """Analyze PSCM/EPAS feedback signals."""
    print(f"\n{'='*80}")
    print(f"  PSCM / EPAS FEEDBACK")
    print(f"{'='*80}")

    if decoded_epas:
        times = np.array([f[0] for f in decoded_epas])
        torques = np.array([f[1].get('SteeringColumnTorque', 0) for f in decoded_epas])
        drv_torques = np.array([f[1].get('DrvSte_Tq_Actl', 0) for f in decoded_epas])
        failures = np.array([f[1].get('EPAS_Failure', 0) for f in decoded_epas])
        module_status = np.array([f[1].get('SteMdule_D_Stat', 0) for f in decoded_epas])
        drv_active = np.array([f[1].get('DrvSteActv_B_Stat', 0) for f in decoded_epas])
        currents = np.array([f[1].get('SteMdule_I_Est', 0) for f in decoded_epas])
        voltages = np.array([f[1].get('SteMdule_U_Meas', 0) for f in decoded_epas])

        print(f"\n  EPAS_INFO (0x{CAN_ID_EPAS_INFO:03X}): {len(decoded_epas):,} frames, "
              f"{times[-1]-times[0]:.1f}s")

        print(f"\n  Steering Column Torque (driver + assist):")
        print(f"    mean:   {np.mean(torques):.4f} Nm")
        print(f"    std:    {np.std(torques):.4f} Nm")
        print(f"    min:    {np.min(torques):.4f} Nm")
        print(f"    max:    {np.max(torques):.4f} Nm")

        print(f"\n  Driver Steering Torque (DrvSte_Tq_Actl):")
        print(f"    mean:   {np.mean(drv_torques):.4f} Nm")
        print(f"    std:    {np.std(drv_torques):.4f} Nm")
        print(f"    min:    {np.min(drv_torques):.4f} Nm")
        print(f"    max:    {np.max(drv_torques):.4f} Nm")

        print(f"\n  Driver Steering Active (DrvSteActv_B_Stat):")
        print(f"    active:   {np.sum(drv_active == 1):,} ({np.mean(drv_active)*100:.1f}%)")
        print(f"    inactive: {np.sum(drv_active == 0):,}")

        print(f"\n  EPAS Module Current (SteMdule_I_Est):")
        print(f"    mean:   {np.mean(currents):.2f} A")
        print(f"    std:    {np.std(currents):.2f} A")
        print(f"    min:    {np.min(currents):.2f} A")
        print(f"    max:    {np.max(currents):.2f} A")

        print(f"\n  EPAS Module Voltage (SteMdule_U_Meas):")
        print(f"    mean:   {np.mean(voltages):.2f} V")
        print(f"    min:    {np.min(voltages):.2f} V")
        print(f"    max:    {np.max(voltages):.2f} V")

        print(f"\n  EPAS_Failure distribution:")
        fail_names = {0: 'NoFailure', 1: 'Temporary', 2: 'Permanent_2', 3: 'Permanent_3'}
        for fv in sorted(np.unique(failures)):
            count = np.sum(failures == fv)
            pct = count / len(failures) * 100
            print(f"    {int(fv)} ({fail_names.get(int(fv), '?')}): {count:,} ({pct:.1f}%)")

        print(f"\n  SteMdule_D_Stat distribution:")
        for sv in sorted(np.unique(module_status)):
            count = np.sum(module_status == sv)
            pct = count / len(module_status) * 100
            print(f"    {int(sv)}: {count:,} ({pct:.1f}%)")

        # Cross-validate: CAN torque vs carState.steeringTorque
        if len(op_data.get('cs_t', [])) > 0:
            print(f"\n  --- CAN torque vs carState.steeringTorque ---")
            cs_torque_interp = np.interp(times, op_data['cs_t'], op_data['cs_steering_torque'])
            diff = torques - cs_torque_interp
            print(f"    Samples:  {len(diff):,}")
            print(f"    RMSE:     {np.sqrt(np.mean(diff**2)):.6f} Nm")
            print(f"    max|diff|: {np.max(np.abs(diff)):.6f} Nm")
    else:
        print("  No EPAS_INFO frames decoded.")

    # Steering Pinion Data
    if decoded_pinion:
        times_p = np.array([f[0] for f in decoded_pinion])
        angles = np.array([f[1].get('StePinComp_An_Est', 0) for f in decoded_pinion])
        angle_raw = np.array([f[1].get('StePinRelInit_An_Sns', 0) for f in decoded_pinion])
        qf = np.array([f[1].get('StePinCompAnEst_D_Qf', 0) for f in decoded_pinion])

        print(f"\n  SteeringPinion_Data (0x{CAN_ID_STEERING_PINION:03X}): {len(decoded_pinion):,} frames, "
              f"{times_p[-1]-times_p[0]:.1f}s")

        print(f"\n  Compensated Steering Angle (StePinComp_An_Est):")
        print(f"    mean:   {np.mean(angles):.2f} deg")
        print(f"    std:    {np.std(angles):.2f} deg")
        print(f"    min:    {np.min(angles):.2f} deg")
        print(f"    max:    {np.max(angles):.2f} deg")

        print(f"\n  Raw Steering Angle (StePinRelInit_An_Sns):")
        print(f"    mean:   {np.mean(angle_raw):.2f} deg")
        print(f"    std:    {np.std(angle_raw):.2f} deg")
        print(f"    min:    {np.min(angle_raw):.2f} deg")
        print(f"    max:    {np.max(angle_raw):.2f} deg")

        print(f"\n  Quality Factor distribution:")
        qf_names = {0: 'NotValid', 1: 'Degraded', 2: 'Temporary', 3: 'FullyValid'}
        for qv in sorted(np.unique(qf)):
            count = np.sum(qf == qv)
            pct = count / len(qf) * 100
            print(f"    {int(qv)} ({qf_names.get(int(qv), '?')}): {count:,} ({pct:.1f}%)")

        # Cross-validate: CAN angle vs carState.steeringAngleDeg
        if len(op_data.get('cs_t', [])) > 0:
            print(f"\n  --- CAN angle vs carState.steeringAngleDeg ---")
            cs_angle_interp = np.interp(times_p, op_data['cs_t'], op_data['cs_steering_angle'])
            diff = angles - cs_angle_interp
            print(f"    Samples:  {len(diff):,}")
            print(f"    RMSE:     {np.sqrt(np.mean(diff**2)):.4f} deg")
            print(f"    max|diff|: {np.max(np.abs(diff)):.4f} deg")
            exact = np.sum(np.abs(diff) < 0.15)  # within one DBC step (0.1 deg)
            print(f"    Match (<0.15 deg): {exact:,} ({exact/len(diff)*100:.1f}%)")
    else:
        print("  No SteeringPinion_Data frames decoded.")


def analyze_vehicle_dynamics(decoded_yaw, decoded_speed, decoded_accel, op_data):
    """Analyze vehicle dynamics sensors from CAN."""
    print(f"\n{'='*80}")
    print(f"  VEHICLE DYNAMICS SENSORS (CAN)")
    print(f"{'='*80}")

    # Yaw Rate
    if decoded_yaw:
        times_y = np.array([f[0] for f in decoded_yaw])
        yaw_rates = np.array([f[1].get('VehYaw_W_Actl', 0) for f in decoded_yaw])
        roll_rates = np.array([f[1].get('VehRol_W_Actl', 0) for f in decoded_yaw])

        print(f"\n  Yaw_Data_FD1 (0x{CAN_ID_YAW_DATA:03X}): {len(decoded_yaw):,} frames")
        print(f"\n  Yaw Rate (VehYaw_W_Actl):")
        print(f"    mean:   {np.mean(yaw_rates):.6f} rad/s")
        print(f"    std:    {np.std(yaw_rates):.6f} rad/s")
        print(f"    min:    {np.min(yaw_rates):.6f} rad/s")
        print(f"    max:    {np.max(yaw_rates):.6f} rad/s")

        print(f"\n  Roll Rate (VehRol_W_Actl):")
        print(f"    mean:   {np.mean(roll_rates):.6f} rad/s")
        print(f"    std:    {np.std(roll_rates):.6f} rad/s")
        print(f"    min:    {np.min(roll_rates):.6f} rad/s")
        print(f"    max:    {np.max(roll_rates):.6f} rad/s")

        # Cross-validate yaw rate
        if len(op_data.get('cs_t', [])) > 0:
            print(f"\n  --- CAN yaw rate vs carState.yawRate ---")
            cs_yaw_interp = np.interp(times_y, op_data['cs_t'], op_data['cs_yaw_rate'])
            diff = yaw_rates - cs_yaw_interp
            print(f"    RMSE:     {np.sqrt(np.mean(diff**2)):.6f} rad/s")
            print(f"    max|diff|: {np.max(np.abs(diff)):.6f} rad/s")
    else:
        print("  No Yaw_Data_FD1 frames decoded.")

    # Vehicle Speed
    if decoded_speed:
        times_s = np.array([f[0] for f in decoded_speed])
        speeds_kph = np.array([f[1].get('Veh_V_ActlBrk', 0) for f in decoded_speed])
        speeds_ms = speeds_kph / 3.6

        print(f"\n  BrakeSysFeatures (0x{CAN_ID_BRAKE_SYS_FEATURES:03X}): {len(decoded_speed):,} frames")
        print(f"\n  Vehicle Speed (Veh_V_ActlBrk):")
        print(f"    mean:   {np.mean(speeds_kph):.1f} kph ({np.mean(speeds_ms):.1f} m/s)")
        print(f"    max:    {np.max(speeds_kph):.1f} kph ({np.max(speeds_ms):.1f} m/s)")

        # Cross-validate speed
        if len(op_data.get('cs_t', [])) > 0:
            print(f"\n  --- CAN speed vs carState.vEgoRaw ---")
            cs_speed_interp = np.interp(times_s, op_data['cs_t'], op_data['cs_v_ego'])
            diff_ms = speeds_ms - cs_speed_interp
            print(f"    RMSE:     {np.sqrt(np.mean(diff_ms**2)):.4f} m/s")
            print(f"    max|diff|: {np.max(np.abs(diff_ms)):.4f} m/s")
    else:
        print("  No BrakeSysFeatures frames decoded.")

    # Lateral Acceleration
    if decoded_accel:
        times_a = np.array([f[0] for f in decoded_accel])
        lat_accels = np.array([f[1].get('VehLatComp_A_Actl', 0) for f in decoded_accel])
        long_accels = np.array([f[1].get('VehLongComp_A_Actl', 0) for f in decoded_accel])

        # Filter out "NoDataExists" sentinel (raw 1022 -> 17.87 m/s^2 ~= 0.035*1022 -17.9)
        # The sentinel decoded value is 0.035*1022 - 17.9 = 35.77 - 17.9 = 17.87
        valid_lat = np.abs(lat_accels) < 15.0  # filter unreasonable values

        print(f"\n  BrakeSnData_3 (0x{CAN_ID_BRAKE_SN_DATA3:03X}): {len(decoded_accel):,} frames")
        print(f"    Valid lat accel frames: {np.sum(valid_lat):,} / {len(lat_accels):,}")

        if np.sum(valid_lat) > 0:
            vla = lat_accels[valid_lat]
            print(f"\n  Lateral Acceleration (VehLatComp_A_Actl):")
            print(f"    mean:   {np.mean(vla):.4f} m/s^2")
            print(f"    std:    {np.std(vla):.4f} m/s^2")
            print(f"    min:    {np.min(vla):.4f} m/s^2")
            print(f"    max:    {np.max(vla):.4f} m/s^2")
            print(f"    |mean|: {np.mean(np.abs(vla)):.4f} m/s^2")

            # Compare with computed lateral accel = yaw_rate * v_ego
            if decoded_yaw and decoded_speed and len(op_data.get('cs_t', [])) > 0:
                print(f"\n  --- Car lat accel sensor vs computed (yaw_rate * v_ego) ---")
                valid_times = times_a[valid_lat]
                cs_yaw_at_accel = np.interp(valid_times, op_data['cs_t'], op_data['cs_yaw_rate'])
                cs_speed_at_accel = np.interp(valid_times, op_data['cs_t'], op_data['cs_v_ego'])
                computed_lat_accel = cs_yaw_at_accel * cs_speed_at_accel
                diff_la = vla - computed_lat_accel
                print(f"    Sensor mean:   {np.mean(vla):.4f} m/s^2")
                print(f"    Computed mean:  {np.mean(computed_lat_accel):.4f} m/s^2")
                print(f"    Difference (sensor - computed):")
                print(f"      mean:   {np.mean(diff_la):.4f} m/s^2")
                print(f"      std:    {np.std(diff_la):.4f} m/s^2")
                print(f"      RMSE:   {np.sqrt(np.mean(diff_la**2)):.4f} m/s^2")
                print(f"    Note: difference includes road bank/superelevation effect")

            # Longitudinal acceleration
            valid_long = np.abs(long_accels) < 15.0
            if np.sum(valid_long) > 0:
                vlo = long_accels[valid_long]
                print(f"\n  Longitudinal Acceleration (VehLongComp_A_Actl):")
                print(f"    mean:   {np.mean(vlo):.4f} m/s^2")
                print(f"    std:    {np.std(vlo):.4f} m/s^2")
                print(f"    min:    {np.min(vlo):.4f} m/s^2")
                print(f"    max:    {np.max(vlo):.4f} m/s^2")
    else:
        print("  No BrakeSnData_3 frames decoded.")


def analyze_pscm_status(decoded_status):
    """Analyze PSCM lateral control status feedback."""
    print(f"\n{'='*80}")
    print(f"  PSCM LATERAL CONTROL STATUS")
    print(f"{'='*80}")

    if not decoded_status:
        print("  No Lane_Assist_Data3_FD1 frames decoded.")
        return

    times = np.array([f[0] for f in decoded_status])
    ste_stat = np.array([f[1].get('LatCtlSte_D_Stat', 0) for f in decoded_status])
    lim_stat = np.array([f[1].get('LatCtlLim_D_Stat', 0) for f in decoded_status])
    cap_stat = np.array([f[1].get('LatCtlCpblty_D_Stat', 0) for f in decoded_status])
    hands_off = np.array([f[1].get('LaHandsOff_B_Actl', 0) for f in decoded_status])
    la_deny = np.array([f[1].get('LaActDeny_B_Actl', 0) for f in decoded_status])

    print(f"\n  Lane_Assist_Data3_FD1 (0x{CAN_ID_LANE_ASSIST_DATA3:03X}): {len(decoded_status):,} frames, "
          f"{times[-1]-times[0]:.1f}s")

    ste_names = {0: 'Unavailable', 1: 'Available', 2: 'ContLatControlInProgress',
                 3: 'RampOut', 4: 'Denied'}
    print(f"\n  LatCtlSte_D_Stat (steering control status):")
    for sv in sorted(np.unique(ste_stat)):
        count = np.sum(ste_stat == sv)
        pct = count / len(ste_stat) * 100
        print(f"    {int(sv)} ({ste_names.get(int(sv), '?')}): {count:,} ({pct:.1f}%)")

    lim_names = {0: 'LimitNotReached', 1: 'LimitClose', 2: 'LimitReached',
                 3: 'LimitWithDriverActive'}
    print(f"\n  LatCtlLim_D_Stat (limit status):")
    for lv in sorted(np.unique(lim_stat)):
        count = np.sum(lim_stat == lv)
        pct = count / len(lim_stat) * 100
        print(f"    {int(lv)} ({lim_names.get(int(lv), '?')}): {count:,} ({pct:.1f}%)")

    print(f"\n  LatCtlCpblty_D_Stat (capability):")
    for cv in sorted(np.unique(cap_stat)):
        count = np.sum(cap_stat == cv)
        pct = count / len(cap_stat) * 100
        print(f"    {int(cv)}: {count:,} ({pct:.1f}%)")

    print(f"\n  LaHandsOff_B_Actl: active={np.sum(hands_off==1):,} "
          f"({np.mean(hands_off)*100:.1f}%)")
    print(f"  LaActDeny_B_Actl:  denied={np.sum(la_deny==1):,} "
          f"({np.mean(la_deny)*100:.1f}%)")

    # Transition analysis
    if len(ste_stat) > 1:
        transitions = []
        for i in range(1, len(ste_stat)):
            if ste_stat[i] != ste_stat[i-1]:
                transitions.append((times[i], int(ste_stat[i-1]), int(ste_stat[i])))
        print(f"\n  Status transitions: {len(transitions)}")
        if len(transitions) <= 30:
            for t_trans, from_s, to_s in transitions:
                print(f"    t={t_trans:.2f}s: {ste_names.get(from_s, '?')} -> {ste_names.get(to_s, '?')}")
        else:
            # Show first 10 and last 10
            print(f"    (showing first 10 and last 10)")
            for t_trans, from_s, to_s in transitions[:10]:
                print(f"    t={t_trans:.2f}s: {ste_names.get(from_s, '?')} -> {ste_names.get(to_s, '?')}")
            print(f"    ...")
            for t_trans, from_s, to_s in transitions[-10:]:
                print(f"    t={t_trans:.2f}s: {ste_names.get(from_s, '?')} -> {ste_names.get(to_s, '?')}")


def analyze_delivery_rate_20hz(tx_frames, rx_frames, op_data):
    """
    Deep analysis: are we actually hitting 20Hz for lateral control?
    Look at sendcan (TX) frames for LateralMotionControl.
    """
    print(f"\n{'='*80}")
    print(f"  20Hz LATERAL CONTROL DELIVERY ANALYSIS")
    print(f"{'='*80}")

    # Check both CAN and CAN FD variants
    lat_id = None
    lat_name = None
    for cid, cname in [(CAN_ID_LATERAL_MOTION_CTRL, 'LateralMotionControl'),
                       (CAN_ID_LATERAL_MOTION_CTRL2, 'LateralMotionControl2')]:
        if cid in tx_frames and len(tx_frames[cid]) > 10:
            lat_id = cid
            lat_name = cname
            break

    if lat_id is None:
        # Maybe they show up in RX (on bus 2 / camera bus coming back)
        for cid, cname in [(CAN_ID_LATERAL_MOTION_CTRL, 'LateralMotionControl'),
                           (CAN_ID_LATERAL_MOTION_CTRL2, 'LateralMotionControl2')]:
            if cid in rx_frames and len(rx_frames[cid]) > 10:
                lat_id = cid
                lat_name = cname
                break

    if lat_id is None:
        print("  No LateralMotionControl frames found in TX or RX.")
        return

    frames = tx_frames.get(lat_id, []) or rx_frames.get(lat_id, [])
    timestamps = np.array([f[0] for f in frames])

    if len(timestamps) < 2:
        print("  Insufficient frames for analysis.")
        return

    duration = timestamps[-1] - timestamps[0]
    overall_rate = len(timestamps) / duration if duration > 0 else 0

    print(f"\n  Source: {'sendcan (TX)' if lat_id in tx_frames else 'can (RX)'}")
    print(f"  Message: {lat_name} (0x{lat_id:03X})")
    print(f"  Total frames: {len(timestamps):,}")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Overall rate: {overall_rate:.2f} Hz")

    # Per-second rate histogram
    dt = np.diff(timestamps)
    dt = dt[dt > 0]

    print(f"\n  Inter-frame interval statistics:")
    print(f"    Target: 50.0 ms (20 Hz)")
    print(f"    Mean:   {np.mean(dt)*1000:.2f} ms ({1/np.mean(dt):.1f} Hz)")
    print(f"    Median: {np.median(dt)*1000:.2f} ms")
    print(f"    Std:    {np.std(dt)*1000:.2f} ms")
    print(f"    Min:    {np.min(dt)*1000:.2f} ms")
    print(f"    Max:    {np.max(dt)*1000:.2f} ms")

    # Histogram of intervals
    bins = [0, 20, 40, 45, 50, 55, 60, 80, 100, 150, 200, 500, 1000, 5000]
    hist, _ = np.histogram(dt * 1000, bins=bins)
    print(f"\n  Interval distribution (ms):")
    for i in range(len(bins)-1):
        if hist[i] > 0:
            print(f"    {bins[i]:5.0f}-{bins[i+1]:5.0f} ms: {hist[i]:6,} ({hist[i]/len(dt)*100:5.1f}%)")

    # Dropped frames: intervals > 75ms suggest a missed 50ms cycle
    dropped = np.sum(dt > 0.075)
    print(f"\n  Dropped cycles (dt > 75ms): {dropped:,} ({dropped/len(dt)*100:.2f}%)")
    if dropped > 0:
        drop_times = timestamps[1:][dt > 0.075]
        drop_dts = dt[dt > 0.075]
        print(f"    Worst gaps:")
        worst_idx = np.argsort(drop_dts)[-min(10, len(drop_dts)):][::-1]
        for idx in worst_idx:
            print(f"      t={drop_times[idx]:.2f}s, gap={drop_dts[idx]*1000:.1f}ms "
                  f"(~{drop_dts[idx]/0.05:.0f} cycles)")

    # Per-second rate over time
    if duration > 5:
        window_start = timestamps[0]
        sec_rates = []
        sec_times = []
        while window_start + 1.0 <= timestamps[-1]:
            count = np.sum((timestamps >= window_start) & (timestamps < window_start + 1.0))
            sec_rates.append(count)
            sec_times.append(window_start - timestamps[0])
            window_start += 1.0

        sr = np.array(sec_rates)
        print(f"\n  Per-second delivery rate:")
        print(f"    Mean:   {np.mean(sr):.1f} Hz")
        print(f"    Min:    {np.min(sr)} Hz (at t={sec_times[np.argmin(sr)]:.0f}s)")
        print(f"    Max:    {np.max(sr)} Hz")
        print(f"    Std:    {np.std(sr):.2f}")
        below_18 = np.sum(sr < 18)
        below_15 = np.sum(sr < 15)
        print(f"    Seconds below 18Hz: {below_18} ({below_18/len(sr)*100:.1f}%)")
        print(f"    Seconds below 15Hz: {below_15} ({below_15/len(sr)*100:.1f}%)")


def summary_report(rx_frames, tx_frames, op_data):
    """Print a compact summary of key findings."""
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")

    # Duration
    all_times = []
    for frames in list(rx_frames.values()) + list(tx_frames.values()):
        if frames:
            all_times.extend([f[0] for f in frames])
    if all_times:
        total_dur = max(all_times) - min(all_times)
        print(f"\n  Total CAN duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")

    # Frame counts
    print(f"\n  Frame counts:")
    print(f"  {'Message':<30} {'RX':>8} {'TX':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for cid in sorted(set(list(rx_frames.keys()) + list(tx_frames.keys()))):
        name = CAN_ID_NAMES.get(cid, f'0x{cid:03X}')
        rx_n = len(rx_frames.get(cid, []))
        tx_n = len(tx_frames.get(cid, []))
        print(f"  {name:<30} {rx_n:>8,} {tx_n:>8,}")

    # OP signal counts
    if op_data:
        print(f"\n  Openpilot signal counts:")
        print(f"    carState:   {len(op_data.get('cs_t', [])):,}")
        print(f"    carControl: {len(op_data.get('cc_t', [])):,}")
        print(f"    carOutput:  {len(op_data.get('co_t', [])):,}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        route_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'route_3d')
    else:
        route_dir = sys.argv[1]

    if not os.path.isdir(route_dir):
        print(f"ERROR: route directory not found: {route_dir}")
        sys.exit(1)

    print(f"Ford Explorer ST — CAN Decoder Analysis v1")
    print(f"Route: {route_dir}")
    print()

    # Load DBC
    db = load_dbc()
    if db:
        print(f"  DBC loaded: {len(db.messages)} messages")
    else:
        print(f"  DBC not loaded — CAN decoding disabled")

    # Load rlogs
    print()
    msgs = load_route(route_dir)
    if not msgs:
        print("ERROR: no messages loaded")
        sys.exit(1)

    # Extract CAN frames
    print("\n  Extracting CAN frames...")
    rx_frames, tx_frames = extract_can_frames(msgs)
    print(f"  RX CAN IDs found: {sorted(rx_frames.keys())}")
    print(f"  TX CAN IDs found: {sorted(tx_frames.keys())}")
    for cid in sorted(set(list(rx_frames.keys()) + list(tx_frames.keys()))):
        name = CAN_ID_NAMES.get(cid, f'0x{cid:03X}')
        print(f"    {name}: RX={len(rx_frames.get(cid, []))}, TX={len(tx_frames.get(cid, []))}")

    # Extract openpilot signals
    print("\n  Extracting openpilot signals...")
    op_data = extract_openpilot_signals(msgs)
    print(f"  carState: {len(op_data.get('cs_t', []))} samples")
    print(f"  carControl: {len(op_data.get('cc_t', []))} samples")
    print(f"  carOutput: {len(op_data.get('co_t', []))} samples")

    # Free memory from raw msgs
    del msgs

    if not db:
        print("\nSkipping CAN decoding (cantools not available).")
        print("Install cantools: pip install cantools")
        sys.exit(0)

    # Decode CAN frames
    print("\n  Decoding CAN frames...")

    # IPMA lateral commands — check TX first (sendcan), then RX
    lat_frames_tx = decode_frames(db, tx_frames, 'LateralMotionControl', CAN_ID_LATERAL_MOTION_CTRL)
    lat_frames_tx2 = decode_frames(db, tx_frames, 'LateralMotionControl2', CAN_ID_LATERAL_MOTION_CTRL2)
    lat_frames_rx = decode_frames(db, rx_frames, 'LateralMotionControl', CAN_ID_LATERAL_MOTION_CTRL)
    lat_frames_rx2 = decode_frames(db, rx_frames, 'LateralMotionControl2', CAN_ID_LATERAL_MOTION_CTRL2)

    # Use TX if available, fall back to RX
    lat_decoded = lat_frames_tx or lat_frames_tx2 or lat_frames_rx or lat_frames_rx2
    lat_source = "TX" if (lat_frames_tx or lat_frames_tx2) else "RX"
    print(f"  LateralMotionControl: {len(lat_decoded)} frames decoded ({lat_source})")

    # EPAS feedback (always RX from car)
    epas_decoded = decode_frames(db, rx_frames, 'EPAS_INFO', CAN_ID_EPAS_INFO)
    print(f"  EPAS_INFO: {len(epas_decoded)} frames decoded")

    # Steering angle (always RX)
    pinion_decoded = decode_frames(db, rx_frames, 'SteeringPinion_Data', CAN_ID_STEERING_PINION)
    print(f"  SteeringPinion_Data: {len(pinion_decoded)} frames decoded")

    # Yaw rate (always RX)
    yaw_decoded = decode_frames(db, rx_frames, 'Yaw_Data_FD1', CAN_ID_YAW_DATA)
    print(f"  Yaw_Data_FD1: {len(yaw_decoded)} frames decoded")

    # Vehicle speed (always RX)
    speed_decoded = decode_frames(db, rx_frames, 'BrakeSysFeatures', CAN_ID_BRAKE_SYS_FEATURES)
    print(f"  BrakeSysFeatures: {len(speed_decoded)} frames decoded")

    # Lateral/longitudinal accel (always RX)
    accel_decoded = decode_frames(db, rx_frames, 'BrakeSnData_3', CAN_ID_BRAKE_SN_DATA3)
    print(f"  BrakeSnData_3: {len(accel_decoded)} frames decoded")

    # PSCM steering status (always RX)
    status_decoded = decode_frames(db, rx_frames, 'Lane_Assist_Data3_FD1', CAN_ID_LANE_ASSIST_DATA3)
    print(f"  Lane_Assist_Data3_FD1: {len(status_decoded)} frames decoded")

    # ─── Run analyses ───

    # 1. CAN timing
    analyze_timing(rx_frames, "RX (from car)")
    analyze_timing(tx_frames, "TX (to car)")

    # 2. IPMA lateral commands
    analyze_lateral_commands(lat_decoded, op_data)

    # 3. EPAS feedback
    analyze_epas_feedback(epas_decoded, pinion_decoded, op_data)

    # 4. Vehicle dynamics
    analyze_vehicle_dynamics(yaw_decoded, speed_decoded, accel_decoded, op_data)

    # 5. PSCM status
    analyze_pscm_status(status_decoded)

    # 6. 20Hz delivery analysis
    analyze_delivery_rate_20hz(tx_frames, rx_frames, op_data)

    # 7. Summary
    summary_report(rx_frames, tx_frames, op_data)

    print(f"\n{'='*80}")
    print(f"  Analysis complete.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
