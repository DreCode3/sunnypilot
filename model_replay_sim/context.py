"""Per-route replay context: calibration, camera offset, lateral delay, device/sensor,
traffic convention, and the active model bundle — everything needed to feed a model
faithfully during a same-scene replay.

FIX 2 (load-bearing): each bundle's model-side ``LAT_SMOOTH_SECONDS`` is sourced from
THAT bundle's pinned commit (``git show <full_sha>:selfdrive/modeld/modeld.py``), NOT a
hardcoded literal. The prior design hardcoded OPM7=0.1, but its pinned commit has 0.0.

Lateral-delay semantics mirror production
(``sunnypilot/livedelay/helpers.py`` + ``sunnypilot/modeld_v2/modeld.py``):
the delay base = ``LagdValueCache`` when ``LagdToggle`` is true, else
``liveDelay.lateralDelay``; the model lateral-delay input = ``base + LAT_SMOOTH_SECONDS``.

Analysis-only. Reads boot rlog (params + first calib/delay/device/camera) via the
openpilot LogReader; never re-runs a model.
"""

from __future__ import annotations

import glob
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from model_replay_sim import config as C

# Reuse the proven bundle-param parser (initData.params.entries pattern).
sys.path.insert(0, str(C.REPO_ROOT))
from retrospective_lateral.code.model_labels import parse_active_bundle

# index 0 = left-hand-drive (LHD), index 1 = right-hand-drive (RHD); see
# sunnypilot/modeld_v2/modeld.py: traffic_convention[int(is_rhd)] = 1
TRAFFIC_CONVENTION_LEN = 2

_LAT_SMOOTH_RE = re.compile(r"^LAT_SMOOTH_SECONDS\s*=\s*([0-9.]+)", re.MULTILINE)


@lru_cache(maxsize=None)
def bundle_lat_smooth_seconds(bundle: str) -> float:
    """Model-side LAT_SMOOTH_SECONDS for ``bundle``, parsed from its pinned commit's
    ``selfdrive/modeld/modeld.py`` (NOT a literal). Raises on unknown bundle or if the
    constant is not found in that commit."""
    if bundle not in C.BUNDLES:
        raise KeyError(f"unknown bundle: {bundle!r} (known: {sorted(C.BUNDLES)})")
    sha = C.BUNDLES[bundle]["full_sha"]
    out = subprocess.run(
        ["git", "show", f"{sha}:selfdrive/modeld/modeld.py"],
        cwd=str(C.REPO_ROOT), capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git show failed for {bundle} ({sha}): {out.stderr.strip()}")
    m = _LAT_SMOOTH_RE.search(out.stdout)
    if m is None:
        raise RuntimeError(f"LAT_SMOOTH_SECONDS not found in {bundle} ({sha}):selfdrive/modeld/modeld.py")
    return float(m.group(1))


@dataclass
class RouteContext:
    route_id: str
    rpy_calib: list[float]                 # liveCalibration.rpyCalib (3 floats)
    height: float                          # liveCalibration.height[0] (m)
    device_type: str                       # deviceState.deviceType
    road_sensor: str                       # roadCameraState.sensor
    live_lateral_delay: float              # liveDelay.lateralDelay (s)
    camera_offset: float = 0.0             # param CameraOffset (m)
    planplus_control: float = 1.0          # param PlanplusControl
    lagd_toggle: bool = False              # param LagdToggle
    lagd_value_cache: float = 0.2          # param LagdValueCache (s)
    active_bundle: dict | None = None      # parsed ModelManager_ActiveBundle
    active_bundle_internal_name: str | None = None
    is_rhd: bool = False                   # traffic convention (default LHD)


def _decode_float(raw: bytes | None, default: float) -> float:
    if raw is None:
        return default
    try:
        return float(raw.decode().strip())
    except (ValueError, AttributeError):
        return default


def _decode_bool(raw: bytes | None, default: bool) -> bool:
    if raw is None:
        return default
    txt = raw.decode().strip().lower()
    if txt in ("1", "true"):
        return True
    if txt in ("0", "false", ""):
        return False
    return default


def _boot_rlog(route_id: str) -> str:
    """Lowest-segment-number rlog for the route (the boot segment carries initData).

    Glob is generic over segment-dir names (same predicate as alignment._seg_dirs) —
    the old f"000000{route_id[len('route_'):]}" form assumed the fork-era route_<2-hex>
    dir convention and broke free-form route dirs (e.g. route_stock05)."""
    matches = glob.glob(str(C.LOG_ROOT / route_id / "000000*--*--*" / "rlog.zst"))
    if not matches:
        raise FileNotFoundError(f"no rlog found for {route_id}")

    def _segnum(p: str) -> int:
        try:
            return int(Path(p).parent.name.split("--")[-1])
        except ValueError:
            return 1 << 30

    return min(set(matches), key=_segnum)


def route_context(route_id: str) -> RouteContext:
    """Build the per-route replay context from the boot rlog: decode the param entries
    (CameraOffset/PlanplusControl/LagdValueCache/LagdToggle/ModelManager_ActiveBundle)
    and the first liveCalibration/liveDelay/deviceState/roadCameraState."""
    sys.path.insert(0, str(C.REPO_ROOT))
    from openpilot.tools.lib.logreader import LogReader

    boot = _boot_rlog(route_id)

    params: dict[str, bytes] = {}
    rpy_calib: list[float] | None = None
    height: float | None = None
    device_type: str | None = None
    road_sensor: str | None = None
    live_lateral_delay: float | None = None

    want = {"CameraOffset", "PlanplusControl", "LagdValueCache", "LagdToggle", "ModelManager_ActiveBundle"}
    seen_initdata = False

    def _done() -> bool:
        return (seen_initdata and rpy_calib is not None and height is not None
                and device_type is not None and road_sensor is not None
                and live_lateral_delay is not None)

    for msg in LogReader(boot):
        try:                       # new-format rlogs contain events .which() throws on
            w = msg.which()        # (KjException "non-union type") — skip the message,
        except Exception:          # same guard as alignment.build_frame_timeline
            continue
        if w == "initData" and not seen_initdata:
            seen_initdata = True
            for e in msg.initData.params.entries:
                if e.key in want:
                    params[e.key] = bytes(e.value)
        elif w == "liveCalibration" and rpy_calib is None:
            rpy_calib = [float(x) for x in msg.liveCalibration.rpyCalib]
            h = list(msg.liveCalibration.height)
            height = float(h[0]) if h else 1.22
        elif w == "liveDelay" and live_lateral_delay is None:
            live_lateral_delay = float(msg.liveDelay.lateralDelay)
        elif w == "deviceState" and device_type is None:
            device_type = str(msg.deviceState.deviceType)
        elif w == "roadCameraState" and road_sensor is None:
            road_sensor = str(msg.roadCameraState.sensor)
        if _done():
            break

    if rpy_calib is None:
        raise RuntimeError(f"{route_id}: no liveCalibration in boot rlog")
    if live_lateral_delay is None:
        raise RuntimeError(f"{route_id}: no liveDelay in boot rlog")

    active = parse_active_bundle(params["ModelManager_ActiveBundle"]) if "ModelManager_ActiveBundle" in params else None

    return RouteContext(
        route_id=route_id,
        rpy_calib=rpy_calib,
        height=height if height is not None else 1.22,
        device_type=device_type if device_type is not None else "",
        road_sensor=road_sensor if road_sensor is not None else "",
        live_lateral_delay=live_lateral_delay,
        camera_offset=_decode_float(params.get("CameraOffset"), 0.0),
        planplus_control=_decode_float(params.get("PlanplusControl"), 1.0),
        lagd_toggle=_decode_bool(params.get("LagdToggle"), False),
        lagd_value_cache=_decode_float(params.get("LagdValueCache"), 0.2),
        active_bundle=active,
        active_bundle_internal_name=(active["internal_name"] if active else None),
    )


def lateral_delay_input(ctx, bundle: str) -> float:
    """Model lateral-delay input = base + bundle LAT_SMOOTH_SECONDS, where
    base = LagdValueCache if LagdToggle else liveDelay.lateralDelay.
    The smooth term is ALWAYS sourced from the bundle's pinned commit (never a literal)."""
    base = ctx.lagd_value_cache if ctx.lagd_toggle else ctx.live_lateral_delay
    return float(base) + bundle_lat_smooth_seconds(bundle)


def traffic_convention_input(ctx) -> list[float]:
    """RHD/LHD one-hot the model expects: a length-2 vector with a 1 at index int(is_rhd)
    (index 0 = LHD, index 1 = RHD). Mirrors sunnypilot/modeld_v2/modeld.py."""
    vec = [0.0] * TRAFFIC_CONVENTION_LEN
    vec[int(getattr(ctx, "is_rhd", False))] = 1.0
    return vec
