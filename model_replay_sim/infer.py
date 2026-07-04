"""Task 8: faithful vision->policy RECURRENT inference loop.

Turns a window of aligned camera frames (Task 3) into a ``desiredCurvature`` time series
by running the SAME math the device runs in ``sunnypilot/modeld_v2/modeld.py``:

  desire roll (no-op on a passive straight replay)
    -> vision model (img, big_img -> hidden_state features)
    -> features_buffer roll + 4x subsample (the 25-deep temporal feature window)
    -> policy model (desire_pulse, traffic_convention, features_buffer -> plan)
    -> curvature post-step (plan -> get_curvature_from_plan -> smooth_value)

All temporal-buffer construction and the per-frame update are COPIED from
``sunnypilot/modeld_v2/modeld.py`` (the ModelState class), parametrized by each bundle's
``input_shapes`` so any bundle's shapes work. Line citations below refer to that file.

pkl execution mirrors ``tinygrad_repo/examples/openpilot/compile3.py`` (the same callable a
compiled openpilot model is) and ``sunnypilot/models/runners/tinygrad/tinygrad_runner.py``:
``run = pickle.load(pkl)``; ``out = run(**named_inputs)``; ``arr = out.numpy()`` -> ONE
concatenated graph output[0]. We slice it into named outputs with the metadata
``output_slices`` (the ``ModelRunner._slice_outputs`` one-liner) then parse with the
production ``parse_model_outputs_split.Parser`` (CD210 ships separate vision+policy onnx,
each parsed by ``parse_{vision,policy}_outputs``).

FIDELITY SCOPE: the faithful, anchor-targeted path is the 2-model (vision+policy) split
used by CD210/Nevada. The OPM7 3-model split (vision + on_policy + off_policy) is a thin
secondary path: for the weave/curvature comparison only ``vision -> on_policy`` matters,
so on_policy slots in exactly where the 2-model policy was and off_policy is intentionally
skipped (see BundleModel.__init__). OPM7 has no same-model fidelity anchor yet, so
replay_window still emits the loud cross-model warning. The 2-model class is structured so
adding the split path is additive.

Analysis-only. Imports tinygrad — call under ``model_replay_sim.env`` semantics (set BEFORE
the first tinygrad import); :func:`model_replay_sim.warp._ensure_tinygrad_env` arranges it.
"""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np

from model_replay_sim import config as C
from model_replay_sim.warp import _ensure_tinygrad_env

# DT_MDL = model step period (modeld imports it from openpilot.common.realtime); the
# curvature lookahead is `lat_delay + DT_MDL` (modeld.py:342).
DT_MDL = 0.05
MIN_SPEED = 1.0  # drive_helpers.MIN_SPEED


# ---------------------------------------------------------------------------
# Faithful copies of the pure curvature/smoothing helpers.
#
# We do NOT import these from the production modules because their import chains pull in
# Cython/native extensions not built in this analysis venv (fill_model_msg -> models.helpers
# -> common.params_pyx; drive_helpers -> common.realtime -> setproctitle). These functions
# are pure numpy. smooth_value/curv_from_psis/get_curvature_from_plan are copied VERBATIM
# (only the import-time DT_MDL default is substituted by the module constant above);
# get_curvature_from_output deliberately deviates (see its docstring). Citations
# (repo-relative paths):
#   smooth_value, curv_from_psis, get_curvature_from_plan
#       -> selfdrive/controls/lib/drive_helpers.py:21-63
#   get_curvature_from_output
#       -> sunnypilot/modeld_v2/fill_model_msg.py:14-20
# ---------------------------------------------------------------------------

def smooth_value(val, prev_val, tau, dt=DT_MDL):
    # drive_helpers.py:21-23
    alpha = 1 - np.exp(-dt / tau) if tau > 0 else 1
    return alpha * val + (1 - alpha) * prev_val


def _curv_from_psis(psi_target, psi_rate, vego, action_t):
    # drive_helpers.py:55-58
    vego = np.clip(vego, MIN_SPEED, np.inf)
    curv_from_psi = psi_target / (vego * action_t)
    return 2 * curv_from_psi - psi_rate / vego


def _get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
    # drive_helpers.py:60-63
    psi_target = np.interp(action_t, t_idxs, yaws)
    psi_rate = yaw_rates[0]
    return _curv_from_psis(psi_target, psi_rate, vego, action_t)


def get_curvature_from_output(output, plan, vego, lat_action_t, mlsim, t_idxs, plan_enum):
    """Adapted from ``sunnypilot/modeld_v2/fill_model_msg.py:14-20`` (t_idxs/plan_enum passed
    in to avoid the heavy production import).

    DELIBERATE DEVIATION FROM PRODUCTION: production tests presence with numpy-array
    truthiness — ``if desired_curv := output.get('desired_curvature'):`` — which raises on a
    multi-element array and, more importantly, treats an all-zeros ``[[0.0]]`` curvature
    output as ABSENT (falls through to the plan path) on its 0.0-array truthiness edge case.
    We use the more-correct ``is not None``: an emitted desired_curvature is used iff the key
    exists, regardless of its value. This matters only for a bundle that BOTH emits
    desired_curvature AND is non-mlsim; CD210 (the anchor) emits none, so both behaviors agree
    and the anchor is unaffected.
    """
    if not mlsim:
        if (desired_curv := output.get('desired_curvature')) is not None:  # model emits curv directly
            return float(desired_curv[0, 0])
    return float(_get_curvature_from_plan(
        plan[:, plan_enum.T_FROM_CURRENT_EULER][:, 2], plan[:, plan_enum.ORIENTATION_RATE][:, 2],
        t_idxs, vego, lat_action_t))


# ---------------------------------------------------------------------------
# Compiled-pkl model wrapper (vision + policy)
# ---------------------------------------------------------------------------

def _bundle_dir(bundle: str) -> Path:
    return C.RESULTS_ROOT / "onnx" / bundle


def _ensure_pkl(onnx: Path, pkl: Path) -> Path:
    """Compile ``onnx`` -> ``pkl`` once (cache). Reuses Task 2's compile_onnx_to_pkl,
    which runs compile3.py under replay_env on CPU. SELFTEST off here (this is the runtime
    path, not the fidelity gate)."""
    if pkl.exists():
        return pkl
    from model_replay_sim.compile_bundle import compile_onnx_to_pkl
    res = compile_onnx_to_pkl(onnx, pkl, compare_onnxruntime=False)
    if not res["compile_ok"] or not pkl.exists():
        raise RuntimeError(f"compile failed for {onnx.name}: rc={res['returncode']} "
                           f"{res['stderr_tail'][-500:]}")
    return pkl


class _CompiledModel:
    """One compiled tinygrad model (vision OR policy): its callable pkl + metadata
    (input_shapes / output_slices / output_shapes). ``run(named_numpy_inputs)`` returns the
    flat graph output[0] as a 1-D float32 array; ``slice_outputs`` cuts it into named
    arrays via ``output_slices`` (the ``ModelRunner._slice_outputs`` convention:
    ``{k: out[np.newaxis, v]}``)."""

    def __init__(self, onnx: Path, pkl: Path, meta_path: Path):
        _ensure_tinygrad_env()
        self.pkl_path = _ensure_pkl(onnx, pkl)
        with open(self.pkl_path, "rb") as f:
            self._run = pickle.load(f)
        meta = pickle.load(open(meta_path, "rb"))
        self.input_shapes: dict = dict(meta["input_shapes"])
        self.output_slices: dict = dict(meta["output_slices"])
        self.output_shapes: dict = dict(meta.get("output_shapes", {}))
        # vision inputs are the image tensors; everything else is a policy/float input.
        self.vision_input_names = [n for n in self.input_shapes if "img" in n]

    def run(self, named_inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Execute the pkl. img inputs go to the DEFAULT (CPU) device realized; non-img
        (policy) inputs go to device='NPY' — matching compile3.py's input construction
        (``'img' in k -> Device.DEFAULT`` else NPY) and tinygrad_runner."""
        from tinygrad.tensor import Tensor
        from tinygrad import Device
        tin = {}
        for k, v in named_inputs.items():
            if "img" in k:
                arr = np.ascontiguousarray(v, dtype=np.uint8)
                tin[k] = Tensor(arr, device=Device.DEFAULT).realize()
            else:
                arr = np.ascontiguousarray(v, dtype=np.float32)
                tin[k] = Tensor(arr, device="NPY").realize()
        out = self._run(**tin)
        return np.asarray(out.numpy(), dtype=np.float32).flatten()

    def slice_outputs(self, flat: np.ndarray) -> dict[str, np.ndarray]:
        """ModelRunner._slice_outputs (model_runner.py:154-166): slice the flat output by
        each named slice and prepend a batch axis."""
        return {k: flat[np.newaxis, v] for k, v in self.output_slices.items()}


class BundleModel:
    """Loads + caches a bundle's compiled vision & policy models and runs them, returning
    sliced+parsed named outputs.

    CD210/Nevada = 2-model split: vision + ``driving_policy.onnx``.
    OPM7 = 3-model split: vision + ``driving_on_policy.onnx`` (+ ``driving_off_policy.onnx``,
    intentionally NOT loaded — see below). For both, ``self.policy`` is THE policy that emits
    the trajectory ``plan`` that drives curvature, so all downstream machinery (``run_policy``,
    ``ReplayState``, ``replay_window``, the buffer-pairing, curvature) is identical.

    Why off_policy is skipped for the split (OPM7) path — the plan/curvature comparison needs
    only ``vision -> on_policy``:
      * The on_policy model produces the trajectory ``plan``; off_policy contributes only
        perception (lane_lines / road_edges / lead). Verified from the materialized metadata:
        on_policy output_slices = {plan, desire_state, pad}; off_policy = {plan, lane_lines,
        lane_lines_prob, road_edges, lead, lead_prob, pad}.
      * The production runner DROPS off_policy's plan when on_policy is present
        (``off_policy_output.pop('plan', None)`` — sunnypilot/models/runners/tinygrad/
        tinygrad_runner.py:116-127), so off_policy never contributes to ``plan``.
      * The runner's ``if 'planplus' in outputs`` combine (tinygrad_runner.py:126-127) does
        NOT fire for this commit: NEITHER on_policy NOR off_policy emits ``planplus`` (no such
        key in either metadata output_slices). So the effective ``plan`` used for curvature =
        on_policy's ``plan`` ALONE, and PLANPLUS_CONTROL is irrelevant for OPM7.
      => off_policy provably cannot affect the plan/curvature; loading/running it would only
         add cost. We skip it entirely. on_policy's input_shapes are identical to CD210's
         policy, so the existing temporal machinery works UNCHANGED.
    """

    _CACHE: dict[str, "BundleModel"] = {}

    def __init__(self, bundle: str):
        if bundle not in C.BUNDLES:
            raise KeyError(f"unknown bundle: {bundle!r} (known: {sorted(C.BUNDLES)})")
        self.bundle = bundle
        self.split = bool(C.BUNDLES[bundle]["split"])
        d = _bundle_dir(bundle)
        _ensure_tinygrad_env()
        from openpilot.sunnypilot.modeld_v2.parse_model_outputs_split import Parser
        self._parser = Parser()
        self.vision = _CompiledModel(d / "driving_vision.onnx", d / "driving_vision.pkl",
                                     d / "driving_vision_metadata.pkl")
        # The policy = the model that emits the plan: CD210/Nevada use driving_policy; the
        # OPM7 split uses driving_on_policy (off_policy is perception-only and skipped, see
        # the class docstring). on_policy's input_shapes / plan output match driving_policy
        # exactly, so it slots in here unchanged.
        if self.split:
            self.policy = _CompiledModel(d / "driving_on_policy.onnx", d / "driving_on_policy.pkl",
                                         d / "driving_on_policy_metadata.pkl")
        else:
            self.policy = _CompiledModel(d / "driving_policy.onnx", d / "driving_policy.pkl",
                                         d / "driving_policy_metadata.pkl")
        # combined input shapes (vision img inputs + policy float inputs) — the source of
        # truth for ReplayState buffer construction (mirrors TinygradSplitRunner.input_shapes).
        self.input_shapes: dict = {**self.vision.input_shapes, **self.policy.input_shapes}
        self.vision_input_names: list[str] = list(self.vision.vision_input_names)

    @classmethod
    def get(cls, bundle: str) -> "BundleModel":
        m = cls._CACHE.get(bundle)
        if m is None:
            m = cls(bundle)
            cls._CACHE[bundle] = m
        return m

    def run_vision(self, img: np.ndarray, big_img: np.ndarray) -> dict[str, np.ndarray]:
        """Run the vision model -> parsed named outputs (incl. raw ``hidden_state`` 512-d
        features used to feed the policy). ``parse_vision_outputs`` leaves ``hidden_state``
        untouched (it is not in the parse list) so we get the raw features."""
        flat = self.vision.run({"img": img, "big_img": big_img})
        sliced = self.vision.slice_outputs(flat)
        return self._parser.parse_vision_outputs(sliced)

    def run_policy(self, desire_pulse: np.ndarray, traffic_convention: np.ndarray,
                   features_buffer: np.ndarray) -> dict[str, np.ndarray]:
        """Run the policy model -> parsed named outputs (incl. ``plan`` reshaped to
        (1, IDX_N, PLAN_WIDTH) by the split parser)."""
        flat = self.policy.run({
            "desire_pulse": desire_pulse,
            "traffic_convention": traffic_convention,
            "features_buffer": features_buffer,
        })
        sliced = self.policy.slice_outputs(flat)
        return self._parser.parse_policy_outputs(sliced)


# ---------------------------------------------------------------------------
# Recurrent replay state (temporal buffers + per-frame step)
# ---------------------------------------------------------------------------

def build_temporal_buffers(input_shapes: dict, vision_input_names) -> tuple[dict, dict, dict]:
    """Build (numpy_inputs, temporal_buffers, temporal_idxs_map) from ``input_shapes``,
    COPIED VERBATIM from ModelState.__init__ (modeld.py:78-104), parametrized by shapes so
    any bundle works. Pure (no tinygrad / no model) -> unit-testable.

    For a 25-deep features_buffer (CD210): buffer_history_len = 25*4 = 100, skip = 4,
    temporal_idxs_map['features_buffer'] = arange(100)[-1-(4*24)::4] = [3,7,...,99] (25 idx).
    """
    vision_input_names = set(vision_input_names)
    numpy_inputs: dict[str, np.ndarray] = {}
    temporal_buffers: dict[str, np.ndarray] = {}
    temporal_idxs_map: dict[str, np.ndarray] = {}
    features_buffer_shape = input_shapes.get("features_buffer")
    for key, shape in input_shapes.items():
        if key not in vision_input_names:  # Policy inputs
            numpy_inputs[key] = np.zeros(shape, dtype=np.float32)
            # Temporal input: shape is [batch, history, features]
            if len(shape) == 3 and shape[1] > 1:
                buffer_history_len = shape[1] * 4 if shape[1] < 99 else shape[1]
                feature_len = shape[2]
                if shape[1] in (24, 25) and features_buffer_shape is not None and features_buffer_shape[1] == 24:  # 20Hz
                    buffer_history_len = (features_buffer_shape[1] + 1) * 4
                    step = int(-buffer_history_len / shape[1])
                    temporal_idxs_map[key] = np.arange(step, step * (shape[1] + 1), step)[::-1]
                elif shape[1] == 25:  # Split
                    skip = buffer_history_len // shape[1]
                    temporal_idxs_map[key] = np.arange(buffer_history_len)[-1 - (skip * (shape[1] - 1))::skip]
                elif shape[1] >= 99:  # non20hz
                    temporal_idxs_map[key] = np.arange(shape[1])
                temporal_buffers[key] = np.zeros((1, buffer_history_len, feature_len), dtype=np.float32)
    return numpy_inputs, temporal_buffers, temporal_idxs_map


class ReplayState:
    """The device's recurrent model state for ONE replay pass: numpy_inputs +
    temporal_buffers + temporal_idxs_map built from ``input_shapes`` (copied from
    ModelState.__init__, modeld.py:80-104) and the per-frame update (modeld.py:142-145).

    A passive straight-road replay has no lane change, so ``vec_desire`` is all-zero every
    frame; we still run the desire roll (a no-op on zeros) so the path is faithful
    (modeld.py:112-125, 304-306).
    """

    def __init__(self, model: BundleModel, ctx, bundle: str):
        _ensure_tinygrad_env()
        from openpilot.sunnypilot.modeld_v2.constants import ModelConstants, Plan

        self.model = model
        self.ctx = ctx
        self.bundle = bundle
        self.DESIRE_LEN = ModelConstants.DESIRE_LEN
        # curvature needs the plan time grid + the Plan output-slice enum (both pure constants)
        self.T_IDXS = ModelConstants.T_IDXS
        self.Plan = Plan

        # lateral-delay lookahead (modeld.py:292,342): lat_action_t = lat_delay + DT_MDL,
        # where lat_delay = base + LAT_SMOOTH_SECONDS (already folded into lateral_delay_input).
        from model_replay_sim.context import lateral_delay_input, bundle_lat_smooth_seconds
        self.lat_action_t = float(lateral_delay_input(ctx, bundle)) + DT_MDL
        self.lat_smooth_seconds = float(bundle_lat_smooth_seconds(bundle))

        # traffic convention one-hot (modeld.py:301-302); Task-4 helper.
        from model_replay_sim.context import traffic_convention_input
        self.traffic_convention = np.array(traffic_convention_input(ctx), dtype=np.float32)[None]

        # CD210 has no 'desired_curvature' output -> get_curvature_from_output always falls
        # through to the plan-based path regardless of mlsim (fill_model_msg.py:14-20).
        # We assert that here so we know we are on the faithful plan path.
        self._has_desired_curv_output = "desired_curvature" in model.policy.output_slices
        # mlsim: the REAL device rule is `generation >= 11` (a download-MANIFEST field, NOT
        # present in the pinned bundle/onnx). We cannot read it here, so we use a PROXY:
        #   mlsim = not has_desired_curvature_output
        # This proxy is correct-and-VALIDATED for CD210 (generation >= 11 AND the policy emits
        # no desired_curvature -> both rules agree: take the plan path) — but it is UNVERIFIED
        # for any other bundle, where the manifest generation could disagree with output-key
        # presence. Hence non-CD210 bundles warn (see replay_window / _warn_if_unvalidated).
        self.mlsim = not self._has_desired_curv_output  # True for CD210 (no desired_curvature)

        # --- temporal buffer construction (modeld.py:78-104) via the pure helper ----------
        self.prev_desire = np.zeros(self.DESIRE_LEN, dtype=np.float32)
        self.numpy_inputs, self.temporal_buffers, self.temporal_idxs_map = build_temporal_buffers(
            model.input_shapes, model.vision_input_names)

        # the desire input key (modeld.py:107-108): the policy input that starts with 'desire'
        self.desire_key = next(k for k in self.numpy_inputs if k.startswith("desire"))

        self.prev_curvature = 0.0  # prev_action.desiredCurvature seed

        # M2a capture hooks: the centering workflow reads these after each step.
        self.last_vision_out: dict | None = None
        self.last_policy_out: dict | None = None

    # ----- the per-frame update --------------------------------------------------------
    def _roll_desire(self, vec_desire: np.ndarray) -> None:
        """Desire pulse roll, copied from modeld.py:112-125. No-op on all-zero vec_desire,
        but run for faithfulness."""
        inp = vec_desire.astype(np.float32).copy()
        inp[0] = 0  # modeld.py:113
        new_desire = np.where(inp - self.prev_desire > .99, inp, 0)
        self.prev_desire[:] = inp
        buf = self.temporal_buffers[self.desire_key]
        buf[0, :-1] = buf[0, 1:]
        buf[0, -1] = new_desire
        # roll buffer and assign based on desire.shape[1] (modeld.py:120-125)
        if buf.shape[1] > self.numpy_inputs[self.desire_key].shape[1]:
            skip = buf.shape[1] // self.numpy_inputs[self.desire_key].shape[1]
            self.numpy_inputs[self.desire_key][:] = (
                buf[0].reshape(self.numpy_inputs[self.desire_key].shape[0],
                               self.numpy_inputs[self.desire_key].shape[1], skip, -1).max(axis=2))
        else:
            self.numpy_inputs[self.desire_key][:] = buf[0, self.temporal_idxs_map[self.desire_key]]

    def step(self, frame_inputs: dict[str, np.ndarray], v_ego: float) -> float:
        """Run ONE frame end-to-end and return the smoothed desiredCurvature.

        ``frame_inputs`` must carry the vision input tensors {'img','big_img'} (uint8
        (1,12,128,256)); replay_window builds them by pairing the oldest+newest sixchan of a
        ring buffer (see :func:`replay_window`). v_ego in m/s; a non-finite v_ego -> nan
        return (the frame is then excluded by the nan-aware metrics).
        """
        # desire roll (passive replay -> all zeros)
        vec_desire = np.zeros(self.DESIRE_LEN, dtype=np.float32)
        self._roll_desire(vec_desire)

        # traffic_convention is constant per-route (modeld.py:301-302,325)
        self.numpy_inputs["traffic_convention"][:] = self.traffic_convention

        # --- vision ---
        img = np.ascontiguousarray(frame_inputs["img"], dtype=np.uint8)
        big_img = np.ascontiguousarray(frame_inputs.get("big_img", frame_inputs["img"]), dtype=np.uint8)
        vision_out = self.model.run_vision(img, big_img)
        hidden_state = vision_out["hidden_state"]  # (1, 512)

        # --- features_buffer roll + subsample (modeld.py:142-145) ---
        fb = self.temporal_buffers["features_buffer"]
        fb[0, :-1] = fb[0, 1:]
        fb[0, -1] = hidden_state[0, :]
        self.numpy_inputs["features_buffer"][:] = fb[0, self.temporal_idxs_map["features_buffer"]]

        # --- policy ---
        policy_out = self.model.run_policy(
            self.numpy_inputs[self.desire_key],
            self.numpy_inputs["traffic_convention"],
            self.numpy_inputs["features_buffer"],
        )

        # M2a capture hooks: the centering workflow reads these after each step.
        self.last_vision_out = vision_out
        self.last_policy_out = policy_out

        # --- curvature post-step (modeld.py:163-178; fill_model_msg.get_curvature_from_output) ---
        plan = policy_out["plan"]            # (1, IDX_N, PLAN_WIDTH)
        # A nan v_ego is our HONEST "no valid speed" signal (_route_window_v_ego). The device
        # never sees nan v_ego, so the production smoothing branch (v_ego > thresh) does not
        # define behavior for it; emit nan (curvature scaling 1/vego is undefined) so the frame
        # is dropped by the nan-aware metrics rather than silently holding prev_curvature.
        if not np.isfinite(v_ego):
            return float("nan")
        curv = get_curvature_from_output(policy_out, plan[0], float(v_ego), self.lat_action_t,
                                         self.mlsim, self.T_IDXS, self.Plan)
        # generation>=10 smoothing (modeld.py:172-176): smooth above MIN_LAT_CONTROL_SPEED,
        # else hold previous. CD210 LAT_SMOOTH_SECONDS=0.0 -> smooth_value alpha=1 (identity).
        MIN_LAT_CONTROL_SPEED = 0.3
        if float(v_ego) > MIN_LAT_CONTROL_SPEED:
            curv = smooth_value(curv, self.prev_curvature, self.lat_smooth_seconds)
        else:
            curv = self.prev_curvature
        self.prev_curvature = float(curv)
        return float(curv)


# ---------------------------------------------------------------------------
# Window replay
# ---------------------------------------------------------------------------

def _warn_if_unvalidated(bundle: str) -> None:
    """Emit ONE loud warning when ``bundle`` has no same-model fidelity anchor.

    Only CD210 is anchor-validated (config.BUNDLES). Cross-model replays (Nevada/OPM7)
    can return silently-wrong weave numbers: img_buffer_length assumes is_20hz and mlsim is
    derived from output-key presence rather than the device's generation>=11 rule — neither
    is proven for those bundles. Pure (no model load) so it can be unit-tested cheaply and
    fires before any heavy compute in replay_window."""
    if not C.BUNDLES.get(bundle, {}).get("anchor_validated", False):
        import warnings
        buf_len = int(C.BUNDLES.get(bundle, {}).get("img_buffer_length", 5))
        warnings.warn(
            f"{bundle}: cross-model replay is NOT fidelity-anchor-validated "
            f"(img_buffer_length={buf_len} assumes is_20hz, and mlsim is derived from "
            f"output-key presence, not the device's generation>=11 rule); the weave number "
            f"may be systematically wrong. Validate with a same-model anchor before trusting it.",
            stacklevel=2)


def _route_window_v_ego(route_id: str, mono_times) -> np.ndarray:
    """v_ego (m/s) for each window mono_time, nearest-sample from the cached route npz
    (the same source assets.py uses).

    HONESTY GUARDS (a silent v_ego=0 corrupts curvature scaling: curv ~ 1/vego, and v_ego=0
    clips to MIN_SPEED=1.0 -> the primary output is silently scaled wrong):
      (a) A frame whose nearest sample is non-finite, OR is farther than ``C.MAX_FRAME_DELTA_S``
          from the request time, emits ``np.nan`` (NOT 0.0). nan v_ego -> nan curvature ->
          the frame is honestly excluded by the nan-aware metrics (parse.py's nan-in-place
          discipline), instead of corrupting the weave number.
      (b) A missing route NPZ, or one lacking ``v_ego``/``mono_time``, raises (no zero array).
      (c) If >25% of the requested frames resolve to nan, warn loudly (route + nan fraction).
    """
    mono_times = [float(m) for m in mono_times]
    npz = C.CACHE_ROOT / f"{route_id}.npz"
    if not npz.exists():
        raise FileNotFoundError(f"route NPZ not found for v_ego lookup: {npz}")
    z = dict(np.load(npz))
    if "mono_time" not in z or "v_ego" not in z:
        raise KeyError(
            f"route NPZ {npz} is missing required key(s) for v_ego lookup "
            f"(need 'mono_time' and 'v_ego'; have {sorted(z)[:8]}...)")
    t = np.asarray(z["mono_time"], float)
    v = np.asarray(z["v_ego"], float)
    out = []
    for m in mono_times:
        i = int(np.argmin(np.abs(t - m)))
        if (not np.isfinite(v[i])) or abs(float(t[i]) - m) > C.MAX_FRAME_DELTA_S:
            out.append(np.nan)        # honest nan: excluded by nan-aware metrics, never a fake 0
        else:
            out.append(float(v[i]))
    arr = np.asarray(out, dtype=np.float32)
    if arr.size:
        nan_frac = float(np.mean(~np.isfinite(arr)))
        if nan_frac > 0.25:
            import warnings
            warnings.warn(
                f"{route_id}: {nan_frac:.0%} of requested frames have no valid v_ego "
                f"(non-finite or > {C.MAX_FRAME_DELTA_S}s from nearest NPZ sample); those "
                f"frames yield nan curvature and are excluded from the weave metric.",
                stacklevel=2)
    return arr


def window_transforms(ctx, camera_offset: float | None = None):
    """The per-window forward warp matrices (M_main, M_extra) + the offset actually used.

    ``camera_offset=None`` -> the route's own logged param (``ctx.camera_offset``); an
    explicit float REPLACES it (S1 counterfactual). Production EMAs the live param
    (0.9*actual + 0.1*param per frame -> steady state == param, camera_offset_helper.py);
    a static-scene replay uses the steady value directly (see warp.model_transform).
    """
    from model_replay_sim.warp import model_transform
    used = float(camera_offset) if camera_offset is not None else float(getattr(ctx, "camera_offset", 0.0))
    M_main = model_transform(ctx, wide=False, camera_offset=used)
    M_extra = model_transform(ctx, wide=True, camera_offset=used)
    return M_main, M_extra, used


def _collect_captured(state, keys) -> dict:
    """Pull the requested PARSED output arrays for the CURRENT frame from a stepped
    ReplayState, dropping the batch axis. Vision outputs win on name collision
    (lane_lines lives in the vision model for SP002/CD210/Nevada — verified from the
    materialized driving_vision_metadata.pkl output_slices); falls back to the policy
    outputs (e.g. 'plan'). Raises KeyError listing what IS available."""
    vis = getattr(state, "last_vision_out", None) or {}
    pol = getattr(state, "last_policy_out", None) or {}
    out = {}
    for k in keys:
        if k in vis:
            out[k] = np.array(vis[k][0], copy=True)
        elif k in pol:
            out[k] = np.array(pol[k][0], copy=True)
        else:
            raise KeyError(f"capture key {k!r} not found; vision has {sorted(vis)}, policy has {sorted(pol)}")
    return out


def replay_window(bundle: str, route_id: str, mono_times,
                  camera_offset: float | None = None,
                  capture_outputs: tuple[str, ...] = ()) -> dict:
    """Replay a window of ``mono_times`` through ``bundle`` on ``route_id`` and return the
    desiredCurvature series.

    Pipeline per frame: map_window_to_frames (Task 3) -> read_frame/read_wide_frame ->
    frame_to_sixchan (warp + sixchan-pack) into a ``buf_len``-deep ring buffer (one for the
    road cam, one for the wide cam) -> ``_pair(buffer[0], buffer[-1])`` = the OLDEST+NEWEST
    sixchan (the device's cat(buffer[:6], buffer[-6:]) 12-channel input) -> ReplayState.step.
    NOTE: this is the production replay path; it does NOT route through
    ``warp.frame_to_model_input`` (that helper pairs only a single (prev, current) and is used
    by tests) — it threads a ring buffer so the two img channels span ``buf_len`` frames.

    camera_offset: None = the route's logged param; float = counterfactual override (S1
    sweep). capture_outputs: names of parsed model outputs captured per frame into
    result["captured"] (vision outputs win on name collision; see _collect_captured).

    Returns ``{"route_id","bundle","mono_time": (N,), "desired_curvature": (N,),
    "v_ego": (N,), "frames": [(seg_num,seg_id),...]}``.
    """
    from model_replay_sim.alignment import map_window_to_frames, read_frame, read_wide_frame
    from model_replay_sim.context import route_context
    from model_replay_sim.warp import frame_to_sixchan

    _warn_if_unvalidated(bundle)  # loud guard for cross-model (non-anchor) bundles, before heavy compute

    mono_times = [float(t) for t in mono_times]
    ctx = route_context(route_id)
    model = BundleModel.get(bundle)
    state = ReplayState(model, ctx, bundle)

    # one forward warp matrix per route (static scene; calib steady)
    M_main, M_extra, camera_offset_used = window_transforms(ctx, camera_offset)

    aligns = map_window_to_frames(route_id, mono_times)
    v_egos = _route_window_v_ego(route_id, mono_times)

    curvs = np.empty(len(mono_times), dtype=np.float64)
    cap_frames: list[dict] = []
    # The device 20Hz vision warp keeps a rolling buffer of `buf_len` frames and feeds the
    # model cat(buffer[:6], buffer[-6:]) = the OLDEST+NEWEST frame (compile_warp.py:101;
    # buffer_length = 5 if is_20hz else 2, modeld.py:75). So the two img channels are frames
    # t-(buf_len-1) and t. Pairing only (t-1, t) gives ~1/4 the inter-frame motion -> the model
    # under-estimates orientation-rate and the replayed weave came out ~2x too small. Threading
    # a buf_len-deep ring buffer and pairing (oldest, newest) restores it (sep-sweep on the
    # CD210 anchor: sep=buf_len-1=4 -> band_ratio 1.008, corr 0.993). Same for the wide cam.
    buf_len = int(C.BUNDLES[bundle].get("img_buffer_length", 5))
    road_buf: deque = deque(maxlen=buf_len)
    wide_buf: deque = deque(maxlen=buf_len)
    for i, (al, t, v) in enumerate(zip(aligns, mono_times, v_egos)):
        nv12 = np.asarray(read_frame(route_id, al.segment_num, al.segment_id), dtype=np.uint8).ravel()
        cam_w, cam_h = _frame_dims(nv12.size)
        road_buf.append(frame_to_sixchan(nv12, cam_w, cam_h, M_main))   # (6,128,256)
        img = _pair(road_buf[0], road_buf[-1])                          # (oldest=t-(buf_len-1), newest=t)

        # wide (big_img): use the REAL wide frame (ecamera.hevc) warped through M_extra; when it
        # is missing/short (al.ecamera_index is None) carry forward the last wide sixchan, and on
        # the very first frame with no history fall back once to the road-through-M_extra proxy.
        if al.ecamera_index is not None:
            wide_nv12 = np.asarray(read_wide_frame(route_id, al.segment_num, al.ecamera_index), dtype=np.uint8).ravel()
            w_w, w_h = _frame_dims(wide_nv12.size)
            cur_big = frame_to_sixchan(wide_nv12, w_w, w_h, M_extra)
        elif wide_buf:
            cur_big = wide_buf[-1]                                      # carry forward last wide
        else:
            cur_big = frame_to_sixchan(nv12, cam_w, cam_h, M_extra)     # one-time proxy fallback
        wide_buf.append(cur_big)
        big_img = _pair(wide_buf[0], wide_buf[-1])

        curvs[i] = state.step({"img": img, "big_img": big_img}, float(v))
        if capture_outputs:
            cap_frames.append(_collect_captured(state, capture_outputs))

    result = {
        "route_id": route_id,
        "bundle": bundle,
        "mono_time": np.asarray(mono_times, dtype=np.float64),
        "desired_curvature": curvs,
        "v_ego": v_egos,
        "frames": [(a.segment_num, a.segment_id) for a in aligns],
        "lat_action_t": state.lat_action_t,
        "camera_offset_used": camera_offset_used,
    }
    if capture_outputs:
        result["captured"] = {k: np.stack([f[k] for f in cap_frames]) for k in capture_outputs}
    return result


def _pair(prev_sixchan: np.ndarray, cur_sixchan: np.ndarray) -> np.ndarray:
    """(prev,cur) sixchan -> (1,12,128,256) uint8 vision input (warp.frame_to_model_input
    convention: channels 0:6 prev, 6:12 current)."""
    prev = np.ascontiguousarray(prev_sixchan, dtype=np.uint8)
    cur = np.ascontiguousarray(cur_sixchan, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([prev, cur], axis=0)[None], dtype=np.uint8)


def _frame_dims(size: int) -> tuple[int, int]:
    """Recover (cam_w, cam_h) from a contiguous nv12 size. The Comma 4 (mici/os04c10) road
    cam is 1344x760 (size 1_532_160); assert that here rather than guess."""
    if size == 1344 * 760 * 3 // 2:
        return 1344, 760
    raise ValueError(f"unrecognized contiguous nv12 size {size}; expected 1344x760 "
                     f"(={1344 * 760 * 3 // 2}) for this device")
