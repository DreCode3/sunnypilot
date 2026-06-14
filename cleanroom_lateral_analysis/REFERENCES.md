# Reference Material (public)

All pointers below are public and independent of any prior analysis of this problem. Use them to
understand the system and the data format; design your own analysis.

## openpilot / sunnypilot
- **openpilot** source & docs: https://github.com/commaai/openpilot — see `docs/`, `selfdrive/`, and
  especially:
  - **`cereal/log.capnp`** — the authoritative schema for every log message type and field (names,
    types, units). The single most important reference for the data.
  - **`tools/lib/logreader.py`** — the reader used to iterate log messages.
  - `selfdrive/controls/` — lateral control implementations (e.g. `latcontrol_*.py`) and the lateral
    planner / model-predictive control.
- **sunnypilot** fork: https://github.com/sunnypilot/sunnypilot
- comma.ai site & engineering blog (architecture, the driving model): https://comma.ai , https://blog.comma.ai

## Data format
- **Cap'n Proto** (the log serialization): https://capnproto.org
- **Zstandard / zstd** (the compression): https://facebook.github.io/zstd/

## Background — vehicle lateral dynamics & feedback control
*(Generic references for reasoning about steering behavior and oscillation in a control loop — not
specific to this problem.)*
- R. Rajamani, *Vehicle Dynamics and Control* — lateral vehicle dynamics, the bicycle model, lane-keeping.
- Standard control-theory texts on PID control, **integrator windup / saturation**, and **limit cycles**
  in feedback systems (useful when reasoning about steering oscillations).
- Signal-processing fundamentals: spectral analysis (PSD / Welch), band-pass filtering, and statistics
  for **autocorrelated time series** (e.g. block bootstrap, effective sample size).

## Vehicle
- 2021 Ford Explorer — manufacturer specifications for wheelbase / steering as needed (or read
  `carParams` directly from the logs).

*(Intentionally omitted: any scripts, notes, metrics, or conclusions from prior work on this issue.)*
