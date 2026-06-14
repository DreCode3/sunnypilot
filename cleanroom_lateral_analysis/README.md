# Clean-Room Lateral-Control Analysis — READ ME FIRST

## What this is
A self-contained, **blinded** dataset and briefing for an **independent third-party analysis** of a
lateral-control (steering) comfort issue on one specific vehicle running an openpilot-derived
driver-assistance system. You are engaged precisely because you have **no prior context** on this
problem. Perform a **clean-room analysis**: design your own methods, reach your own conclusions.

## What you are given
- `drives/` — raw openpilot drive logs from 6 drives (`drive_01` … `drive_06`) recorded under
  different software configurations, plus `drives/METADATA.csv` describing each drive factually.
- `PROBLEM_STATEMENT.md` — the physical symptom to investigate and the questions to answer.
- `SYSTEM_OVERVIEW.md` — how this class of system works (background).
- `CUSTOMIZATIONS.md` — how this vehicle's software differs from stock, incl. the two variables
  that differ across the drives.
- `HOW_TO_READ_LOGS.md` — the log format and how to load it.
- `REFERENCES.md` — public reference material.

## What is deliberately withheld (and why)
To keep this a true clean room:
- **No analysis logic, code, metrics, or methods** are provided — design your own from first principles.
- **No findings or conclusions** from any prior analysis are provided.
- **The driver's subjective opinion of which configuration is best is withheld.** Do NOT assume any
  configuration is a "baseline," "control," "stock," or "improved" version. They are simply different
  configurations to be compared objectively.

## Rules of engagement
1. **Derive everything yourself** — signals, metrics, frequency bands, statistics.
2. **Do not assume a conclusion.** Let the data tell you whether configurations differ, and how.
3. **Confounds first.** Before claiming ANY difference between configurations, characterize and address
   confounds (vehicle speed, road geometry/location, traffic, drive-to-drive and day-to-day variation,
   sample size). Report whether a valid comparison is even possible with this data, *before* difference claims.
4. **Two explanatory variables.** The drives differ in (a) the driving-model version and (b) the
   lateral-controller parameters (see `CUSTOMIZATIONS.md` / `METADATA.csv`). Where you find a difference,
   try to attribute it to one variable vs the other.
5. **Quantify uncertainty.** Robust statistics; confidence intervals / significance; honor signal
   autocorrelation (samples within a drive are highly correlated).
6. **If the data is insufficient, say so.** State plainly if a confident answer is not possible from this
   data, and specify exactly what additional data or controlled experiment would resolve it.
7. **Make it reproducible.** Document your method so your numbers can be re-derived.

## Suggested read order
1. `PROBLEM_STATEMENT.md`  2. `SYSTEM_OVERVIEW.md`  3. `CUSTOMIZATIONS.md`
4. `HOW_TO_READ_LOGS.md`  5. `drives/METADATA.csv`  6. `REFERENCES.md` (as needed)

## Suggested deliverable
A written report containing: your method (and why), the metric(s) you chose to capture the symptom,
your per-configuration results with uncertainty, your treatment of confounds, your causal attribution
(model vs controller), a clear yes/no/insufficient verdict on whether configurations differ in the
symptom, and — if insufficient — the specific experiment that would settle it.
