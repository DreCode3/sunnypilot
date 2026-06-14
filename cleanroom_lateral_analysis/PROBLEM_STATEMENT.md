# Problem Statement

## Background
This concerns one specific vehicle (a 2021 Ford Explorer ST) running an openpilot-derived lateral
driver-assistance system (see `SYSTEM_OVERVIEW.md`). When engaged, the system steers the car to keep it
in its lane. The investigation was prompted by a **driver-reported comfort issue**.

## The symptom to investigate
While the lateral system is **engaged**, on **straight roads and very gentle curves**, there is a
**small but perceptible left-right oscillation ("weave")** of the steering wheel and of the car's path
within the lane:
- **Low frequency / slow** — a back-and-forth that plays out over a few seconds (not a fast vibration).
- **Small amplitude,** but noticeable.
- **Perceptible to the driver** (mildly uncomfortable, especially over longer drives) and **visible to
  surrounding traffic** (the car can be seen subtly weaving within its lane).
- Occurs on **straights and gentle curves** — i.e., when the car should be tracking a nearly-straight path.

This is the specific behavior to quantify and compare. It is distinct from (a) steady-state lane
*position* (whether the car rides left/right of lane center) and (b) behavior in sharper curves —
though you may investigate those too if you find them relevant.

## Your objective
Using methods of your own design, analyze the provided drives and determine:
1. **Quantify the symptom.** How would you measure this weave? Characterize its amplitude, frequency
   content, and when/where it occurs. Which signal(s) best capture what the driver feels and what is
   visible externally?
2. **Compare configurations.** Do the configurations (see `drives/METADATA.csv` and `CUSTOMIZATIONS.md`)
   differ in this symptom? By how much?
3. **Confidence.** What is the statistical confidence of any difference (robust, autocorrelation-aware)?
4. **Causal attribution.** Is any difference attributable to the configuration, or to confounds (speed,
   road/location, traffic, drive-to-drive variation)? The drives differ along **two** variables — the
   driving-model version and the lateral-controller parameters — so where there is a difference, which
   variable is responsible?
5. **Sufficiency.** If the data cannot support a confident answer, say so explicitly and specify exactly
   what additional data or controlled experiment would resolve it.

## Important — blinding
The driver's subjective impression of which configuration is best is **deliberately withheld** so it
cannot bias your analysis. Treat all configurations as equals to be compared objectively. Report what
the data shows — including "no detectable difference" or "confounded / insufficient data" if that is the
honest result. A rigorous "we cannot tell from this data, and here is exactly what would settle it" is a
fully acceptable — and valuable — outcome.
