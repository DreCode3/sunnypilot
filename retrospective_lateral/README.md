# Retrospective Lateral Weave Analysis

This package analyzes existing Explorer ST openpilot/sunnypilot logs for two symptoms:

- Low-speed steering-wheel swing at 1-10 mph.
- Straight/gentle-section path and wheel weave at 10-70 mph.

Run commands from the repository root with `.venv311/bin/python`.

The package is analysis-only. It must not modify vehicle-control code.
