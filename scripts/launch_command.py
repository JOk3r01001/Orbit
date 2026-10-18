"""Dependency-free validation for the existing 16-observation launch policy."""

import math


def validate_launch_command(options):
    if not isinstance(options, dict):
        raise ValueError("Launch command must be an object")
    defaults = {
        "target_pe": 70_000.0,
        "min_fuel_fraction": 0.0,
        "urgency": 0.5,
        "ap_tolerance": 10_000.0,
        "pe_tolerance": 10_000.0,
    }
    unknown = options.keys() - (defaults.keys() | {"target_ap"})
    if unknown or "target_ap" not in options:
        raise ValueError(f"Launch command requires target_ap; unknown fields: {sorted(unknown)}")
    values = {**defaults, **options}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError(f"{name} must be a finite number")
    limits = {
        "target_ap": (80_000, 180_000),
        "target_pe": (70_000, values["target_ap"]),
        "min_fuel_fraction": (0, 0.15),
        "urgency": (0, 1),
        "ap_tolerance": (1, 10_000),
        "pe_tolerance": (1, 10_000),
    }
    for name, (low, high) in limits.items():
        if not low <= values[name] <= high:
            raise ValueError(f"{name} must be between {low} and {high}; request was not changed")
    return {name: float(value) for name, value in values.items()}
