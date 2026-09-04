"""Logistics offline oracle.

``ortools_reference`` retains its historical module name but formal replay now
uses one dependency-invariant, bounded capacity partition followed by
nearest-neighbour plus single-pass 2-opt. The result is content-cached across
repeated episode and counterfactual resets and is never recomputed live per
tick.
"""

from .ortools_reference import (
    ORTOOLS_AVAILABLE,
    compute_reference_optimum,
)

__all__ = ["ORTOOLS_AVAILABLE", "compute_reference_optimum"]
