"""Hawkes Process model skeleton module."""

from sentinel_analysis.hawkes.calibration import (
    HawkesCalibrationResult,
    calibrate_hawkes,
)
from sentinel_analysis.hawkes.kernels import ExponentialKernel, Kernel, PowerLawKernel
from sentinel_analysis.hawkes.model import HawkesProcess
from sentinel_analysis.hawkes.validation import (
    GoodnessOfFitResult,
    compute_transformed_residuals,
    evaluate_goodness_of_fit,
    plot_qq_residuals,
)

__all__ = [
    "ExponentialKernel",
    "GoodnessOfFitResult",
    "HawkesCalibrationResult",
    "HawkesProcess",
    "Kernel",
    "PowerLawKernel",
    "calibrate_hawkes",
    "compute_transformed_residuals",
    "evaluate_goodness_of_fit",
    "plot_qq_residuals",
]
