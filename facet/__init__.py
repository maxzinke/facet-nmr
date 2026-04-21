"""FACET: Fold And Conformation Estimation Tool.

Predict backbone torsion angles, secondary structure, and chi1 rotamers
from NMR chemical shifts.

Quick start::

    facet predict shifts.tab

Python API::

    from facet import predict
    result = predict("shifts.tab")
    print(result.summary())
    result.to_tbl("restraints.tbl")
"""
from __future__ import annotations

__version__ = "0.3.0"

from .inference import predict
from .io.formats import FACETResult, ResiduePrediction, ShiftList
from .ss_populations import (
    SSPopulationResult,
    predict_ss_populations,
)


def plot_sequence_ss(*args, **kwargs):
    """Lazy import wrapper — matplotlib is an optional dependency."""
    from .visualization import plot_sequence_ss as _impl
    return _impl(*args, **kwargs)


def plot_residual_ss(*args, **kwargs):
    """Lazy import wrapper — matplotlib is an optional dependency.

    IDP-mode figure: four per-basin tracks along the sequence. Use for
    flexible / disordered samples where plot_sequence_ss degenerates.
    """
    from .visualization import plot_residual_ss as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "__version__",
    "predict",
    "predict_ss_populations",
    "FACETResult",
    "ResiduePrediction",
    "SSPopulationResult",
    "ShiftList",
    "plot_sequence_ss",
    "plot_residual_ss",
]
