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

__version__ = "0.2.0"

from .inference import predict
from .io.formats import FACETResult, ResiduePrediction, ShiftList


def plot_sequence_ss(*args, **kwargs):
    """Lazy import wrapper — matplotlib is an optional dependency."""
    from .visualization import plot_sequence_ss as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "__version__",
    "predict",
    "FACETResult",
    "ResiduePrediction",
    "ShiftList",
    "plot_sequence_ss",
]
