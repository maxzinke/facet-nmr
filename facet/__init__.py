"""FACET: Fold And Conformation Estimation Tool.

Predict backbone torsion angles, secondary structure, and chi1 rotamers
from NMR chemical shifts. Surpasses TALOS-N on 27,461 matched residues
(13.4 vs 14.4 deg median error).

Quick start::

    facet predict shifts.tab

Python API::

    from facet import predict
    result = predict("shifts.tab")
    print(result.summary())
    result.to_tbl("restraints.tbl")
"""
from __future__ import annotations

__version__ = "0.1.0"

from .inference import predict
from .io.formats import FACETResult, ResiduePrediction, ShiftList

__all__ = [
    "__version__",
    "predict",
    "FACETResult",
    "ResiduePrediction",
    "ShiftList",
]
