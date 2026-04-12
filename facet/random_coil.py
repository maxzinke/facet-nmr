"""Random coil chemical shift reference values for secondary shift calculation.

Secondary shifts (delta = observed - random_coil) are the primary input features
for FACET. Values from Wishart et al. (1995) and Schwarzinger et al. (2001).

All values in ppm. pH ~5-7, 25C, DSS reference.
"""
from __future__ import annotations

import numpy as np

from .io.formats import BACKBONE_NUCLEI

# Wishart/Schwarzinger consensus random coil shifts (ppm)
RANDOM_COIL_SHIFTS: dict[str, dict[str, float]] = {
    "ALA": {"H": 8.24, "HA": 4.32, "N": 123.8, "CA": 52.5, "CB": 19.1, "C": 177.8},
    "ARG": {"H": 8.23, "HA": 4.34, "N": 120.5, "CA": 56.0, "CB": 30.7, "C": 176.3},
    "ASN": {"H": 8.38, "HA": 4.74, "N": 118.7, "CA": 53.1, "CB": 38.7, "C": 175.2},
    "ASP": {"H": 8.34, "HA": 4.64, "N": 120.4, "CA": 54.0, "CB": 41.1, "C": 176.3},
    "CYS": {"H": 8.32, "HA": 4.55, "N": 118.8, "CA": 58.2, "CB": 28.0, "C": 174.6},
    "GLN": {"H": 8.27, "HA": 4.34, "N": 119.8, "CA": 55.7, "CB": 29.4, "C": 176.0},
    "GLU": {"H": 8.42, "HA": 4.29, "N": 120.2, "CA": 56.6, "CB": 30.0, "C": 176.6},
    "GLY": {"H": 8.33, "HA": 3.96, "N": 109.9, "CA": 45.4,              "C": 174.9},
    "HIS": {"H": 8.42, "HA": 4.73, "N": 118.2, "CA": 55.0, "CB": 29.0, "C": 174.1},
    "ILE": {"H": 8.00, "HA": 4.17, "N": 119.9, "CA": 61.1, "CB": 38.8, "C": 176.4},
    "LEU": {"H": 8.16, "HA": 4.34, "N": 121.8, "CA": 55.1, "CB": 42.4, "C": 177.6},
    "LYS": {"H": 8.29, "HA": 4.32, "N": 120.4, "CA": 56.5, "CB": 32.7, "C": 176.6},
    "MET": {"H": 8.28, "HA": 4.48, "N": 119.6, "CA": 55.4, "CB": 32.9, "C": 176.3},
    "PHE": {"H": 8.30, "HA": 4.62, "N": 120.3, "CA": 57.7, "CB": 39.6, "C": 175.8},
    "PRO": {           "HA": 4.42, "N": 136.6, "CA": 63.3, "CB": 32.1, "C": 177.3},
    "SER": {"H": 8.31, "HA": 4.47, "N": 115.7, "CA": 58.3, "CB": 63.8, "C": 174.6},
    "THR": {"H": 8.15, "HA": 4.35, "N": 113.6, "CA": 61.8, "CB": 69.8, "C": 174.7},
    "TRP": {"H": 8.25, "HA": 4.66, "N": 121.3, "CA": 57.5, "CB": 29.6, "C": 176.1},
    "TYR": {"H": 8.12, "HA": 4.55, "N": 120.3, "CA": 57.9, "CB": 38.8, "C": 175.9},
    "VAL": {"H": 8.03, "HA": 4.12, "N": 119.2, "CA": 62.4, "CB": 32.9, "C": 176.3},
}


def to_secondary_shifts(
    shifts: np.ndarray,
    masks: np.ndarray,
    comp_ids: list[str],
) -> np.ndarray:
    """Convert raw ppm shifts to secondary shifts (observed - random_coil).

    Args:
        shifts: (N, 6) raw ppm, NaN for missing.
        masks: (N, 6) observation masks.
        comp_ids: 3-letter AA codes.

    Returns:
        (N, 6) secondary shifts. NaN where no RC value or shift missing.
    """
    n = len(comp_ids)
    sec = np.full((n, 6), np.nan, dtype=np.float32)
    for i in range(n):
        rc = RANDOM_COIL_SHIFTS.get(comp_ids[i], {})
        for j, nuc in enumerate(BACKBONE_NUCLEI):
            if masks[i, j] > 0 and nuc in rc and not np.isnan(shifts[i, j]):
                sec[i, j] = shifts[i, j] - rc[nuc]
    # Replace NaN with 0 (mask handles missing)
    return np.where(np.isnan(sec), 0.0, sec).astype(np.float32)
