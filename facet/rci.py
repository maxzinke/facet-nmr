"""Random coil index (RCI) and derived per-residue S^2 order parameter.

Berjanskii & Wishart (2005) JACS 127, 14970; Berjanskii & Wishart (2008)
J. Biomol. NMR 40, 31 — RCI converts backbone chemical shift deviations
from random coil into an approximate NMR order parameter (S^2).

Physical meaning: S^2 is a model-free order parameter in [0, 1]. A rigid
residue in a folded element has S^2 ~ 0.85-0.90. A flexible loop has
S^2 ~ 0.65-0.80. A disordered IDR residue has S^2 < 0.5.

Algorithm (simplified Berjanskii-Wishart with empirical calibration):
  1. Secondary shifts: delta(nucleus) = observed - random_coil
  2. Per-nucleus |delta| scaled by the nucleus' typical deviation range
     so different nuclei contribute comparably to the aggregate.
  3. 3-residue smoothing window (|delta_i| averaged with neighbors).
  4. Weighted RCI = weighted mean of smoothed scaled |deltas| across
     observed nuclei. Characteristic values:
       ~0.05 — pure random coil (no structural signature)
       ~0.3  — flexible loop
       ~0.7  — structured α-helix / β-strand
  5. S^2 = tanh(2 * RCI), giving values ~0.1 for disordered up to ~0.9
     for rigid structured elements — matches the phenomenological range
     cited in Wishart 2008.

The exact Berjanskii-Wishart polynomial uses per-nucleus calibrated
coefficients from a large dataset; our simplified form produces the
same qualitative pattern with cleaner code. Use this output as an
order-of-magnitude rigidity indicator, not an absolute S^2 replacement.
"""
from __future__ import annotations

import math

import numpy as np

from .io.formats import BACKBONE_NUCLEI
from .random_coil import RANDOM_COIL_SHIFTS


# Per-nucleus weights. Proton and 13C get higher weight (largest dynamic
# range in secondary shifts); 15N has wide intrinsic spread so smaller
# weight. Weights don't need to sum to 1; we normalize by observed weights
# per residue at the aggregation step.
_WEIGHTS = {
    "H":  0.3,
    "HA": 0.5,
    "N":  0.2,
    "CA": 1.0,
    "CB": 1.0,
    "C":  0.8,
}

# Per-nucleus scaling factor (approximate typical maximum secondary shift,
# used to bring all nuclei to a comparable ~0-1 range before weighting).
_SCALE_PPM = {
    "H":  1.2,   # |ΔH|   typically 0-1.2 ppm
    "HA": 0.8,   # |ΔHA|  typically 0-0.8 ppm
    "N":  6.0,   # |ΔN|   typically 0-6 ppm
    "CA": 4.5,   # |ΔCA|  typically 0-4.5 ppm (helix +3, strand -1.5)
    "CB": 3.0,   # |ΔCB|  typically 0-3 ppm
    "C":  3.5,   # |ΔC'|  typically 0-3.5 ppm
}


def compute_rci_s2(
    shifts: np.ndarray,   # (N, 6), ppm, NaN for missing
    masks: np.ndarray,    # (N, 6), 1=observed
    comp_ids: list[str],  # 3-letter AA codes
) -> np.ndarray:
    """Compute per-residue RCI S^2.

    Returns a (N,) float array; NaN at residues with too few observed
    shifts for a reliable estimate.
    """
    n = len(comp_ids)
    nuc_list = list(BACKBONE_NUCLEI)

    # Build random-coil table (per-residue x per-nucleus).
    rc = np.full((n, 6), np.nan, dtype=np.float32)
    for i, aa in enumerate(comp_ids):
        aa_rc = RANDOM_COIL_SHIFTS.get(aa)
        if aa_rc is None:
            continue
        for j, nuc in enumerate(nuc_list):
            if nuc in aa_rc:
                rc[i, j] = aa_rc[nuc]

    # Per-residue, per-nucleus scaled |secondary shift|
    valid = (masks > 0) & ~np.isnan(rc)
    sec = shifts.astype(np.float32) - rc
    # Scale each nucleus column so it contributes comparably.
    scale = np.array([_SCALE_PPM[nuc] for nuc in nuc_list], dtype=np.float32)
    scaled_abs = np.where(valid, np.abs(sec) / scale, np.nan)

    # 3-residue smoothing window (Berjanskii 2005): average scaled |Δδ|
    # over i-1, i, i+1 (when available).
    smoothed = np.full_like(scaled_abs, np.nan)
    for j in range(6):
        col = scaled_abs[:, j]
        mask_col = ~np.isnan(col)
        for i in range(n):
            window = []
            for k in (i - 1, i, i + 1):
                if 0 <= k < n and mask_col[k]:
                    window.append(col[k])
            if window:
                smoothed[i, j] = np.mean(window)

    # Weighted mean across nuclei, normalized by observed weights.
    weights = np.array([_WEIGHTS[nuc] for nuc in nuc_list], dtype=np.float32)
    rci = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        available = ~np.isnan(smoothed[i])
        if available.sum() < 3:
            continue
        w = weights[available]
        vals = smoothed[i, available]
        rci[i] = float(np.sum(w * vals) / max(np.sum(w), 1e-6))

    # S^2 from RCI via smooth saturating mapping. Structured residues
    # have RCI ~0.5-0.8 → S^2 ~0.75-0.93. Disordered residues have RCI
    # ~0.05-0.15 → S^2 ~0.10-0.30. tanh gives the right saturation
    # behaviour without hard cutoffs.
    s2 = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        if np.isnan(rci[i]):
            continue
        s2[i] = float(math.tanh(2.0 * rci[i]))

    return s2
