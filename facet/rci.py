"""Random Coil Index (RCI) and derived per-residue S^2 order parameter.

Faithful implementation of Berjanskii & Wishart (2008) J. Biomol. NMR
40, 31 (doi:10.1007/s10858-007-9208-0), with the simplifications noted
below.

Physical meaning: S^2 is a model-free order parameter in [0, 1]. A rigid
residue in a folded element has S^2 ~ 0.85; a flexible loop ~ 0.65; a
disordered IDR residue <0.5. Computed here directly from backbone
chemical shifts without NMR relaxation data.

Algorithm (faithful to Berjanskii-Wishart 2008 Eq. 2 and Eq. 3):

  1. Secondary shifts per residue: Δδ(nucleus) = observed - random_coil.
  2. 3-residue smoothing of |Δδ|: average over i-1, i, i+1.
  3. Weighted sum across observed nuclei with the all-6 weights from
     the paper's Eq. 2:
         0.74 Cα, 0.72 C', 0.13 Cβ, 0.38 N, 0.15 NH, 0.91 Hα
     (from Supplemental Table 1, optimised against the 33-protein
     training set for best correlation with MD RMSF and experimental
     order parameters, r = 0.81).
  4. RCI = 1 / (weighted_sum * 6) — the inverse of the weighted
     average. Higher Δδ magnitudes → smaller RCI → more rigid.
  5. S^2 = 1 - 0.4 * ln(1 + 17.7 * RCI) — Eq. 3 from the paper.

Simplifications vs. the paper:
  - We do NOT implement the per-combination weight optimisation from
    SI Table 1 (weights adjust when some nuclei are missing). We always
    use the all-6 weights; missing nuclei contribute 0 to the weighted
    sum. On residues with incomplete data this biases RCI slightly
    upward (S^2 slightly downward).
  - We do NOT apply sequential i±1 neighbour corrections, chemical-
    shift re-referencing (REFCOR), or the end-effect correction.
    Re-referencing is handled separately by
    ``facet.referencing.check_referencing`` if the user enables
    ``auto_reference`` in ``predict()``.

These simplifications keep the implementation small and stable. For
S^2 values calibrated to the Berjanskii-Wishart training set exactly,
users should go to the original RCI web server at
http://wishart.biology.ualberta.ca/rci .
"""
from __future__ import annotations

import math

import numpy as np

from .io.formats import BACKBONE_NUCLEI
from .random_coil import RANDOM_COIL_SHIFTS


# Berjanskii-Wishart 2008 Eq. 2 weights for the all-6-nuclei case.
# Paper names → FACET names:
#   Cα    → "CA"
#   C'/CO → "C"
#   Cβ    → "CB"
#   N     → "N"
#   NH    → "H"   (amide proton)
#   Hα    → "HA"  (alpha proton)
_WEIGHTS_BW2008 = {
    "CA": 0.74,
    "C":  0.72,
    "CB": 0.13,
    "N":  0.38,
    "H":  0.15,
    "HA": 0.91,
}

# Normalising factor in Eq. 2 — always 6, independent of how many
# nuclei are actually observed at a given residue.
_N_NUCLEI = 6

# Eq. 3 conversion constants.
_S2_A = 0.4
_S2_B = 17.7


def compute_rci_s2(
    shifts: np.ndarray,   # (N, 6), ppm, NaN for missing
    masks: np.ndarray,    # (N, 6), 1=observed
    comp_ids: list[str],  # 3-letter AA codes
) -> np.ndarray:
    """Compute per-residue RCI S^2 following Berjanskii-Wishart 2008.

    Returns a (N,) float array of S^2 values in [0, 1]; NaN at residues
    with fewer than 3 observed nuclei (insufficient for a stable RCI).
    """
    n = len(comp_ids)
    nuc_list = list(BACKBONE_NUCLEI)   # ("H", "HA", "N", "CA", "CB", "C")

    # Build random-coil table (per-residue × per-nucleus).
    rc = np.full((n, 6), np.nan, dtype=np.float32)
    for i, aa in enumerate(comp_ids):
        aa_rc = RANDOM_COIL_SHIFTS.get(aa)
        if aa_rc is None:
            continue
        for j, nuc in enumerate(nuc_list):
            if nuc in aa_rc:
                rc[i, j] = aa_rc[nuc]

    # |secondary shift| per residue/nucleus; NaN where unobserved or
    # no RC reference for that AA.
    valid = (masks > 0) & ~np.isnan(rc)
    sec = shifts.astype(np.float32) - rc
    abs_sec = np.where(valid, np.abs(sec), np.nan)

    # 3-residue smoothing: average |Δδ| over i-1, i, i+1 (only observed
    # neighbours count). This is the smoothing step from the paper's
    # protocol section.
    smoothed = np.full_like(abs_sec, np.nan)
    for j in range(6):
        col = abs_sec[:, j]
        mask_col = ~np.isnan(col)
        for i in range(n):
            window = []
            for k in (i - 1, i, i + 1):
                if 0 <= k < n and mask_col[k]:
                    window.append(col[k])
            if window:
                smoothed[i, j] = float(np.mean(window))

    # Weighted sum per residue using the all-6 weights. Missing nuclei
    # contribute 0 (which biases RCI slightly upward — see module
    # docstring for the simplification note).
    weights = np.array([_WEIGHTS_BW2008[nuc] for nuc in nuc_list],
                       dtype=np.float32)

    s2 = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        observed_this_residue = ~np.isnan(smoothed[i])
        if observed_this_residue.sum() < 3:
            # Too few nuclei for a stable RCI estimate.
            continue

        # Zero out missing nuclei, compute weighted sum.
        row = np.where(observed_this_residue, smoothed[i], 0.0)
        weighted_sum = float(np.sum(weights * row))
        if weighted_sum <= 0.0:
            # All |Δδ| are exactly zero — extreme random-coil limit.
            # RCI saturates; S^2 → 0.
            s2[i] = 0.0
            continue

        # Eq. 2: RCI = 1 / (weighted_sum * 6)
        rci_val = 1.0 / (weighted_sum * _N_NUCLEI)

        # Eq. 3: S^2 = 1 - 0.4 * ln(1 + 17.7 * RCI)
        s2_val = 1.0 - _S2_A * math.log(1.0 + _S2_B * rci_val)
        s2[i] = float(np.clip(s2_val, 0.0, 1.0))

    return s2
