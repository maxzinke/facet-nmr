"""Shift referencing sanity check.

Flags inputs whose per-nucleus mean secondary shift deviates from what
would be expected given the protein's apparent secondary-structure
composition. A systematic offset is almost always a referencing error
(DSS vs TMS mismatch, wrong carrier frequency, copy-paste mistake in
the assignment table) rather than a real structural signal.

Algorithm (composition-adaptive):
  1. Per-residue secondary shift = observed - random-coil[AA].
  2. Robust SS estimate from the CA-CB CSI difference: CA_sec - CB_sec.
     Helix residues have CSI >> 0, strand residues CSI << 0. The
     difference is invariant under a constant 13C offset applied to
     both CA and CB together, so the SS estimate is not biased by the
     very referencing error we're trying to detect.
  3. Expected mean per nucleus = composition-weighted sum of per-SS
     canonical secondary shifts (Wishart et al. 1991 / Wang & Jardetzky
     2002).
  4. Warn when |observed_mean - expected_mean| > tolerance.

Tolerances are per-nucleus: tighter on 13C (well-calibrated DSS-
referenced data has |residual| < 0.3 ppm), looser on 15N.

**Limitations:**

The CSI step uses CA_sec - CB_sec to infer SS composition. A systematic
offset applied to *only* CA (e.g. a CA-column assignment-table error)
will push the CSI classifier toward "more helix" and the composition-
adaptive expected CA mean will follow the offset, masking the fault.
This check catches miscalibration on N, HA, H, C', and CB reliably.
CA-only offsets below ~2 ppm may slip through. Users should always
cross-reference with an external tool (LACS: Wang & Markley 2009) for
the most rigorous referencing sanity check before structure-quality
work.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .io.formats import BACKBONE_NUCLEI
from .random_coil import RANDOM_COIL_SHIFTS


# Canonical secondary-shift means per SS class (ppm).
# Sources: Wishart & Sykes 1994, Wang & Jardetzky 2002, CSI literature.
_SS_MEAN_SECONDARY = {
    "CA": {"H": +3.10, "E": -1.50, "C":  0.00},
    "CB": {"H": -0.40, "E": +1.90, "C":  0.00},
    "C":  {"H": +1.80, "E": -1.00, "C":  0.00},
    "N":  {"H": -2.00, "E": +2.00, "C":  0.00},
    "HA": {"H": -0.30, "E": +0.30, "C":  0.00},
    "H":  {"H":  0.00, "E":  0.00, "C":  0.00},
}

# Per-nucleus tolerance for the residual offset (observed - expected).
# Below these thresholds the deviation is dominated by per-residue noise
# and intrinsic dispersion; above, systematic miscalibration is the
# most likely cause.
#
# Tolerances are wide enough to accommodate natural composition variation
# (an all-helix protein's mean CA secondary shift sits ~2.3 ppm above a
# typical-composition baseline). The composition-adaptive step below
# already subtracts the expected shift for the inferred H/E/C mix, so
# these are residual tolerances on top of that correction.
_TOLERANCE_PPM = {
    "CA": 0.8,
    "CB": 0.8,
    "C":  0.6,
    "N":  1.5,   # 15N: noisier, wider dispersion
    # 1H RC values vary by ~0.1 ppm between Wishart 1995 / Schwarzinger
    # 2001 / BMRB consensus. Use a margin above that so we don't false-
    # alarm on correctly-referenced samples that used a slightly
    # different RC table than ours.
    "HA": 0.35,
    "H":  0.35,
}

# CSI thresholds on CA_sec - CB_sec for tentative SS classification.
_CSI_HELIX_THRESHOLD = +2.0  # CSI > +2 → helix
_CSI_STRAND_THRESHOLD = -2.0  # CSI < −2 → strand


@dataclass
class ReferencingReport:
    """Result of the referencing sanity check.

    Fields:
      composition:    {"H", "E", "C"} → fraction (0..1) of residues in
                      each class estimated from the CSI (composition-
                      adaptive baseline for the check).
      observed_means: per-nucleus mean secondary shift (ppm).
      expected_means: per-nucleus expected mean given the composition.
      offsets:        observed - expected, per nucleus.
      warnings:       list of plain-English messages, one per out-of-
                      tolerance nucleus, each including a suggested
                      additive correction.
    """

    composition: dict[str, float] = field(default_factory=dict)
    observed_means: dict[str, float] = field(default_factory=dict)
    expected_means: dict[str, float] = field(default_factory=dict)
    offsets: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    n_residues: int = 0
    n_csi_classified: int = 0

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    def summary(self) -> str:
        """Short human-readable summary for logs / UI."""
        if not self.offsets:
            return "Referencing check: insufficient data."
        parts = []
        for nuc in ("H", "HA", "N", "CA", "CB", "C"):
            if nuc not in self.offsets:
                continue
            off = self.offsets[nuc]
            tol = _TOLERANCE_PPM.get(nuc, 0.5)
            flag = "!" if abs(off) > tol else "."
            parts.append(f"{nuc}{flag}{off:+.2f}")
        comp_str = (
            f"H{100*self.composition.get('H', 0):.0f}/"
            f"E{100*self.composition.get('E', 0):.0f}/"
            f"C{100*self.composition.get('C', 0):.0f}"
        )
        return f"Referencing (obs-exp, ppm) [{comp_str}]: " + " ".join(parts)


def check_referencing(
    shifts: np.ndarray,   # (N, 6), ppm, NaN for missing
    masks: np.ndarray,    # (N, 6), 1=observed
    comp_ids: list[str],  # 3-letter AA codes
) -> ReferencingReport:
    """Composition-adaptive referencing sanity check.

    Returns a ReferencingReport. Always returns successfully; absence of
    warnings means the input looks well-calibrated. Callers should pass
    ``report.warnings`` to the user and surface ``report.summary()`` in
    status displays.
    """
    n = len(comp_ids)
    report = ReferencingReport(n_residues=n)
    nuc_list = list(BACKBONE_NUCLEI)   # ("H", "HA", "N", "CA", "CB", "C")
    col = {nuc: j for j, nuc in enumerate(nuc_list)}

    # Composition estimation needs enough data for stable statistics.
    # On very small proteins (< 20 residues) the observed means have
    # high variance and would false-alarm on correctly-referenced data.
    # Skip the check rather than mislead the user.
    MIN_RESIDUES_FOR_CHECK = 20
    if n < MIN_RESIDUES_FOR_CHECK:
        return report

    # Build per-residue random-coil table.
    rc = np.full((n, 6), np.nan, dtype=np.float32)
    for i, aa in enumerate(comp_ids):
        aa_rc = RANDOM_COIL_SHIFTS.get(aa)
        if aa_rc is None:
            continue
        for j, nuc in enumerate(nuc_list):
            if nuc in aa_rc:
                rc[i, j] = aa_rc[nuc]

    # Per-residue secondary shifts — valid only where both the shift
    # is observed (mask) and an RC value exists for that AA + nucleus.
    valid = (masks > 0) & ~np.isnan(rc)
    sec = shifts.astype(np.float32) - rc

    # ── Step 1: CSI-based tentative SS classification ──────────────
    has_ca_cb = valid[:, col["CA"]] & valid[:, col["CB"]]
    csi = sec[:, col["CA"]] - sec[:, col["CB"]]
    csi_valid = csi[has_ca_cb]
    n_csi = len(csi_valid)
    if n_csi >= 5:
        n_helix = int((csi_valid > _CSI_HELIX_THRESHOLD).sum())
        n_strand = int((csi_valid < _CSI_STRAND_THRESHOLD).sum())
        n_coil = n_csi - n_helix - n_strand
        frac_h = n_helix / n_csi
        frac_e = n_strand / n_csi
        frac_c = n_coil / n_csi
    else:
        # Not enough CA+CB data — fall back to a typical composition.
        # Values chosen to be mildly H-biased since ~35/20/45 is the
        # average across PDB.
        frac_h, frac_e, frac_c = 0.35, 0.20, 0.45

    report.composition = {"H": frac_h, "E": frac_e, "C": frac_c}
    report.n_csi_classified = n_csi

    # ── Step 2: per-nucleus observed vs expected mean ──────────────
    # Per-nucleus minimum sample: we need enough observed residues for
    # the mean to be a stable estimate of the population mean. With 20
    # observations and expected per-residue noise of ~1 ppm on 13C, the
    # standard error on the mean is ~0.22 ppm — small enough to
    # distinguish a 0.4-ppm systematic offset from noise.
    MIN_OBS_PER_NUCLEUS = 20
    for nuc in nuc_list:
        j = col[nuc]
        col_mask = valid[:, j]
        n_obs = int(col_mask.sum())
        if n_obs < MIN_OBS_PER_NUCLEUS:
            continue
        observed = float(sec[col_mask, j].mean())
        ss_means = _SS_MEAN_SECONDARY.get(nuc, {"H": 0.0, "E": 0.0, "C": 0.0})
        expected = (
            frac_h * ss_means["H"]
            + frac_e * ss_means["E"]
            + frac_c * ss_means["C"]
        )
        offset = observed - expected
        report.observed_means[nuc] = observed
        report.expected_means[nuc] = expected
        report.offsets[nuc] = offset

        tol = _TOLERANCE_PPM.get(nuc, 0.5)
        if abs(offset) > tol:
            # Phrase correction direction clearly: if observed is TOO
            # HIGH, user should subtract; if TOO LOW, add.
            if offset > 0:
                action = f"subtract {offset:+.2f} ppm from every {nuc} shift"
            else:
                action = f"add {abs(offset):+.2f} ppm to every {nuc} shift"
            report.warnings.append(
                f"{nuc}: mean secondary shift {observed:+.2f} ppm (expected "
                f"{expected:+.2f} for this protein's apparent composition). "
                f"Off by {offset:+.2f} ppm - likely referencing error. "
                f"Suggested fix: {action}."
            )

    return report
