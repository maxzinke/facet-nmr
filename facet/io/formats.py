"""Data classes for shift lists and FACET predictions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Backbone nuclei in canonical order (matches model input columns)
BACKBONE_NUCLEI: tuple[str, ...] = ("H", "HA", "N", "CA", "CB", "C")

# 20 canonical amino acids (index 0..19; matches model AA embedding)
CANONICAL_AA: tuple[str, ...] = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
AA_ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}
AA_THREE_TO_ONE = {v: k for k, v in AA_ONE_TO_THREE.items()}

# FACET confidence tiers — single unified vocabulary across the model.
# Calibration source differs by mode:
#   * Retrieval on (default): derived from DBSCAN cluster agreement among
#     the top-k retrieved neighbors.
#   * Retrieval off: derived from the entropy of the coarse Ramachandran head.
#
# Calibrated numbers on the 745-protein clean benchmark (retrieval mode):
#   High     — 50.5% of residues, 9.6 deg median, 14.5% fail25  (restraint-quality)
#   Medium   — 47.5% of residues, 16.5 deg median, 38.2% fail25 (cautious use)
#   Low      —  1.7% of residues, 73.8 deg median, 76.5% fail25 (multi-modal)
#   Flexible —  0.3% of residues                                (disordered / no cluster)
#
# Flexible is not a failure state — it flags residues where neighbor shifts
# disagree, i.e. the chemical shifts are consistent with conformational
# averaging rather than rigid geometry.
CONF_HIGH = "High"
CONF_MEDIUM = "Medium"
CONF_LOW = "Low"
CONF_FLEXIBLE = "Flexible"


@dataclass
class Residue:
    """One residue's chemical shifts."""

    seq_id: int
    comp_id: str  # 3-letter AA code
    shifts: dict[str, float] = field(default_factory=dict)  # nucleus → ppm


@dataclass
class ShiftList:
    """A protein's backbone chemical shift assignments.

    This is the primary input to FACET. Constructed by IO readers or
    directly by the user.
    """

    residues: list[Residue]
    sequence: str = ""  # one-letter sequence covering the assigned range plus any
                        # unassigned flanking/interior residues known to the reader.
                        # Unknown positions within the range use "X".
    seq_id_start: int = 1  # seq_id that sequence[0] corresponds to
    source: str = ""  # filename or BMRB ID

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        """Convert to (shifts, masks, comp_ids, seq_ids) arrays for the model.

        Returns:
            shifts: (N, 6) float32 — raw ppm, NaN for missing
            masks: (N, 6) float32 — 1.0 if observed, 0.0 if missing
            comp_ids: list of 3-letter AA codes
            seq_ids: list of sequence IDs (ints)
        """
        n = len(self.residues)
        shifts = np.full((n, 6), np.nan, dtype=np.float32)
        masks = np.zeros((n, 6), dtype=np.float32)
        comp_ids: list[str] = []
        seq_ids: list[int] = []

        for i, res in enumerate(self.residues):
            comp_ids.append(res.comp_id)
            seq_ids.append(res.seq_id)
            for j, nuc in enumerate(BACKBONE_NUCLEI):
                if nuc in res.shifts:
                    shifts[i, j] = res.shifts[nuc]
                    masks[i, j] = 1.0

        return shifts, masks, comp_ids, seq_ids


@dataclass
class ResiduePrediction:
    """FACET prediction for one residue."""

    seq_id: int
    comp_id: str  # 3-letter
    phi: float  # degrees
    psi: float  # degrees
    confidence: float  # higher = more confident (negative coarse entropy or retrieval-mass)
    confidence_class: str  # High / Medium / Low / Flexible
    ss: str  # H / E / C
    chi1: int | None = None  # 0=g+, 1=g-, 2=trans, None=undefined
    phi_err: float = 0.0  # 1-sigma circular std of the retrieval cluster in
                          # degrees (tier-bound fallback when retrieval off).
    psi_err: float = 0.0  # 1-sigma; restraint writers emit 2*err as the
                          # half-width for ~95% coverage.
    chi1_probs: tuple[float, float, float] | None = None  # softmax over (g+, g-, t)

    # ── Retrieval-augmented output (v0.2+) ──
    # Only populated when predict(..., use_retrieval=True). Absent fields
    # default to None so existing downstream consumers keep working.
    basin_populations: tuple[float, float, float, float] | None = None
    # [alpha_R, beta, PPII, other] fractions summing to 1. For flex residues
    # spectroscopists get a proper distribution, not a single-point fiction.
    alt_clusters: list[tuple[float, float, float]] | None = None
    # For residues with multi-modal support: list of (phi_deg, psi_deg, weight)
    # for alternative clusters beyond the primary one. Empty list when
    # prediction is single-modal.


@dataclass
class FACETResult:
    """Full FACET prediction for a protein."""

    residues: list[ResiduePrediction]
    source: str = ""
    sequence: str = ""  # full one-letter sequence as reported by the reader, or
                        # empty if the source format doesn't carry it.
    seq_id_start: int = 1  # seq_id that sequence[0] corresponds to
    # Retrieval-index provenance (v0.2+). Populated when retrieval is used.
    index_version: str = ""
    index_n_residues: int = 0
    # Referencing sanity-check output (v0.2.2+). List of one-line warnings
    # plus a short summary string for status displays. Empty list means
    # the input passed the check.
    referencing_warnings: list[str] = field(default_factory=list)
    referencing_summary: str = ""

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def high(self) -> list[ResiduePrediction]:
        """Residues in the High tier — restraint-quality.

        In retrieval mode: 9.6 deg median / 14.5% fail25 / ~50% of residues.
        """
        return [r for r in self.residues if r.confidence_class == CONF_HIGH]

    def accepted(self, include_medium: bool = False) -> list[ResiduePrediction]:
        """Residues to include in restraint files.

        Default: High tier only (9.6 deg / 14.5% fail25). This is the
        restraint-quality set — directly usable for XPLOR / CYANA / HADDOCK.

        ``include_medium=True`` adds the Medium tier (16.5 deg / 38% fail25),
        bringing coverage to ~98% of residues. Use with caution — Medium-tier
        residues fail more often and widen the restraint bounds.
        """
        keep = {CONF_HIGH}
        if include_medium:
            keep.add(CONF_MEDIUM)
        return [r for r in self.residues if r.confidence_class in keep]

    def flexible(self) -> list[ResiduePrediction]:
        """Residues flagged as flexible / disordered (Low + Flexible tiers).

        These should be excluded from structure-calculation restraints, but
        the flexible label is itself a biologically meaningful signal.
        """
        return [r for r in self.residues
                if r.confidence_class in (CONF_LOW, CONF_FLEXIBLE)]

    def to_tbl(self, path: str, **kw) -> None:
        """Write XPLOR/CNS .tbl restraints."""
        from .writers import write_tbl
        write_tbl(self, path, **kw)

    def to_aco(self, path: str, **kw) -> None:
        """Write CYANA .aco angle constraints."""
        from .writers import write_aco
        write_aco(self, path, **kw)

    def to_nef(self, path: str, **kw) -> None:
        """Write NEF dihedral restraints."""
        from .writers import write_nef
        write_nef(self, path, **kw)

    def to_predtab(self, path: str, **kw) -> None:
        """Write pred.tab summary table."""
        from .writers import write_predtab
        write_predtab(self, path, **kw)

    def to_csv(self, path: str, **kw) -> None:
        """Write CSV."""
        from .writers import write_csv
        write_csv(self, path, **kw)

    def to_json(self, path: str, **kw) -> None:
        """Write JSON."""
        from .writers import write_json
        write_json(self, path, **kw)

    def summary(self) -> str:
        """Format a per-residue prediction summary as a plain-text table."""
        CHI1_NAMES = {0: "g+", 1: "g-", 2: "t"}
        header = f"FACET prediction: {self.n_residues} residues from {self.source}"
        if self.index_version:
            header += (
                f" | retrieval index: {self.index_version} "
                f"({self.index_n_residues:,} residues)"
            )
        lines = [
            header,
            "",
            f"  {'ResID':>5s} {'AA':>4s} {'PHI':>8s} {'PSI':>8s} {'dPHI':>6s} {'dPSI':>6s} {'SS':>3s} {'CHI1':>5s} {'Tier':>9s}",
        ]
        for r in self.residues:
            chi1_str = CHI1_NAMES.get(r.chi1, ".") if r.chi1 is not None else "."
            lines.append(
                f"  {r.seq_id:5d} {r.comp_id:>4s} {r.phi:8.1f} {r.psi:8.1f} "
                f"{r.phi_err:6.1f} {r.psi_err:6.1f} {r.ss:>3s} {chi1_str:>5s} {r.confidence_class:>9s}"
            )
        n = max(self.n_residues, 1)
        from collections import Counter
        counts = Counter(r.confidence_class for r in self.residues)
        lines.append("")
        lines.append(
            f"  High: {counts.get(CONF_HIGH,0)} ({100*counts.get(CONF_HIGH,0)/n:.0f}%)  "
            f"Medium: {counts.get(CONF_MEDIUM,0)} ({100*counts.get(CONF_MEDIUM,0)/n:.0f}%)  "
            f"Low: {counts.get(CONF_LOW,0)} ({100*counts.get(CONF_LOW,0)/n:.0f}%)  "
            f"Flexible: {counts.get(CONF_FLEXIBLE,0)} ({100*counts.get(CONF_FLEXIBLE,0)/n:.0f}%)"
        )
        return "\n".join(lines)
