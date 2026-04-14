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

# Confidence classes — independently calibrated from the v3 risk-coverage curve.
#   High:     confident rigid geometry — use for restraints
#   Medium:   moderate confidence — use cautiously
#   Low:      uncertain — interpret with care
#   Flexible: biologically flexible / disordered region (NOT a failure)
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
    confidence: float  # higher = more confident (negative coarse entropy)
    confidence_class: str  # High / Medium / Low / Flexible
    ss: str  # H / E / C
    chi1: int | None = None  # 0=g+, 1=g-, 2=trans, None=undefined
    phi_err: float = 0.0  # estimated error bound (degrees)
    psi_err: float = 0.0
    chi1_probs: tuple[float, float, float] | None = None  # softmax over (g+, g-, t)


@dataclass
class FACETResult:
    """Full FACET prediction for a protein."""

    residues: list[ResiduePrediction]
    source: str = ""
    sequence: str = ""  # full one-letter sequence as reported by the reader, or
                        # empty if the source format doesn't carry it.
    seq_id_start: int = 1  # seq_id that sequence[0] corresponds to

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def high(self) -> list[ResiduePrediction]:
        """Residues with High confidence (narrowest restraints, safest use)."""
        return [r for r in self.residues if r.confidence_class == CONF_HIGH]

    def accepted(self) -> list[ResiduePrediction]:
        """Residues with High or Medium confidence (use for restraints)."""
        return [r for r in self.residues
                if r.confidence_class in (CONF_HIGH, CONF_MEDIUM)]

    def flexible(self) -> list[ResiduePrediction]:
        """Residues flagged as flexible / disordered."""
        return [r for r in self.residues if r.confidence_class == CONF_FLEXIBLE]

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
        lines = [
            f"FACET prediction: {self.n_residues} residues from {self.source}",
            "",
            f"  {'ResID':>5s} {'AA':>4s} {'PHI':>8s} {'PSI':>8s} {'dPHI':>6s} {'dPSI':>6s} {'SS':>3s} {'CHI1':>5s} {'Class':>8s}",
        ]
        for r in self.residues:
            chi1_str = CHI1_NAMES.get(r.chi1, ".") if r.chi1 is not None else "."
            lines.append(
                f"  {r.seq_id:5d} {r.comp_id:>4s} {r.phi:8.1f} {r.psi:8.1f} "
                f"{r.phi_err:6.1f} {r.psi_err:6.1f} {r.ss:>3s} {chi1_str:>5s} {r.confidence_class:>8s}"
            )
        n_high = len(self.high())
        n_accepted = len(self.accepted())
        n_flexible = len(self.flexible())
        lines.append("")
        lines.append(
            f"  High: {n_high} ({100*n_high/max(self.n_residues,1):.0f}%)  "
            f"Accepted: {n_accepted} ({100*n_accepted/max(self.n_residues,1):.0f}%)  "
            f"Flexible: {n_flexible} ({100*n_flexible/max(self.n_residues,1):.0f}%)"
        )
        return "\n".join(lines)
