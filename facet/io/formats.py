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

# Confidence classes (TALOS-N-compatible naming)
CONF_STRONG = "Strong"
CONF_GOOD = "Good"
CONF_WARN = "Warn"
CONF_DYNAMIC = "Dynamic"


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
    sequence: str = ""  # optional one-letter sequence (for display)
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
    confidence_class: str  # Strong / Good / Warn / Dynamic
    ss: str  # H / E / C
    chi1: int | None = None  # 0=g+, 1=g-, 2=trans, None=undefined
    phi_err: float = 0.0  # estimated error bound (degrees)
    psi_err: float = 0.0


@dataclass
class FACETResult:
    """Full FACET prediction for a protein."""

    residues: list[ResiduePrediction]
    source: str = ""

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def strong(self) -> list[ResiduePrediction]:
        """Residues with Strong confidence."""
        return [r for r in self.residues if r.confidence_class == CONF_STRONG]

    def accepted(self) -> list[ResiduePrediction]:
        """Residues with Strong or Good confidence (use for restraints)."""
        return [r for r in self.residues
                if r.confidence_class in (CONF_STRONG, CONF_GOOD)]

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
        """Write TALOS-N-style pred.tab summary."""
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
        """Print a TALOS-N-style summary to string."""
        lines = [
            f"FACET prediction: {self.n_residues} residues from {self.source}",
            "",
            f"  {'ResID':>5s} {'AA':>4s} {'PHI':>8s} {'PSI':>8s} {'dPHI':>6s} {'dPSI':>6s} {'SS':>3s} {'Class':>8s}",
        ]
        for r in self.residues:
            lines.append(
                f"  {r.seq_id:5d} {r.comp_id:>4s} {r.phi:8.1f} {r.psi:8.1f} "
                f"{r.phi_err:6.1f} {r.psi_err:6.1f} {r.ss:>3s} {r.confidence_class:>8s}"
            )
        n_strong = len(self.strong())
        n_accepted = len(self.accepted())
        lines.append("")
        lines.append(f"  Strong: {n_strong} ({100*n_strong/max(self.n_residues,1):.0f}%)  "
                      f"Accepted: {n_accepted} ({100*n_accepted/max(self.n_residues,1):.0f}%)")
        return "\n".join(lines)
