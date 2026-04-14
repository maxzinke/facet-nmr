"""Deuterium isotope shift correction for 13C backbone shifts.

Perdeuterated proteins (common for large systems studied via TROSY)
exhibit systematic upfield shifts on 13C nuclei adjacent to deuterium.
This module applies an analytical correction that converts observed
shifts back to their protonated-equivalent values, which is what
FACET was trained on.

Reference coefficients: Venters et al. (1996) JACS 118, 12000;
Hansen (1988). The one-bond effect is ~-0.30 ppm per bonded 2H and
the two-bond effect is ~-0.10 ppm per 2-bond 2H neighbour.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeuterationVector:
    """Three-component deuteration description for a sample.

    Each component is the fraction of that hydrogen class replaced by
    deuterium, in [0, 1]. A fully protonated sample is (0, 0, 0); a
    perdeuterated sample with amide back-exchange (standard for NMR of
    large proteins) is approximately (0, 0.97, 0.97).
    """

    frac_amide: float = 0.0       # HN -> DN fraction
    frac_alpha: float = 0.0       # HA -> DA fraction
    frac_sidechain: float = 0.0   # sidechain H -> D fraction


# Well-known presets covering the common labelling schemes.
PRESETS: dict[str, DeuterationVector] = {
    "protonated":          DeuterationVector(0.0, 0.0, 0.0),
    "perdeut_exchanged":   DeuterationVector(0.0, 0.97, 0.97),
    "perdeut_unexchanged": DeuterationVector(0.97, 0.97, 0.97),
    "ilv_methyl":          DeuterationVector(0.0, 0.97, 0.70),
}


def resolve_deuteration(
    value: "DeuterationVector | str | tuple | list | None",
) -> DeuterationVector:
    """Coerce a user-supplied deuteration specifier to a DeuterationVector.

    Accepts:
        - None                    → protonated (zero vector)
        - DeuterationVector       → returned as-is
        - preset name (str)       → one of ``PRESETS``; dashes and
                                    underscores are interchangeable
        - 3-tuple/list of floats  → ``(frac_amide, frac_alpha, frac_sidechain)``
    """
    if value is None:
        return PRESETS["protonated"]
    if isinstance(value, DeuterationVector):
        return value
    if isinstance(value, str):
        key = value.replace("-", "_").lower()
        if key not in PRESETS:
            raise ValueError(
                f"Unknown deuteration preset: {value!r}. "
                f"Valid: {sorted(PRESETS.keys())}"
            )
        return PRESETS[key]
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return DeuterationVector(
            float(value[0]), float(value[1]), float(value[2])
        )
    raise TypeError(
        f"Cannot interpret deuteration={value!r} "
        f"(type {type(value).__name__}). Expected None, str preset, "
        f"DeuterationVector, or 3-tuple."
    )


# ──────────────────────────────────────────────────────────────────
# Analytical isotope correction coefficients
# ──────────────────────────────────────────────────────────────────

_ONE_BOND_ISOTOPE_EFFECT = -0.30  # ppm per bonded 2H (upfield)
_TWO_BOND_ISOTOPE_EFFECT = -0.10  # ppm per 2-bond 2H neighbour

# Directly bonded H atoms per CA (always 1 except GLY which has 2).
_CA_BONDED_H: dict[str, int] = {aa: 1 for aa in "ACDEFHIKLMNPQRSTVWY"}
_CA_BONDED_H["G"] = 2

# Directly bonded H atoms on CB, by AA one-letter code.
_CB_BONDED_H: dict[str, int] = {
    "A": 3,  # CH3
    "C": 2, "D": 2, "E": 2, "F": 2, "H": 2,
    "K": 2, "L": 2, "M": 2, "N": 2, "Q": 2,
    "P": 2, "R": 2, "S": 2, "W": 2, "Y": 2,
    "I": 1, "T": 1, "V": 1,  # CH
    "G": 0,  # no CB
}

_CA_TWO_BOND_FROM_AMIDE = 1  # CA sees one 2-bond amide H
_C_TWO_BOND_FROM_ALPHA = 1   # C' sees one 2-bond alpha H


def compute_isotope_correction(
    aa_1letter: str,
    deut: DeuterationVector,
) -> dict[str, float]:
    """Per-residue 13C isotope correction for CA, CB, and C'.

    Returns the correction to ADD to observed shifts to recover the
    protonated-equivalent values that FACET was trained on.
    """
    corrections: dict[str, float] = {}

    n_ca_h = _CA_BONDED_H.get(aa_1letter, 1)
    ca_1bond = n_ca_h * deut.frac_alpha * _ONE_BOND_ISOTOPE_EFFECT
    ca_2bond = _CA_TWO_BOND_FROM_AMIDE * deut.frac_amide * _TWO_BOND_ISOTOPE_EFFECT
    corrections["CA"] = -(ca_1bond + ca_2bond)

    n_cb_h = _CB_BONDED_H.get(aa_1letter, 0)
    if n_cb_h > 0:
        cb_1bond = n_cb_h * deut.frac_sidechain * _ONE_BOND_ISOTOPE_EFFECT
        cb_2bond = 1 * deut.frac_alpha * _TWO_BOND_ISOTOPE_EFFECT
        corrections["CB"] = -(cb_1bond + cb_2bond)
    else:
        corrections["CB"] = 0.0

    c_2bond = _C_TWO_BOND_FROM_ALPHA * deut.frac_alpha * _TWO_BOND_ISOTOPE_EFFECT
    corrections["C"] = -(c_2bond)

    return corrections
