"""Output format writers for FACET predictions.

Supported formats:
  - XPLOR/CNS .tbl   (XPLOR-NIH, CNS, HADDOCK, ARIA)
  - CYANA .aco        (CYANA angle constraints)
  - NEF .nef          (wwPDB NMR Exchange Format dihedral restraints)
  - pred.tab          (TALOS-N-style summary table)
  - CSV               (simple comma-separated)
  - JSON              (machine-readable)

All writers take a FACETResult and an output path. Only residues with
Strong or Good confidence are included in restraint files (.tbl, .aco,
.nef) by default. Summary formats (pred.tab, CSV, JSON) include all
residues.
"""
from __future__ import annotations

import json as json_mod
from pathlib import Path

from .formats import (
    CONF_GOOD,
    CONF_STRONG,
    FACETResult,
    ResiduePrediction,
)

# ─────────────────────── XPLOR/CNS .tbl ────────────────────────

def write_tbl(
    result: FACETResult,
    path: str | Path,
    *,
    accepted_only: bool = True,
    chain: str = "A",
) -> Path:
    """Write XPLOR/CNS dihedral restraint file.

    Format (one restraint per line)::

        assign (resid 5 and name C)  (resid 6 and name N)
               (resid 6 and name CA) (resid 6 and name C)  1.0 -64.8 15.0
        ! PHI for residue 6 ALA (Strong, FACET)

    The four atoms define the dihedral: C(i-1)-N(i)-CA(i)-C(i) for phi,
    N(i)-CA(i)-C(i)-N(i+1) for psi. Error bounds from FACET confidence.

    Terminal residues are handled correctly: PHI omitted when residue
    (i-1) is not in the shift list (N-terminus or gap), PSI omitted when
    residue (i+1) is not in the shift list (C-terminus or gap).
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    residues = result.accepted() if accepted_only else result.residues
    present_sids = {r.seq_id for r in result.residues}

    lines = [f"! FACET dihedral restraints — {len(residues)} residues\n"]
    n_phi_skip = 0
    n_psi_skip = 0

    for r in residues:
        sid = r.seq_id

        # PHI: C(i-1) - N(i) - CA(i) - C(i)  — needs (i-1) present
        if (sid - 1) in present_sids:
            lines.append(
                f"assign (resid {sid - 1} and name C)  (resid {sid} and name N)\n"
                f"       (resid {sid} and name CA) (resid {sid} and name C)"
                f"  1.0 {r.phi:7.1f} {r.phi_err:5.1f}"
            )
            lines.append(f"! PHI {sid} {r.comp_id} ({r.confidence_class})")
        else:
            n_phi_skip += 1

        # PSI: N(i) - CA(i) - C(i) - N(i+1)  — needs (i+1) present
        if (sid + 1) in present_sids:
            lines.append(
                f"assign (resid {sid} and name N)  (resid {sid} and name CA)\n"
                f"       (resid {sid} and name C)  (resid {sid + 1} and name N)"
                f"  1.0 {r.psi:7.1f} {r.psi_err:5.1f}"
            )
            lines.append(f"! PSI {sid} {r.comp_id} ({r.confidence_class})")
        else:
            n_psi_skip += 1
        lines.append("")

    if n_phi_skip or n_psi_skip:
        lines.append(f"! Skipped terminal/gap: {n_phi_skip} PHI, {n_psi_skip} PSI")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ────────────────────────── CYANA .aco ──────────────────────────

def write_aco(
    result: FACETResult,
    path: str | Path,
    *,
    accepted_only: bool = True,
) -> Path:
    """Write CYANA angle constraint file.

    Format::

        # FACET torsion angle restraints
          6 ALA  PHI    -79.8   -49.8
          6 ALA  PSI    -60.2   -30.2
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    residues = result.accepted() if accepted_only else result.residues

    lines = ["# FACET torsion angle restraints"]
    lines.append(f"# {len(residues)} residues")
    lines.append("")

    for r in residues:
        phi_lo = r.phi - r.phi_err
        phi_hi = r.phi + r.phi_err
        psi_lo = r.psi - r.psi_err
        psi_hi = r.psi + r.psi_err
        lines.append(f"{r.seq_id:5d} {r.comp_id:4s} PHI  {phi_lo:8.1f} {phi_hi:8.1f}")
        lines.append(f"{r.seq_id:5d} {r.comp_id:4s} PSI  {psi_lo:8.1f} {psi_hi:8.1f}")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ──────────────────────── NEF dihedrals ─────────────────────────

def write_nef(
    result: FACETResult,
    path: str | Path,
    *,
    accepted_only: bool = True,
    chain_code: str = "A",
    program_version: str = "0.1.0",
) -> Path:
    """Write NEF dihedral restraint file.

    Uses the ``nef_dihedral_restraint_list`` saveframe with the
    ``_nef_dihedral_restraint`` loop. Follows NEF 1.1 specification
    and atom naming conventions from docs/NEF_ATOM_NAMING_CONVENTIONS.md.

    Each restraint row defines four atoms (the dihedral) plus target
    angle and bounds. PHI = C(i-1)-N(i)-CA(i)-C(i), PSI = N(i)-CA(i)-C(i)-N(i+1).
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    residues = result.accepted() if accepted_only else result.residues
    present_sids = {r.seq_id for r in result.residues}

    lines: list[str] = []

    # Data block
    lines.append("data_facet_restraints")
    lines.append("")

    # Metadata saveframe
    lines.append("save_nef_nmr_meta_data")
    lines.append("   _nef_nmr_meta_data.sf_category       nef_nmr_meta_data")
    lines.append("   _nef_nmr_meta_data.sf_framecode      nef_nmr_meta_data")
    lines.append("   _nef_nmr_meta_data.format_name       nmr_exchange_format")
    lines.append("   _nef_nmr_meta_data.format_version    1.1")
    lines.append(f"   _nef_nmr_meta_data.program_name      facet")
    lines.append(f"   _nef_nmr_meta_data.program_version   {program_version}")
    lines.append("save_")
    lines.append("")

    # Dihedral restraint saveframe
    lines.append("save_nef_dihedral_restraint_list_1")
    lines.append("   _nef_dihedral_restraint_list.sf_category     nef_dihedral_restraint_list")
    lines.append("   _nef_dihedral_restraint_list.sf_framecode    nef_dihedral_restraint_list_1")
    lines.append("")
    lines.append("   loop_")
    lines.append("      _nef_dihedral_restraint.index")
    lines.append("      _nef_dihedral_restraint.restraint_id")
    lines.append("      _nef_dihedral_restraint.restraint_combination_id")
    lines.append("      _nef_dihedral_restraint.chain_code_1")
    lines.append("      _nef_dihedral_restraint.sequence_code_1")
    lines.append("      _nef_dihedral_restraint.residue_name_1")
    lines.append("      _nef_dihedral_restraint.atom_name_1")
    lines.append("      _nef_dihedral_restraint.chain_code_2")
    lines.append("      _nef_dihedral_restraint.sequence_code_2")
    lines.append("      _nef_dihedral_restraint.residue_name_2")
    lines.append("      _nef_dihedral_restraint.atom_name_2")
    lines.append("      _nef_dihedral_restraint.chain_code_3")
    lines.append("      _nef_dihedral_restraint.sequence_code_3")
    lines.append("      _nef_dihedral_restraint.residue_name_3")
    lines.append("      _nef_dihedral_restraint.atom_name_3")
    lines.append("      _nef_dihedral_restraint.chain_code_4")
    lines.append("      _nef_dihedral_restraint.sequence_code_4")
    lines.append("      _nef_dihedral_restraint.residue_name_4")
    lines.append("      _nef_dihedral_restraint.atom_name_4")
    lines.append("      _nef_dihedral_restraint.name")
    lines.append("      _nef_dihedral_restraint.target_value")
    lines.append("      _nef_dihedral_restraint.lower_limit")
    lines.append("      _nef_dihedral_restraint.upper_limit")
    lines.append("")

    idx = 1
    restraint_id = 1
    for r in residues:
        sid = r.seq_id
        cc = chain_code

        # PHI: C(i-1) - N(i) - CA(i) - C(i) — skip if (i-1) missing
        if (sid - 1) in present_sids:
            phi_lo = r.phi - r.phi_err
            phi_hi = r.phi + r.phi_err
            lines.append(
                f"      {idx} {restraint_id} . "
                f"{cc} {sid - 1} . C "
                f"{cc} {sid} {r.comp_id} N "
                f"{cc} {sid} {r.comp_id} CA "
                f"{cc} {sid} {r.comp_id} C "
                f"PHI {r.phi:.1f} {phi_lo:.1f} {phi_hi:.1f}"
            )
            idx += 1

        # PSI: N(i) - CA(i) - C(i) - N(i+1) — skip if (i+1) missing
        if (sid + 1) in present_sids:
            psi_lo = r.psi - r.psi_err
            psi_hi = r.psi + r.psi_err
            lines.append(
                f"      {idx} {restraint_id} . "
                f"{cc} {sid} {r.comp_id} N "
                f"{cc} {sid} {r.comp_id} CA "
                f"{cc} {sid} {r.comp_id} C "
                f"{cc} {sid + 1} . N "
                f"PSI {r.psi:.1f} {psi_lo:.1f} {psi_hi:.1f}"
            )
            idx += 1
        restraint_id += 1

    lines.append("")
    lines.append("   stop_")
    lines.append("save_")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ─────────────────────── pred.tab (TALOS-N style) ───────────────

def write_predtab(
    result: FACETResult,
    path: str | Path,
) -> Path:
    """Write TALOS-N-style prediction summary.

    Format::

        REMARK FACET backbone torsion angle prediction
        VARS   RESID RESNAME PHI PSI DPHI DPSI SS CHI1 CLASS
        FORMAT %4d %s %8.1f %8.1f %6.1f %6.1f %s %s %s

           1 M   -64.8   -45.2   40.0   40.0 C  .  Warn
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    from .formats import AA_THREE_TO_ONE

    CHI1_NAMES = {0: "g+", 1: "g-", 2: "t"}

    lines = [
        "REMARK FACET backbone torsion angle prediction",
        "REMARK https://github.com/bluegems661/facet-nmr",
        "",
        "VARS   RESID RESNAME PHI PSI DPHI DPSI SS CHI1 CLASS",
        "FORMAT %4d %s %8.1f %8.1f %6.1f %6.1f %s %s %s",
        "",
    ]

    for r in result.residues:
        aa1 = AA_THREE_TO_ONE.get(r.comp_id, "X")
        chi1_str = CHI1_NAMES.get(r.chi1, ".") if r.chi1 is not None else "."
        lines.append(
            f" {r.seq_id:4d} {aa1} {r.phi:8.1f} {r.psi:8.1f} "
            f"{r.phi_err:6.1f} {r.psi_err:6.1f} {r.ss} {chi1_str:>3s} {r.confidence_class}"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ──────────────────────────── CSV ───────────────────────────────

def write_csv(
    result: FACETResult,
    path: str | Path,
) -> Path:
    """Write a simple CSV with all predictions."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    CHI1_NAMES = {0: "g+", 1: "g-", 2: "t"}
    lines = ["ResID,AA,PHI,PSI,dPHI,dPSI,SS,Chi1,Confidence,Class"]
    for r in result.residues:
        chi1_str = CHI1_NAMES.get(r.chi1, "") if r.chi1 is not None else ""
        lines.append(
            f"{r.seq_id},{r.comp_id},{r.phi:.1f},{r.psi:.1f},"
            f"{r.phi_err:.1f},{r.psi_err:.1f},{r.ss},{chi1_str},"
            f"{r.confidence:.3f},{r.confidence_class}"
        )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ──────────────────────────── JSON ──────────────────────────────

def write_json(
    result: FACETResult,
    path: str | Path,
) -> Path:
    """Write all predictions as JSON."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source": result.source,
        "n_residues": result.n_residues,
        "residues": [
            {
                "seq_id": r.seq_id,
                "comp_id": r.comp_id,
                "phi": round(r.phi, 1),
                "psi": round(r.psi, 1),
                "phi_err": round(r.phi_err, 1),
                "psi_err": round(r.psi_err, 1),
                "ss": r.ss,
                "chi1": r.chi1,
                "confidence": round(r.confidence, 3),
                "confidence_class": r.confidence_class,
            }
            for r in result.residues
        ],
    }

    out.write_text(json_mod.dumps(data, indent=2), encoding="utf-8")
    return out
