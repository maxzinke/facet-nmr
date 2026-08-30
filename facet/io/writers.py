"""Output format writers for FACET predictions.

Supported formats:
  - XPLOR/CNS .tbl   (XPLOR-NIH, CNS, HADDOCK, ARIA)
  - CYANA .aco        (CYANA angle constraints)
  - NEF .nef          (wwPDB NMR Exchange Format dihedral restraints)
  - pred.tab          (per-residue summary table)
  - CSV               (simple comma-separated)
  - JSON              (machine-readable)

All writers take a FACETResult and an output path. Only residues with
High confidence are included in restraint files (.tbl, .aco, .nef) by
default; pass ``include_medium=True`` to also include Medium-tier
residues. Summary formats (pred.tab, CSV, JSON) include all residues.
"""
from __future__ import annotations

import json as json_mod
from pathlib import Path

from .. import __version__
from .formats import (
    FACETResult,
)

# ─────────────────────── XPLOR/CNS .tbl ────────────────────────

def write_tbl(
    result: FACETResult,
    path: str | Path,
    *,
    accepted_only: bool = True,
    include_medium: bool = False,
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

    residues = result.accepted(include_medium=include_medium) if accepted_only else result.residues
    present_sids = {r.seq_id for r in result.residues}

    lines = [f"! FACET dihedral restraints — {len(residues)} residues\n"]
    n_phi_skip = 0
    n_psi_skip = 0

    # Restraint half-widths are 2 * (1-sigma per-residue err) so the
    # emitted ± bound covers ~95% of the cluster distribution.
    for r in residues:
        sid = r.seq_id
        phi_bound = 2.0 * r.phi_err
        psi_bound = 2.0 * r.psi_err

        # PHI: C(i-1) - N(i) - CA(i) - C(i)  — needs (i-1) present
        # XPLOR DIHE syntax: <Kf> <target_deg> <range_deg> <exponent>
        if (sid - 1) in present_sids:
            lines.append(
                f"assign (resid {sid - 1} and name C)  (resid {sid} and name N)\n"
                f"       (resid {sid} and name CA) (resid {sid} and name C)"
                f"  1.0 {r.phi:7.1f} {phi_bound:5.1f} 2"
            )
            lines.append(f"! PHI {sid} {r.comp_id} ({r.confidence_class})")
        else:
            n_phi_skip += 1

        # PSI: N(i) - CA(i) - C(i) - N(i+1)  — needs (i+1) present
        if (sid + 1) in present_sids:
            lines.append(
                f"assign (resid {sid} and name N)  (resid {sid} and name CA)\n"
                f"       (resid {sid} and name C)  (resid {sid + 1} and name N)"
                f"  1.0 {r.psi:7.1f} {psi_bound:5.1f} 2"
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
    include_medium: bool = False,
) -> Path:
    """Write CYANA angle constraint file.

    Format::

        # FACET torsion angle restraints
          6 ALA  PHI    -79.8   -49.8
          6 ALA  PSI    -60.2   -30.2
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    residues = result.accepted(include_medium=include_medium) if accepted_only else result.residues

    lines = ["# FACET torsion angle restraints",
             f"# facet-nmr {__version__}"
             + (f" | retrieval index {result.index_version}" if result.index_version else "")]
    lines.append(f"# {len(residues)} residues")
    lines.append("")

    # CYANA .aco expects hard lo/hi bounds per restraint. Emit 2-sigma
    # bounds (matches the .tbl format half-widths).
    for r in residues:
        phi_bound = 2.0 * r.phi_err
        psi_bound = 2.0 * r.psi_err
        phi_lo = r.phi - phi_bound
        phi_hi = r.phi + phi_bound
        psi_lo = r.psi - psi_bound
        psi_hi = r.psi + psi_bound
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
    include_medium: bool = False,
    chain_code: str = "A",
    program_version: str = "0.2.0",
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

    residues = result.accepted(include_medium=include_medium) if accepted_only else result.residues
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
    lines.append("   _nef_nmr_meta_data.program_name      facet")
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
    # Use 2-sigma bounds for consistency with .tbl and .aco writers.
    for r in residues:
        sid = r.seq_id
        cc = chain_code
        phi_bound = 2.0 * r.phi_err
        psi_bound = 2.0 * r.psi_err

        # PHI: C(i-1) - N(i) - CA(i) - C(i) — skip if (i-1) missing
        if (sid - 1) in present_sids:
            phi_lo = r.phi - phi_bound
            phi_hi = r.phi + phi_bound
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
            psi_lo = r.psi - psi_bound
            psi_hi = r.psi + psi_bound
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


# ─────────────────────── pred.tab summary ──────────────────────

def write_predtab(
    result: FACETResult,
    path: str | Path,
) -> Path:
    """Write per-residue prediction summary table.

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
        f"REMARK facet-nmr {__version__}"
        + (f" | retrieval index {result.index_version}" if result.index_version else ""),
        "REMARK https://github.com/maxzinke/facet-nmr",
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
    lines = [
        "ResID,AA,PHI,PSI,dPHI,dPSI,SS,Chi1,Chi1_prob,"
        "Chi1_p_gplus,Chi1_p_gminus,Chi1_p_trans,"
        "RCI_S2,"
        "struct_helix,struct_beta,struct_ppii,struct_coil,struct_effN,"
        "Confidence,Class"
    ]
    for r in result.residues:
        chi1_str = CHI1_NAMES.get(r.chi1, "") if r.chi1 is not None else ""
        if r.chi1_probs is not None and r.chi1 is not None:
            p_top = r.chi1_probs[r.chi1]
            p_gp, p_gm, p_t = r.chi1_probs
            probs_str = f"{p_top:.3f},{p_gp:.3f},{p_gm:.3f},{p_t:.3f}"
        else:
            probs_str = ",,,"
        rci_str = f"{r.rci_s2:.3f}" if r.rci_s2 is not None else ""
        if r.structural_populations is not None:
            h, b, p, c = r.structural_populations
            struct_str = f"{h:.3f},{b:.3f},{p:.3f},{c:.3f}"
            effn = r.structural_populations_eff_n or 0.0
            struct_str += f",{effn:.1f}"
        else:
            struct_str = ",,,,"
        lines.append(
            f"{r.seq_id},{r.comp_id},{r.phi:.1f},{r.psi:.1f},"
            f"{r.phi_err:.1f},{r.psi_err:.1f},{r.ss},{chi1_str},"
            f"{probs_str},{rci_str},{struct_str},"
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

    def _residue_dict(r):
        d = {
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
        if r.chi1_probs is not None:
            d["chi1_probs"] = {
                "g+": round(r.chi1_probs[0], 3),
                "g-": round(r.chi1_probs[1], 3),
                "t":  round(r.chi1_probs[2], 3),
            }
        if r.rci_s2 is not None:
            d["rci_s2"] = round(r.rci_s2, 3)
        if r.structural_populations is not None:
            h, b, p, c = r.structural_populations
            d["structural_populations"] = {
                "helix": round(h, 3),
                "beta":  round(b, 3),
                "ppii":  round(p, 3),
                "coil":  round(c, 3),
            }
            if r.structural_populations_eff_n is not None:
                d["structural_eff_N"] = round(r.structural_populations_eff_n, 1)
        if r.basin_populations is not None:
            a, b, p, o = r.basin_populations
            d["basin_populations"] = {
                "alpha_R": round(a, 3),
                "beta": round(b, 3),
                "PPII": round(p, 3),
                "other": round(o, 3),
            }
        if r.alt_clusters:
            d["alt_clusters"] = [
                {"phi": round(ph, 1), "psi": round(ps, 1), "weight": round(w, 3)}
                for ph, ps, w in r.alt_clusters
            ]
        return d

    data = {
        "source": result.source,
        "facet_version": __version__,
        "n_residues": result.n_residues,
        "index_version": result.index_version,
        "deuteration_preset": result.deuteration_preset,
        "deuteration_corrections_ppm": result.deuteration_corrections_ppm,
        "referencing_summary": result.referencing_summary,
        "referencing_corrections_applied": result.referencing_corrections_applied,
        "residues": [_residue_dict(r) for r in result.residues],
    }

    out.write_text(json_mod.dumps(data, indent=2), encoding="utf-8")
    return out
