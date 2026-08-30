#!/usr/bin/env python
"""Write the benchmark tables with the ground-truth units defect repaired.

6,978 scored residues in 63 single-model (X-ray) entries were stored with phi/psi
that had been converted to radians twice (docs/BENCHMARKS.md section 7). The fix is
to multiply those values by 180/pi; it was validated residue-by-residue against
phi/psi recomputed with gemmi from the deposited mmCIF files (median deviation
0.02 degrees, 100 % of residues within 1 degree, six entries / 1,648 residues).

Two outputs, both in results/talosn_clean/:

  per_residue_corrected.csv
      The benchmark of record with the flagged truth rescaled. TALOS-N's error on
      the flagged rows is recomputed from its stored angles. The record stores no
      FACET angles, so FACET's error on the flagged rows is taken from the public-
      path re-run (per_residue_rerun.csv); the column ``facet_err_source`` says
      which rows are ``record`` and which are ``rerun``. Unflagged rows are left
      exactly as recorded.

  per_residue_rerun_corrected.csv
      The public-path re-run with the flagged truth rescaled and BOTH errors
      recomputed from stored angles. Pure, single-source; this is the table the
      released package reproduces.

    python benchmarks/build_corrected_truth.py
    python benchmarks/rescore.py --csv benchmarks/results/talosn_clean/per_residue_corrected.csv
    python benchmarks/rescore.py --csv benchmarks/results/talosn_clean/per_residue_rerun_corrected.csv --rerun
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "talosn_clean"
SCALE = 180.0 / math.pi


def rms(p_phi, p_psi, t_phi, t_psi) -> float:
    dphi = abs(p_phi - t_phi) % 360.0
    dphi = min(dphi, 360.0 - dphi)
    dpsi = abs(p_psi - t_psi) % 360.0
    dpsi = min(dpsi, 360.0 - dpsi)
    return math.sqrt((dphi * dphi + dpsi * dpsi) / 2.0)


def fnum(s: str):
    return float(s) if s not in ("", "nan") else None


def main() -> None:
    record = list(csv.DictReader(open(RES / "per_residue.csv", newline="")))
    rerun = list(csv.DictReader(open(RES / "per_residue_rerun.csv", newline="")))
    rerun_by = {(r["bmrb_id"], r["seq_id"]): r for r in rerun}

    # ---- hybrid: record + rescaled truth ------------------------------------------
    out_rows = []
    n_flag = n_facet_from_rerun = n_facet_missing = 0
    for r in record:
        row = dict(r)
        row["facet_err_source"] = "record"
        if r["truth_units_suspect"] == "1":
            n_flag += 1
            t_phi, t_psi = float(r["phi_true"]) * SCALE, float(r["psi_true"]) * SCALE
            row["phi_true"], row["psi_true"] = f"{t_phi:.3f}", f"{t_psi:.3f}"
            n_phi, n_psi = fnum(r["phi_talosn"]), fnum(r["psi_talosn"])
            if r["talosn_has_pred"] == "1" and n_phi is not None:
                row["talosn_err"] = f"{rms(n_phi, n_psi, t_phi, t_psi):.3f}"
            rr = rerun_by.get((r["bmrb_id"], r["seq_id"]))
            f_phi = fnum(rr["phi_facet"]) if rr else None
            if f_phi is not None:
                row["facet_err"] = f"{rms(f_phi, fnum(rr['psi_facet']), t_phi, t_psi):.3f}"
                row["facet_tier"] = rr["facet_tier_rerun"]
                row["facet_tier_public"] = {"Strong": "High", "Generous": "Medium",
                                            "Ambiguous": "Low", "None": "Flexible"}.get(rr["facet_tier_rerun"], "Flexible")
                row["facet_has_pred"] = "1"
                row["facet_err_source"] = "rerun"
                n_facet_from_rerun += 1
            else:
                row["facet_err"] = ""
                row["facet_has_pred"] = "0"
                n_facet_missing += 1
        out_rows.append(row)
    fields = list(record[0].keys()) + ["facet_err_source"]
    with open(RES / "per_residue_corrected.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"per_residue_corrected.csv: {len(out_rows):,} rows; {n_flag:,} flagged rows rescaled; "
          f"FACET error from the re-run on {n_facet_from_rerun:,} of them, unavailable on {n_facet_missing}")

    # ---- pure public path + rescaled truth --------------------------------------------
    out_rows = []
    for r in rerun:
        row = dict(r)
        if r["truth_units_suspect"] == "1":
            t_phi, t_psi = float(r["phi_true"]) * SCALE, float(r["psi_true"]) * SCALE
            row["phi_true"], row["psi_true"] = f"{t_phi:.3f}", f"{t_psi:.3f}"
            n_phi, n_psi = fnum(r["phi_talosn"]), fnum(r["psi_talosn"])
            if r["talosn_has_pred"] == "1" and n_phi is not None:
                row["talosn_err"] = f"{rms(n_phi, n_psi, t_phi, t_psi):.3f}"
            f_phi, f_psi = fnum(r["phi_facet"]), fnum(r["psi_facet"])
            if f_phi is not None:
                row["facet_err_rerun"] = f"{rms(f_phi, f_psi, t_phi, t_psi):.3f}"
                # the record column is not meaningful once the truth changes
                row["facet_err"] = ""
        out_rows.append(row)
    with open(RES / "per_residue_rerun_corrected.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rerun[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"per_residue_rerun_corrected.csv: {len(out_rows):,} rows written")


if __name__ == "__main__":
    main()
