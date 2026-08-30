#!/usr/bin/env python
"""Re-run FACET on the benchmark inputs and score it against TALOS-N and the truth.

This is the public, self-contained version of the script that produced the benchmark
of record. It depends only on the ``facet`` package (plus its downloaded assets) and
NumPy. It reads:

  * ``benchmarks/data/inputs/<id>.tab``   — the exact shift tables both methods saw
  * ``benchmarks/data/test_set_745.txt``  — which entries to run (or ``--ids``)
  * ``per_residue.csv``                    — ground truth (``phi_true``/``psi_true``),
                                             SS, flexibility and TALOS-N columns
  * ``--talosn-dir <dir>``                 — optional: a directory of TALOS-N outputs
                                             laid out as ``<dir>/<id>/pred.tab``. If
                                             given, TALOS-N angles are re-parsed from
                                             there; otherwise the TALOS-N columns are
                                             carried over from ``per_residue.csv``

and writes a CSV with the same columns as ``per_residue.csv`` plus
``phi_facet`` / ``psi_facet`` / ``facet_phi_err`` / ``facet_psi_err``, which
``rescore.py --csv`` can score directly.

    python benchmarks/run_talosn_comparison.py --ids 10034 10040 10046       # smoke
    python benchmarks/run_talosn_comparison.py --out benchmarks/results/talosn_clean/per_residue_rerun.csv

The benchmark of record was produced inside the training harness (batched windows
over the curated shift arrays, with the same weights and the same index). This
script goes through the public ``facet.predict`` path — file parsing, secondary-shift
conversion, the mask-safe fallback — so it reproduces the record closely but not
bit-for-bit; ``benchmarks/README.md`` reports the measured agreement.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

PUBLIC_TO_INTERNAL = {"High": "Strong", "Medium": "Generous", "Low": "Ambiguous", "Flexible": "None"}


def wrapped_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def rms_error(p_phi, p_psi, t_phi, t_psi) -> float:
    dphi = wrapped_delta(p_phi, t_phi)
    dpsi = wrapped_delta(p_psi, t_psi)
    return math.sqrt(dphi * dphi + dpsi * dpsi) / math.sqrt(2)


def parse_pred_tab(path: Path) -> dict[int, dict]:
    """TALOS-N pred.tab -> {resid: {phi, psi, class}}; 9999 = no prediction."""
    out: dict[int, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line[0] in "RDVF":
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            resid = int(parts[0])
        except ValueError:
            continue
        phi, psi, cls = float(parts[2]), float(parts[3]), parts[10]
        if abs(phi) > 9000 or abs(psi) > 9000:
            out[resid] = {"phi": None, "psi": None, "class": "None"}
        else:
            out[resid] = {"phi": phi, "psi": psi, "class": cls}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--truth", type=Path, default=HERE / "results" / "talosn_clean" / "per_residue.csv")
    ap.add_argument("--inputs", type=Path, default=HERE / "data" / "inputs")
    ap.add_argument("--ids", nargs="*", help="subset of BMRB ids (default: every id in --id-list)")
    ap.add_argument("--id-list", type=Path, default=HERE / "data" / "test_set_745.txt")
    ap.add_argument("--talosn-dir", type=Path, help="<dir>/<id>/pred.tab; not redistributed")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "talosn_clean" / "per_residue_rerun.csv")
    ap.add_argument("--no-fallback", action="store_true", help="disable the mask-safe fallback")
    args = ap.parse_args()

    from facet import predict

    with open(args.truth, newline="") as fh:
        truth_rows = list(csv.DictReader(fh))
    truth: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in truth_rows:
        truth[r["bmrb_id"]][int(r["seq_id"])] = r
    cols = list(truth_rows[0].keys())
    extra = ["phi_facet", "psi_facet", "facet_phi_err", "facet_psi_err", "facet_tier_rerun", "facet_err_rerun"]

    ids = args.ids or [ln.strip() for ln in args.id_list.read_text().splitlines() if ln.strip()]
    ids = [i for i in ids if i in truth]  # entries without any scorable residue are skipped
    print(f"running FACET on {len(ids)} entries ...", flush=True)

    out_rows = []
    n_err_match = n_cmp = 0
    t0 = time.time()
    for k, eid in enumerate(ids, 1):
        tab = args.inputs / f"{eid}.tab"
        if not tab.exists():
            print(f"  {eid}: no input file, skipped")
            continue
        try:
            res = predict(tab, mask_safe_fallback=not args.no_fallback)
        except Exception as exc:  # noqa: BLE001 — report and keep going
            print(f"  {eid}: FACET failed: {exc}")
            continue
        tn = parse_pred_tab(args.talosn_dir / eid / "pred.tab") if args.talosn_dir else None
        by_sid = {r.seq_id: r for r in res.residues}
        for sid, t in sorted(truth[eid].items()):
            r = by_sid.get(sid)
            row = dict(t)
            if tn is not None:
                p = tn.get(sid)
                if p is not None and p["phi"] is not None:
                    row["phi_talosn"], row["psi_talosn"], row["talosn_class"] = f"{p['phi']:.3f}", f"{p['psi']:.3f}", p["class"]
                    row["talosn_has_pred"] = "1"
                    row["talosn_err"] = f"{rms_error(p['phi'], p['psi'], float(t['phi_true']), float(t['psi_true'])):.3f}"
                else:
                    row["phi_talosn"] = row["psi_talosn"] = row["talosn_err"] = ""
                    row["talosn_class"] = "None"
                    row["talosn_has_pred"] = "0"
            if r is None:
                for c in extra:
                    row[c] = ""
                out_rows.append(row)
                continue
            t_phi, t_psi = float(t["phi_true"]), float(t["psi_true"])
            err = rms_error(r.phi, r.psi, t_phi, t_psi)
            row["phi_facet"] = f"{r.phi:.3f}"
            row["psi_facet"] = f"{r.psi:.3f}"
            row["facet_phi_err"] = f"{r.phi_err:.3f}"
            row["facet_psi_err"] = f"{r.psi_err:.3f}"
            row["facet_tier_rerun"] = PUBLIC_TO_INTERNAL[r.confidence_class]
            row["facet_err_rerun"] = f"{err:.3f}"
            # agreement with the record, where the record emitted a prediction
            if t["facet_err"] != "":
                n_cmp += 1
                n_err_match += abs(err - float(t["facet_err"])) < 0.5
            out_rows.append(row)
        if k % 25 == 0 or k == len(ids):
            print(f"  {k}/{len(ids)} entries, {time.time() - t0:.0f}s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols + extra)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out} ({len(out_rows)} rows)")
    if n_cmp:
        print(f"agreement with the record: {n_err_match}/{n_cmp} residues "
              f"({100 * n_err_match / n_cmp:.1f}%) within 0.5° of the recorded FACET error")
    print("score it with:  python benchmarks/rescore.py --csv", args.out,
          "\n(rescore.py reads facet_err; to score the re-run itself, pass --rerun)")


if __name__ == "__main__":
    main()
