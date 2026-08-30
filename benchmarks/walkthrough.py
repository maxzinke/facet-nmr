#!/usr/bin/env python
"""One benchmark protein, end to end: input shifts -> FACET -> compare with TALOS-N and truth.

Produces the material quoted in WALKTHROUGH.md and the figure
``figures/walkthrough_<id>.png`` (per-residue error along the sequence for both
methods, plus a Ramachandran plot of FACET's prediction coloured by error).

    python benchmarks/walkthrough.py            # BMRB 15232
    python benchmarks/walkthrough.py --id 10034

Needs the ``facet`` package and matplotlib. TALOS-N angles come from
``per_residue.csv`` (the benchmark of record), so no TALOS-N installation is needed.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def wrapped_delta(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def rms_error(p_phi, p_psi, t_phi, t_psi) -> float:
    return math.sqrt(wrapped_delta(p_phi, t_phi) ** 2 + wrapped_delta(p_psi, t_psi) ** 2) / math.sqrt(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="15232")
    ap.add_argument("--out-dir", type=Path, default=None, help="where to write the predtab (default: temp dir)")
    args = ap.parse_args()

    from facet import predict

    tab = HERE / "data" / "inputs" / f"{args.id}.tab"
    truth = {int(r["seq_id"]): r
             for r in csv.DictReader(open(HERE / "results" / "talosn_clean" / "per_residue.csv", newline=""))
             if r["bmrb_id"] == args.id}
    if not truth:
        raise SystemExit(f"{args.id} has no rows in per_residue.csv")

    res = predict(tab)
    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="facet_walkthrough_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    predtab = out_dir / f"{args.id}_facet.predtab"
    res.to_predtab(predtab)

    # ── per-residue comparison ───────────────────────────────────────────
    print(f"BMRB {args.id}: {len(res.residues)} residues predicted, {len(truth)} with ground truth\n")
    print("resid AA  ss   phi_true  psi_true |  phi_facet psi_facet tier      err | phi_talosn psi_talosn class     err")
    seq_ids, e_f, e_t, f_phi, f_psi, tiers = [], [], [], [], [], []
    rec_f, rec_t = [], []
    for r in res.residues:
        t = truth.get(r.seq_id)
        if t is None:
            continue
        tp, ts = float(t["phi_true"]), float(t["psi_true"])
        ef = rms_error(r.phi, r.psi, tp, ts)
        if t["talosn_has_pred"] == "1":
            et = float(t["talosn_err"])
            tn = f"{float(t['phi_talosn']):9.1f} {float(t['psi_talosn']):9.1f} {t['talosn_class']:<9s} {et:5.1f}"
        else:
            et = float("nan")
            tn = f"{'—':>9s} {'—':>9s} {t['talosn_class']:<9s}   —"
        print(f"{r.seq_id:5d} {r.comp_id} {t['ss']:>3s} {tp:9.1f} {ts:9.1f} | {r.phi:9.1f} {r.psi:9.1f} "
              f"{r.confidence_class:<9s} {ef:5.1f} | {tn}")
        seq_ids.append(r.seq_id)
        e_f.append(ef)
        e_t.append(et)
        f_phi.append(r.phi)
        f_psi.append(r.psi)
        tiers.append(r.confidence_class)
        rec_f.append(float(t["facet_err"]) if t["facet_err"] else float("nan"))
        rec_t.append(et)

    e_f, e_t, rec_f = np.array(e_f), np.array(e_t), np.array(rec_f)
    both = ~np.isnan(e_t)
    print(f"\nthis run     : FACET median {np.median(e_f[both]):.1f}°, TALOS-N median {np.nanmedian(e_t):.1f}°, "
          f"FACET lower on {np.mean(e_f[both] < e_t[both]) * 100:.0f}% of {both.sum()} paired residues")
    okr = both & ~np.isnan(rec_f)
    print(f"record (CSV) : FACET median {np.median(rec_f[okr]):.1f}° on the same residues; "
          f"{np.mean(np.abs(rec_f[okr] - e_f[okr]) < 0.5) * 100:.0f}% of residues within 0.5° of this run")
    print(f"predtab written to {predtab}")

    # ── figure ───────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c_f, c_t = "#1E5F8C", "#B23A2E"
    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1])
    ax = fig.add_subplot(gs[0])
    ax.plot(seq_ids, e_t, color=c_t, lw=1.2, label="TALOS-N", alpha=0.85)
    ax.plot(seq_ids, e_f, color=c_f, lw=1.6, label="FACET")
    ax.axhline(25, color="#999", ls=":", lw=1)
    ss = [truth[s]["ss"] for s in seq_ids]
    for i, s in enumerate(seq_ids):
        if ss[i] == "H":
            ax.axvspan(s - 0.5, s + 0.5, color="#F2D3CF", lw=0, zorder=0)
        elif ss[i] == "E":
            ax.axvspan(s - 0.5, s + 0.5, color="#D6E4F0", lw=0, zorder=0)
    ax.set_xlabel("residue")
    ax.set_ylabel("φ/ψ error (°)")
    ax.set_ylim(0, max(60, np.nanmax(np.concatenate([e_f, e_t])) * 1.05))
    ax.set_title(f"BMRB {args.id}: per-residue error (shaded: helix red, strand blue)", fontsize=10)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[1])
    sc = ax2.scatter(f_phi, f_psi, c=e_f, cmap="viridis_r", vmin=0, vmax=60, s=16, edgecolors="none")
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(-180, 180)
    ax2.set_xticks([-180, -90, 0, 90, 180])
    ax2.set_yticks([-180, -90, 0, 90, 180])
    ax2.set_xlabel("φ (°)")
    ax2.set_ylabel("ψ (°)")
    ax2.set_title("FACET prediction, coloured by error", fontsize=10)
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.2)
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04, label="error (°)")
    fig.tight_layout()
    fig_path = HERE / "figures" / f"walkthrough_{args.id}.png"
    fig_path.parent.mkdir(exist_ok=True)
    fig.savefig(fig_path, dpi=150)
    print(f"figure written to {fig_path}")


if __name__ == "__main__":
    main()
