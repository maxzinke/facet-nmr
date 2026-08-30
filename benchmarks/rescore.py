#!/usr/bin/env python
"""Recompute every benchmark table in the README from ``per_residue.csv``.

This script needs no model, no weights and no network: it reads one CSV and prints
the numbers. If the numbers it prints disagree with the README, the README is wrong.

    python benchmarks/rescore.py                 # print the tables
    python benchmarks/rescore.py --figures       # also write benchmarks/figures/*.png
    python benchmarks/rescore.py --csv other.csv # score a re-run instead

Definitions (see docs/BENCHMARKS.md):
    error   = sqrt((dphi^2 + dpsi^2) / 2), each delta wrapped to [0, 180] degrees
    fail25  = share of residues with error > 25 degrees
    paired  = residues for which BOTH methods emit a prediction; every head-to-head
              number is computed on that set so the comparison is like-for-like
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "results" / "talosn_clean" / "per_residue.csv"
ABLATION_JSON = HERE / "results" / "coverage_ablation" / "results.json"
FIG_DIR = HERE / "figures"

TIERS = ["High", "Medium", "Low", "Flexible"]
RESTRAINT_BOUND = {"High": "±20°", "Medium": "±35°", "Low": "—", "Flexible": "—"}
SS_NAME = {"H": "Helix", "E": "Strand", "C": "Coil"}


INTERNAL_TO_PUBLIC = {"Strong": "High", "Generous": "Medium", "Ambiguous": "Low", "None": "Flexible"}


def load(path: Path, rerun: bool = False, corrected_truth: bool = False) -> dict[str, np.ndarray]:
    """Read per_residue.csv. With ``rerun=True`` score the ``*_rerun`` columns
    written by run_talosn_comparison.py instead of the benchmark of record. With
    ``corrected_truth=True`` rescale the flagged ground truth (see docs/BENCHMARKS.md
    §7) and recompute both methods' errors from the stored angles."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{path} is empty")

    def col(name, conv):
        return np.array([conv(r[name]) if r[name] != "" else np.nan for r in rows])

    if rerun:
        if "facet_err_rerun" not in rows[0]:
            raise SystemExit("--rerun needs the columns written by run_talosn_comparison.py")
        facet_err = col("facet_err_rerun", float)
        tier = np.array([INTERNAL_TO_PUBLIC.get(r["facet_tier_rerun"], "Flexible") for r in rows])
        # the public path always emits an angle; "Flexible" residues are still
        # counted as predicted, exactly as the record counts tier None with an angle
        facet_has = ~np.isnan(facet_err)
    else:
        facet_err = col("facet_err", float)
        tier = np.array([r["facet_tier_public"] for r in rows])
        facet_has = col("facet_has_pred", int).astype(bool)

    talosn_err = col("talosn_err", float)
    suspect = (col("truth_units_suspect", int).astype(bool) if "truth_units_suspect" in rows[0]
               else np.zeros(len(rows), dtype=bool))

    if corrected_truth:
        # Re-derive BOTH errors from stored angles against the rescaled truth. Needs the
        # FACET angle columns, i.e. a per_residue_rerun.csv.
        if "phi_facet" not in rows[0]:
            raise SystemExit("--corrected-truth needs phi_facet/psi_facet (a run_talosn_comparison.py output)")
        scale = 180.0 / np.pi
        t_phi = col("phi_true", float)
        t_psi = col("psi_true", float)
        t_phi = np.where(suspect, t_phi * scale, t_phi)
        t_psi = np.where(suspect, t_psi * scale, t_psi)
        f_phi, f_psi = col("phi_facet", float), col("psi_facet", float)
        n_phi, n_psi = col("phi_talosn", float), col("psi_talosn", float)
        facet_err = rms_error(f_phi, f_psi, t_phi, t_psi)
        talosn_err = rms_error(n_phi, n_psi, t_phi, t_psi)
        facet_has = ~np.isnan(facet_err)
        tier = np.array([INTERNAL_TO_PUBLIC.get(r["facet_tier_rerun"], "Flexible") for r in rows])

    return {
        "bmrb_id": np.array([r["bmrb_id"] for r in rows]),
        "ss": np.array([r["ss"] for r in rows]),
        "flexible": col("flexible", int).astype(bool),
        "suspect": suspect,
        "tier": tier,
        "facet_has": facet_has,
        "facet_err": facet_err,
        "talosn_has": col("talosn_has_pred", int).astype(bool) & ~np.isnan(talosn_err),
        "talosn_err": talosn_err,
    }


def rms_error(p_phi, p_psi, t_phi, t_psi):
    """sqrt((dphi^2 + dpsi^2)/2) with each delta wrapped into [0, 180]; NaN-propagating."""
    dphi = np.abs(p_phi - t_phi) % 360.0
    dphi = np.minimum(dphi, 360.0 - dphi)
    dpsi = np.abs(p_psi - t_psi) % 360.0
    dpsi = np.minimum(dpsi, 360.0 - dpsi)
    return np.sqrt((dphi ** 2 + dpsi ** 2) / 2.0)


def drop_suspect(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keep = ~d["suspect"]
    return {k: v[keep] for k, v in d.items()}


def stats(err: np.ndarray) -> dict:
    return {
        "n": int(err.size),
        "median": float(np.median(err)),
        "mean": float(np.mean(err)),
        "p90": float(np.percentile(err, 90)),
        "fail25": float(np.mean(err > 25.0) * 100),
    }


def table(d: dict[str, np.ndarray]) -> dict:
    both = d["facet_has"] & d["talosn_has"]
    fe, te = d["facet_err"], d["talosn_err"]
    out: dict = {
        "n_rows": int(len(fe)),
        "coverage_facet_pct": float(d["facet_has"].mean() * 100),
        "coverage_talosn_pct": float(d["talosn_has"].mean() * 100),
        "n_paired": int(both.sum()),
        "overall": {"facet": stats(fe[both]), "talosn": stats(te[both])},
        "by_ss": {},
        "by_flex": {},
        "win_rate_facet_pct": float(np.mean(fe[both] < te[both]) * 100),
        "win_rate_talosn_pct": float(np.mean(te[both] < fe[both]) * 100),
        "n_ties": int(np.sum(fe[both] == te[both])),
        "tiers": {},
    }
    for ss in ("C", "H", "E"):
        m = both & (d["ss"] == ss)
        out["by_ss"][SS_NAME[ss]] = {"facet": stats(fe[m]), "talosn": stats(te[m])}
    for name, m in (("Rigid", both & ~d["flexible"]), ("Flexible", both & d["flexible"])):
        out["by_flex"][name] = {"facet": stats(fe[m]), "talosn": stats(te[m])}
    for t in TIERS:
        m = d["tier"] == t
        mp = m & both
        out["tiers"][t] = {
            "coverage_pct": float(m.mean() * 100),
            "median": float(np.median(fe[mp])) if mp.any() else None,
            "fail25": float(np.mean(fe[mp] > 25.0) * 100) if mp.any() else None,
            "n": int(m.sum()),
        }
    return out


def print_tables(t: dict) -> None:
    o = t["overall"]
    print(f"Rows scored: {t['n_rows']:,}   coverage FACET {t['coverage_facet_pct']:.1f}% / "
          f"TALOS-N {t['coverage_talosn_pct']:.1f}%   paired (both predict): {t['n_paired']:,}\n")
    print("| Metric | TALOS-N | FACET |")
    print("|---|---|---|")
    print(f"| All-residue median | {o['talosn']['median']:.2f}° | {o['facet']['median']:.2f}° "
          f"({o['facet']['median'] - o['talosn']['median']:+.2f}°) |")
    print(f"| fail25 rate | {o['talosn']['fail25']:.1f}% | {o['facet']['fail25']:.1f}% |")
    print(f"| Mean | {o['talosn']['mean']:.1f}° | {o['facet']['mean']:.1f}° |")
    print(f"| p90 | {o['talosn']['p90']:.1f}° | {o['facet']['p90']:.1f}° |")
    for name in ("Coil", "Helix", "Strand"):
        s = t["by_ss"][name]
        print(f"| {name} median (n={s['facet']['n']:,}) | {s['talosn']['median']:.1f}° | "
              f"{s['facet']['median']:.1f}° ({s['facet']['median'] - s['talosn']['median']:+.2f}°) |")
    for name in ("Rigid", "Flexible"):
        s = t["by_flex"][name]
        print(f"| {name} (n={s['facet']['n']:,}) | {s['talosn']['median']:.1f}° | {s['facet']['median']:.1f}° |")
    print(f"| Head-to-head win rate | {t['win_rate_talosn_pct']:.1f}% | {t['win_rate_facet_pct']:.1f}% "
          f"(ties: {t['n_ties']}) |")
    print("\nTier calibration (paired residues; coverage over all scored residues):\n")
    print("| Tier | Coverage | Median err | fail25 | Restraint bound |")
    print("|---|---|---|---|---|")
    for name in TIERS:
        s = t["tiers"][name]
        med = f"{s['median']:.1f}°" if s["median"] is not None else "—"
        f25 = f"{s['fail25']:.1f}%" if s["fail25"] is not None else "—"
        print(f"| {name} | {s['coverage_pct']:.1f}% | {med} | {f25} | {RESTRAINT_BOUND[name]} |")


def print_ablation() -> None:
    if not ABLATION_JSON.exists():
        return
    a = json.load(open(ABLATION_JSON))
    print(f"\nCoverage ablation ({a['n_test_entries']} entries, {a['n_valid']:,} scored residues; "
          f"metric = (|dphi| + |dpsi|) / 2 — NOT the RMS metric above):\n")
    print("| available shifts | mask-safe fallback median | fail25 |")
    print("|---|---|---|")
    label = {"full": "all backbone shifts", "-HA": "HA absent", "-H": "H absent",
             "-H-HA": "H and HA absent", "-CB": "CB absent", "-H-HA-CB": "H, HA and CB absent"}
    for k, v in a["scenarios"].items():
        mm = v.get("masked_module", {}).get("overall")
        if mm:
            print(f"| {label.get(k, k)} | {mm['median']:.1f}° | {mm['fail25']:.1f}% |")


def figures(d: dict[str, np.ndarray], t: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(exist_ok=True)
    both = d["facet_has"] & d["talosn_has"]
    fe, te = d["facet_err"][both], d["talosn_err"][both]
    c_f, c_t = "#1E5F8C", "#B23A2E"

    # 1. error CDF
    fig, ax = plt.subplots(figsize=(6, 4.2))
    grid = np.linspace(0, 180, 721)
    for err, lab, c in ((te, "TALOS-N", c_t), (fe, "FACET", c_f)):
        ax.plot(grid, [np.mean(err <= g) for g in grid], label=lab, color=c, lw=1.8)
    ax.axvline(25, color="#999", ls=":", lw=1)
    ax.text(26, 0.03, "25° (fail25 threshold)", fontsize=8, color="#666")
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 1)
    ax.set_xlabel("φ/ψ error (degrees, RMS of wrapped deltas)")
    ax.set_ylabel("fraction of paired residues ≤ error")
    ax.set_title(f"Error distribution, {both.sum():,} paired residues", fontsize=10)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_cdf.png", dpi=150)
    plt.close(fig)

    # 2. tier calibration
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    tiers = [x for x in TIERS if t["tiers"][x]["median"] is not None]
    cov = [t["tiers"][x]["coverage_pct"] for x in tiers]
    med = [t["tiers"][x]["median"] for x in tiers]
    f25 = [t["tiers"][x]["fail25"] for x in tiers]
    axes[0].bar(tiers, med, color=c_f)
    axes[0].set_ylabel("median error (°)")
    axes[0].set_title("Median error by tier", fontsize=10)
    for i, (m, cv) in enumerate(zip(med, cov)):
        axes[0].text(i, m + 1, f"{m:.1f}°\n{cv:.1f}% of residues", ha="center", fontsize=8)
    axes[0].set_ylim(0, max(med) * 1.35)
    axes[1].bar(tiers, f25, color="#7A8492")
    axes[1].set_ylabel("fail25 (%)")
    axes[1].set_title("Share of residues > 25° by tier", fontsize=10)
    for i, v in enumerate(f25):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tier_calibration.png", dpi=150)
    plt.close(fig)

    # 3. per-protein scatter
    ids = d["bmrb_id"][both]
    fm, tm = [], []
    for e in np.unique(ids):
        m = ids == e
        if m.sum() >= 10:
            fm.append(np.median(fe[m]))
            tm.append(np.median(te[m]))
    fm, tm = np.array(fm), np.array(tm)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    lim = max(fm.max(), tm.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#999", lw=1, ls="--")
    ax.scatter(tm, fm, s=9, alpha=0.55, color=c_f, edgecolors="none")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("TALOS-N per-protein median error (°)")
    ax.set_ylabel("FACET per-protein median error (°)")
    ax.set_title(f"{len(fm)} proteins with ≥10 paired residues; "
                 f"FACET lower in {np.mean(fm < tm) * 100:.0f}%", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_protein_scatter.png", dpi=150)
    plt.close(fig)

    # 4. coverage ablation
    if ABLATION_JSON.exists():
        a = json.load(open(ABLATION_JSON))
        names = list(a["scenarios"].keys())
        retr = [a["scenarios"][k]["retrieval"]["overall"]["median"] for k in names]
        mm = [a["scenarios"][k]["masked_module"]["overall"]["median"] for k in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.bar(x - 0.2, retr, 0.4, label="training-harness reference retrieval (mean-z scorer)", color="#B4BCC7")
        ax.bar(x + 0.2, mm, 0.4, label="shipped facet.masked_retrieval (coverage-first ranking)", color=c_f)
        ax.set_xticks(x)
        ax.set_xticklabels(["all shifts" if n == "full" else n for n in names])
        ax.set_ylabel("median error (°), (|Δφ|+|Δψ|)/2")
        ax.set_title(f"Missing-atom ablation, {a['n_test_entries']} held-out entries, "
                     f"{a['n_valid']:,} residues", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "coverage_ablation.png", dpi=150)
        plt.close(fig)
    print(f"\nfigures written to {FIG_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--json", type=Path, help="also dump the computed numbers as JSON")
    ap.add_argument("--rerun", action="store_true",
                    help="score the facet_*_rerun columns of a run_talosn_comparison.py output")
    ap.add_argument("--exclude-suspect", action="store_true",
                    help="drop residues whose ground truth is flagged truth_units_suspect")
    ap.add_argument("--corrected-truth", action="store_true",
                    help="rescale flagged ground truth by 180/pi and re-derive both errors "
                         "from stored angles (needs a run_talosn_comparison.py output)")
    args = ap.parse_args()
    d = load(args.csv, rerun=args.rerun, corrected_truth=args.corrected_truth)
    n_sus = int(d["suspect"].sum())
    if args.exclude_suspect:
        d = drop_suspect(d)
        print(f"[excluded {n_sus:,} residues flagged truth_units_suspect]\n")
    elif n_sus and not args.corrected_truth:
        print(f"[note: {n_sus:,} residues are flagged truth_units_suspect and are INCLUDED below; "
              f"see --exclude-suspect / --corrected-truth and docs/BENCHMARKS.md §7]\n")
    elif args.corrected_truth:
        print(f"[ground truth rescaled for {n_sus:,} flagged residues; both errors re-derived from stored angles]\n")
    t = table(d)
    print_tables(t)
    print_ablation()
    if args.json:
        json.dump(t, open(args.json, "w"), indent=2)
    if args.figures:
        figures(d, t)


if __name__ == "__main__":
    main()
