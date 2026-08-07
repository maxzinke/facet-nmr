"""Rebuild the mask-safe shift reference as aggregated statistics.

WHY
---
The shipped reference held 310,923 rows of per-residue secondary shifts with the
BMRB deposition id and residue number attached, so every one of ~4.8 million values
was traceable to a specific deposition. A secondary shift is one observation minus a
constant, and the package ships that constant (``RANDOM_COIL_SHIFTS``), so raw ppm
were recoverable. That is a database extract, not a statistic, and it was being
served publicly under an MIT header.

Aggregation fixes what the file *is*: each output row is the mean of many
observations grouped by (centre residue, secondary structure) and shift-space
cluster -- the same kind of summary the BMRB publishes itself. No deposition id, no
residue number, no single measurement survives.

MEASURED, not assumed
---------------------
Compared against the full record set under a homology-aware holdout: entries were
clustered by 5-mer sequence similarity and whole clusters held out, so no query
could retrieve a relative of itself. 1,200 queries from 520 held-out entries,
evaluated with HA masked on every residue (the perdeuterated regime this file
serves, since it is only consulted for HA-missing residues).

    index                    phi med   phi mean   phi p90   psi med   rows
    full record set            0.38       0.61      1.43      0.76    264,571
    centroids <=60/(aa,ss)     0.32       0.55      1.37      0.58      3,524

Better on every metric and every secondary-structure class, coverage unchanged at
100%. Averaging ~75 observations suppresses measurement noise, so a centroid is a
cleaner estimate of the shift pattern for a conformation than any single residue.

Those absolute numbers measure index self-consistency, NOT field accuracy -- they
are not comparable to the published 18 deg for this path, which comes from real
inputs through the full pipeline. Re-run the project benchmark before quoting any
accuracy figure for the rebuilt index.

USAGE
-----
    python scripts/build_centroid_reference.py --in facet/weights/facet_shift_reference.npz \
        --out facet/weights/facet_shift_reference_centroids.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#: Clusters per (centre amino acid, secondary structure) cell. 60 was the setting
#: measured above; raising it trades size for a resolution gain that did not appear.
PER_CELL = 60

#: A centroid over one observation is that observation. Rows below this are dropped
#: rather than emitted, which is what keeps the output a summary rather than a
#: passthrough for sparsely populated cells.
MIN_OBS = 2

KMEANS_ITERS = 12


def circ_mean(deg: np.ndarray) -> float:
    r = np.radians(np.asarray(deg, dtype=float))
    return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))


def circ_std(deg: np.ndarray) -> float:
    """Circular standard deviation in degrees; 0 for a single observation."""
    r = np.radians(np.asarray(deg, dtype=float))
    R = np.hypot(np.sin(r).mean(), np.cos(r).mean())
    return float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))))


def build(src: Path, out: Path, per_cell: int, seed: int) -> None:
    z = np.load(src, allow_pickle=False)
    S = z["triplet_shifts"].astype(np.float64)
    M = z["triplet_masks"].astype(np.float64)
    PHI = z["phi"].astype(np.float64)
    PSI = z["psi"].astype(np.float64)
    SS = z["ss_labels"].astype(np.int64)
    AA = z["aa_triplets"].astype(np.int64)
    W = z["weights"].astype(np.float64)
    rng = np.random.default_rng(seed)

    cs, cm, cp, cq, css, caa, cw, cn, cphi_sd, cpsi_sd = [], [], [], [], [], [], [], [], [], []
    all_idx = np.arange(len(PHI))

    for aa in np.unique(AA[:, 1]):
        for ss in np.unique(SS):
            sel = all_idx[(AA[:, 1] == aa) & (SS == ss)]
            if sel.size < MIN_OBS + 1:
                continue
            k = min(per_cell, sel.size)
            X = S[sel] * M[sel]
            C = X[rng.choice(sel.size, k, replace=False)].copy()
            lbl = np.zeros(sel.size, dtype=int)
            for _ in range(KMEANS_ITERS):
                lbl = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
                for c in range(k):
                    if (lbl == c).any():
                        C[c] = X[lbl == c].mean(0)
            for c in range(k):
                m = lbl == c
                if m.sum() < MIN_OBS:
                    continue
                g = sel[m]
                cs.append(S[g].mean(0))
                # An atom is present in the summary when most members observed it.
                cm.append((M[g].mean(0) > 0.5).astype(np.float64))
                cp.append(circ_mean(PHI[g])); cq.append(circ_mean(PSI[g]))
                cphi_sd.append(circ_std(PHI[g])); cpsi_sd.append(circ_std(PSI[g]))
                css.append(ss); caa.append(AA[g][0]); cw.append(W[g].mean())
                cn.append(int(m.sum()))

    n = len(cs)
    if n == 0:
        raise SystemExit("no centroids produced - check the input file")

    # Keys mirror the source so MaskedShiftRetrieval loads this unchanged.
    #
    # entry_ids becomes an OPAQUE cluster label. The query path uses it only as an
    # equality key, to avoid returning several correlated hits from one source; it
    # never reads the value. Deposition identity is therefore not needed and is gone.
    #
    # seq_ids is written as zeros. It is used when *building* triplets (to find
    # sequence-adjacent neighbours) and never read at query time, so the residue
    # number does not have to ship.
    np.savez_compressed(
        out,
        triplet_shifts=np.asarray(cs, dtype=np.float32),
        triplet_masks=np.asarray(cm, dtype=np.float32),
        phi=np.asarray(cp, dtype=np.float32),
        psi=np.asarray(cq, dtype=np.float32),
        ss_labels=np.asarray(css, dtype=np.int8),
        aa_triplets=np.asarray(caa, dtype=np.int16),
        weights=np.asarray(cw, dtype=np.float32),
        atom_stds=z["atom_stds"],
        entry_ids=np.asarray([f"c{i:05d}" for i in range(n)], dtype="<U6"),
        seq_ids=np.zeros(n, dtype=np.int64),
        # Extra, for transparency about what each row summarises.
        n_observations=np.asarray(cn, dtype=np.int32),
        phi_circ_std=np.asarray(cphi_sd, dtype=np.float32),
        psi_circ_std=np.asarray(cpsi_sd, dtype=np.float32),
        index_kind=np.array("aggregated-centroids", dtype="<U24"),
    )

    obs = np.asarray(cn)
    src_mb, out_mb = src.stat().st_size / 1e6, out.stat().st_size / 1e6
    print(f"{len(PHI):,} observations -> {n:,} centroids ({len(PHI)/n:.0f}x)")
    print(f"observations per centroid: mean {obs.mean():.0f}, median {int(np.median(obs))}, min {obs.min()}")
    print(f"{src_mb:.1f} MB -> {out_mb:.2f} MB")
    print("dropped: entry_ids (now opaque cluster labels), seq_ids (now zeros)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--in", dest="src", type=Path, default=root / "facet/weights/facet_shift_reference.npz")
    ap.add_argument("--out", type=Path, default=root / "facet/weights/facet_shift_reference_centroids.npz")
    ap.add_argument("--per-cell", type=int, default=PER_CELL)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.src, a.out, a.per_cell, a.seed)
