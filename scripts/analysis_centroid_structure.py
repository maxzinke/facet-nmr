#!/usr/bin/env python
"""Structural analysis of the aggregated (centroid) references vs the full record set.

Reconstructs the k-means assignment used by scripts/build_centroid_reference.py
(exact: the builder's feature matrix X == triplet_shifts, and the stored
triplet_shifts of a centroid ARE the k-means centre), then measures, per centroid:
  * n_obs
  * within-cluster std of each shift column  -> the noise that averaging removes
  * within-cluster circular std of phi/psi   -> the signal that averaging blurs
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np

W = Path(r"C:\Users\maxim\Documents\coding\facet-nmr\facet\weights")
FULL = W / "facet_shift_reference.npz"
CENT = {"cent14x": W / "cent_pc400.npz", "cent88x": W / "facet_shift_reference_centroids.npz"}
LBL = ["H-1", "HA-1", "N-1", "CA-1", "CB-1", "C-1",
       "H", "HA", "N", "CA", "CB", "C",
       "H+1", "HA+1", "N+1", "CA+1", "CB+1", "C+1"]


def circ_std_deg(rad):
    if len(rad) < 2:
        return 0.0
    R = math.hypot(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.degrees(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12))))))


z = np.load(FULL, allow_pickle=False)
S = z["triplet_shifts"].astype(np.float64)
M = z["triplet_masks"].astype(np.float64)
PHI = z["phi"].astype(np.float64); PSI = z["psi"].astype(np.float64)
SS = z["ss_labels"].astype(np.int64); AA = z["aa_triplets"].astype(np.int64)
STD = z["atom_stds"].astype(np.float64)
EID = z["entry_ids"]

print("=" * 78)
print("A. FULL RECORD SET")
print("=" * 78)
print(f"rows={len(PHI):,}  entries={len(set(EID.tolist())):,}")
occ = M.mean(0)
print("column occupancy (fraction of rows with the atom observed):")
for j in range(18):
    print(f"  {LBL[j]:>6s} occ={occ[j]:.3f}  atom_std={STD[j]:8.3f}")
print(f"mean observed columns per row: {M.sum(1).mean():.2f}   "
      f"median {np.median(M.sum(1)):.0f}")
print("centre-block (6 cols) occupancy:", np.round(M[:, 6:12].mean(0), 3))

# rows per (aa,ss) cell
cells = {}
for aa in np.unique(AA[:, 1]):
    for ss in np.unique(SS):
        cells[(aa, ss)] = int(((AA[:, 1] == aa) & (SS == ss)).sum())
cv = np.array(sorted(cells.values()))
print(f"(aa,ss) cells: {len(cells)}  rows/cell min={cv.min()} med={int(np.median(cv))} max={cv.max()}")

for cname, cpath in CENT.items():
    print()
    print("=" * 78)
    print(f"B. {cname}  ({cpath.name})")
    print("=" * 78)
    zc = np.load(cpath, allow_pickle=False)
    CS = zc["triplet_shifts"].astype(np.float64)
    CM = zc["triplet_masks"].astype(np.float64)
    CPHI = zc["phi"].astype(np.float64); CPSI = zc["psi"].astype(np.float64)
    CSS = zc["ss_labels"].astype(np.int64); CAA = zc["aa_triplets"].astype(np.int64)
    NOBS = zc["n_observations"].astype(np.int64)
    PSD = zc["phi_circ_std"].astype(np.float64); QSD = zc["psi_circ_std"].astype(np.float64)
    print(f"rows={len(CPHI):,}  compression={len(PHI)/len(CPHI):.1f}x")
    print(f"n_obs per centroid: mean={NOBS.mean():.1f} med={int(np.median(NOBS))} "
          f"p10={int(np.percentile(NOBS,10))} p90={int(np.percentile(NOBS,90))} max={NOBS.max()}")
    print(f"rows covered by centroids: {NOBS.sum():,} of {len(PHI):,} "
          f"({100*NOBS.sum()/len(PHI):.1f}%)")
    print(f"unique entry_ids in index: {len(set(zc['entry_ids'].tolist()))} "
          f"(= n rows -> dedup by entry_id is a NO-OP here)")

    # --- exact re-assignment of full rows to their centroid ---
    assign = np.full(len(PHI), -1, np.int64)
    for aa in np.unique(AA[:, 1]):
        for ss in np.unique(SS):
            sel = np.where((AA[:, 1] == aa) & (SS == ss))[0]
            csel = np.where((CAA[:, 1] == aa) & (CSS == ss))[0]
            if sel.size == 0 or csel.size == 0:
                continue
            d = ((S[sel][:, None, :] - CS[csel][None]) ** 2).sum(-1)
            assign[sel] = csel[np.argmin(d, 1)]
    print(f"assigned {int((assign>=0).sum()):,} / {len(PHI):,} rows")

    # per-centroid within-cluster stats
    wstd = np.full((len(CPHI), 18), np.nan)
    wphi = np.full(len(CPHI), np.nan); wpsi = np.full(len(CPHI), np.nan)
    nmem = np.zeros(len(CPHI), np.int64)
    for c in range(len(CPHI)):
        g = np.where(assign == c)[0]
        nmem[c] = len(g)
        if len(g) < 2:
            continue
        for j in range(18):
            v = S[g, j][M[g, j] > 0]
            if v.size >= 2:
                wstd[c, j] = v.std()
        wphi[c] = circ_std_deg(PHI[g]); wpsi[c] = circ_std_deg(PSI[g])

    ok = nmem >= 5
    print(f"\ncentroids with >=5 reassigned members: {ok.sum()}")
    print("\n  WITHIN-CLUSTER SHIFT SPREAD vs GLOBAL SPREAD (ppm)")
    print(f"  {'col':>6s} {'global sd':>10s} {'within sd':>10s} {'within/global':>14s} "
          f"{'SE of centroid':>15s}")
    for j in range(18):
        w = np.nanmedian(wstd[ok, j])
        se = np.nanmedian(wstd[ok, j] / np.sqrt(np.maximum(nmem[ok], 1)))
        print(f"  {LBL[j]:>6s} {STD[j]:10.3f} {w:10.3f} {w/STD[j]:14.3f} {se:15.4f}")

    print("\n  WITHIN-CLUSTER ANGULAR SPREAD (deg, circular std)")
    for nm, arr, stored in (("phi", wphi, PSD), ("psi", wpsi, QSD)):
        a = arr[ok]
        print(f"  {nm}: med={np.nanmedian(a):6.1f}  p25={np.nanpercentile(a,25):6.1f}  "
              f"p75={np.nanpercentile(a,75):6.1f}  p90={np.nanpercentile(a,90):6.1f}   "
              f"[stored in file: med={np.median(stored):.1f}]")
    # weighted by members (what a query actually meets)
    wt = nmem[ok].astype(float)
    print(f"  member-weighted median phi spread: "
          f"{np.nansum(np.where(np.isnan(wphi[ok]),0,wphi[ok])*wt)/wt.sum():.1f} deg (mean)")

    print("\n  MASK BINARISATION (an atom is 'present' when >50% of members had it)")
    print("  centroid occupancy:", np.round(CM.mean(0), 3))
    print("  full   occupancy:  ", np.round(M.mean(0), 3))
    print(f"  mean observed columns per centroid row: {CM.sum(1).mean():.2f} "
          f"(full: {M.sum(1).mean():.2f})")

    print("\n  NEIGHBOUR-AA FIELD (used for the x0.6 distance bonus)")
    # how often does the stored aa_triplet neighbour match the cluster majority?
    agree_prev, agree_next, npair = 0, 0, 0
    for c in np.where(ok)[0]:
        g = np.where(assign == c)[0]
        for k_, col in ((0, 0), (2, 2)):
            vals = AA[g, col]
            vals = vals[vals >= 0]
            if vals.size == 0:
                continue
            maj = np.bincount(vals).max() / vals.size
            if col == 0:
                agree_prev += maj
            else:
                agree_next += maj
        npair += 1
    print(f"  mean purity of the i-1 aa within a centroid: {agree_prev/npair:.3f}")
    print(f"  mean purity of the i+1 aa within a centroid: {agree_next/npair:.3f}")
    print("  (1/20 = 0.05 would be chance; the stored value is member[0]'s, "
          "so the bonus fires on ~this fraction of a cell)")
