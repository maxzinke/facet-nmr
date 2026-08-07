#!/usr/bin/env python
"""HARNESS EXTENSION of noft/benchmarks/facet_coverage_ablation.py.

WHAT CHANGED vs the benchmark
-----------------------------
* Loading, leak-safe split, --max-test entry selection, ablation and the
  angular-error metric are COPIED VERBATIM from facet_coverage_ablation.py.
* The Tier-A `HierarchicalDatabase` path and the parametric foil are dropped
  (not needed here).
* `MaskedShiftRetrieval._query` is re-implemented line-for-line as
  `_query_instrumented`, adding ONLY diagnostic outputs (no change to the
  distance, the dedup, the candidate cap, or the aa bonus). A `no_dedup`
  switch is added and is OFF unless explicitly requested.
* Per-query diagnostics are written to an npz instead of medians to stdout.

Everything the benchmark reports is reproducible from the dump, so the
headline medians can be checked against the harness.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

NOFT = Path(r"C:\Users\maxim\Documents\coding\noft")
sys.path.insert(0, str(NOFT))

CACHE_DIR = Path(os.path.expanduser("~/.crystalline_fid/crystalline_cache"))
ATOM_ORDER = ["H", "HA", "N", "CA", "CB", "C"]
ATOM_IDX = {a: i for i, a in enumerate(ATOM_ORDER)}

SCENARIOS = {
    "full": [],
    "-HA": ["HA"],
    "-H": ["H"],
    "-H-HA": ["H", "HA"],
    "-CB": ["CB"],
    "-H-HA-CB": ["H", "HA", "CB"],
}

INDICES = {
    "full_records": NOFT / "checkpoints/facet_shift_reference.npz",
    "cent14x": Path(r"C:\Users\maxim\Documents\coding\facet-nmr\facet\weights\cent_pc400.npz"),
    "cent88x": Path(r"C:\Users\maxim\Documents\coding\facet-nmr\facet\weights\facet_shift_reference_centroids.npz"),
}


def angular_errors_deg(pred_phi, pred_psi, true_phi, true_psi):
    dphi = np.abs(pred_phi - true_phi)
    dphi = np.minimum(dphi, 360.0 - dphi)
    dpsi = np.abs(pred_psi - true_psi)
    dpsi = np.minimum(dpsi, 360.0 - dpsi)
    return (dphi + dpsi) / 2.0


def ablate(shifts, masks, atoms):
    s = shifts.copy()
    m = masks.copy()
    for a in atoms:
        j = ATOM_IDX[a]
        s[:, j] = 0.0
        m[:, j] = 0.0
    return s, m


def load_ground_truth():
    for fn in ("crystalline_all_phipsi_ensemble.npz", "crystalline_all_phipsi.npz"):
        p = CACHE_DIR / fn
        if p.is_file():
            z = np.load(p)
            pp = z["phi_psi"]
            wd = z["well_defined"].astype(bool) if "well_defined" in z.files else None
            return pp, wd, fn
    raise FileNotFoundError("no phi_psi npz")


# ── instrumented query (verbatim logic + diagnostics) ────────────────────
def query_instrumented(R, q_shifts, q_masks, aa_triplet, k, no_dedup=False):
    if aa_triplet is not None and aa_triplet[1] >= 0:
        center_match = R._center_aa == aa_triplet[1]
        if center_match.sum() < k:
            center_match = np.ones(R.size, dtype=bool)
    else:
        center_match = np.ones(R.size, dtype=bool)
    idx_pool = np.where(center_match)[0]
    if len(idx_pool) == 0:
        return [], {}

    pool_shifts = R.triplet_shifts[idx_pool]
    pool_masks = R.triplet_masks[idx_pool]
    pool_weights = R.weights[idx_pool]

    shared = pool_masks * q_masks[np.newaxis, :]
    diff = (pool_shifts - q_shifts[np.newaxis, :]) * shared
    standardized = diff / R.atom_stds[np.newaxis, :]
    n_shared = shared.sum(axis=1)
    dists = np.sum(standardized ** 2, axis=1)
    valid = n_shared >= 2
    dists[~valid] = 1e10
    dists[valid] /= n_shared[valid]

    n_bonus = 0
    if aa_triplet is not None:
        pool_aa = R.aa_triplets[idx_pool]
        full_match = np.ones(len(idx_pool), dtype=bool)
        if aa_triplet[0] >= 0:
            full_match &= pool_aa[:, 0] == aa_triplet[0]
        if aa_triplet[2] >= 0:
            full_match &= pool_aa[:, 2] == aa_triplet[2]
        dists[full_match] *= 0.6
        n_bonus = int(full_match.sum())

    n_cand = min(k * 5, len(dists))
    if n_cand == 0:
        return [], {}
    top_local = np.argpartition(dists, min(n_cand - 1, len(dists) - 1))[:n_cand]
    top_local = top_local[np.argsort(dists[top_local])]

    matches = []
    seen = set()
    n_skipped = 0
    for li in top_local:
        d = dists[li]
        if d >= 1e10:
            break
        gi = idx_pool[li]
        eid = R.entry_ids[gi]
        if (not no_dedup) and eid in seen:
            n_skipped += 1
            continue
        seen.add(eid)
        matches.append({
            "distance": float(d),
            "phi": float(R.phi[gi]),
            "psi": float(R.psi[gi]),
            "ss_label": int(R.ss_labels[gi]),
            "lookup_weight": float(pool_weights[li]),
            "n_shared": float(n_shared[li]),
            "gi": int(gi),
        })
        if len(matches) >= k:
            break

    diag = {
        "pool_size": int(len(idx_pool)),
        "n_valid_pool": int(valid.sum()),
        "n_bonus": n_bonus,
        "n_skipped_dedup": n_skipped,
        "n_cand_scanned": int(len(top_local)),
    }
    return matches, diag


def circ_std_deg(a_deg):
    if len(a_deg) < 2:
        return 0.0
    r = np.radians(a_deg)
    R = math.hypot(np.sin(r).mean(), np.cos(r).mean())
    return float(np.degrees(math.sqrt(max(0.0, -2 * math.log(max(R, 1e-12))))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-test", type=int, default=4000)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--out", default=str(Path(os.environ.get("SCRATCH", ".")) / "crossover_dump.npz"))
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    meta = json.load(open(CACHE_DIR / "crystalline_all_meta.json"))
    npz = np.load(CACHE_DIR / "crystalline_all.npz")
    shifts_all = npz["shifts"]; masks_all = npz["masks"]; labels_all = npz["labels"]
    pp_all, wd_all, gt_src = load_ground_truth()
    sequences = meta.get("sequences", {})
    entry_ids_all = meta["entry_ids"]
    comp_ids_all = meta["comp_ids"]
    seq_ids_all = [int(s) for s in meta["seq_ids"]]

    from crystalline_fid.crystalline.data.splits import cluster_by_similarity, split_clusters
    clusters = cluster_by_similarity(sequences, threshold=0.5, linkage="connected_components")
    train_list, _val, test_list = split_clusters(clusters)
    test_entries = set(test_list)

    test_by_entry = defaultdict(list)
    for i, e in enumerate(entry_ids_all):
        if e in test_entries:
            test_by_entry[e].append(i)
    selected_entries = sorted(test_by_entry.keys())
    if args.max_test:
        chosen, total = [], 0
        for e in selected_entries:
            chosen.append(e); total += len(test_by_entry[e])
            if total >= args.max_test:
                break
        selected_entries = chosen
    ordered_idx = []
    for e in selected_entries:
        ordered_idx.extend(sorted(test_by_entry[e], key=lambda i: seq_ids_all[i]))
    ordered_idx = np.array(ordered_idx, dtype=np.int64)

    t_shifts = shifts_all[ordered_idx].astype(np.float64)
    t_masks = masks_all[ordered_idx].astype(np.float64)
    t_comp = [comp_ids_all[i] for i in ordered_idx]
    t_entry = [entry_ids_all[i] for i in ordered_idx]
    t_seq = [seq_ids_all[i] for i in ordered_idx]
    t_labels = labels_all[ordered_idx]
    t_pp = pp_all[ordered_idx]
    valid = np.isfinite(t_pp[:, 0]) & np.isfinite(t_pp[:, 1])
    if wd_all is not None:
        valid &= wd_all[ordered_idx]
    gt_phi = np.degrees(t_pp[:, 0]); gt_psi = np.degrees(t_pp[:, 1])
    n = len(ordered_idx)
    print(f"[split] test={n} residues, {len(selected_entries)} entries, valid={valid.sum()}",
          flush=True)

    from crystalline_fid.crystalline.baselines.masked_retrieval import (
        MaskedShiftRetrieval, to_secondary_shifts, build_triplets,
    )

    out = {
        "valid": valid, "gt_phi": gt_phi, "gt_psi": gt_psi, "labels": t_labels,
        "entry": np.array(t_entry), "comp": np.array(t_comp), "seq": np.array(t_seq),
    }

    K = args.k
    for iname, ipath in INDICES.items():
        R = MaskedShiftRetrieval(ipath)
        print(f"[index] {iname}: {R.size} rows ({time.time()-t0:.0f}s)", flush=True)
        for sname, atoms in SCENARIOS.items():
            ts = time.time()
            s, m = ablate(t_shifts, t_masks, atoms)
            sec = to_secondary_shifts(s, m, t_comp)
            tri_s, tri_m, aa_trip = build_triplets(sec, m, t_comp, t_entry, t_seq)

            pf = np.zeros(n); pq = np.zeros(n)
            nmatch = np.zeros(n, np.int32)
            d1 = np.full(n, np.nan); dk = np.full(n, np.nan); dmean = np.full(n, np.nan)
            top1_err = np.full(n, np.nan)
            oracle_err = np.full(n, np.nan)
            mean_nb_err = np.full(n, np.nan)
            med_nb_err = np.full(n, np.nan)
            spread = np.full(n, np.nan)
            nshared1 = np.full(n, np.nan)
            nobs18 = tri_m.sum(1)
            nobs_center = tri_m[:, 6:12].sum(1)
            pool_sz = np.zeros(n, np.int32)
            n_skip = np.zeros(n, np.int32)
            n_bonus = np.zeros(n, np.int32)
            nsupport = np.zeros(n, np.int32)
            phistd = np.full(n, np.nan)
            conf = np.zeros(n, np.int8)
            CONF = {"strong": 3, "good": 2, "weak": 1, "insufficient": 0}

            for i in range(n):
                matches, diag = query_instrumented(
                    R, tri_s[i], tri_m[i], aa_trip[i], K, no_dedup=args.no_dedup)
                pred = R._build_prediction(matches)
                pf[i] = pred["phi_deg"]; pq[i] = pred["psi_deg"]
                nsupport[i] = pred["n_support"]; phistd[i] = pred["phi_std_deg"]
                conf[i] = CONF[pred["confidence"]]
                if not matches:
                    continue
                nmatch[i] = len(matches)
                dd = np.array([mm["distance"] for mm in matches])
                d1[i] = dd[0]; dk[i] = dd[-1]; dmean[i] = dd.mean()
                nshared1[i] = matches[0]["n_shared"]
                mp = np.degrees(np.array([mm["phi"] for mm in matches]))
                ms = np.degrees(np.array([mm["psi"] for mm in matches]))
                e = angular_errors_deg(mp, ms, gt_phi[i], gt_psi[i])
                top1_err[i] = e[0]; oracle_err[i] = e.min()
                mean_nb_err[i] = e.mean(); med_nb_err[i] = np.median(e)
                spread[i] = 0.5 * (circ_std_deg(mp) + circ_std_deg(ms))
                pool_sz[i] = diag["pool_size"]; n_skip[i] = diag["n_skipped_dedup"]
                n_bonus[i] = diag["n_bonus"]

            err = angular_errors_deg(pf, pq, gt_phi, gt_psi)
            pre = f"{iname}|{sname}"
            out[f"{pre}|err"] = err
            out[f"{pre}|phi"] = pf; out[f"{pre}|psi"] = pq
            out[f"{pre}|nmatch"] = nmatch
            out[f"{pre}|d1"] = d1; out[f"{pre}|dk"] = dk; out[f"{pre}|dmean"] = dmean
            out[f"{pre}|top1_err"] = top1_err
            out[f"{pre}|oracle_err"] = oracle_err
            out[f"{pre}|mean_nb_err"] = mean_nb_err
            out[f"{pre}|med_nb_err"] = med_nb_err
            out[f"{pre}|spread"] = spread
            out[f"{pre}|nshared1"] = nshared1
            out[f"{pre}|pool"] = pool_sz
            out[f"{pre}|nskip"] = n_skip
            out[f"{pre}|nbonus"] = n_bonus
            out[f"{pre}|nsupport"] = nsupport
            out[f"{pre}|phistd"] = phistd
            out[f"{pre}|conf"] = conf
            out[f"nobs18|{sname}"] = nobs18
            out[f"nobs_center|{sname}"] = nobs_center
            print(f"  [{sname:>9}] median={np.median(err[valid]):6.2f} "
                  f"mean={np.mean(err[valid]):6.2f} ({time.time()-ts:.0f}s)", flush=True)

    np.savez_compressed(args.out, **out)
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
