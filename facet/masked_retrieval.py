"""Mask-safe shift-retrieval fallback for FACET backbone torsion prediction.

When required backbone shifts are missing (e.g. HA in perdeuterated samples),
the parametric FACET model collapses toward ~0 deg because it was trained
never to lose H/HA. This module predicts phi/psi via **mask-aware kNN** over a
reference index of secondary-shift triplets: the distance intersects the query
and reference masks and normalizes by the shared-atom count, so an absent atom
simply drops out of the match instead of poisoning it.

Pure NumPy, no torch — self-contained so it can be dropped into the external
`facet` package and the HF Space unchanged. Faithful to the validated
HierarchicalDatabase merged-query (Tier A) path; the same code builds the
index and queries it, so build and query conversions are identical by
construction.

Ships with ``facet_shift_reference.npz``. Build it via
``scripts/build_facet_shift_reference.py``.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Crystalline atom order: H=0, HA=1, N=2, CA=3, CB=4, C=5
BACKBONE_ATOMS: tuple[str, ...] = ("H", "HA", "N", "CA", "CB", "C")

# 0-indexed AA map (matches crystalline_fid.crystalline.data.features.AA_TO_INDEX)
AA_ORDER = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

# Consensus random-coil shifts (ppm), used to convert observed shifts to secondary
# shifts. Static branch of
# crystalline_fid.crystalline.baselines.database_lookup._to_secondary_shifts.
#
# Values follow the standard random-coil references:
#   Wishart, D. S., Bigam, C. G., Holm, A., Hodges, R. S. & Sykes, B. D.
#   "1H, 13C and 15N random coil NMR chemical shifts of the common amino acids.
#   I. Investigations of nearest-neighbor effects."
#   J. Biomol. NMR 5, 67-81 (1995). doi:10.1007/BF00227471
#
#   Schwarzinger, S., Kroon, G. J. A., Foss, T. R., Chung, J., Wright, P. E. &
#   Dyson, H. J. "Sequence-dependent correction of random coil NMR chemical shifts."
#   J. Am. Chem. Soc. 123, 2970-2978 (2001). doi:10.1021/ja003760i
#
# Published numeric reference values from the literature, cited rather than merely
# attributed by surname -- the same standard applied to the other borrowed constants
# in this codebase.
RANDOM_COIL_SHIFTS: dict[str, dict[str, float]] = {
    "ALA": {"H": 8.24, "HA": 4.32, "N": 123.8, "CA": 52.5, "CB": 19.1, "C": 177.8},
    "ARG": {"H": 8.23, "HA": 4.34, "N": 120.5, "CA": 56.0, "CB": 30.7, "C": 176.3},
    "ASN": {"H": 8.38, "HA": 4.74, "N": 118.7, "CA": 53.1, "CB": 38.7, "C": 175.2},
    "ASP": {"H": 8.34, "HA": 4.64, "N": 120.4, "CA": 54.0, "CB": 41.1, "C": 176.3},
    "CYS": {"H": 8.32, "HA": 4.55, "N": 118.8, "CA": 58.2, "CB": 28.0, "C": 174.6},
    "GLN": {"H": 8.27, "HA": 4.34, "N": 119.8, "CA": 55.7, "CB": 29.4, "C": 176.0},
    "GLU": {"H": 8.42, "HA": 4.29, "N": 120.2, "CA": 56.6, "CB": 30.0, "C": 176.6},
    "GLY": {"H": 8.33, "HA": 3.96, "N": 109.9, "CA": 45.4, "C": 174.9},
    "HIS": {"H": 8.42, "HA": 4.73, "N": 118.2, "CA": 55.0, "CB": 29.0, "C": 174.1},
    "ILE": {"H": 8.00, "HA": 4.17, "N": 119.9, "CA": 61.1, "CB": 38.8, "C": 176.4},
    "LEU": {"H": 8.16, "HA": 4.34, "N": 121.8, "CA": 55.1, "CB": 42.4, "C": 177.6},
    "LYS": {"H": 8.29, "HA": 4.32, "N": 120.4, "CA": 56.5, "CB": 32.7, "C": 176.6},
    "MET": {"H": 8.28, "HA": 4.48, "N": 119.6, "CA": 55.4, "CB": 32.9, "C": 176.3},
    "PHE": {"H": 8.30, "HA": 4.62, "N": 120.3, "CA": 57.7, "CB": 39.6, "C": 175.8},
    "PRO": {"HA": 4.42, "N": 136.6, "CA": 63.3, "CB": 32.1, "C": 177.3},
    "SER": {"H": 8.31, "HA": 4.47, "N": 115.7, "CA": 58.3, "CB": 63.8, "C": 174.6},
    "THR": {"H": 8.15, "HA": 4.35, "N": 113.6, "CA": 61.8, "CB": 69.8, "C": 174.7},
    "TRP": {"H": 8.25, "HA": 4.66, "N": 121.3, "CA": 57.5, "CB": 29.6, "C": 176.1},
    "TYR": {"H": 8.12, "HA": 4.55, "N": 120.3, "CA": 57.9, "CB": 38.8, "C": 175.9},
    "VAL": {"H": 8.03, "HA": 4.12, "N": 119.2, "CA": 62.4, "CB": 32.9, "C": 176.3},
}

_SS_LABELS = ("H", "E", "C")


# ─────────────────────────────────────────────────────────────
# Secondary shifts + triplet construction (shared by build & query)
# ─────────────────────────────────────────────────────────────

def to_secondary_shifts(
    shifts: np.ndarray, masks: np.ndarray, comp_ids: list[str],
) -> np.ndarray:
    """Raw ppm -> secondary shifts (observed - static random coil). 0 where absent."""
    sec = np.zeros_like(shifts, dtype=np.float64)
    for i, c in enumerate(comp_ids):
        rc = RANDOM_COIL_SHIFTS.get(str(c).upper(), {}) or {}
        for j, atom in enumerate(BACKBONE_ATOMS):
            if masks[i, j] > 0 and atom in rc:
                sec[i, j] = float(shifts[i, j]) - float(rc[atom])
    return sec


def build_triplets(
    sec_shifts: np.ndarray,
    masks: np.ndarray,
    comp_ids: list[str],
    entry_ids: list[str],
    seq_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (i-1, i, i+1) triplet feature rows per residue.

    Returns triplet_shifts (n,18), triplet_masks (n,18), aa_triplets (n,3).
    Neighbours are only used when they are sequence-adjacent within the same
    entry (seq_id +/- 1). Edge / missing neighbours stay zero-filled (mask 0).
    Mirrors the triplet loop in hierarchical_database.predict_hierarchical.
    """
    n = len(comp_ids)
    triplet_s = np.zeros((n, 18), dtype=np.float64)
    triplet_m = np.zeros((n, 18), dtype=np.float64)
    aa_trip = np.full((n, 3), -1, dtype=np.int32)

    # group residue rows by entry, ordered by seq_id
    groups: dict[Any, list[int]] = {}
    for i in range(n):
        groups.setdefault(entry_ids[i], []).append(i)
    for eid in groups:
        groups[eid].sort(key=lambda i: int(seq_ids[i]))
    pos_in_group = {}
    for eid, g in groups.items():
        for p, idx in enumerate(g):
            pos_in_group[idx] = (eid, p)

    for i in range(n):
        sid = int(seq_ids[i])
        for j in range(6):
            if masks[i, j] > 0:
                triplet_s[i, 6 + j] = sec_shifts[i, j]
                triplet_m[i, 6 + j] = 1.0
        aa_trip[i, 1] = AA_TO_INDEX.get(str(comp_ids[i]).upper(), -1)

        eid, p = pos_in_group[i]
        group = groups[eid]
        if p > 0:
            prev = group[p - 1]
            if int(seq_ids[prev]) == sid - 1:
                for j in range(6):
                    if masks[prev, j] > 0:
                        triplet_s[i, j] = sec_shifts[prev, j]
                        triplet_m[i, j] = 1.0
                aa_trip[i, 0] = AA_TO_INDEX.get(str(comp_ids[prev]).upper(), -1)
        if p < len(group) - 1:
            nxt = group[p + 1]
            if int(seq_ids[nxt]) == sid + 1:
                for j in range(6):
                    if masks[nxt, j] > 0:
                        triplet_s[i, 12 + j] = sec_shifts[nxt, j]
                        triplet_m[i, 12 + j] = 1.0
                aa_trip[i, 2] = AA_TO_INDEX.get(str(comp_ids[nxt]).upper(), -1)

    return triplet_s, triplet_m, aa_trip


# ─────────────────────────────────────────────────────────────
# Circular + clustering helpers (copied from fragment_clustering.py)
# ─────────────────────────────────────────────────────────────

def _angular_distance(phi1, psi1, phi2, psi2) -> float:
    dphi = abs(phi1 - phi2)
    if dphi > math.pi:
        dphi = 2 * math.pi - dphi
    dpsi = abs(psi1 - psi2)
    if dpsi > math.pi:
        dpsi = 2 * math.pi - dpsi
    return math.sqrt(dphi ** 2 + dpsi ** 2)


def _weighted_circular_mean(phis, psis, weights):
    w = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
    return (
        math.atan2(np.sum(w * np.sin(phis)), np.sum(w * np.cos(phis))),
        math.atan2(np.sum(w * np.sin(psis)), np.sum(w * np.cos(psis))),
    )


def _circular_std(angles) -> float:
    if len(angles) < 2:
        return 0.0
    R = math.sqrt(np.sin(angles).mean() ** 2 + np.cos(angles).mean() ** 2)
    return 0.0 if R >= 1.0 else math.sqrt(-2 * math.log(max(R, 1e-10)))


def _cluster_greedy(phis, psis, weights, threshold_deg=30.0) -> list[dict]:
    n = len(phis)
    if n == 0:
        return []
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    threshold = math.radians(threshold_deg)
    clusters: list[list[int]] = []
    centers: list[tuple[float, float]] = []
    for i in range(n):
        assigned = False
        for c_idx, (cphi, cpsi) in enumerate(centers):
            if _angular_distance(phis[i], psis[i], cphi, cpsi) < threshold:
                clusters[c_idx].append(i)
                centers[c_idx] = _weighted_circular_mean(
                    phis[clusters[c_idx]], psis[clusters[c_idx]], weights[clusters[c_idx]],
                )
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            centers.append((phis[i], psis[i]))
    results = []
    for members in clusters:
        m_phis, m_psis, m_w = phis[members], psis[members], weights[members]
        phi_c, psi_c = _weighted_circular_mean(m_phis, m_psis, m_w)
        results.append({
            "phi_center": phi_c, "psi_center": psi_c,
            "phi_std": _circular_std(m_phis), "psi_std": _circular_std(m_psis),
            "size": len(members), "member_indices": members,
            "total_weight": float(m_w.sum()),
        })
    results.sort(key=lambda c: c["size"], reverse=True)
    return results


def _assess_confidence(matches: list[dict]) -> str:
    """strong / good / weak / insufficient from match agreement (Tier A logic)."""
    if not matches:
        return "insufficient"
    n = len(matches)
    ss_counts = np.zeros(3)
    for m in matches:
        ss_counts[m["ss_label"]] += 1.0 / (m["distance"] + 1e-8)
    total = ss_counts.sum()
    ss_agreement = ss_counts.max() / total if total > 0 else 0.0
    phi_std = _circular_std(np.array([m["phi"] for m in matches]))
    psi_std = _circular_std(np.array([m["psi"] for m in matches]))
    angle_consistent = phi_std < math.radians(30) and psi_std < math.radians(30)
    mean_dist = float(np.mean([m["distance"] for m in matches]))
    if n >= 7 and ss_agreement > 0.8 and angle_consistent and mean_dist < 1.5:
        return "strong"
    if n >= 5 and ss_agreement > 0.6 and mean_dist < 3.0:
        return "good"
    if n >= 3 and ss_agreement > 0.4:
        return "weak"
    return "insufficient"


# ─────────────────────────────────────────────────────────────
# Index build + serialization
# ─────────────────────────────────────────────────────────────

_NPZ_ARRAYS = (
    "triplet_shifts", "triplet_masks", "phi", "psi",
    "ss_labels", "aa_triplets", "weights", "atom_stds",
)


def build_index(
    out_path: str | Path,
    shifts: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    phi_psi: np.ndarray,
    comp_ids: list[str],
    entry_ids: list[str],
    seq_ids: list[int],
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build the reference index from per-residue arrays and save to ``out_path``.

    ``shifts`` are RAW ppm (n,6); ``phi_psi`` radians (n,2). Each residue with
    a finite phi/psi becomes one fragment.
    """
    sec = to_secondary_shifts(shifts, masks, comp_ids)
    tri_s, tri_m, aa_trip = build_triplets(sec, masks, comp_ids, entry_ids, seq_ids)

    keep = np.isfinite(phi_psi[:, 0]) & np.isfinite(phi_psi[:, 1])
    idx = np.where(keep)[0]

    # per-position normalization: std over observed secondary-shift values
    obs = tri_m[idx] > 0
    atom_stds = np.ones(18, dtype=np.float64)
    for j in range(18):
        vals = tri_s[idx, j][obs[:, j]]
        if vals.size > 8:
            s = float(np.std(vals))
            atom_stds[j] = s if s > 1e-6 else 1.0

    w = np.ones(len(idx), dtype=np.float64) if weights is None else weights[idx].astype(np.float64)

    payload = {
        "triplet_shifts": tri_s[idx].astype(np.float32),
        "triplet_masks": tri_m[idx].astype(np.float32),
        "phi": phi_psi[idx, 0].astype(np.float32),
        "psi": phi_psi[idx, 1].astype(np.float32),
        "ss_labels": labels[idx].astype(np.int8),
        "aa_triplets": aa_trip[idx].astype(np.int16),
        "weights": w.astype(np.float32),
        "atom_stds": atom_stds.astype(np.float64),
        "entry_ids": np.array([str(entry_ids[i]) for i in idx]),
        "seq_ids": np.array([int(seq_ids[i]) for i in idx], dtype=np.int64),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return {"n_fragments": int(len(idx)), "path": str(out_path)}


# ─────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────

class MaskedShiftRetrieval:
    """Mask-aware shift-triplet kNN over a prebuilt reference index."""

    def __init__(self, index_path: str | Path):
        z = np.load(Path(index_path), allow_pickle=False)
        self.triplet_shifts = z["triplet_shifts"].astype(np.float64)
        self.triplet_masks = z["triplet_masks"].astype(np.float64)
        self.phi = z["phi"].astype(np.float64)
        self.psi = z["psi"].astype(np.float64)
        self.ss_labels = z["ss_labels"].astype(np.int64)
        self.aa_triplets = z["aa_triplets"].astype(np.int64)
        self.weights = z["weights"].astype(np.float64)
        self.atom_stds = z["atom_stds"].astype(np.float64)
        self.entry_ids = list(z["entry_ids"])
        self.seq_ids = z["seq_ids"].astype(np.int64)
        self.size = int(self.phi.shape[0])
        self._center_aa = self.aa_triplets[:, 1]

    # ---- single-residue query (mirrors HierarchicalDatabase._query_tier) ----
    def _query(self, q_shifts: np.ndarray, q_masks: np.ndarray,
               aa_triplet: np.ndarray, k: int) -> list[dict]:
        if aa_triplet is not None and aa_triplet[1] >= 0:
            center_match = self._center_aa == aa_triplet[1]
            if center_match.sum() < k:
                center_match = np.ones(self.size, dtype=bool)
        else:
            center_match = np.ones(self.size, dtype=bool)
        idx_pool = np.where(center_match)[0]
        if len(idx_pool) == 0:
            return []

        pool_shifts = self.triplet_shifts[idx_pool]
        pool_masks = self.triplet_masks[idx_pool]
        pool_weights = self.weights[idx_pool]

        shared = pool_masks * q_masks[np.newaxis, :]
        diff = (pool_shifts - q_shifts[np.newaxis, :]) * shared
        standardized = diff / self.atom_stds[np.newaxis, :]
        n_shared = shared.sum(axis=1)
        dists = np.sum(standardized ** 2, axis=1)

        # Rank by how much of the QUERY a candidate covers before ranking by distance.
        #
        # The score is a MEAN squared z over shared columns, so a row sharing only two
        # columns is judged on two samples and wins the argmin far too often -- a
        # winner's curse. Measured on the coverage-ablation benchmark: at -H-HA, 88% of
        # queries were won by a row covering under 40% of the query, and the Spearman
        # correlation between rank-1 distance and rank-1 error was +0.007. The distance
        # carried no quality information at all.
        #
        # tau is the coverage of the n_cand-th best-covering candidate, so the
        # threshold adapts to the pool instead of being a tuned constant, and reuses
        # the k*5 candidate cap that already existed. If too few rows survive, it
        # relaxes to the original n_shared >= 2 so a query can never be starved.
        n_cand = min(k * 5, len(dists))
        if len(n_shared) > n_cand:
            tau = np.partition(n_shared, len(n_shared) - n_cand)[len(n_shared) - n_cand]
            need = max(2.0, tau)
        else:
            need = 2.0
        valid = n_shared >= need
        if valid.sum() < min(k, len(n_shared)):
            valid = n_shared >= 2.0

        dists = np.where(valid, dists, np.inf)
        dists[valid] /= n_shared[valid]

        if aa_triplet is not None:
            pool_aa = self.aa_triplets[idx_pool]
            full_match = np.ones(len(idx_pool), dtype=bool)
            if aa_triplet[0] >= 0:
                full_match &= pool_aa[:, 0] == aa_triplet[0]
            if aa_triplet[2] >= 0:
                full_match &= pool_aa[:, 2] == aa_triplet[2]
            # Restricted to valid rows: scaling an inf sentinel produced a finite
            # value, so the 'no usable match' break below could never fire.
            dists[full_match & valid] *= 0.6

        n_cand = min(k * 5, len(dists))
        if n_cand == 0:
            return []
        top_local = np.argpartition(dists, min(n_cand - 1, len(dists) - 1))[:n_cand]
        top_local = top_local[np.argsort(dists[top_local])]

        matches: list[dict] = []
        seen: set = set()
        for li in top_local:
            d = dists[li]
            if not np.isfinite(d):
                break
            gi = idx_pool[li]
            eid = self.entry_ids[gi]
            if eid in seen:
                continue
            seen.add(eid)
            matches.append({
                "distance": float(d),
                "phi": float(self.phi[gi]),
                "psi": float(self.psi[gi]),
                "ss_label": int(self.ss_labels[gi]),
                "lookup_weight": float(pool_weights[li]),
            })
            if len(matches) >= k:
                break
        return matches

    @staticmethod
    def _build_prediction(matches: list[dict], mode_cluster_deg: float = 30.0) -> dict:
        if not matches:
            return {
                "phi_deg": 0.0, "psi_deg": 0.0, "phi_std_deg": 180.0, "psi_std_deg": 180.0,
                "ss_label": 2, "ss_probs": [0.0, 0.0, 1.0],
                "confidence": "insufficient", "n_modes": 0, "n_support": 0,
            }
        for m in matches:
            m["effective_weight"] = m["lookup_weight"] / (m["distance"] + 1e-8)
        ss_scores = np.zeros(3)
        for m in matches:
            ss_scores[m["ss_label"]] += m["effective_weight"]
        total = ss_scores.sum()
        ss_probs = ss_scores / total if total > 0 else np.array([0.0, 0.0, 1.0])

        phis = np.array([m["phi"] for m in matches])
        psis = np.array([m["psi"] for m in matches])
        eff = np.array([m["effective_weight"] for m in matches])
        clusters = _cluster_greedy(phis, psis, eff, mode_cluster_deg)
        total_cw = sum(c["total_weight"] for c in clusters)
        modes = []
        for c in clusters:
            modes.append({
                "phi": c["phi_center"], "psi": c["psi_center"],
                "phi_std": c["phi_std"], "psi_std": c["psi_std"],
                "weight": (c["total_weight"] / total_cw) if total_cw > 0 else 0.0,
                "size": c["size"],
            })
        modes.sort(key=lambda m: m["weight"], reverse=True)
        dom = modes[0]
        return {
            "phi_deg": math.degrees(dom["phi"]),
            "psi_deg": math.degrees(dom["psi"]),
            "phi_std_deg": math.degrees(dom["phi_std"]),
            "psi_std_deg": math.degrees(dom["psi_std"]),
            "ss_label": int(np.argmax(ss_probs)),
            "ss_probs": [float(x) for x in ss_probs],
            "confidence": _assess_confidence(matches),
            "n_modes": len(modes),
            "n_support": dom["size"],
        }

    def predict_residues(
        self,
        shifts: np.ndarray,
        masks: np.ndarray,
        comp_ids: list[str],
        seq_ids: list[int],
        entry_ids: list[str] | None = None,
        k: int = 25,
    ) -> list[dict]:
        """Predict phi/psi for each query residue via mask-aware retrieval.

        ``shifts`` raw ppm (n,6); ``masks`` (n,6). All residues are treated as
        one entry unless ``entry_ids`` is given. Returns one dict per residue
        with phi_deg/psi_deg/phi_std_deg/psi_std_deg/ss_label/ss_probs/
        confidence/n_modes/n_support (degrees).
        """
        n = len(comp_ids)
        if entry_ids is None:
            entry_ids = ["_q"] * n
        sec = to_secondary_shifts(np.asarray(shifts, dtype=np.float64),
                                  np.asarray(masks, dtype=np.float64), comp_ids)
        tri_s, tri_m, aa_trip = build_triplets(sec, np.asarray(masks), comp_ids, entry_ids, seq_ids)
        out = []
        for i in range(n):
            matches = self._query(tri_s[i], tri_m[i], aa_trip[i], k=k)
            out.append(self._build_prediction(matches))
        return out
