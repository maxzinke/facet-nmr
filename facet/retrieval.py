"""Retrieval-augmented FACET inference — kNN over a bundled reference index.

Replaces parametric argmax decoding with a kNN search over ~253K BMRB
training-residue embeddings + DBSCAN clustering of retrieved neighbors'
(phi, psi). Emits multi-modal predictions with TALOS-N-style confidence
tiers (Strong / Generous / Ambiguous / None) when the retrieved cluster
agreement warrants abstention.

**Why this helps**: chemical shifts are time-averaged observables, so
for flexible residues no single (phi, psi) fully explains them. Retrieval
finds precedents — individual conformers from deposited structures —
and aggregates their angles. Aggregate beats any single parametric
point prediction, especially on coil.

**Usage**:

    from facet.retrieval import FACETRetrieval

    retr = FACETRetrieval(
        checkpoint_path="facet/weights/facet_v3.pt",
        index_path="facet/weights/facet_retrieval_index.npz",
    )
    results = retr.predict(shifts, masks, aa_idx, flags, k=25)
    # results[i]: RetrievalResult with .clusters (list of RetrievalCluster)
    # and .tier ∈ {"Strong", "Generous", "Ambiguous", "None"}.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────
# DBSCAN (simple O(n^2), fine for top-25)
# ─────────────────────────────────────────────────────────────


def _dbscan_simple(
    points_rad: np.ndarray,
    eps_deg: float = 30.0,
    min_size: int = 3,
) -> np.ndarray:
    """Minimal DBSCAN-style clustering on 2D periodic (phi, psi) in radians.

    Returns an int array of cluster labels (-1 for noise). Uses an eps
    threshold on angular distance (deg). For small n (<=25) this is fine
    at O(n^2).
    """
    n = len(points_rad)
    if n == 0:
        return np.array([], dtype=np.int32)

    phi = points_rad[:, 0]
    psi = points_rad[:, 1]
    dphi = np.abs(phi[:, None] - phi[None, :])
    dphi = np.minimum(dphi, 2 * np.pi - dphi)
    dpsi = np.abs(psi[:, None] - psi[None, :])
    dpsi = np.minimum(dpsi, 2 * np.pi - dpsi)
    d_deg = np.sqrt(dphi ** 2 + dpsi ** 2) * (180.0 / np.pi)
    adj = d_deg <= eps_deg

    labels = np.full(n, -1, dtype=np.int32)
    cluster_id = 0
    for seed in range(n):
        if labels[seed] != -1:
            continue
        frontier = [seed]
        members: list[int] = []
        visited = {seed}
        while frontier:
            pt = frontier.pop()
            members.append(pt)
            for nb in np.where(adj[pt])[0]:
                if nb not in visited:
                    visited.add(nb)
                    frontier.append(nb)
        if len(members) >= min_size:
            labels[members] = cluster_id
            cluster_id += 1
    return labels


def _circular_mean_std_deg(angles_rad: np.ndarray) -> tuple[float, float]:
    """Circular mean + angular std deviation in degrees."""
    sin_m = float(np.sin(angles_rad).mean())
    cos_m = float(np.cos(angles_rad).mean())
    mean = math.atan2(sin_m, cos_m)
    r = math.sqrt(sin_m * sin_m + cos_m * cos_m)
    r = min(r, 1.0)
    std = math.sqrt(-2 * math.log(max(r, 1e-6)))
    return math.degrees(mean), math.degrees(std)


# ─────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────


@dataclass
class RetrievalCluster:
    """One mode of a multi-modal retrieval prediction."""
    phi_deg: float
    psi_deg: float
    phi_std_deg: float
    psi_std_deg: float
    size: int           # number of neighbors in this cluster
    weight: float       # fraction of top-k neighbors that joined this cluster
    basin: int          # most common basin label among cluster members (0-3)


@dataclass
class RetrievalResult:
    clusters: list[RetrievalCluster]
    tier: str            # "Strong" / "Generous" / "Ambiguous" / "None"
    n_neighbors: int     # how many neighbors retrieved
    basin_populations: list[float] = field(default_factory=list)
    # 4-element list of basin population fractions: [alpha, beta, PPII, other].
    # Derived from cluster weights × cluster basin assignments.


# ─────────────────────────────────────────────────────────────
# Retrieval wrapper
# ─────────────────────────────────────────────────────────────


class FACETRetrieval:
    """Retrieval-augmented FACET inference.

    Loads an encoder checkpoint + a reference index (phi/psi/ss/basin
    metadata). At inference: encode the query residues, do cosine kNN
    over the index, DBSCAN-cluster the retrieved neighbors' (phi, psi),
    return ranked clusters + a confidence tier.

    Same encoder can be used for either retrieval or parametric (argmax)
    inference — the ``FACETv3.encode`` method is shared.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        index_path: str | Path,
        device: str | None = None,
    ):
        from .model import FACETv3, FACETv3Config

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Load encoder. The bundled checkpoint has the ExpectedErrorHead
        # (added 2026-04-17 for calibration), so use_error_head=True.
        config = FACETv3Config(retrieval_dim=0, use_error_head=True)
        self.model = FACETv3(config).to(self.device)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

        # Load + normalize index
        data = np.load(index_path)
        emb = data["embeddings"].astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings_normed = emb / norms
        self.phi = data["phi"].astype(np.float32)
        self.psi = data["psi"].astype(np.float32)
        self.ss = data["ss"]
        self.basin = data["basin"]
        self.aa_idx = data["aa_idx"]
        # Optional source flag (0=BMRB, 1=Phase2.1) — safe default if absent
        self.source = data["source"] if "source" in data.files else np.zeros(len(emb), dtype=np.int8)
        self.n_index = len(self.embeddings_normed)
        self.embed_dim = int(self.embeddings_normed.shape[1])

    @torch.no_grad()
    def _encode(self, shifts, masks, aa_idx, flags) -> np.ndarray:
        """Run the encoder center-residue output (pre-SS-conditioning)."""
        h_seq = self.model.encode(shifts, masks, aa_idx, flags)
        center = shifts.shape[1] // 2
        h = h_seq[:, center, :]  # (B, D)
        return h.cpu().numpy().astype(np.float32)

    def _knn_batch(self, queries_normed: np.ndarray, k: int) -> np.ndarray:
        """Batched kNN: returns (B, k) indices, sorted by similarity desc."""
        sims = self.embeddings_normed @ queries_normed.T  # (N_index, B)
        top = np.argpartition(-sims, k, axis=0)[:k]        # (k, B)
        out = np.empty((queries_normed.shape[0], k), dtype=np.int64)
        for b in range(queries_normed.shape[0]):
            order = np.argsort(-sims[top[:, b], b])
            out[b] = top[order, b]
        return out

    def predict(
        self,
        shifts: torch.Tensor,
        masks: torch.Tensor,
        aa_idx: torch.Tensor,
        flags: torch.Tensor,
        k: int = 25,
        dbscan_eps_deg: float = 30.0,
        dbscan_min_size: int = 3,
    ) -> list[RetrievalResult]:
        """Predict multi-modal (phi, psi) for each residue in the batch.

        Returns a list of ``RetrievalResult``, one per batch element.
        Each result has:
          - ``clusters``: sorted by size desc; top cluster is the primary
            prediction. Each cluster carries per-basin classification.
          - ``basin_populations``: [alpha_R, beta, PPII, other] fractions
            summing to 1 (over clustered neighbors only).
          - ``tier``: TALOS-N-style ``Strong``/``Generous``/``Ambiguous``/``None``.
        """
        h = self._encode(
            shifts.to(self.device), masks.to(self.device),
            aa_idx.to(self.device), flags.to(self.device),
        )
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        h_normed = h / norms

        top_indices_batch = self._knn_batch(h_normed, k)

        results: list[RetrievalResult] = []
        for b in range(len(h_normed)):
            top_idx = top_indices_batch[b]
            phi_nb = self.phi[top_idx]
            psi_nb = self.psi[top_idx]
            basin_nb = self.basin[top_idx]

            pts = np.stack([phi_nb, psi_nb], axis=1)
            labels = _dbscan_simple(pts, dbscan_eps_deg, dbscan_min_size)

            clusters: list[RetrievalCluster] = []
            max_label = int(labels.max()) if labels.size and labels.max() >= 0 else -1
            for cid in range(max_label + 1):
                mask = labels == cid
                if mask.sum() < dbscan_min_size:
                    continue
                phi_mean, phi_std = _circular_mean_std_deg(phi_nb[mask])
                psi_mean, psi_std = _circular_mean_std_deg(psi_nb[mask])
                basin_counts = np.bincount(basin_nb[mask], minlength=4)
                dominant_basin = int(basin_counts.argmax())
                clusters.append(RetrievalCluster(
                    phi_deg=phi_mean,
                    psi_deg=psi_mean,
                    phi_std_deg=phi_std,
                    psi_std_deg=psi_std,
                    size=int(mask.sum()),
                    weight=float(mask.sum()) / k,
                    basin=dominant_basin,
                ))
            clusters.sort(key=lambda c: -c.size)

            # Basin populations (over clustered neighbors only, normalized to 1)
            clustered_mask = labels >= 0
            if clustered_mask.sum() > 0:
                basins_clustered = basin_nb[clustered_mask]
                bp = np.bincount(basins_clustered, minlength=4)[:4].astype(np.float32)
                bp /= bp.sum()
                basin_populations = bp.tolist()
            else:
                basin_populations = [0.0, 0.0, 0.0, 0.0]

            # Tier assignment
            if len(clusters) == 1 and clusters[0].size >= k - 2:
                tier = "Strong"
            elif len(clusters) >= 1 and clusters[0].size >= 10:
                tier = "Generous"
            elif len(clusters) >= 2 and clusters[0].size < 2 * clusters[1].size:
                tier = "Ambiguous"
            else:
                tier = "None"

            results.append(RetrievalResult(
                clusters=clusters, tier=tier, n_neighbors=k,
                basin_populations=basin_populations,
            ))

        return results
