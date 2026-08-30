"""Kernel-weighted Bayesian structural SS-state populations.

Companion to facet.retrieval. Where the geometric basin populations
(facet/retrieval.py) answer "which region of Ramachandran do the
retrieved neighbors sample", this module answers the d2D/CheSPI
question: "what fraction of the ensemble is in canonical helix /
strand / PPII / coil STATE".

Algorithm — kernel-weighted Bayesian inference over the full retrieval
index, using the learned embedding as a non-parametric likelihood::

    P(state = s | query, AA) = [Σᵢ K(h_q, hᵢ) · 1[stateᵢ = s ∧ AAᵢ = AA]]
                             / [Σᵢ K(h_q, hᵢ) · 1[AAᵢ = AA]]

    K(h_q, hᵢ) = exp(β · cos_sim(h_q, hᵢ))

Where:
  - i runs over all training residues in the retrieval index (~220K).
  - stateᵢ ∈ {H, E, P, C}:
      H if SSᵢ = H  (helix, from the deposited PDB secondary-structure records)
      E if SSᵢ = E  (strand)
      P if SSᵢ = C (loop) AND basinᵢ = PPII — matches d2D's PPII defn
      C otherwise (loop not in PPII region)
  - β is a softmax temperature hyperparameter (default 15).

Conceptually this is "non-parametric Bayesian d2D using FACET's learned
shift embeddings instead of hand-crafted per-AA Gaussians". Advantages:

  - Non-parametric in shift space: no Gaussian assumption.
  - Per-AA conditioning: matches d2D's Bayesian factorisation exactly.
  - Low variance: softmax over all 220K rows, not a count out of 25.
  - Unified pipeline: one forward pass, two complementary outputs.

Validation target: PED ensemble-derived populations at least as well
as d2D does. See ``docs/future_dreams/260420_facet_v02_gaps_plan.md``
Phase 3.5.3 for the calibration plan.
"""
from __future__ import annotations


import numpy as np


# Per-residue SS-state indices for the structural populations vector.
STATE_HELIX = 0
STATE_BETA  = 1
STATE_PPII  = 2
STATE_COIL  = 3

# State order in the returned 4-vector (matches d2D's output convention).
STATE_NAMES = ("helix", "beta", "ppii", "coil")


def _derive_state_labels(ss: np.ndarray, basin: np.ndarray) -> np.ndarray:
    """Map (3-state SS, basin 4-state) index labels to d2D's 4-state.

    Args:
        ss: int array of 3-state SS labels (0=H, 1=E, 2=C), from the PDB
            secondary-structure records of the reference structure.
        basin: int array of Ramachandran basin labels (0=α_R, 1=β,
            2=PPII, 3=other).

    Returns:
        int array of state labels (0=Helix, 1=Beta, 2=PPII, 3=Coil)
        of the same shape.
    """
    state = np.full_like(ss, STATE_COIL, dtype=np.int8)
    state[ss == 0] = STATE_HELIX
    state[ss == 1] = STATE_BETA
    # Loop (C) further split: PPII-region residues → PPII, else → Coil.
    loop_mask = (ss == 2)
    ppii_mask = loop_mask & (basin == 2)
    state[ppii_mask] = STATE_PPII
    return state


def compute_structural_populations(
    retriever,
    query_embeddings: np.ndarray,   # (B, D), not necessarily normalized
    query_aa_idx: np.ndarray,        # (B,) int labels into retriever.aa_idx space
    beta: float = 15.0,
    topk_truncate: int | None = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d2D-style structural SS-state populations per residue.

    Args:
        retriever: a ``facet.retrieval.FACETRetrieval`` instance (loaded
            index with embeddings, ss, basin, aa_idx arrays).
        query_embeddings: (B, D) raw embeddings for each query residue.
            Cosine similarity is computed internally (we re-normalize).
        query_aa_idx: (B,) AA identity index per query; conditioning on
            this filters the denominator + numerator to same-AA training
            residues (matches d2D's per-AA Bayesian factorisation).
        beta: softmax temperature for the similarity kernel. Higher β →
            sharper, more deterministic; lower → softer, more uniform.
            Default 15 gives effective neighbor counts in the
            O(10²–10³) range on typical queries.
        topk_truncate: if set, only the top-K most similar index rows
            contribute to the softmax. This captures > 99% of the mass
            at typical β while cutting cost dramatically. Set to None
            to use all ~220K rows.

    Returns:
        pops: (B, 4) float32 array, each row summing to 1.0. Columns
            are (helix, beta, ppii, coil) per STATE_NAMES.
        eff_n: (B,) float32 effective sample size per query (exp of
            entropy of the per-same-AA softmax weights). Useful as a
            precision proxy — low values mean the estimate relies on
            few neighbors and should be trusted less.
    """
    emb_index = retriever.embeddings_normed    # (N, D), already normalized
    ss_index = np.asarray(retriever.ss).astype(np.int8)
    basin_index = np.asarray(retriever.basin).astype(np.int8)
    aa_index = np.asarray(retriever.aa_idx).astype(np.int8)

    # Precompute state labels over the full index (once, not per query).
    states_index = _derive_state_labels(ss_index, basin_index)

    # Normalize queries for cosine similarity.
    q = np.asarray(query_embeddings, dtype=np.float32)
    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    q_normed = q / q_norms

    B = q_normed.shape[0]
    pops = np.zeros((B, 4), dtype=np.float32)
    eff_n = np.zeros(B, dtype=np.float32)

    for b in range(B):
        sims = emb_index @ q_normed[b]   # (N,)

        # Optionally truncate to top-K similar index rows BEFORE per-AA
        # masking — the softmax tail beyond the top few thousand
        # contributes negligible mass at β ≥ 10.
        if topk_truncate is not None and topk_truncate < len(sims):
            top_idx = np.argpartition(-sims, topk_truncate)[:topk_truncate]
            local_sims = sims[top_idx]
            local_states = states_index[top_idx]
            local_aa = aa_index[top_idx]
        else:
            top_idx = np.arange(len(sims))
            local_sims = sims
            local_states = states_index
            local_aa = aa_index

        # AA-conditioned softmax. Rows with AA != query AA are excluded
        # from both numerator and denominator (same effect as multiplying
        # their kernel weight by 0).
        aa_mask = (local_aa == int(query_aa_idx[b]))
        if not aa_mask.any():
            # No same-AA training residues among the top-K (extremely
            # rare). Fall back to all training data for this residue.
            aa_mask = np.ones_like(local_aa, dtype=bool)

        # Softmax weights with numerical stabilization (subtract max).
        scaled = beta * local_sims[aa_mask]
        scaled = scaled - scaled.max()
        w = np.exp(scaled, dtype=np.float64)
        w_sum = w.sum()
        if w_sum <= 0:
            continue

        w_norm = (w / w_sum).astype(np.float32)
        filtered_states = local_states[aa_mask]

        # Accumulate into 4 state bins.
        for s in range(4):
            pops[b, s] = float(w_norm[filtered_states == s].sum())

        # Effective sample size via exp of Shannon entropy of weights.
        entropy = -float((w_norm * np.log(w_norm + 1e-20)).sum())
        eff_n[b] = float(np.exp(entropy))

    return pops, eff_n
