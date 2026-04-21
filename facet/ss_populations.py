"""Per-AA Gaussian SS-state population predictor — d2D engine, modern data.

This module is the "D2" option from docs/future_dreams/
260420_facet_v02_gaps_plan.md. It re-derives d2D's Gaussian Bayesian
inference on FACET's own 220K-residue modern, re-referenced index
instead of relying on Camilloni 2012's published parameter tables.

Complements facet.structural (kernel-weighted retrieval populations) by
offering a field-standard, retrieval-free readout of SS populations.
Useful for IDPs where the retrieval approach inherits folded-training
bias.

Inference is a per-residue multivariate Gaussian over the observed
backbone shifts::

    For state s in {H, E, P, C}:
        v_s = observed_shifts - mu[aa, s, :]
        L_s = N(0 | v_s, Sigma)  =  (2pi)^{-n/2} * |Sigma|^{-1/2}
                                    * exp(-1/2 * v_s^T Sigma^{-1} v_s)
    pops[s] = L_s / sum_s' L_s'

Where Sigma[a, b] = sigma[aa, a] * sigma[aa, b] * corr[a, b] — per-AA
sigmas (state-pooled, d2D convention) coupled by the pooled nucleus
correlation. Missing nuclei are dropped from the vector + matrix.

Optionally applies a 5-residue triangular smoothing (d2D does the same)
before normalisation, to reduce per-residue noise without muddling
genuine transitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .io.formats import BACKBONE_NUCLEI, CANONICAL_AA, ShiftList


_PARAMS_PATH = Path(__file__).parent / "data" / "ss_popn_params.npz"

# State order in the returned 4-vector (helix, beta, ppii, coil).
STATE_NAMES = ("helix", "beta", "ppii", "coil")
N_STATES = 4
N_NUC = len(BACKBONE_NUCLEI)  # 6


@dataclass(frozen=True)
class _GaussianParams:
    """Loaded d2D-style emission parameters."""

    mu: np.ndarray          # (20, 4, 6) per-(AA, state, nucleus) mean
    mu_valid: np.ndarray    # (20, 4, 6) bool: enough data for this cell
    sigma: np.ndarray       # (20, 6) per-AA per-nucleus pooled std
    corr: np.ndarray        # (6, 6) pooled inter-nucleus correlation of residuals
    # Neighbor-context corrections: (4, N_STATES, N_NUC, N_AA). Axis 0 is
    # offset ∈ {-2, -1, +1, +2}; value is the mean shift residual when a
    # given AA sits at that offset relative to the centre residue. Added
    # to mu at inference time to produce per-residue state-conditioned
    # predicted shifts.
    context: np.ndarray     # (4, 4, 6, 20)
    context_offsets: np.ndarray  # (4,) offsets in {-2, -1, 1, 2}


@lru_cache(maxsize=1)
def _load_params() -> _GaussianParams:
    if not _PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"SS-population parameter file not found at {_PARAMS_PATH}. "
            "Run scripts/fit_ss_populations.py in the noft repo to generate it."
        )
    data = np.load(_PARAMS_PATH, allow_pickle=True)
    # Backward-compat: accept parameter files fit before task #3 (no context).
    if "context" in data.files:
        context = data["context"].astype(np.float64)
        offsets = data["context_offsets"].astype(np.int64)
    else:
        context = np.zeros((4, 4, 6, 20), dtype=np.float64)
        offsets = np.array([-2, -1, 1, 2], dtype=np.int64)
    return _GaussianParams(
        mu=data["mu"].astype(np.float64),
        mu_valid=data["mu_valid"].astype(bool),
        sigma=data["sigma"].astype(np.float64),
        corr=data["corr"].astype(np.float64),
        context=context,
        context_offsets=offsets,
    )


@dataclass
class SSPopulationResult:
    """Per-residue SS-state populations from the d2D-style Gaussian engine.

    Attributes:
        seq_ids:   (N,) sequence numbers
        comp_ids:  length-N list of 3-letter AA codes
        populations: (N, 4) float — (helix, beta, ppii, coil) per residue.
                   Rows sum to 1. Residues with no usable shifts carry NaN.
        n_nuclei:  (N,) int — number of nuclei contributing to the fit
                   (residues with <3 are flagged as low-confidence).
    """

    seq_ids: np.ndarray
    comp_ids: list[str]
    populations: np.ndarray
    n_nuclei: np.ndarray

    @property
    def mean_populations(self) -> np.ndarray:
        """Sequence-average (H, E, P, C) ignoring NaN residues."""
        finite = np.isfinite(self.populations).all(axis=1)
        if not finite.any():
            return np.full(4, np.nan)
        return self.populations[finite].mean(axis=0)

    def per_residue(self, seq_id: int) -> np.ndarray:
        """Populations for a given residue; returns (4,) float."""
        idx = int(np.where(self.seq_ids == seq_id)[0][0])
        return self.populations[idx]


def predict_ss_populations(
    shift_list: ShiftList,
    smooth: bool = True,
    min_nuclei: int = 3,
) -> SSPopulationResult:
    """Compute per-residue (H, E, P, C) populations from shifts.

    Args:
        shift_list: the observed chemical shifts + sequence context.
        smooth: apply d2D-style 5-residue triangular smoothing after
            the per-residue Gaussian inference. Default True.
        min_nuclei: minimum number of observed backbone nuclei for a
            residue to be fitted. Residues below this report NaN.

    Returns:
        SSPopulationResult with per-residue populations.
    """
    params = _load_params()
    aa_to_idx = {aa: i for i, aa in enumerate(CANONICAL_AA)}

    shifts_arr, masks_arr, comp_ids, seq_ids = shift_list.to_arrays()
    n_res = len(seq_ids)
    pops = np.full((n_res, N_STATES), np.nan, dtype=np.float64)
    n_nuclei = np.zeros(n_res, dtype=np.int32)

    # Pre-compute AA index per residue for neighbor lookup
    aa_idx_per_res = np.array(
        [aa_to_idx.get(c, -1) for c in comp_ids], dtype=np.int32
    )
    seq_ids_arr = np.asarray(seq_ids, dtype=np.int64)

    for i in range(n_res):
        aa = int(aa_idx_per_res[i])
        if aa < 0:
            continue

        # Select usable nuclei: mask on, shift not NaN, μ known for ALL
        # four states (else we can't compare — drop the nucleus).
        usable = []
        for n in range(N_NUC):
            if masks_arr[i, n] == 0:
                continue
            if not np.isfinite(shifts_arr[i, n]):
                continue
            # Require at least H and E states to have a known mean — those
            # are the two cells where we actually discriminate. Missing P
            # or C is backstopped by sigma pooling.
            if not (params.mu_valid[aa, 0, n] and params.mu_valid[aa, 1, n]):
                continue
            usable.append(n)
        if len(usable) < min_nuclei:
            continue

        n_nuclei[i] = len(usable)
        obs = shifts_arr[i, usable]                     # (k,)
        # Per-AA per-state means (4, k)
        mu_s = params.mu[aa, :, usable].T
        mu_s = np.where(np.isfinite(mu_s), mu_s, params.mu[aa, 3, usable])

        # Apply neighbor-context corrections (task #3): for each offset in
        # {-2, -1, +1, +2}, look up the neighbor AA and add its correction
        # to the per-state predicted shift. This narrows σ by removing
        # intra-state variance due to sequence context.
        for off_idx, off in enumerate(params.context_offsets):
            j = i + off
            if j < 0 or j >= n_res:
                continue
            # Require seq_id to actually be contiguous (no assignment gaps)
            if seq_ids_arr[j] != seq_ids_arr[i] + off:
                continue
            nbr_aa = int(aa_idx_per_res[j])
            if nbr_aa < 0:
                continue
            # params.context shape (4, 4, 6, 20); we pick offset, all states,
            # selected nuclei, and the specific neighbor AA.
            correction = params.context[off_idx, :, :, nbr_aa][:, usable]  # (4, k)
            mu_s = mu_s + correction

        # Build reduced covariance: Σ_jk = σ_j · σ_k · corr_jk  (k×k).
        sig = params.sigma[aa, usable]                  # (k,)
        cov = np.outer(sig, sig) * params.corr[np.ix_(usable, usable)]

        # Regularise to keep numerically PD (rare; <1% of cases).
        cov = cov + 1e-4 * np.eye(len(usable))

        try:
            cov_inv = np.linalg.inv(cov)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                continue
        except np.linalg.LinAlgError:
            continue

        log_lik = np.empty(N_STATES, dtype=np.float64)
        for s in range(N_STATES):
            v = obs - mu_s[s]
            q = float(v @ cov_inv @ v)
            log_lik[s] = -0.5 * q  # the (2π)^(-n/2) |Σ|^(-1/2) factor is state-independent

        # Normalise in log-space
        log_lik -= log_lik.max()
        lik = np.exp(log_lik)
        pops[i] = lik / lik.sum()

    if smooth:
        pops = _smooth_populations_triangular(pops)

    return SSPopulationResult(
        seq_ids=np.asarray(seq_ids, dtype=np.int32),
        comp_ids=list(comp_ids),
        populations=pops,
        n_nuclei=n_nuclei,
    )


def _smooth_populations_triangular(pops: np.ndarray) -> np.ndarray:
    """Apply d2D-style triangular 5-residue smoothing to populations.

    Weights: centre=1.0, ±1=0.5, ±2=0.25. Finite rows only; NaN residues
    break the kernel (output stays NaN at those positions).
    """
    weights = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    n = len(pops)
    out = np.full_like(pops, np.nan)
    for i in range(n):
        num = np.zeros(pops.shape[1])
        denom = 0.0
        for off, w in zip((-2, -1, 0, 1, 2), weights):
            j = i + off
            if not (0 <= j < n):
                continue
            if not np.all(np.isfinite(pops[j])):
                continue
            num += w * pops[j]
            denom += w
        if denom == 0:
            continue
        s = num / num.sum()
        out[i] = s
    return out
