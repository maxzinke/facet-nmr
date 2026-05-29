"""FACET inference: predict backbone torsion angles from chemical shifts.

Public API::

    from facet import predict
    result = predict("shifts.tab")
    result = predict(shift_list)  # ShiftList object
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


# Physical ranges for backbone chemical shifts (in ppm)
# Used for sanity-checking input data — shifts far outside these ranges
# almost always indicate a units error, referencing problem, or the wrong
# nucleus being labeled as one of ours.
# Sources: Wishart random-coil tables + BMRB statistics (mean ± ~8σ)
_SHIFT_RANGE = {
    "H":  (5.0, 12.0),     # amide 1H
    "HA": (1.5, 6.5),      # alpha 1H
    "N":  (95.0, 140.0),   # backbone 15N
    "CA": (40.0, 75.0),    # alpha 13C
    "CB": (15.0, 80.0),    # beta 13C (wide: CB of T/S is 60-70)
    "C":  (168.0, 185.0),  # carbonyl 13C
}

_MIN_RESIDUES = 3  # need at least 3 residues for a pentapeptide center + neighbors

from .io.formats import (
    BACKBONE_NUCLEI,
    CANONICAL_AA,
    CONF_FLEXIBLE,
    CONF_HIGH,
    CONF_LOW,
    CONF_MEDIUM,
    FACETResult,
    ResiduePrediction,
    ShiftList,
)
from .random_coil import to_secondary_shifts

# AA 3-letter → index (1-20, 0=pad)
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(CANONICAL_AA)}

# Aromatic AAs for the neighbor flag
AROMATIC_AA = {"PHE", "TRP", "TYR", "HIS"}

# FACET confidence tiers — independently calibrated from the v3
# risk-coverage curve on a 26,944-residue held-out test set (2026-04-12).
#
# Confidence = negative entropy of the 36x36 coarse Ramachandran grid.
# Sorted from most confident (least negative) to least.
#
# Coverage summary (cumulative, sorted by confidence descending):
#   top 10%:  conf >= -3.02,  median 6.9°,  fail25  5.1%,  fail30  4.1%
#   top 30%:  conf >= -3.72,  median 7.9°,  fail25  8.2%,  fail30  6.9%
#   top 60%:  conf >= -4.36,  median 10.8°, fail25 15.0%,  fail30 11.5%
#   top 85%:  conf >= -5.02,  median 12.9°, fail25 22.0%,  fail30 17.3%
#
# Tier mapping:
#   HIGH     (top 30%): 8% fail25 — safe for structure-calculation restraints
#   MEDIUM   (30-60%): 15% fail25 — use with wider error bounds
#   LOW      (60-85%): 22% fail25 — interpret cautiously
#   FLEXIBLE (bottom 15%): biologically flexible / disordered regions
_CONF_THRESHOLDS = [
    (-3.72, CONF_HIGH),     # top 30%: fail25 ~8%
    (-4.36, CONF_MEDIUM),   # top 60%: fail25 ~15%
    (-5.02, CONF_LOW),      # top 85%: fail25 ~22%
]
# Below all thresholds → Flexible (bottom 15%)


def _classify_confidence(conf: float) -> str:
    """Map negative-entropy confidence to a tier label."""
    for threshold, label in _CONF_THRESHOLDS:
        if conf >= threshold:
            return label
    return CONF_FLEXIBLE


def _estimate_error_bound(conf: float) -> float:
    """Per-tier error bound (degrees) for restraint generation.

    Derived from the test-set risk-coverage curve (2x the median error
    within each tier — the conventional half-width for dihedral
    restraints in structure calculation).
    """
    if conf >= -3.72:
        return 16.0  # HIGH:   2x median (8°)
    elif conf >= -4.36:
        return 22.0  # MEDIUM: 2x median (11°)
    elif conf >= -5.02:
        return 26.0  # LOW:    2x median (13°)
    return 40.0      # FLEXIBLE: wide bounds (usually not used for restraints)


def _build_windows(
    sec_shifts: np.ndarray,
    masks: np.ndarray,
    comp_ids: list[str],
    seq_ids: list[int],
    half_window: int = 2,
) -> list[dict[str, torch.Tensor]]:
    """Build pentapeptide/heptapeptide windows for inference.

    Returns a list of dicts, one per residue, each with:
        shifts (W, 6), masks (W, 6), aa_idx (W,), flags (W, 6)
    """
    n = len(comp_ids)
    W = 2 * half_window + 1

    # Per-residue metadata
    aa_idx = np.array([AA_TO_IDX.get(c, 0) for c in comp_ids], dtype=np.int64)
    flags_base = np.zeros((n, 6), dtype=np.float32)
    for i, c in enumerate(comp_ids):
        flags_base[i, 0] = 1.0 if c == "GLY" else 0.0
        flags_base[i, 1] = 1.0 if c == "PRO" else 0.0
        flags_base[i, 4] = 1.0 if c in AROMATIC_AA else 0.0

    # Build seq_id → position index for neighbor lookup
    sid_to_pos = {sid: i for i, sid in enumerate(seq_ids)}

    windows = []
    for i in range(n):
        center_sid = seq_ids[i]
        shifts_w = np.zeros((W, 6), dtype=np.float32)
        masks_w = np.zeros((W, 6), dtype=np.float32)
        aa_w = np.zeros(W, dtype=np.int64)
        flags_w = np.zeros((W, 6), dtype=np.float32)

        for w, offset in enumerate(range(-half_window, half_window + 1)):
            neighbor_sid = center_sid + offset
            j = sid_to_pos.get(neighbor_sid)
            if j is not None:
                shifts_w[w] = sec_shifts[j]
                masks_w[w] = masks[j]
                aa_w[w] = aa_idx[j]
                flags_w[w] = flags_base[j]
            else:
                flags_w[w, 5] = 1.0  # missing neighbor

        # Pre-Pro flag
        for w in range(W - 1):
            next_sid = center_sid + (w - half_window + 1)
            j = sid_to_pos.get(next_sid)
            if j is not None and aa_idx[j] == AA_TO_IDX.get("PRO", 0):
                flags_w[w, 2] = 1.0

        # Aromatic neighbor flag
        for w in range(W):
            for dw in [-1, 1]:
                nw = w + dw
                if 0 <= nw < W:
                    nsid = center_sid + (nw - half_window)
                    j = sid_to_pos.get(nsid)
                    if j is not None and flags_base[j, 4] > 0:
                        flags_w[w, 3] = 1.0

        # Missing neighbor flag
        for w in range(W):
            nsid = center_sid + (w - half_window)
            if nsid not in sid_to_pos:
                flags_w[w, 5] = 1.0

        windows.append({
            "shifts": torch.from_numpy(shifts_w),
            "masks": torch.from_numpy(masks_w),
            "aa_idx": torch.from_numpy(aa_w),
            "flags": torch.from_numpy(flags_w),
        })

    return windows


@lru_cache(maxsize=4)
def _load_cached_model(checkpoint_path: str, device_str: str):
    """Load a FACETv3 checkpoint. Cached per (path, device) so repeated
    predict() calls in the same process don't re-read the weights file."""
    from .model import FACETv3, FACETv3Config
    dev = torch.device(device_str)
    # v0.2+: checkpoints carry the ExpectedErrorHead for calibrated confidence.
    # use_error_head=True is a superset load — older checkpoints still load
    # via strict=False.
    config = FACETv3Config(use_error_head=True)
    model = FACETv3(config).to(dev)
    state = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    # strict=False: checkpoint may predate the chi1/error-head additions
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@lru_cache(maxsize=2)
def _load_cached_retrieval(checkpoint_path: str, index_path: str, device_str: str):
    """Load FACETRetrieval (encoder + reference index). Cached across calls."""
    from .retrieval import FACETRetrieval
    return FACETRetrieval(checkpoint_path, index_path, device=device_str)


@lru_cache(maxsize=1)
def _load_cached_masked_retrieval(index_path: str):
    """Load the mask-safe shift-retrieval fallback (numpy-only). Cached."""
    from .masked_retrieval import MaskedShiftRetrieval
    return MaskedShiftRetrieval(index_path)


def _find_index() -> Path | None:
    """Locate the bundled retrieval index, if any."""
    import os
    pkg = Path(__file__).resolve().parent
    for candidate in [
        pkg / "weights" / "facet_retrieval_index.npz",
        Path(os.environ.get("FACET_INDEX", "")) if os.environ.get("FACET_INDEX") else None,
        Path.home() / ".facet" / "facet_retrieval_index.npz",
    ]:
        if candidate is not None and candidate.exists():
            return candidate
    return None


# Maps the mask-safe retrieval confidence tier onto the DBSCAN tier vocabulary
# so the existing _RETRIEVAL_TIER_TO_CONF mapping yields a confidence_class.
_MASKED_CONF_TO_TIER = {
    "strong": "Strong", "good": "Generous", "weak": "Ambiguous", "insufficient": "None",
}


def _find_shift_reference() -> Path | None:
    """Locate the bundled mask-safe shift-retrieval reference index, if any."""
    import os
    pkg = Path(__file__).resolve().parent
    for candidate in [
        pkg / "weights" / "facet_shift_reference.npz",
        Path(os.environ["FACET_SHIFT_REFERENCE"]) if os.environ.get("FACET_SHIFT_REFERENCE") else None,
        Path.home() / ".facet" / "facet_shift_reference.npz",
    ]:
        if candidate is not None and candidate.exists():
            return candidate
    return None


def _find_checkpoint() -> Path:
    """Locate the bundled model weights."""
    import os
    pkg = Path(__file__).resolve().parent
    for candidate in [
        pkg / "weights" / "facet_v3.pt",
        pkg / "weights" / "best.onnx",
        Path(os.environ.get("FACET_WEIGHTS", "")) if os.environ.get("FACET_WEIGHTS") else None,
        Path.home() / ".facet" / "facet_v3.pt",
    ]:
        if candidate is not None and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "FACET weights not found. Set FACET_WEIGHTS env var or place "
        "facet_v3.pt in ~/.facet/ or <package>/weights/."
    )


@torch.no_grad()
def predict(
    input_: str | Path | ShiftList,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    half_window: int = 2,
    batch_size: int = 512,
    deuteration=None,
    use_retrieval: bool | None = None,
    retrieval_k: int = 25,
    auto_reference: bool = False,
    structural_beta: float = 15.0,
    mask_safe_fallback: bool = True,
) -> FACETResult:
    """Predict backbone torsion angles from chemical shifts.

    Args:
        input_: Path to a shift list file (auto-detects format), or a
            ShiftList object directly.
        checkpoint: Path to model weights (.pt). If None, uses bundled weights.
        device: "cuda", "cpu", or None (auto).
        half_window: Context window (2 = pentapeptide, 3 = heptapeptide).
        batch_size: Inference batch size.
        deuteration: Optional sample deuteration specifier for 13C isotope
            shift correction.
        use_retrieval: If True, use retrieval-augmented inference (kNN + DBSCAN
            over a bundled 253K-residue reference index). Produces multi-modal
            predictions with FACET confidence tiers (High / Medium / Low /
            Flexible) calibrated from cluster agreement, and per-residue
            basin populations (alpha/beta/PPII/other). Requires the bundled
            retrieval index file. If None (default), auto-selects retrieval
            when the index is available, parametric otherwise.
        retrieval_k: Number of nearest neighbors to retrieve (default 25).
        auto_reference: If True, run the shift-referencing sanity check and
            apply its suggested per-nucleus corrections before prediction.
            Only corrections that exceeded the tolerance threshold are
            applied (well-referenced channels stay untouched). The applied
            offsets are reported in ``result.referencing_corrections_applied``.

    Returns:
        FACETResult with per-residue phi, psi, confidence, confidence_class
        (tier), SS, chi1. When retrieval is used, each prediction also
        carries basin_populations and alt_clusters.
    """
    import logging
    logger = logging.getLogger("facet")

    # Load shift list
    if isinstance(input_, ShiftList):
        shift_list = input_
    else:
        from .io.readers import read_auto
        shift_list = read_auto(input_)

    shifts, masks, comp_ids, seq_ids = shift_list.to_arrays()

    # ── Deuterium isotope correction ──
    # FACET was trained on shifts that are protonated-equivalent (the data
    # pipeline applies the same correction at build time). Perdeuterated
    # samples must be corrected here or predictions will be biased by up
    # to ~0.5 ppm on Cα — enough to swap helix/strand calls near the boundary.
    deuteration_preset = "protonated"
    deuteration_corrections_ppm: dict[str, float] = {}
    if deuteration is not None:
        from .deuteration import compute_isotope_correction, resolve_deuteration
        from .io.formats import AA_THREE_TO_ONE
        dv = resolve_deuteration(deuteration)
        deuteration_preset = (
            deuteration if isinstance(deuteration, str) else "custom"
        )
        if dv.frac_amide > 0 or dv.frac_alpha > 0 or dv.frac_sidechain > 0:
            logger.info(
                "Applying deuterium isotope correction: "
                "frac_amide=%.2f, frac_alpha=%.2f, frac_sidechain=%.2f",
                dv.frac_amide, dv.frac_alpha, dv.frac_sidechain,
            )
            _nuc_cols = {"CA": 3, "CB": 4, "C": 5}
            # Track the mean per-nucleus correction in ppm for reporting.
            _nuc_sums = {nuc: 0.0 for nuc in _nuc_cols}
            _nuc_counts = {nuc: 0 for nuc in _nuc_cols}
            for i, comp in enumerate(comp_ids):
                aa1 = AA_THREE_TO_ONE.get(comp)
                if aa1 is None:
                    continue
                corr = compute_isotope_correction(aa1, dv)
                for nuc, col in _nuc_cols.items():
                    if masks[i, col] > 0 and abs(corr[nuc]) > 0.001:
                        shifts[i, col] += corr[nuc]
                        _nuc_sums[nuc] += corr[nuc]
                        _nuc_counts[nuc] += 1
            for nuc, total in _nuc_sums.items():
                if _nuc_counts[nuc] > 0:
                    deuteration_corrections_ppm[nuc] = total / _nuc_counts[nuc]
            logger.info(
                "Deuteration correction per-nucleus mean (ppm): %s",
                ", ".join(
                    f"{n} {v:+.2f}" for n, v in deuteration_corrections_ppm.items()
                ),
            )

    # ── Minimum size ──
    if len(comp_ids) < _MIN_RESIDUES:
        raise ValueError(
            f"Too few residues ({len(comp_ids)}). FACET needs at least "
            f"{_MIN_RESIDUES} residues to form a pentapeptide window."
        )

    # ── Shift range sanity check ──
    # Detect obviously miscalibrated data (e.g. 13C shifts where 15N should be)
    n_out_of_range = 0
    out_of_range_nuclei: dict[str, int] = {}
    for i in range(len(comp_ids)):
        for j, nuc in enumerate(BACKBONE_NUCLEI):
            if masks[i, j] < 0.5:
                continue
            val = float(shifts[i, j])
            lo, hi = _SHIFT_RANGE[nuc]
            if val < lo or val > hi:
                n_out_of_range += 1
                out_of_range_nuclei[nuc] = out_of_range_nuclei.get(nuc, 0) + 1

    total_observed = int(masks.sum())
    if total_observed > 0 and n_out_of_range / total_observed > 0.1:
        # More than 10% of shifts are wildly off — probably a units issue
        detail = ", ".join(f"{nuc}={n}" for nuc, n in sorted(out_of_range_nuclei.items()))
        raise ValueError(
            f"{n_out_of_range}/{total_observed} ({100*n_out_of_range/total_observed:.0f}%) "
            f"of chemical shifts are outside physical ranges ({detail}). "
            f"This usually indicates a referencing error, wrong nucleus labels, "
            f"or a non-protein sample. Expected ranges (ppm): "
            f"H=5-12, HA=1.5-6.5, N=95-140, CA=40-75, CB=15-80, C=168-185."
        )
    elif n_out_of_range > 0:
        logger.warning(
            "%d individual shifts outside typical ranges (%s) — likely outliers",
            n_out_of_range, ", ".join(f"{k}={v}" for k, v in out_of_range_nuclei.items()),
        )

    # ── Validate AA composition ──
    # Non-canonical residues (modified / xeno AAs) are silently skipped
    # from prediction — they're removed from shift_list.residues below.
    # Before v0.2.2 we raised ValueError at >50% non-canonical; that was
    # too aggressive for real NMR-STAR files (some deposits use MSE for
    # selenomet, PTR for phosphotyrosine, etc.). Now we skip them with
    # a warning, predict on the standard residues, and let the user
    # interpret the partial output.
    unknown_aas = [c for c in comp_ids if c not in CANONICAL_AA]
    skipped_noncanonical: list[tuple[int, str]] = []
    if unknown_aas:
        unknown_set = sorted(set(unknown_aas))
        frac = len(unknown_aas) / len(comp_ids)
        logger.warning(
            "%d/%d residues (%.0f%%) use non-canonical amino acids not "
            "supported by FACET: %s%s. Skipping those residues; predicting "
            "on the %d standard residues.",
            len(unknown_aas), len(comp_ids), 100 * frac,
            unknown_set[:10], "..." if len(unknown_set) > 10 else "",
            len(comp_ids) - len(unknown_aas),
        )
        # Filter shift_list in place to drop non-canonical residues, then
        # rebuild the derived arrays so all downstream code sees only
        # standard AAs.
        kept_residues = [r for r in shift_list.residues if r.comp_id in CANONICAL_AA]
        for r in shift_list.residues:
            if r.comp_id not in CANONICAL_AA:
                skipped_noncanonical.append((r.seq_id, r.comp_id))
        shift_list = type(shift_list)(
            residues=kept_residues,
            sequence=shift_list.sequence,
            seq_id_start=shift_list.seq_id_start,
            source=shift_list.source,
        )
        shifts, masks, comp_ids, seq_ids = shift_list.to_arrays()
        if len(comp_ids) < _MIN_RESIDUES:
            raise ValueError(
                f"After skipping {len(unknown_aas)} non-canonical residues, "
                f"only {len(comp_ids)} standard residues remain — fewer than "
                f"FACET's minimum of {_MIN_RESIDUES}. Use a sample with more "
                f"standard AAs."
            )

    # ── Validate shift coverage ──
    # FACET needs heavy atoms (N, CA, CB, C) — proton-only data is insufficient
    # Column order: H, HA, N, CA, CB, C
    heavy_mask = masks[:, 2:].sum(axis=1)  # N, CA, CB, C observed count
    n_heavy_starved = int((heavy_mask < 1).sum())
    if n_heavy_starved > len(comp_ids) * 0.5:
        raise ValueError(
            f"{n_heavy_starved}/{len(comp_ids)} residues have no heavy-atom "
            f"(N/CA/CB/C') shifts. FACET requires at least one heavy-atom "
            f"chemical shift per residue — proton-only assignments (H/HA) "
            f"are insufficient to predict backbone torsion angles. "
            f"This dataset appears to contain only 1H assignments."
        )
    if n_heavy_starved > 0:
        logger.warning(
            "%d residues have no heavy-atom shifts — predictions will be poor",
            n_heavy_starved,
        )

    # ── Referencing sanity check ──
    # Composition-adaptive: if mean secondary shift per nucleus deviates
    # from what the protein's apparent H/E/C mix would predict, warn the
    # user (likely miscalibration). Non-fatal — predictions still run.
    from .referencing import check_referencing, _TOLERANCE_PPM
    ref_report = check_referencing(shifts, masks, comp_ids)

    applied_corrections: dict[str, float] = {}
    if auto_reference and ref_report.has_warnings:
        # Apply the suggested corrections IN PLACE on the shift array.
        # Only per-nucleus offsets that exceed the tolerance are applied
        # (well-referenced channels stay untouched). We then re-run the
        # check so the FACETResult carries the POST-correction summary
        # and warnings list — users see what we fixed, not what was wrong.
        _NUC_COL = {"H": 0, "HA": 1, "N": 2, "CA": 3, "CB": 4, "C": 5}
        for nuc, off in ref_report.offsets.items():
            if abs(off) > _TOLERANCE_PPM.get(nuc, 0.5):
                col = _NUC_COL.get(nuc)
                if col is None:
                    continue
                # Subtract the observed offset from every observed shift
                # on that nucleus.
                col_mask = masks[:, col] > 0
                shifts[col_mask, col] = shifts[col_mask, col] - off
                applied_corrections[nuc] = float(off)
        if applied_corrections:
            logger.info(
                "auto_reference: applied corrections "
                + ", ".join(f"{n} {-v:+.2f}" for n, v in applied_corrections.items())
            )
            # Re-run the check to produce post-correction summary
            ref_report = check_referencing(shifts, masks, comp_ids)

    if ref_report.has_warnings:
        logger.warning("Referencing sanity check raised warnings:")
        for w in ref_report.warnings:
            logger.warning("  %s", w)
    else:
        logger.info(ref_report.summary())

    # Random-coil-index S^2 per residue (Berjanskii-Wishart). Cheap to
    # compute from the already-corrected shifts; attached to each residue
    # below. Returned as float in [0, 1], NaN when too few shifts.
    from .rci import compute_rci_s2
    rci_s2_arr = compute_rci_s2(shifts, masks, comp_ids)

    # Convert to secondary shifts
    sec_shifts = to_secondary_shifts(shifts, masks, comp_ids)

    # Build windows
    windows = _build_windows(sec_shifts, masks, comp_ids, seq_ids, half_window)

    # Load model
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    if checkpoint is None:
        checkpoint = _find_checkpoint()
    checkpoint = Path(checkpoint).resolve()

    # LRU-cached load — reuses the same model across successive predict() calls
    model = _load_cached_model(str(checkpoint), device)

    # ── Decide retrieval vs parametric ──
    index_path = _find_index()
    if use_retrieval is None:
        use_retrieval = index_path is not None
    if use_retrieval and index_path is None:
        logger.warning(
            "use_retrieval=True but no bundled index found; falling back to parametric. "
            "Set FACET_INDEX or place facet_retrieval_index.npz in the weights dir."
        )
        use_retrieval = False

    retriever = None
    if use_retrieval:
        retriever = _load_cached_retrieval(str(checkpoint), str(index_path), device)
        logger.info("Using retrieval-augmented inference (index: %d residues)",
                    retriever.n_index)

    # Batch inference
    n = len(windows)
    all_phi = np.zeros(n, dtype=np.float64)
    all_psi = np.zeros(n, dtype=np.float64)
    all_conf = np.zeros(n, dtype=np.float64)
    all_ss = np.zeros(n, dtype=np.int64)
    all_chi1 = np.zeros(n, dtype=np.int64)
    all_chi1_probs = np.zeros((n, 3), dtype=np.float32)
    has_chi1_probs = False
    # Retrieval-only outputs (stays None for all residues when use_retrieval=False;
    # in retrieval mode the DBSCAN tier is folded into confidence_class below).
    all_tier: list[str | None] = [None] * n
    all_basin_pops: list[tuple[float, float, float, float] | None] = [None] * n
    all_alt_clusters: list[list[tuple[float, float, float]] | None] = [None] * n
    all_top_neighbors: list[list[dict] | None] = [None] * n
    # Structural populations (v0.4+, phase 3.5): filled only when
    # retrieval is enabled. Kernel-weighted Bayesian SS-state populations
    # in d2D convention (helix, beta, ppii, coil). See facet/structural.py.
    all_structural_pops: list[tuple[float, float, float, float] | None] = [None] * n
    all_structural_effn: list[float | None] = [None] * n
    # Per-residue 1-sigma error bounds (degrees). Populated from the
    # retrieval cluster's circular std when retrieval is used and a
    # tight cluster is found; otherwise falls back to a tier-based
    # entropy bound from _estimate_error_bound. Restraint writers then
    # widen to 2-sigma for the emitted bound.
    all_phi_err = np.zeros(n, dtype=np.float64)
    all_psi_err = np.zeros(n, dtype=np.float64)

    # DBSCAN cluster-quality labels map onto FACET's native tier vocabulary.
    _RETRIEVAL_TIER_TO_CONF = {
        "Strong":    "High",      # tight single cluster, restraint-quality
        "Generous":  "Medium",    # wider cluster, cautious use
        "Ambiguous": "Low",       # multi-modal — omit from restraints
        "None":      "Flexible",  # no confident cluster — disordered / flexible
    }

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_w = windows[start:end]

        b_shifts = torch.stack([w["shifts"] for w in batch_w]).to(dev)
        b_masks = torch.stack([w["masks"] for w in batch_w]).to(dev)
        b_aa = torch.stack([w["aa_idx"] for w in batch_w]).to(dev)
        b_flags = torch.stack([w["flags"] for w in batch_w]).to(dev)

        # Parametric heads (SS + chi1) always come from the encoder —
        # those outputs are valid regardless of whether (phi, psi) comes
        # from argmax or retrieval.
        out = model.predict(b_shifts, b_masks, b_aa, b_flags)
        all_conf[start:end] = out["confidence"].cpu().numpy()
        all_ss[start:end] = out["ss_pred"].cpu().numpy()
        if "chi1_pred" in out:
            all_chi1[start:end] = out["chi1_pred"].cpu().numpy()
        if "chi1_probs" in out:
            all_chi1_probs[start:end] = out["chi1_probs"].cpu().numpy()
            has_chi1_probs = True

        if use_retrieval:
            # Replace (phi, psi) with retrieval's top-cluster answer;
            # attach tier + basin populations + alternative clusters.
            r_results = retriever.predict(b_shifts, b_masks, b_aa, b_flags, k=retrieval_k)
            # Also compute d2D-style structural populations via
            # kernel-weighted Bayesian softmax over the full index.
            # We re-encode here (cheap; the encoder is already warm).
            from .structural import compute_structural_populations
            batch_embeds = retriever._encode(
                b_shifts.to(retriever.device),
                b_masks.to(retriever.device),
                b_aa.to(retriever.device),
                b_flags.to(retriever.device),
            )
            # Extract the center-residue AA index per batch element.
            batch_aa_center = b_aa[:, b_aa.shape[1] // 2].cpu().numpy()
            struct_pops, struct_effn = compute_structural_populations(
                retriever, batch_embeds, batch_aa_center,
                beta=structural_beta,
            )
            # Expected basin by encoder SS (basin index: 0=alpha_R, 1=beta,
            # 2=PPII, 3=other). We use this to catch cases where retrieval
            # lands in the wrong Ramachandran region for a residue whose SS
            # head clearly says otherwise (typical on OOD inputs).
            _SS_EXPECTED_BASINS = {0: {0}, 1: {1, 2}, 2: {0, 1, 2, 3}}
            for i, r in enumerate(r_results):
                idx = start + i
                if r.clusters:
                    top = r.clusters[0]
                    ss_idx = int(out["ss_pred"][i].cpu().item())
                    expected = _SS_EXPECTED_BASINS.get(ss_idx, {0, 1, 2, 3})
                    # Catch residual sentinel-attractor hits: a top cluster
                    # within 10 deg of the Ramachandran origin in basin=other
                    # is almost always a pull on leftover near-(0,0) entries
                    # rather than a real bridge/turn. Demote regardless of SS
                    # (SS=C otherwise accepts any basin and lets this slip).
                    near_origin_other = (
                        abs(top.phi_deg) < 10.0
                        and abs(top.psi_deg) < 10.0
                        and top.basin == 3
                    )
                    if top.basin not in expected or near_origin_other:
                        # Either SS head and retrieved basin disagree, or the
                        # cluster hit the near-origin "other" attractor. Don't
                        # trust the cluster point estimate — fall back to
                        # encoder argmax and mark Flexible.
                        all_phi[idx] = float(np.degrees(out["phi"][i].cpu().numpy()))
                        all_psi[idx] = float(np.degrees(out["psi"][i].cpu().numpy()))
                        all_alt_clusters[idx] = [
                            (c.phi_deg, c.psi_deg, c.weight) for c in r.clusters
                        ]
                        all_tier[idx] = "None"
                        # No trustworthy cluster — leave per-residue err at 0
                        # so the fallback tier bound kicks in below.
                    else:
                        all_phi[idx] = top.phi_deg
                        all_psi[idx] = top.psi_deg
                        all_alt_clusters[idx] = [
                            (c.phi_deg, c.psi_deg, c.weight) for c in r.clusters[1:]
                        ]
                        all_tier[idx] = r.tier
                        # 1-sigma per-residue uncertainty from the cluster's
                        # circular std. Floor at 3 deg so restraint writers
                        # never emit a bound smaller than the model's coarse
                        # Ramachandran grid resolution (10 deg / 2 = 5 deg, with
                        # some margin). Clip at 25 deg: above that we cap
                        # because the 2-sigma restraint would exceed 50 deg
                        # and be meaningless for structure calculation.
                        all_phi_err[idx] = max(3.0, min(25.0, top.phi_std_deg))
                        all_psi_err[idx] = max(3.0, min(25.0, top.psi_std_deg))
                else:
                    # No cluster — fall back to encoder argmax for this residue
                    all_phi[idx] = float(np.degrees(out["phi"][i].cpu().numpy()))
                    all_psi[idx] = float(np.degrees(out["psi"][i].cpu().numpy()))
                    all_alt_clusters[idx] = []
                    all_tier[idx] = r.tier
                bp = r.basin_populations
                all_basin_pops[idx] = (
                    (float(bp[0]), float(bp[1]), float(bp[2]), float(bp[3]))
                    if bp else None
                )
                all_top_neighbors[idx] = list(r.top_neighbors) if r.top_neighbors else None
                # Structural populations from the kernel-weighted softmax.
                sp = struct_pops[i]
                all_structural_pops[idx] = (
                    float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])
                )
                all_structural_effn[idx] = float(struct_effn[i])
        else:
            # Parametric argmax path (v0.1 compatible)
            all_phi[start:end] = np.degrees(out["phi"].cpu().numpy())
            all_psi[start:end] = np.degrees(out["psi"].cpu().numpy())

    # ── Mask-safe shift-retrieval fallback for HA-missing residues ──
    # The parametric phi/psi head (and the embedding retrieval, which uses the
    # same encoder) collapse toward ~0 deg when HA is absent — the model was
    # trained never to lose H/HA. For those residues, predict phi/psi via
    # mask-aware shift retrieval, whose distance drops the missing atom instead
    # of collapsing. Validated to recover ~18 deg median error vs ~53 deg for
    # the parametric head on held-out perdeuterated-style inputs.
    if mask_safe_fallback:
        ha_missing = masks[:, 1] < 0.5
        n_missing = int(ha_missing.sum())
        if n_missing > 0:
            shift_ref = _find_shift_reference()
            if shift_ref is not None:
                mr = _load_cached_masked_retrieval(str(shift_ref))
                mpreds = mr.predict_residues(
                    shifts, masks, list(comp_ids),
                    [int(s) for s in seq_ids], k=retrieval_k,
                )
                for i in range(n):
                    if not ha_missing[i]:
                        continue
                    mp = mpreds[i]
                    all_phi[i] = mp["phi_deg"]
                    all_psi[i] = mp["psi_deg"]
                    all_phi_err[i] = max(3.0, min(25.0, mp["phi_std_deg"]))
                    all_psi_err[i] = max(3.0, min(25.0, mp["psi_std_deg"]))
                    all_tier[i] = _MASKED_CONF_TO_TIER.get(mp["confidence"], "None")
                logger.info(
                    "Mask-safe fallback: recovered phi/psi for %d HA-missing residue(s)",
                    n_missing,
                )
            else:
                logger.warning(
                    "%d residue(s) missing HA but no shift-reference index found "
                    "(weights/facet_shift_reference.npz) — phi/psi may be unreliable",
                    n_missing,
                )

    # Build result
    SS_LABELS = {0: "H", 1: "E", 2: "C"}
    residues: list[ResiduePrediction] = []
    for i in range(n):
        conf = float(all_conf[i])
        fallback_err = _estimate_error_bound(conf) / 2.0  # half-width -> 1-sigma
        phi_err = all_phi_err[i] if all_phi_err[i] > 0 else fallback_err
        psi_err = all_psi_err[i] if all_psi_err[i] > 0 else fallback_err
        has_chi1 = comp_ids[i] not in ("GLY", "ALA")
        # In retrieval mode, DBSCAN cluster quality drives the tier; the
        # entropy-based class is used only when retrieval is disabled.
        if all_tier[i] is not None:
            tier = _RETRIEVAL_TIER_TO_CONF.get(all_tier[i], "Flexible")
        else:
            tier = _classify_confidence(conf)
        residues.append(ResiduePrediction(
            seq_id=seq_ids[i],
            comp_id=comp_ids[i],
            phi=float(all_phi[i]),
            psi=float(all_psi[i]),
            confidence=conf,
            confidence_class=tier,
            ss=SS_LABELS.get(int(all_ss[i]), "C"),
            chi1=int(all_chi1[i]) if has_chi1 else None,
            phi_err=float(phi_err),
            psi_err=float(psi_err),
            basin_populations=all_basin_pops[i],
            alt_clusters=all_alt_clusters[i],
            top_neighbors=all_top_neighbors[i],
            rci_s2=(
                float(rci_s2_arr[i])
                if i < len(rci_s2_arr) and not np.isnan(rci_s2_arr[i])
                else None
            ),
            structural_populations=all_structural_pops[i],
            structural_populations_eff_n=all_structural_effn[i],
            chi1_probs=(
                (float(all_chi1_probs[i, 0]),
                 float(all_chi1_probs[i, 1]),
                 float(all_chi1_probs[i, 2]))
                if has_chi1 and has_chi1_probs else None
            ),
        ))

    return FACETResult(
        residues=residues,
        source=shift_list.source,
        sequence=shift_list.sequence,
        seq_id_start=shift_list.seq_id_start,
        index_version=(retriever.index_version if retriever is not None else ""),
        index_n_residues=(retriever.n_index if retriever is not None else 0),
        referencing_warnings=ref_report.warnings,
        referencing_summary=ref_report.summary(),
        referencing_corrections_applied=applied_corrections,
        deuteration_preset=deuteration_preset,
        deuteration_corrections_ppm=deuteration_corrections_ppm,
    )
