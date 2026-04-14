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
    config = FACETv3Config()
    model = FACETv3(config).to(dev)
    state = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    # strict=False: checkpoint may predate the chi1 head addition
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


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
            shift correction. Accepts ``None`` (protonated, default), a
            preset name (``"protonated"``, ``"perdeut_exchanged"``,
            ``"perdeut_unexchanged"``, ``"ilv_methyl"``), a ``DeuterationVector``,
            or a 3-tuple ``(frac_amide, frac_alpha, frac_sidechain)``. When
            set, CA/CB/C' shifts are corrected to protonated-equivalent values
            before inference (FACET is trained on protonated-equivalent data).

    Returns:
        FACETResult with per-residue phi, psi, confidence, SS, chi1.
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
    if deuteration is not None:
        from .deuteration import compute_isotope_correction, resolve_deuteration
        from .io.formats import AA_THREE_TO_ONE
        dv = resolve_deuteration(deuteration)
        if dv.frac_amide > 0 or dv.frac_alpha > 0 or dv.frac_sidechain > 0:
            logger.info(
                "Applying deuterium isotope correction: "
                "frac_amide=%.2f, frac_alpha=%.2f, frac_sidechain=%.2f",
                dv.frac_amide, dv.frac_alpha, dv.frac_sidechain,
            )
            _nuc_cols = {"CA": 3, "CB": 4, "C": 5}
            for i, comp in enumerate(comp_ids):
                aa1 = AA_THREE_TO_ONE.get(comp)
                if aa1 is None:
                    continue
                corr = compute_isotope_correction(aa1, dv)
                for nuc, col in _nuc_cols.items():
                    if masks[i, col] > 0 and abs(corr[nuc]) > 0.001:
                        shifts[i, col] += corr[nuc]

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
    unknown_aas = [c for c in comp_ids if c not in CANONICAL_AA]
    n_unknown = len(unknown_aas)
    if n_unknown > 0:
        unknown_set = sorted(set(unknown_aas))
        frac = n_unknown / len(comp_ids)
        msg = (
            f"{n_unknown}/{len(comp_ids)} residues ({100*frac:.0f}%) use "
            f"non-canonical amino acids not supported by FACET: "
            f"{unknown_set[:10]}{'...' if len(unknown_set) > 10 else ''}. "
            f"FACET is trained on the 20 standard proteinogenic AAs only."
        )
        if frac > 0.5:
            raise ValueError(
                msg + "\n\nThis file appears to contain a synthetic peptide "
                "with xeno or modified amino acids. FACET cannot predict "
                "torsion angles for these residues — the model has not been "
                "trained on their chemistry. Please use a standard-AA "
                "protein or a method designed for xeno peptides."
            )
        else:
            logger.warning(msg)
            logger.warning(
                "Non-canonical residues will be assigned Dynamic confidence "
                "and should not be used for structure calculation."
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

    # Batch inference
    n = len(windows)
    all_phi = np.zeros(n, dtype=np.float64)
    all_psi = np.zeros(n, dtype=np.float64)
    all_conf = np.zeros(n, dtype=np.float64)
    all_ss = np.zeros(n, dtype=np.int64)
    all_chi1 = np.zeros(n, dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_w = windows[start:end]

        b_shifts = torch.stack([w["shifts"] for w in batch_w]).to(dev)
        b_masks = torch.stack([w["masks"] for w in batch_w]).to(dev)
        b_aa = torch.stack([w["aa_idx"] for w in batch_w]).to(dev)
        b_flags = torch.stack([w["flags"] for w in batch_w]).to(dev)

        out = model.predict(b_shifts, b_masks, b_aa, b_flags)

        all_phi[start:end] = np.degrees(out["phi"].cpu().numpy())
        all_psi[start:end] = np.degrees(out["psi"].cpu().numpy())
        all_conf[start:end] = out["confidence"].cpu().numpy()
        all_ss[start:end] = out["ss_pred"].cpu().numpy()
        if "chi1_pred" in out:
            all_chi1[start:end] = out["chi1_pred"].cpu().numpy()

    # Build result
    SS_LABELS = {0: "H", 1: "E", 2: "C"}
    residues: list[ResiduePrediction] = []
    for i in range(n):
        conf = float(all_conf[i])
        err_bound = _estimate_error_bound(conf)
        residues.append(ResiduePrediction(
            seq_id=seq_ids[i],
            comp_id=comp_ids[i],
            phi=float(all_phi[i]),
            psi=float(all_psi[i]),
            confidence=conf,
            confidence_class=_classify_confidence(conf),
            ss=SS_LABELS.get(int(all_ss[i]), "C"),
            chi1=int(all_chi1[i]) if comp_ids[i] not in ("GLY", "ALA") else None,
            phi_err=err_bound,
            psi_err=err_bound,
        ))

    return FACETResult(residues=residues, source=shift_list.source)
