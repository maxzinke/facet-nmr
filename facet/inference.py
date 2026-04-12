"""FACET inference: predict backbone torsion angles from chemical shifts.

Public API::

    from facet import predict
    result = predict("shifts.tab")
    result = predict(shift_list)  # ShiftList object
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .io.formats import (
    BACKBONE_NUCLEI,
    CANONICAL_AA,
    CONF_DYNAMIC,
    CONF_GOOD,
    CONF_STRONG,
    CONF_WARN,
    FACETResult,
    ResiduePrediction,
    ShiftList,
)
from .model import FACETv3, FACETv3Config
from .random_coil import to_secondary_shifts

# AA 3-letter → index (1-20, 0=pad)
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(CANONICAL_AA)}

# Aromatic AAs for the neighbor flag
AROMATIC_AA = {"PHE", "TRP", "TYR", "HIS"}

# Confidence thresholds (calibrated from v3 risk-coverage curves)
# These map negative-entropy confidence → TALOS-N-style classes
_CONF_THRESHOLDS = [
    (-3.0, CONF_STRONG),   # highest confidence (low entropy)
    (-4.5, CONF_GOOD),
    (-5.5, CONF_WARN),
]
# Below all thresholds → Dynamic


def _classify_confidence(conf: float) -> str:
    """Map negative-entropy confidence to a class label."""
    for threshold, label in _CONF_THRESHOLDS:
        if conf >= threshold:
            return label
    return CONF_DYNAMIC


def _estimate_error_bound(conf: float) -> float:
    """Rough error bound (degrees) from confidence, for restraint generation."""
    # Empirical mapping from risk-coverage curve (approximate)
    if conf >= -3.0:
        return 15.0
    elif conf >= -4.5:
        return 25.0
    elif conf >= -5.5:
        return 40.0
    return 60.0


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
) -> FACETResult:
    """Predict backbone torsion angles from chemical shifts.

    Args:
        input_: Path to a shift list file (auto-detects format), or a
            ShiftList object directly.
        checkpoint: Path to model weights (.pt). If None, uses bundled weights.
        device: "cuda", "cpu", or None (auto).
        half_window: Context window (2 = pentapeptide, 3 = heptapeptide).
        batch_size: Inference batch size.

    Returns:
        FACETResult with per-residue phi, psi, confidence, SS, chi1.
    """
    # Load shift list
    if isinstance(input_, ShiftList):
        shift_list = input_
    else:
        from .io.readers import read_auto
        shift_list = read_auto(input_)

    shifts, masks, comp_ids, seq_ids = shift_list.to_arrays()

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
    checkpoint = Path(checkpoint)

    config = FACETv3Config()
    model = FACETv3(config).to(dev)
    state = torch.load(checkpoint, map_location=dev, weights_only=True)
    # strict=False: checkpoint may predate chi1 head addition
    model.load_state_dict(state, strict=False)
    model.eval()

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
