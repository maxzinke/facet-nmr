"""ONNX export and inference for FACET.

FACET's input is simple fixed-shape tensors (B, L, 50) → easy ONNX export
unlike Caustic's graph-based model. The wrapper flattens the dict output
into a fixed tuple of tensors.

Use::

    from facet.export import export_to_onnx, load_onnx_session, run_onnx
    export_to_onnx("facet_v3.pt", "facet_v3.onnx")

    session = load_onnx_session("facet_v3.onnx")
    result = run_onnx(session, shifts, masks, aa_idx, flags)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class _FACETONNXWrapper:
    """Torch wrapper that returns flat tensors instead of a dict."""

    def __init__(self, model):
        import torch.nn as nn
        self.model = model
        # Make it a proper nn.Module subclass for export
        self._module = type(
            "_Wrapper",
            (nn.Module,),
            {"forward": lambda self_inner, s, m, a, f: self_inner._run(s, m, a, f)},
        )()
        self._module._run = self._forward
        self._module.model = model

    def _forward(self, shifts, masks, aa_idx, flags):
        out = self.model.forward(shifts, masks, aa_idx, flags)
        return (
            out["coarse_logits"],       # (B, 1296)
            out["fine_delta_phi"],       # (B,)
            out["fine_delta_psi"],       # (B,)
            out["ss_logits"],            # (B, 3)
            out["confidence"],           # (B,)
        )


def export_to_onnx(
    checkpoint_path: str | Path,
    onnx_path: str | Path,
    opset_version: int = 17,
) -> Path:
    """Export a FACET v3 checkpoint to ONNX.

    Args:
        checkpoint_path: Path to .pt state_dict.
        onnx_path: Output .onnx path.
        opset_version: ONNX opset (default 17).

    Returns:
        Resolved path to the .onnx file.
    """
    import torch

    from .model import FACETv3, FACETv3Config

    out = Path(onnx_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Production configuration: the error head defines ``confidence``. Exporting
    # with the default config silently swapped in the entropy fallback, so the
    # ONNX confidence disagreed with the PyTorch path by up to ~2 units.
    config = FACETv3Config(use_error_head=True)
    model = FACETv3(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    wrapper = _FACETONNXWrapper(model)

    # Dummy input: batch=2, window=5 (pentapeptide)
    B, L = 2, 5
    dummy_shifts = torch.randn(B, L, 6)
    dummy_masks = torch.ones(B, L, 6)
    dummy_aa = torch.randint(1, 20, (B, L))
    dummy_flags = torch.zeros(B, L, 6)

    torch.onnx.export(
        wrapper._module,
        (dummy_shifts, dummy_masks, dummy_aa, dummy_flags),
        str(out),
        input_names=["shifts", "masks", "aa_idx", "flags"],
        output_names=["coarse_logits", "fine_delta_phi", "fine_delta_psi",
                       "ss_logits", "confidence"],
        dynamic_axes={
            "shifts": {0: "batch", 1: "window"},
            "masks": {0: "batch", 1: "window"},
            "aa_idx": {0: "batch", 1: "window"},
            "flags": {0: "batch", 1: "window"},
            "coarse_logits": {0: "batch"},
            "fine_delta_phi": {0: "batch"},
            "fine_delta_psi": {0: "batch"},
            "ss_logits": {0: "batch"},
            "confidence": {0: "batch"},
        },
        opset_version=opset_version,
    )
    print(f"Exported ONNX to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def load_onnx_session(onnx_path: str | Path):
    """Load an ONNX Runtime InferenceSession."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), opts)


def run_onnx(
    session,
    shifts: np.ndarray,
    masks: np.ndarray,
    aa_idx: np.ndarray,
    flags: np.ndarray,
) -> dict[str, np.ndarray]:
    """Run ONNX inference. Returns dict with coarse_logits, fine_delta_phi/psi, etc."""
    feeds = {
        "shifts": shifts.astype(np.float32),
        "masks": masks.astype(np.float32),
        "aa_idx": aa_idx.astype(np.int64),
        "flags": flags.astype(np.float32),
    }
    outputs = session.run(None, feeds)
    return {
        "coarse_logits": outputs[0],
        "fine_delta_phi": outputs[1],
        "fine_delta_psi": outputs[2],
        "ss_logits": outputs[3],
        "confidence": outputs[4],
    }
