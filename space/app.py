"""FACET HuggingFace Space — Gradio UI.

Predict backbone torsion angles from NMR chemical shifts.
Upload a shift list, click Predict, download restraint files.

Launch locally:
    cd space && python app.py

Deploy to HF Spaces:
    Copy space/ contents + facet/ package to a HF Space repo, push.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add parent dir so `import facet` works in the Space environment
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("facet-space")

# Model checkpoint — set FACET_CHECKPOINT env var or place in default location
CHECKPOINT = os.environ.get(
    "FACET_CHECKPOINT",
    str(Path.home() / ".facet" / "facet_v3.pt"),
)

# Cache the model after first load
_CACHED_RESULT = {}


def predict_and_format(file_obj, output_formats: list[str]):
    """Run FACET on an uploaded shift list file."""
    if file_obj is None:
        raise gr.Error("Please upload a shift list file (.tab, .csv, .nef, .str)")

    from facet import predict
    from facet.io.writers import (
        write_tbl, write_aco, write_nef, write_predtab, write_csv, write_json,
    )
    from facet.visualization import plot_sequence_ss

    input_path = file_obj if isinstance(file_obj, str) else file_obj.name
    logger.info("Predicting from %s", input_path)

    try:
        result = predict(input_path, checkpoint=CHECKPOINT, device="cpu")
    except Exception as e:
        raise gr.Error(f"Prediction failed: {e}")

    n_accepted = len(result.accepted())
    summary_text = result.summary()

    # Publication-grade sequence + SS figure
    seq_ss_path = os.path.join(tempfile.gettempdir(), "facet_sequence_ss.png")
    try:
        fig_seq = plot_sequence_ss(
            result, seq_ss_path,
            title=f"FACET prediction — {Path(input_path).stem}",
        )
        plt.close(fig_seq)
    except Exception as e:
        logger.warning("Sequence figure failed: %s", e)
        seq_ss_path = None

    # Ramachandran plot
    fig = _plot_ramachandran(result)
    rama_path = os.path.join(tempfile.gettempdir(), "facet_rama.png")
    fig.savefig(rama_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Generate output files
    tmpdir = tempfile.mkdtemp(prefix="facet_")
    stem = Path(input_path).stem
    output_files = []

    writer_map = {
        "XPLOR .tbl": ("tbl", write_tbl, True),
        "CYANA .aco": ("aco", write_aco, True),
        "NEF .nef": ("nef", write_nef, True),
        "pred.tab": ("predtab", write_predtab, False),
        "CSV": ("csv", write_csv, False),
        "JSON": ("json", write_json, False),
    }

    for fmt_name in output_formats:
        if fmt_name not in writer_map:
            continue
        ext, writer, needs_accepted = writer_map[fmt_name]
        out_path = os.path.join(tmpdir, f"{stem}_facet.{ext}")
        if needs_accepted:
            writer(result, out_path, accepted_only=True)
        else:
            writer(result, out_path)
        output_files.append(out_path)

    return summary_text, seq_ss_path, rama_path, output_files


def _plot_ramachandran(result):
    """Ramachandran plot colored by SS and sized by confidence."""
    from facet.io.formats import CONF_STRONG, CONF_GOOD

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ss_colors = {"H": "#E74C3C", "E": "#3498DB", "C": "#95A5A6"}
    ss_labels = {"H": "Helix", "E": "Strand", "C": "Coil"}

    for ss in ["C", "E", "H"]:
        residues = [r for r in result.residues if r.ss == ss]
        if not residues:
            continue
        phi = [r.phi for r in residues]
        psi = [r.psi for r in residues]
        sizes = [
            40 if r.confidence_class in (CONF_STRONG, CONF_GOOD) else 15
            for r in residues
        ]
        alpha = [
            0.8 if r.confidence_class in (CONF_STRONG, CONF_GOOD) else 0.3
            for r in residues
        ]
        ax.scatter(phi, psi, c=ss_colors[ss], s=sizes, alpha=alpha,
                   label=f"{ss_labels[ss]} ({len(residues)})", edgecolors="none")

    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xlabel("Phi (degrees)")
    ax.set_ylabel("Psi (degrees)")
    ax.set_title("FACET Ramachandran Plot")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.15)

    return fig


# ── Gradio interface ──

DESCRIPTION = """
# FACET: Backbone Torsion Angle Prediction from NMR Chemical Shifts

Upload a chemical shift list (.tab or .csv format) to predict backbone
phi/psi torsion angles, secondary structure, and chi1 rotamers.

**Outputs:** XPLOR/CNS restraints (.tbl), CYANA constraints (.aco),
NEF dihedral restraints, TALOS-N-style summary, and Ramachandran plot.

FACET v3 surpasses TALOS-N: **13.4 vs 14.4 deg** median error on 27,461
matched residues.
"""

FORMAT_CHOICES = ["XPLOR .tbl", "CYANA .aco", "NEF .nef", "pred.tab", "CSV", "JSON"]

demo = gr.Interface(
    fn=predict_and_format,
    inputs=[
        gr.File(label="Shift list (.tab or .csv)", file_types=[".tab", ".csv", ".txt"]),
        gr.CheckboxGroup(
            choices=FORMAT_CHOICES,
            value=["XPLOR .tbl", "CYANA .aco", "pred.tab"],
            label="Output formats",
        ),
    ],
    outputs=[
        gr.Textbox(label="Summary", lines=20),
        gr.Image(label="Sequence + Secondary Structure"),
        gr.Image(label="Ramachandran Plot"),
        gr.Files(label="Download restraint files"),
    ],
    title="FACET",
    description=DESCRIPTION,
    examples=[
        ["examples/ubiquitin.tab", ["XPLOR .tbl", "CYANA .aco", "pred.tab"]],
    ],
    cache_examples=False,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
