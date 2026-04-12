"""Publication-grade figures for FACET predictions.

Reproduces the sequence-with-SS-cartoon layout commonly used in NMR
publications: amino-acid sequence in rows, secondary-structure elements
(beta arrows, alpha cylinders) underneath, labeled β1..βN / α1..αN.

Requires matplotlib (optional dependency; install via ``facet-nmr[plot]``).
"""
from __future__ import annotations

import math
from pathlib import Path

from .io.formats import AA_THREE_TO_ONE, FACETResult, ResiduePrediction


# Colors (publication palette)
COLOR_HELIX = "#C0392B"      # red — alpha helix
COLOR_STRAND = "#2C3E50"     # dark blue/gray — beta strand
COLOR_COIL = "#95A5A6"       # neutral gray — coil/loop
COLOR_STRONG = "#27AE60"     # green — high-confidence
COLOR_GOOD = "#3498DB"       # blue — moderate
COLOR_WARN = "#E67E22"       # orange — caution
COLOR_DYNAMIC = "#95A5A6"    # gray — low confidence


def _ss_elements(residues: list[ResiduePrediction]) -> list[tuple[str, int, int, str]]:
    """Identify contiguous SS elements (H or E runs).

    Returns list of (ss_type, start_idx, end_idx_inclusive, label).
    """
    elements: list[tuple[str, int, int, str]] = []
    n_h = 0
    n_e = 0
    i = 0
    n = len(residues)
    while i < n:
        ss = residues[i].ss
        if ss in ("H", "E"):
            j = i
            while j < n and residues[j].ss == ss:
                j += 1
            # Minimum length filter: single-residue H/E is noise
            if j - i >= 2:
                if ss == "H":
                    n_h += 1
                    label = f"α{n_h}"
                else:
                    n_e += 1
                    label = f"β{n_e}"
                elements.append((ss, i, j - 1, label))
            i = j
        else:
            i += 1
    return elements


def _draw_helix_cylinder(ax, x_start, x_end, y_center, height=0.4):
    """Draw a rounded rectangle (cylinder) for a helix element."""
    from matplotlib.patches import FancyBboxPatch
    width = x_end - x_start + 1
    patch = FancyBboxPatch(
        (x_start - 0.5, y_center - height / 2),
        width, height,
        boxstyle="round,pad=0,rounding_size=0.15",
        facecolor=COLOR_HELIX,
        edgecolor="black",
        linewidth=1.0,
    )
    ax.add_patch(patch)


def _draw_strand_arrow(ax, x_start, x_end, y_center, height=0.4):
    """Draw an arrow shape for a beta strand element."""
    from matplotlib.patches import Polygon
    # Arrow: rectangle body + triangle head at the C-terminal end
    width = x_end - x_start + 1
    head_width = min(0.8, width * 0.35)  # tip length in residues
    body_end = x_end + 0.5 - head_width

    # Flare: wider at the head
    body_half = height / 2 * 0.7
    head_half = height / 2

    points = [
        (x_start - 0.5, y_center - body_half),          # body bottom-left
        (body_end,      y_center - body_half),          # body bottom-right
        (body_end,      y_center - head_half),          # flare bottom
        (x_end + 0.5,   y_center),                      # arrow tip
        (body_end,      y_center + head_half),          # flare top
        (body_end,      y_center + body_half),          # body top-right
        (x_start - 0.5, y_center + body_half),          # body top-left
    ]
    patch = Polygon(
        points,
        closed=True,
        facecolor=COLOR_STRAND,
        edgecolor="black",
        linewidth=1.0,
    )
    ax.add_patch(patch)


def _conf_color(conf_class: str) -> str:
    return {
        "Strong": COLOR_STRONG,
        "Good": COLOR_GOOD,
        "Warn": COLOR_WARN,
        "Dynamic": COLOR_DYNAMIC,
    }.get(conf_class, COLOR_COIL)


def plot_sequence_ss(
    result: FACETResult,
    path: str | Path | None = None,
    residues_per_row: int = 60,
    title: str | None = None,
    show_confidence: bool = True,
):
    """Publication-grade sequence + secondary structure figure.

    Lays out the protein sequence in rows of ``residues_per_row`` with:
      - confidence bars above each residue (optional, colored by tier)
      - one-letter sequence + residue numbers
      - SS cartoon below (β arrows, α cylinders) with labels

    Args:
        result: FACETResult from predict()
        path: Output path (.png / .pdf / .svg). None → don't save, return fig.
        residues_per_row: Residues per row (default 60)
        title: Figure title (default: derived from result.source)
        show_confidence: If True, draw confidence bars above sequence

    Returns:
        matplotlib Figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np

    residues = result.residues
    n = len(residues)
    if n == 0:
        raise ValueError("No residues to plot")

    n_rows = math.ceil(n / residues_per_row)
    row_height = 2.0 if show_confidence else 1.3
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, row_height * n_rows + 0.5),
        squeeze=False,
    )
    axes = axes.flatten()

    # Precompute SS elements over the whole protein
    elements = _ss_elements(residues)

    # Map seq_id → 0-indexed position for column alignment
    seq_ids = [r.seq_id for r in residues]
    min_sid = seq_ids[0]

    for row_idx in range(n_rows):
        ax = axes[row_idx]
        r0 = row_idx * residues_per_row
        r1 = min(r0 + residues_per_row, n)
        row_residues = residues[r0:r1]
        row_len = len(row_residues)

        # X-axis: local column 0 .. row_len - 1
        x = np.arange(row_len)

        # ── Confidence bars (top) ──
        if show_confidence:
            bar_y_top = 1.0
            bar_y_bot = 0.3
            bar_heights = []
            bar_colors = []
            for r in row_residues:
                # Normalize confidence to [0, 1] for bar height
                # Strong ≈ 0.9, Good ≈ 0.6, Warn ≈ 0.4, Dynamic ≈ 0.15
                cls_height = {
                    "Strong": 0.9,
                    "Good": 0.65,
                    "Warn": 0.45,
                    "Dynamic": 0.20,
                }.get(r.confidence_class, 0.2)
                bar_heights.append(cls_height)
                bar_colors.append(_conf_color(r.confidence_class))

            ax.bar(
                x, bar_heights,
                width=0.85,
                bottom=bar_y_bot,
                color=bar_colors,
                edgecolor="none",
            )

        # ── Sequence letters ──
        seq_y = -0.25 if show_confidence else 0.5
        for i, r in enumerate(row_residues):
            aa1 = AA_THREE_TO_ONE.get(r.comp_id, "X")
            ax.text(
                i, seq_y, aa1,
                ha="center", va="center",
                fontsize=10,
                fontfamily="monospace",
                fontweight="bold",
            )

        # Residue numbering every 10 residues (within the row)
        num_y = seq_y - 0.6
        for i, r in enumerate(row_residues):
            if r.seq_id % 10 == 0 or i == 0 or i == row_len - 1:
                ax.text(
                    i, num_y, str(r.seq_id),
                    ha="center", va="top",
                    fontsize=7,
                    color="gray",
                )

        # ── SS cartoon (bottom) ──
        ss_y = -1.3 if show_confidence else -0.5

        # Draw a thin baseline for coil
        ax.plot(
            [-0.5, row_len - 0.5],
            [ss_y, ss_y],
            color=COLOR_COIL,
            linewidth=1.0,
            zorder=1,
        )

        # Draw SS elements that overlap this row
        for ss_type, start, end, label in elements:
            if end < r0 or start >= r1:
                continue  # outside this row
            # Clip to row
            row_start = max(start, r0) - r0
            row_end = min(end, r1 - 1) - r0

            if ss_type == "H":
                _draw_helix_cylinder(ax, row_start, row_end, ss_y, height=0.55)
            elif ss_type == "E":
                _draw_strand_arrow(ax, row_start, row_end, ss_y, height=0.55)

            # Label (centered under the element within this row)
            label_x = (row_start + row_end) / 2
            ax.text(
                label_x, ss_y - 0.85, label,
                ha="center", va="top",
                fontsize=9,
                fontweight="bold",
                color=COLOR_HELIX if ss_type == "H" else COLOR_STRAND,
            )

        # ── Axis styling ──
        ax.set_xlim(-1, residues_per_row)
        y_min = ss_y - 1.3
        y_max = 1.3 if show_confidence else 0.9
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("auto")
        ax.axis("off")

    # Title
    if title is None:
        source = result.source or "FACET prediction"
        title = f"FACET secondary structure — {source}"
    fig.suptitle(title, fontsize=12, y=0.995)

    # Legend at the top
    if show_confidence:
        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor=COLOR_STRONG, label="Strong"),
            Patch(facecolor=COLOR_GOOD, label="Good"),
            Patch(facecolor=COLOR_WARN, label="Warn"),
            Patch(facecolor=COLOR_DYNAMIC, label="Dynamic"),
            Patch(facecolor=COLOR_HELIX, label="α helix"),
            Patch(facecolor=COLOR_STRAND, label="β strand"),
        ]
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.985),
            ncol=6,
            frameon=False,
            fontsize=8,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")

    return fig
