"""Publication-grade figures for FACET predictions.

Produces a clean sequence + secondary-structure cartoon in the style of
NMR/structural biology publications (see e.g., Brown et al., Angew. Chem.
Int. Ed. 2017). Design principles:

  - One semantic axis per visual channel:
      * Bar height   = continuous confidence (not class)
      * Bar color    = single neutral (charcoal)
      * Dashed line  = "Strong" threshold reference
      * SS shape     = helix cylinder vs strand arrow (convention)
      * SS color     = red helix / navy strand (convention)
      * Dynamic residues = absence of bar + gray tick marker

  - Strict vertical layering (no overlap zones between sequence,
    numbering, SS cartoon, and element labels).

  - DSSP-like SS smoothing: α-helix ≥4 consecutive H, β-strand ≥2
    consecutive E. Shorter runs are treated as coil.

Requires matplotlib (install via ``facet-nmr[plot]``).
"""
from __future__ import annotations

import math
from pathlib import Path

from .io.formats import AA_THREE_TO_ONE, FACETResult, ResiduePrediction


# ─────────────────────────── Style ───────────────────────────

# Single palette — minimal colors to keep publication aesthetic.
# Confidence bars are neutral slate gray (distinct from navy strand)
# so the three semantic roles (confidence / strand / helix) each have
# their own visually unambiguous color.
COLOR_BAR = "#566573"           # slate gray — confidence bars
COLOR_FLEXIBLE = "#1ABC9C"      # teal — flexible / disordered residues
COLOR_THRESHOLD = "#7F8C8D"     # mid-gray — "High tier" dashed reference
COLOR_HELIX = "#C0392B"         # red — alpha helix (convention)
COLOR_STRAND = "#1F3A5F"        # navy — beta strand (convention)
COLOR_BASELINE = "#95A5A6"      # light gray — coil baseline line
COLOR_SEQ = "#1C2833"           # near-black — sequence letters
COLOR_NUM = "#7F8C8D"           # mid-gray — residue numbers
COLOR_UNASSIGNED = "#C8CDD0"    # very light gray — unassigned residue placeholder

# DSSP-like minimum lengths for SS elements
MIN_HELIX_LEN = 4   # α-helix = one full turn (≥4 residues)
MIN_STRAND_LEN = 2  # β-strand = ≥2 to distinguish from isolated H-bond

# Confidence score range for visualization. Theoretical range is
# [-log(1296), 0] ≈ [-7.17, 0], but the empirical 10th–90th percentile
# range on calibrated FACET v3 is [-5.2, -3.0]. Using the empirical
# range gives bars with visible dynamic range instead of all bunched
# near the top.
CONF_MIN = -5.2              # bottom 10% of valid predictions
CONF_MAX = -2.5              # top 10% of valid predictions
CONF_HIGH_THRESHOLD = -3.72  # High tier cutoff (8.2% fail25)

# Vertical layout — strict layering with clear gaps
Y_BAR_BASE = 3.6
Y_BAR_TOP = 5.0
Y_NUM = 3.25
Y_SEQ = 2.35
Y_SS_CENTER = 1.15
Y_LABEL = 0.35
Y_MIN = -0.1

SS_ELEMENT_HEIGHT = 0.55


# ───────────────── SS element identification ─────────────────

def _ss_elements(
    residues: list[ResiduePrediction],
    seq_id_base: int,
) -> list[tuple[str, int, int, str]]:
    """Identify SS elements with DSSP-like minimum-length smoothing.

    Returns list of (ss_type, start_pos, end_pos_inclusive, label) where
    ss_type ∈ {"H", "E"}, positions are sequence-space (seq_id - seq_id_base),
    and label is "α1", "β1", etc.

    Runs are broken at non-consecutive seq_ids so an element cannot span an
    unassigned gap. Runs shorter than the minimum length are demoted to coil
    (not emitted).
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
                if j > i and residues[j].seq_id != residues[j - 1].seq_id + 1:
                    break
                j += 1
            run_len = j - i
            min_len = MIN_HELIX_LEN if ss == "H" else MIN_STRAND_LEN
            if run_len >= min_len:
                start_pos = residues[i].seq_id - seq_id_base
                end_pos = residues[j - 1].seq_id - seq_id_base
                if ss == "H":
                    n_h += 1
                    label = f"α{n_h}"
                else:
                    n_e += 1
                    label = f"β{n_e}"
                elements.append((ss, start_pos, end_pos, label))
            i = j
        else:
            i += 1
    return elements


def _normalize_confidence(conf: float) -> float:
    """Map a raw confidence value (negative entropy) to [0, 1] for bar height."""
    x = (conf - CONF_MIN) / (CONF_MAX - CONF_MIN)
    return max(0.0, min(1.0, x))


# ─────────────────────────── Shapes ──────────────────────────

def _draw_helix(ax, x_start, x_end, y_center):
    """Draw a filled rounded rectangle (cylinder) for a helix."""
    from matplotlib.patches import FancyBboxPatch
    width = x_end - x_start + 1
    patch = FancyBboxPatch(
        (x_start - 0.5, y_center - SS_ELEMENT_HEIGHT / 2),
        width, SS_ELEMENT_HEIGHT,
        boxstyle="round,pad=0,rounding_size=0.18",
        facecolor=COLOR_HELIX,
        edgecolor="#5D2116",  # darker red outline
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(patch)


def _draw_strand(ax, x_start, x_end, y_center):
    """Draw a filled arrow for a beta strand, pointing N→C."""
    from matplotlib.patches import Polygon
    width = x_end - x_start + 1
    head_width = min(0.9, max(0.4, width * 0.30))
    body_end = x_end + 0.5 - head_width

    body_half = SS_ELEMENT_HEIGHT * 0.35
    head_half = SS_ELEMENT_HEIGHT * 0.55

    points = [
        (x_start - 0.5, y_center - body_half),
        (body_end,      y_center - body_half),
        (body_end,      y_center - head_half),
        (x_end + 0.5,   y_center),
        (body_end,      y_center + head_half),
        (body_end,      y_center + body_half),
        (x_start - 0.5, y_center + body_half),
    ]
    patch = Polygon(
        points, closed=True,
        facecolor=COLOR_STRAND,
        edgecolor="#0F1419",
        linewidth=0.8,
        zorder=3,
    )
    ax.add_patch(patch)


def _draw_coil(ax, x_start, x_end, y_center):
    """Draw a soft wavy line for a coil run.

    Distinct from the 'no data' background — coil is a positive visual
    signal (wave glyph), not the absence of a patch. Important for IDP
    outputs where almost every residue is coil.
    """
    import numpy as np
    # Enough samples for smooth curves even over short runs.
    n = max(40, int((x_end - x_start + 1) * 10))
    x = np.linspace(x_start - 0.5, x_end + 0.5, n)
    amplitude = SS_ELEMENT_HEIGHT * 0.18
    wavelength = 1.0   # one cycle per residue → looks like a random-coil wiggle
    y = y_center + amplitude * np.sin(2 * np.pi * x / wavelength)
    ax.plot(
        x, y,
        color=COLOR_BASELINE,
        linewidth=1.3,
        solid_capstyle="round",
        zorder=2,
    )


# ─────────────────────────── Main ────────────────────────────

def plot_sequence_ss(
    result: FACETResult,
    path: str | Path | None = None,
    residues_per_row: int = 60,
    title: str | None = None,
    sequence: str | None = None,
    seq_id_start: int | None = None,
):
    """Publication-grade sequence + secondary structure figure.

    Layers (top to bottom):
      1. Confidence bars (continuous height, single color, + threshold line)
      2. Residue numbering (every 10, small gray)
      3. One-letter sequence (bold monospace; dim for unassigned residues)
      4. SS cartoon (helix cylinders, strand arrows)
      5. SS element labels (β1, α1, ...)

    Args:
        result: FACETResult from predict()
        path: Output path (.png/.pdf/.svg). None → return fig without saving.
        residues_per_row: Residues per row (default 60)
        title: Figure title (default: derived from result.source)
        sequence: Optional one-letter sequence override. If provided, takes
            precedence over ``result.sequence``. Used to render real amino-acid
            letters in unassigned slots and to extend the plot range across
            unassigned N/C termini.
        seq_id_start: Optional seq_id corresponding to ``sequence[0]``. Only
            meaningful when ``sequence`` is given; defaults to
            ``result.seq_id_start`` and falls back to 1.

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if len(result.residues) == 0:
        raise ValueError("No residues to plot")

    # Sort by seq_id so gap detection and position math work regardless of
    # the order the predictor emitted.
    residues = sorted(result.residues, key=lambda r: r.seq_id)

    # Resolve the display sequence. Priority: explicit param → result field →
    # none (fall back to placeholder dots for unassigned slots).
    display_sequence = sequence if sequence is not None else (result.sequence or "")
    display_seq_id_start = (
        seq_id_start
        if seq_id_start is not None
        else (result.seq_id_start if display_sequence else 1)
    )

    # Sanity-check the sequence length against the assignment span. NMR-STAR
    # readers sometimes return a single-character placeholder (e.g. ".") as
    # a polymer sequence, which would collapse the plot to one position and
    # render it empty. Reject any sequence too short to cover the assigned
    # residues.
    if display_sequence:
        min_required = residues[-1].seq_id - display_seq_id_start + 1
        if len(display_sequence) < max(min_required, len(residues)):
            import logging
            logging.getLogger("facet").warning(
                "Reader-supplied sequence (len=%d) is shorter than the "
                "assigned residue span (%d) — ignoring and using "
                "assignment-driven rendering.",
                len(display_sequence), min_required,
            )
            display_sequence = ""

    # Sanity-check the sequence against the assigned residues. If > 20% of
    # assigned positions disagree with the sequence letter, drop the sequence
    # and fall back to placeholder rendering — most likely the reader picked
    # the wrong entity or the seq_id_start is wrong.
    if display_sequence:
        seq_end = display_seq_id_start + len(display_sequence) - 1
        n_check = 0
        n_mismatch = 0
        for r in residues:
            idx = r.seq_id - display_seq_id_start
            if 0 <= idx < len(display_sequence):
                n_check += 1
                expected = AA_THREE_TO_ONE.get(r.comp_id, "X")
                if display_sequence[idx] != expected and display_sequence[idx] != "X":
                    n_mismatch += 1
        if n_check > 0 and n_mismatch / n_check > 0.2:
            import logging
            logging.getLogger("facet").warning(
                "Sequence/assignment mismatch (%d/%d positions disagree) — "
                "falling back to placeholder rendering",
                n_mismatch, n_check,
            )
            display_sequence = ""

    # Position space: position = seq_id - seq_id_base.
    if display_sequence:
        # Sequence-driven: span the full known polymer, not just the assigned range.
        seq_id_base = display_seq_id_start
        total_positions = len(display_sequence)
    else:
        # Assignment-driven: start at first assigned, end at last. Unassigned
        # internal gaps still render as empty slots (via the position map).
        seq_id_base = residues[0].seq_id
        total_positions = residues[-1].seq_id - seq_id_base + 1

    position_to_residue: dict[int, ResiduePrediction] = {
        r.seq_id - seq_id_base: r for r in residues
        if 0 <= r.seq_id - seq_id_base < total_positions
    }

    n_rows = math.ceil(total_positions / residues_per_row)
    row_height_in = 1.6  # inches per row

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, row_height_in * n_rows + 0.6),
        squeeze=False,
    )
    axes = axes.flatten()

    elements = _ss_elements(residues, seq_id_base)

    for row_idx in range(n_rows):
        ax = axes[row_idx]
        p0 = row_idx * residues_per_row
        p1 = min(p0 + residues_per_row, total_positions)
        row_len = p1 - p0

        x = np.arange(row_len)

        # ── Layer 1: confidence bars ──
        bar_range = Y_BAR_TOP - Y_BAR_BASE
        bar_heights: list[float] = []
        is_flexible: list[bool] = []
        is_unassigned: list[bool] = []
        for pos in range(p0, p1):
            r = position_to_residue.get(pos)
            if r is None:
                bar_heights.append(0.0)
                is_flexible.append(False)
                is_unassigned.append(True)
            elif r.confidence_class == "Flexible":
                bar_heights.append(0.0)
                is_flexible.append(True)
                is_unassigned.append(False)
            else:
                h = _normalize_confidence(r.confidence) * bar_range
                bar_heights.append(h)
                is_flexible.append(False)
                is_unassigned.append(False)

        # Main bars (uniform charcoal color)
        ax.bar(
            x, bar_heights,
            width=0.82,
            bottom=Y_BAR_BASE,
            color=COLOR_BAR,
            edgecolor="none",
            zorder=2,
        )

        # Flexible residues: teal "×" where the bar would be — clearly
        # distinguishable from a low-confidence short bar. "Flexible"
        # means biologically disordered, not a model failure.
        flex_x = [i for i, f in enumerate(is_flexible) if f]
        if flex_x:
            ax.scatter(
                flex_x,
                [Y_BAR_BASE + 0.25] * len(flex_x),
                s=40,
                color=COLOR_FLEXIBLE,
                marker="x",
                linewidths=1.5,
                zorder=2,
            )

        # Unassigned residues: small light-gray dash at bar base to mark
        # the missing slot without filling it.
        unassigned_x = [i for i, u in enumerate(is_unassigned) if u]
        if unassigned_x:
            ax.scatter(
                unassigned_x,
                [Y_BAR_BASE] * len(unassigned_x),
                s=12,
                color=COLOR_UNASSIGNED,
                marker="_",
                linewidths=1.0,
                zorder=2,
            )

        # High-tier threshold dashed reference line
        high_norm = _normalize_confidence(CONF_HIGH_THRESHOLD) * bar_range
        ax.axhline(
            Y_BAR_BASE + high_norm,
            xmin=0, xmax=1,
            linestyle=(0, (2, 3)),
            color=COLOR_THRESHOLD,
            linewidth=0.8,
            zorder=1,
            alpha=0.7,
        )

        # ── Layer 2: residue numbers (every 10 + row termini) ──
        for i, pos in enumerate(range(p0, p1)):
            seq_id = pos + seq_id_base
            if seq_id % 10 == 0 or i == 0 or i == row_len - 1:
                ax.text(
                    i, Y_NUM, str(seq_id),
                    ha="center", va="center",
                    fontsize=7,
                    color=COLOR_NUM,
                )

        # ── Layer 3: sequence letters ──
        for i, pos in enumerate(range(p0, p1)):
            r = position_to_residue.get(pos)
            if r is None:
                # Unassigned slot. If a reader-supplied sequence is available
                # we render the real AA letter in dim gray; otherwise fall
                # back to a neutral dot placeholder.
                if display_sequence and 0 <= pos < len(display_sequence):
                    letter = display_sequence[pos]
                    if letter == "X":
                        letter = "·"
                else:
                    letter = "·"
                ax.text(
                    i, Y_SEQ, letter,
                    ha="center", va="center",
                    fontsize=10,
                    fontfamily="monospace",
                    color=COLOR_UNASSIGNED,
                )
            else:
                aa1 = AA_THREE_TO_ONE.get(r.comp_id, "X")
                ax.text(
                    i, Y_SEQ, aa1,
                    ha="center", va="center",
                    fontsize=10,
                    fontfamily="monospace",
                    fontweight="bold",
                    color=COLOR_SEQ,
                )

        # ── Layer 4: SS cartoon ──
        # Build coil runs for this row = everything in [0, row_len) not
        # covered by any H/E element. Drawn as a soft wavy line so coil
        # is a positive glyph, not just "no patch here" — important for
        # IDP outputs where nearly every residue is coil.
        row_elements = []
        for ss_type, start, end, label in elements:
            if end < p0 or start >= p1:
                continue
            row_elements.append((
                ss_type,
                max(start, p0) - p0,
                min(end, p1 - 1) - p0,
                label,
            ))
        row_elements.sort(key=lambda e: e[1])

        cursor = 0
        for _, rs, re_, _ in row_elements:
            if rs > cursor:
                _draw_coil(ax, cursor, rs - 1, Y_SS_CENTER)
            cursor = re_ + 1
        if cursor < row_len:
            _draw_coil(ax, cursor, row_len - 1, Y_SS_CENTER)

        # Draw H/E elements on top of the coil wave
        for ss_type, row_start, row_end, label in row_elements:
            if ss_type == "H":
                _draw_helix(ax, row_start, row_end, Y_SS_CENTER)
            else:
                _draw_strand(ax, row_start, row_end, Y_SS_CENTER)

            # ── Layer 5: element label ──
            label_x = (row_start + row_end) / 2
            label_color = COLOR_HELIX if ss_type == "H" else COLOR_STRAND
            ax.text(
                label_x, Y_LABEL, label,
                ha="center", va="center",
                fontsize=9,
                fontweight="bold",
                color=label_color,
            )

        # ── Axis ──
        ax.set_xlim(-1.0, residues_per_row)
        ax.set_ylim(Y_MIN, Y_BAR_TOP + 0.15)
        ax.axis("off")

    # Title (top, centered)
    if title is None:
        source = result.source or "FACET"
        title = f"FACET backbone prediction — {Path(source).stem if source else 'prediction'}"
    fig.suptitle(title, fontsize=12, y=0.99, fontweight="bold", color=COLOR_SEQ)

    # Legend at bottom — keeps the title band clean.
    # "Flexible" = biologically disordered region (real feature, not
    # a model failure). FACET correctly flags these rather than
    # assigning spurious rigid angles.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    # Use a short wavy-line sample as the legend marker for coil so the
    # legend shape matches what's drawn in the panel.
    import numpy as np
    _coil_x = np.linspace(0, 1, 60)
    _coil_y = 0.4 * np.sin(2 * np.pi * _coil_x / 0.3)
    handles = [
        Patch(facecolor=COLOR_BAR, label="Confidence"),
        Line2D([0], [0], linestyle=(0, (2, 3)), color=COLOR_THRESHOLD,
               linewidth=1.2, label="High-tier threshold"),
        Line2D([0], [0], marker="x", color=COLOR_FLEXIBLE,
               markersize=7, linewidth=0, markeredgewidth=1.8,
               label="Flexible"),
        Line2D([0], [0], marker="_", color=COLOR_UNASSIGNED,
               markersize=9, linewidth=0, markeredgewidth=1.4,
               label="Unassigned"),
        Patch(facecolor=COLOR_HELIX, label="α helix"),
        Patch(facecolor=COLOR_STRAND, label="β strand"),
        Line2D(_coil_x, _coil_y, color=COLOR_BASELINE,
               linewidth=1.3, label="coil"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=7,
        frameon=False,
        fontsize=9,
    )

    # Leave room at top for title and at bottom for legend
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")

    return fig


# ───────────────────── IDP residual-SS figure ─────────────────────

# Palette for the four basins — each visually distinct, helix/strand
# match the main plot's conventions, PPII and "other" get their own hues.
BASIN_COLORS = {
    "alpha": "#C0392B",   # red — α_R, same as helix in main plot
    "beta":  "#1F3A5F",   # navy — β, same as strand
    "ppii":  "#16A085",   # teal-green — PPII (extended disordered)
    "other": "#7F8C8D",   # gray — everything else
}
BASIN_LABELS_GEOM = {
    "alpha": "α_R",
    "beta":  "β",
    "ppii":  "PPII",
    "other": "other",
}
BASIN_LABELS_STRUCT = {
    "alpha": "Helix",
    "beta":  "Beta",
    "ppii":  "PPII",
    "other": "Coil",
}
# Backwards-compat alias pointing at the geometric labels.
BASIN_LABELS = BASIN_LABELS_GEOM


def plot_residual_ss(
    result: FACETResult,
    path: str | Path | None = None,
    residues_per_row: int = 60,
    title: str | None = None,
    mode: str = "geometric",
):
    """Per-residue basin-population figure for flexible / disordered samples.

    Two modes are supported:

    - **mode="geometric"** (default): uses the ``basin_populations``
      vector from retrieval — fraction of retrieved neighbours whose
      phi/psi fall in each Ramachandran region. Answers "which region
      of the plot is the ensemble in". Tracks labelled α_R / β / PPII /
      other.
    - **mode="structural"**: uses ``structural_populations`` —
      kernel-weighted Bayesian estimate of canonical SS-state
      populations (helix / beta / PPII / coil), comparable to
      d2D. Tracks labelled Helix / Beta / PPII / Coil. More directly
      comparable to published IDP state populations; magnitudes agree
      with d2D to within ~0.10 per state.

    Visual twin of ``plot_sequence_ss`` — same figure width, row wrap,
    fonts, and palette so the two figures can be shown side-by-side
    without jarring style mismatches.

    For samples where nearly every residue is coil (IDPs, unfolded
    peptides, disordered regions of folded proteins), the H/E/C cartoon
    from ``plot_sequence_ss`` carries almost no information. The residual
    structural signal lives in the **basin populations** — the per-residue
    α / β / PPII / other fractions from the retrieval neighbours.

    This figure renders that signal as four horizontal tracks (one per
    basin) along the sequence. Each residue contributes a filled tile
    whose height is proportional to the basin fraction (0 to 1). Users
    can read off transient-helix segments (α track elevated), PPII-rich
    regions, nascent β, etc. directly from the track profiles.

    Requires that the prediction was run with retrieval enabled (the
    default); residues without ``basin_populations`` render as blank.

    Args:
        result: FACETResult from ``predict()``.
        path: Output .png/.pdf/.svg path. None → return fig without saving.
        residues_per_row: Wrap the sequence after this many residues
            (default 60 — matches ``plot_sequence_ss``).
        title: Optional title override.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not result.residues:
        raise ValueError("result has no residues")

    # Sort by seq_id so gap handling and position math match plot_sequence_ss.
    residues = sorted(result.residues, key=lambda r: r.seq_id)

    # Resolve display sequence (mirror plot_sequence_ss logic) so unassigned
    # slots can render the real AA letter in dim gray instead of a bare
    # placeholder dot. Same length-and-mismatch sanity checks.
    display_sequence = (result.sequence or "")
    display_seq_id_start = result.seq_id_start if display_sequence else 1
    if display_sequence:
        min_required = residues[-1].seq_id - display_seq_id_start + 1
        if len(display_sequence) < max(min_required, len(residues)):
            display_sequence = ""
    if display_sequence:
        n_check = 0
        n_mismatch = 0
        for r in residues:
            idx = r.seq_id - display_seq_id_start
            if 0 <= idx < len(display_sequence):
                n_check += 1
                expected = AA_THREE_TO_ONE.get(r.comp_id, "X")
                if display_sequence[idx] != expected and display_sequence[idx] != "X":
                    n_mismatch += 1
        if n_check > 0 and n_mismatch / n_check > 0.2:
            display_sequence = ""

    if display_sequence:
        seq_id_base = display_seq_id_start
        total_positions = len(display_sequence)
    else:
        seq_id_base = residues[0].seq_id
        total_positions = residues[-1].seq_id - seq_id_base + 1

    position_to_residue = {
        r.seq_id - seq_id_base: r for r in residues
        if 0 <= r.seq_id - seq_id_base < total_positions
    }

    # Build per-position population arrays (NaN where residue is unassigned
    # or lacks the requested populations). Mode selects which per-residue
    # field to read from.
    if mode not in ("geometric", "structural"):
        raise ValueError(
            f"plot_residual_ss: mode must be 'geometric' or 'structural', got {mode!r}"
        )
    alpha_arr = np.full(total_positions, np.nan)
    beta_arr = np.full(total_positions, np.nan)
    ppii_arr = np.full(total_positions, np.nan)
    other_arr = np.full(total_positions, np.nan)
    for pos, r in position_to_residue.items():
        source = (
            r.structural_populations
            if mode == "structural"
            else r.basin_populations
        )
        if source is not None:
            a, b, p, o = source
            alpha_arr[pos] = a
            beta_arr[pos] = b
            ppii_arr[pos] = p
            other_arr[pos] = o

    n_rows = math.ceil(total_positions / residues_per_row)
    # Stacked-bar layout is more compact than the old 4-track version —
    # one track per row instead of four — so row height shrinks accordingly.
    row_height_in = 0.95

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(14, row_height_in * n_rows + 0.6),
        squeeze=False,
    )
    axes = axes.flatten()

    # Single stacked-bar track per row: each residue is one vertical bar
    # segmented by basin fraction. We only paint the *informative* stack
    # — alpha / beta / PPII — and leave the remainder (other / coil) as
    # blank background. This keeps the figure visually clean: mostly-coil
    # IDPs render as short bars with only the transient-structure signal
    # showing, instead of a wall of gray "coil" blocks swamping the real
    # signal. A faint 1.0 baseline line anchors the max stack height.
    _labels_for_mode = (
        BASIN_LABELS_STRUCT if mode == "structural" else BASIN_LABELS_GEOM
    )
    STACK_ORDER = ("alpha", "beta", "ppii")   # 'other' intentionally omitted
    STACK_HEIGHT = 1.0
    Y_TOP = STACK_HEIGHT
    Y_SEQ_IDP = -0.32
    Y_NUM_IDP = -0.72

    for row_idx in range(n_rows):
        ax = axes[row_idx]
        p0 = row_idx * residues_per_row
        p1 = min(p0 + residues_per_row, total_positions)
        row_len = p1 - p0
        xs = np.arange(row_len)

        # Draw stacked bars, alpha at the bottom, skipping the 'other'
        # baseline — mostly-coil residues render as short stacks so the
        # real signal (α/β/PPII peaks) is visually dominant.
        bottom = np.zeros(row_len, dtype=np.float64)
        arr_by_name = {"alpha": alpha_arr, "beta": beta_arr, "ppii": ppii_arr}
        for name in STACK_ORDER:
            vals = arr_by_name[name][p0:p1]
            heights = np.where(np.isnan(vals), 0.0, vals)
            ax.bar(
                xs, heights,
                width=0.9,
                bottom=bottom,
                color=BASIN_COLORS[name],
                edgecolor="none",
                zorder=2,
            )
            bottom = bottom + heights

        # Baseline at y=0 and a faint y=1.0 anchor so readers can still
        # read absolute fractions off the bars.
        ax.plot(
            [-0.5, row_len - 0.5], [0.0, 0.0],
            color="#D5D8DC", linewidth=0.7, zorder=1,
        )
        ax.plot(
            [-0.5, row_len - 0.5], [1.0, 1.0],
            color="#E8EAED", linewidth=0.6, linestyle=(0, (2, 2)), zorder=1,
        )

        # Sequence letters below the bar band.
        for i, pos in enumerate(range(p0, p1)):
            r = position_to_residue.get(pos)
            if r is None:
                if display_sequence and 0 <= pos < len(display_sequence):
                    letter = display_sequence[pos]
                    if letter == "X":
                        letter = "·"
                else:
                    letter = "·"
                ax.text(
                    i, Y_SEQ_IDP, letter,
                    ha="center", va="center",
                    fontsize=10, fontfamily="monospace",
                    color=COLOR_UNASSIGNED,
                )
            else:
                aa1 = AA_THREE_TO_ONE.get(r.comp_id, "X")
                ax.text(
                    i, Y_SEQ_IDP, aa1,
                    ha="center", va="center",
                    fontsize=10, fontfamily="monospace",
                    color=COLOR_SEQ,
                )

        # Residue numbers — every 10 + row termini.
        for i, pos in enumerate(range(p0, p1)):
            seq_id = pos + seq_id_base
            if seq_id % 10 == 0 or i == 0 or i == row_len - 1:
                ax.text(
                    i, Y_NUM_IDP, str(seq_id),
                    ha="center", va="center",
                    fontsize=7,
                    color=COLOR_NUM,
                )

        # Keep x-range identical across rows so partial last row aligns.
        ax.set_xlim(-2.5, residues_per_row)
        ax.set_ylim(Y_NUM_IDP - 0.18, Y_TOP + 0.08)
        ax.axis("off")

    if title is None:
        source = result.source or "FACET"
        mode_suffix = (
            "Cooperative SS — retrieval-based"
            if mode == "structural"
            else "Basin sampling (φ/ψ fingerprint)"
        )
        title = (
            f"{mode_suffix} — "
            f"{Path(source).stem if source else 'prediction'}"
        )
    # Match plot_sequence_ss title styling.
    fig.suptitle(title, fontsize=12, y=0.99, fontweight="bold", color=COLOR_SEQ)

    # Legend block (basin colors) — placed bottom-center to mirror the
    # sequence-plot legend. 'other' is intentionally omitted as a colored
    # swatch; it's the unpainted remainder of each stack (bar top → 1.0).
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=BASIN_COLORS["alpha"], label=_labels_for_mode["alpha"]),
        Patch(facecolor=BASIN_COLORS["beta"],  label=_labels_for_mode["beta"]),
        Patch(facecolor=BASIN_COLORS["ppii"],  label=_labels_for_mode["ppii"]),
        Patch(facecolor="white", edgecolor="#B0B0B0", linewidth=0.8,
              label=f"{_labels_for_mode['other']} (blank)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.055, 1, 0.96])

    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")

    return fig
