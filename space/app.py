"""FACET HuggingFace Space — interactive Gradio UI.

Workflow:
  1. Upload a shift list (.tab/.csv/.nef/.str) or enter a BMRB ID
  2. Click Predict
  3. Inspect the per-residue table and Ramachandran (Tier 4: density background)
  4. Click any row or point to drill into a residue (Tier 3: AA-specific
     Ramachandran context, error ellipse, input shifts, region/outlier label)
  5. Download restraint files in any of 6 formats

The publication-quality sequence + secondary-structure figure is generated
by ``facet.visualization.plot_sequence_ss`` and rendered untouched at the
top of the page so users can save it for figures.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("facet-space")

CHECKPOINT = os.environ.get(
    "FACET_CHECKPOINT",
    str(_HERE / "facet" / "weights" / "facet_v3.pt"),
)


# ─────────────────── Constants ────────────────────────────────

SS_COLORS = {"H": "#C0392B", "E": "#1F3A5F", "C": "#7F8C8D"}
SS_NAMES = {"H": "α-Helix", "E": "β-Strand", "C": "Coil"}

CHI1_NAMES_FULL = {0: "g+ (gauche+)", 1: "g- (gauche-)", 2: "t (trans)"}
CHI1_NAMES_SHORT = {0: "g+", 1: "g-", 2: "t"}

FORMAT_CHOICES = ["XPLOR .tbl", "CYANA .aco", "NEF .nef", "pred.tab", "CSV", "JSON"]

DATAFRAME_COLUMNS = ["ResID", "AA", "phi", "psi", "err", "SS", "chi1", "Class"]
SHIFTS_COLUMNS = ["HN", "HA", "N", "CA", "CB", "C'"]

# Trace order in the global Ramachandran figure. Trace 0 is always the
# density background (toggle-controlled visibility); traces 1-3 are the
# per-SS scatter layers in this fixed order. The on_rama_select handler
# relies on these indices.
_RAMA_SS_ORDER = ["C", "E", "H"]


# ─────────────────── Plot builders ────────────────────────────


def build_global_ramachandran(result, selected_seq_id=None, show_background=True):
    """Global Ramachandran with per-residue scatter and density background."""
    from facet.ramachandran import (
        GRID_PHI,
        GRID_PSI,
        density_contour_levels,
        get_global_density,
    )

    fig = go.Figure()

    # Trace 0: density background (always present, visibility-toggled).
    density = get_global_density()
    levels = density_contour_levels(density)
    fig.add_trace(
        go.Contour(
            x=GRID_PHI,
            y=GRID_PSI,
            z=density,
            contours=dict(
                start=levels[0],
                end=levels[2],
                size=(levels[2] - levels[0]) / 6,
                showlines=True,
                showlabels=False,
                coloring="lines",
            ),
            line=dict(color="#bdbdbd", width=1),
            showscale=False,
            hoverinfo="skip",
            visible=show_background,
            name="Density",
            showlegend=False,
        )
    )

    # Traces 1-3: per-SS scatter (always added, even if empty, so trace
    # indices stay stable for the click handler).
    by_ss = {"H": [], "E": [], "C": []}
    for r in result.residues:
        by_ss[r.ss].append(r)

    for ss in _RAMA_SS_ORDER:
        residues = by_ss.get(ss, [])
        if residues:
            sizes = [
                14 if r.seq_id == selected_seq_id else 9 for r in residues
            ]
            line_widths = [
                2.5 if r.seq_id == selected_seq_id else 0 for r in residues
            ]
            opacities = [
                1.0
                if r.seq_id == selected_seq_id or selected_seq_id is None
                else 0.45
                for r in residues
            ]
            hover_text = [
                f"<b>{r.seq_id} {r.comp_id}</b><br>"
                f"phi = {r.phi:.0f}° ± {r.phi_err:.0f}°<br>"
                f"psi = {r.psi:.0f}° ± {r.psi_err:.0f}°<br>"
                f"SS: {SS_NAMES[r.ss]}<br>"
                f"chi1: {CHI1_NAMES_FULL.get(r.chi1, '—')}<br>"
                f"Class: {r.confidence_class}"
                for r in residues
            ]
            fig.add_trace(
                go.Scatter(
                    x=[r.phi for r in residues],
                    y=[r.psi for r in residues],
                    mode="markers",
                    marker=dict(
                        color=SS_COLORS[ss],
                        size=sizes,
                        opacity=opacities,
                        line=dict(color="black", width=line_widths),
                    ),
                    name=f"{SS_NAMES[ss]} ({len(residues)})",
                    text=hover_text,
                    hoverinfo="text",
                    customdata=[[r.seq_id] for r in residues],
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker=dict(color=SS_COLORS[ss], size=9),
                    name=f"{SS_NAMES[ss]} (0)",
                    showlegend=False,
                    customdata=[],
                )
            )

    fig.update_layout(
        xaxis=dict(
            title="φ (degrees)",
            range=[-180, 180],
            tickmode="linear",
            tick0=-180,
            dtick=60,
            zeroline=True,
            zerolinecolor="#cccccc",
            gridcolor="#eeeeee",
        ),
        yaxis=dict(
            title="ψ (degrees)",
            range=[-180, 180],
            tickmode="linear",
            tick0=-180,
            dtick=60,
            zeroline=True,
            zerolinecolor="#cccccc",
            gridcolor="#eeeeee",
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=10, t=10, b=50),
        legend=dict(
            x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc"
        ),
        height=520,
    )
    return fig


def build_residue_ramachandran(result, selected_seq_id):
    """AA-specific Ramachandran for a single residue (Tier 3 inspector)."""
    from facet.ramachandran import (
        GRID_PHI,
        GRID_PSI,
        density_contour_levels,
        get_density,
    )

    if selected_seq_id is None or result is None:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="Select a residue to inspect",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="#999999"),
                )
            ],
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        return fig

    selected = next((r for r in result.residues if r.seq_id == selected_seq_id), None)
    if selected is None:
        return build_residue_ramachandran(result, None)

    aa = selected.comp_id
    density = get_density(aa)
    levels = density_contour_levels(density)
    ss_color = SS_COLORS[selected.ss]

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=GRID_PHI,
            y=GRID_PSI,
            z=density,
            contours=dict(
                start=levels[0],
                end=levels[2],
                size=(levels[2] - levels[0]) / 6,
                showlines=True,
                showlabels=False,
                coloring="lines",
            ),
            line=dict(color="#9b9b9b", width=1.2),
            showscale=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[selected.phi],
            y=[selected.psi],
            mode="markers",
            marker=dict(
                color=ss_color, size=18, line=dict(color="black", width=2)
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Error ring at the displayed ±err bound. phi_err == psi_err in v1
    # so this is a circle, but the ellipse math handles the general case.
    theta = np.linspace(0, 2 * np.pi, 80)
    ell_x = selected.phi + selected.phi_err * np.cos(theta)
    ell_y = selected.psi + selected.psi_err * np.sin(theta)
    fig.add_trace(
        go.Scatter(
            x=ell_x,
            y=ell_y,
            mode="lines",
            line=dict(color=ss_color, width=2, dash="dot"),
            fill="toself",
            fillcolor="rgba(0,0,0,0.05)",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{selected.comp_id}</b> — backbone Ramachandran density",
            x=0.5,
            xanchor="center",
            font=dict(size=12),
        ),
        xaxis=dict(
            title="φ (deg)",
            range=[-180, 180],
            tickmode="linear",
            tick0=-180,
            dtick=90,
            zeroline=True,
            zerolinecolor="#cccccc",
            gridcolor="#eeeeee",
        ),
        yaxis=dict(
            title="ψ (deg)",
            range=[-180, 180],
            tickmode="linear",
            tick0=-180,
            dtick=90,
            zeroline=True,
            zerolinecolor="#cccccc",
            gridcolor="#eeeeee",
            scaleanchor="x",
            scaleratio=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=10, t=40, b=40),
        height=380,
    )
    return fig


# ─────────────────── Data conversion ───────────────────────────


def result_to_table(result):
    rows = []
    for r in result.residues:
        chi1_str = (
            CHI1_NAMES_SHORT.get(r.chi1, "—") if r.chi1 is not None else "—"
        )
        rows.append(
            [
                r.seq_id,
                r.comp_id,
                f"{r.phi:.1f}",
                f"{r.psi:.1f}",
                f"±{r.phi_err:.0f}",
                r.ss,
                chi1_str,
                r.confidence_class,
            ]
        )
    return rows


def get_residue_shifts(shift_list, seq_id):
    if shift_list is None or seq_id is None:
        return {}
    for res in shift_list.residues:
        if res.seq_id == seq_id:
            return res.shifts
    return {}


def build_inspector_markdown(result, selected_seq_id, shift_list):
    if selected_seq_id is None or result is None:
        return "*Click a residue in the table or Ramachandran plot to inspect.*"

    selected = next(
        (r for r in result.residues if r.seq_id == selected_seq_id), None
    )
    if selected is None:
        return "*Residue not found.*"

    from facet.ramachandran import classify_outlier, classify_region

    region = classify_region(selected.phi, selected.psi)
    outlier = classify_outlier(selected.phi, selected.psi, selected.comp_id)

    chi1_label = (
        CHI1_NAMES_FULL.get(selected.chi1, "—")
        if selected.chi1 is not None
        else "— (no χ1 dihedral)"
    )

    outlier_text = {
        "core": "in the core (50% mass) region",
        "allowed": "in the allowed (90% mass) region",
        "marginal": "in the marginal (99% mass) region",
        "outlier": "outside the 99% density (potential outlier)",
    }.get(outlier, outlier)

    return f"""### Residue **{selected.seq_id} {selected.comp_id}**

| | |
|---|---|
| **φ** | `{selected.phi:.1f}° ± {selected.phi_err:.0f}°` |
| **ψ** | `{selected.psi:.1f}° ± {selected.psi_err:.0f}°` |
| **SS** | {SS_NAMES[selected.ss]} (`{selected.ss}`) |
| **χ1** | {chi1_label} |
| **Confidence** | {selected.confidence_class} |
| **Region** | {region} |
| **Density** | {outlier_text} |
"""


def build_shifts_table(shift_list, selected_seq_id):
    shifts = get_residue_shifts(shift_list, selected_seq_id)
    if not shifts:
        return [["—"] * 6]

    def fmt(nuc):
        v = shifts.get(nuc)
        return f"{v:.2f}" if v is not None else "—"

    return [[fmt("H"), fmt("HA"), fmt("N"), fmt("CA"), fmt("CB"), fmt("C")]]


# ─────────────────── Predict callback ─────────────────────────


def predict_and_format(file_obj, bmrb_id, output_formats):
    if not file_obj and not (bmrb_id and bmrb_id.strip()):
        raise gr.Error("Upload a shift list or enter a BMRB entry ID.")

    from facet import predict
    from facet.io.readers import read_auto
    from facet.io.writers import (
        write_aco,
        write_csv,
        write_json,
        write_nef,
        write_predtab,
        write_tbl,
    )
    from facet.visualization import plot_sequence_ss

    if bmrb_id and bmrb_id.strip():
        try:
            from facet.io.bmrb import fetch_bmrb
        except ImportError:
            raise gr.Error("BMRB fetch is not available in this build.")
        try:
            shift_list = fetch_bmrb(bmrb_id.strip())
        except Exception as e:
            raise gr.Error(f"BMRB fetch failed: {e}")
        predict_input = shift_list
        stem = f"bmrb_{bmrb_id.strip()}"
        input_label = f"BMRB:{bmrb_id.strip()}"
    else:
        input_path = file_obj if isinstance(file_obj, str) else file_obj.name
        try:
            shift_list = read_auto(input_path)
        except Exception as e:
            raise gr.Error(f"Could not parse input file: {e}")
        predict_input = input_path
        stem = Path(input_path).stem
        input_label = stem

    try:
        result = predict(predict_input, checkpoint=CHECKPOINT, device="cpu")
    except Exception as e:
        raise gr.Error(f"Prediction failed: {e}")

    seq_path = os.path.join(tempfile.gettempdir(), f"facet_seq_ss_{stem}.png")
    try:
        fig_seq = plot_sequence_ss(
            result, seq_path, title=f"FACET prediction — {input_label}"
        )
        plt.close(fig_seq)
    except Exception as e:
        logger.warning("Sequence figure failed: %s", e)
        seq_path = None

    rama_fig = build_global_ramachandran(
        result, selected_seq_id=None, show_background=True
    )
    table_rows = result_to_table(result)

    tmpdir = tempfile.mkdtemp(prefix="facet_")
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

    inspector_md = build_inspector_markdown(result, None, shift_list)
    inspector_rama = build_residue_ramachandran(result, None)
    inspector_shifts = build_shifts_table(shift_list, None)

    n_high = len(result.high())
    n_acc = len(result.accepted())
    n_flex = len(result.flexible())
    status_md = (
        f"**{result.n_residues}** residues predicted  ·  "
        f"**{n_high}** High  ·  **{n_acc}** Accepted (High+Medium)  ·  "
        f"**{n_flex}** Flexible"
    )

    return (
        result,
        shift_list,
        None,
        seq_path,
        rama_fig,
        table_rows,
        inspector_md,
        inspector_rama,
        inspector_shifts,
        output_files,
        status_md,
        gr.update(open=False),
    )


# ─────────────────── Selection callbacks ──────────────────────


def _empty_selection_outputs():
    return None, gr.update(), gr.update(), gr.update(), gr.update()


def _selection_outputs(result, shift_list, seq_id, show_background):
    rama_fig = build_global_ramachandran(
        result, selected_seq_id=seq_id, show_background=show_background
    )
    inspector_md = build_inspector_markdown(result, seq_id, shift_list)
    inspector_rama = build_residue_ramachandran(result, seq_id)
    inspector_shifts = build_shifts_table(shift_list, seq_id)
    return seq_id, rama_fig, inspector_md, inspector_rama, inspector_shifts


def on_table_select(result, shift_list, show_background, evt: gr.SelectData):
    if result is None or evt is None:
        return _empty_selection_outputs()
    idx = evt.index
    if isinstance(idx, (list, tuple)):
        row_idx = idx[0]
    else:
        row_idx = idx
    if row_idx is None or row_idx < 0 or row_idx >= len(result.residues):
        return _empty_selection_outputs()
    seq_id = result.residues[row_idx].seq_id
    return _selection_outputs(result, shift_list, seq_id, show_background)


def navigate_residue(result, shift_list, current_seq_id, show_background, direction):
    if result is None or not result.residues:
        return _empty_selection_outputs()
    seq_ids = [r.seq_id for r in result.residues]
    if current_seq_id is None or current_seq_id not in seq_ids:
        new_idx = 0
    else:
        idx = seq_ids.index(current_seq_id)
        new_idx = (idx + direction) % len(seq_ids)
    return _selection_outputs(result, shift_list, seq_ids[new_idx], show_background)


def on_bg_toggle(result, selected_seq_id, show_bg):
    if result is None:
        return gr.update()
    return build_global_ramachandran(result, selected_seq_id, show_background=show_bg)


# ─────────────────── Layout ───────────────────────────────────

DESCRIPTION = """
# FACET — Backbone Torsion Angle Prediction from NMR Chemical Shifts

Upload a chemical shift list or enter a BMRB entry ID to predict
per-residue **φ/ψ torsion angles**, **secondary structure**, and
**χ1 rotamers**. Hover any point on the Ramachandran for details, or
click a row in the table to inspect a residue in depth.

FACET v3 median error: **13.4°** on 27,461 matched residues.
"""


def build_demo():
    with gr.Blocks(title="FACET") as demo:

        result_state = gr.State(value=None)
        shift_list_state = gr.State(value=None)
        selected_seq_id_state = gr.State(value=None)

        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1, min_width=280):
                with gr.Accordion("Inputs", open=True) as input_accordion:
                    file_input = gr.File(
                        label="Shift list (.tab, .csv, .nef, .str)",
                        file_types=[".tab", ".csv", ".nef", ".str", ".txt"],
                    )
                    gr.Markdown("**— or —**")
                    bmrb_input = gr.Textbox(
                        label="BMRB entry ID",
                        placeholder="e.g. 4493",
                    )
                    format_choices = gr.CheckboxGroup(
                        choices=FORMAT_CHOICES,
                        value=["XPLOR .tbl", "CYANA .aco", "pred.tab"],
                        label="Output formats",
                    )
                    predict_btn = gr.Button("Predict", variant="primary")

                gr.Markdown("---")
                gr.Markdown(
                    "**Tip:** Hover any point on the Ramachandran for a "
                    "tooltip, or click a row in the table for the full "
                    "per-residue inspector."
                )

            with gr.Column(scale=4):

                status_bar = gr.Markdown("*Run a prediction to begin.*")

                sequence_plot = gr.Image(
                    label="Sequence + Secondary Structure  (download for figures)",
                    interactive=False,
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        residue_table = gr.DataFrame(
                            headers=DATAFRAME_COLUMNS,
                            interactive=False,
                            label="Per-residue predictions  (click a row to inspect)",
                        )
                    with gr.Column(scale=4):
                        rama_bg_toggle = gr.Checkbox(
                            label="Show Ramachandran density background",
                            value=True,
                        )
                        global_rama = gr.Plot(
                            label="Ramachandran  (hover any point for details)",
                        )

                gr.Markdown("### Per-residue inspector")
                with gr.Row():
                    with gr.Column(scale=2):
                        inspector_rama = gr.Plot(label="")
                    with gr.Column(scale=3):
                        inspector_md = gr.Markdown(
                            "*Click a residue in the table or Ramachandran "
                            "plot to inspect.*"
                        )
                        gr.Markdown("**Input shifts (ppm):**")
                        inspector_shifts = gr.DataFrame(
                            headers=SHIFTS_COLUMNS,
                            interactive=False,
                        )
                        with gr.Row():
                            prev_btn = gr.Button("◀ Prev", size="sm")
                            next_btn = gr.Button("Next ▶", size="sm")

                output_files = gr.Files(label="Download restraint files")

        # ── Event wiring ──

        predict_btn.click(
            fn=predict_and_format,
            inputs=[file_input, bmrb_input, format_choices],
            outputs=[
                result_state,
                shift_list_state,
                selected_seq_id_state,
                sequence_plot,
                global_rama,
                residue_table,
                inspector_md,
                inspector_rama,
                inspector_shifts,
                output_files,
                status_bar,
                input_accordion,
            ],
        )

        residue_table.select(
            fn=on_table_select,
            inputs=[result_state, shift_list_state, rama_bg_toggle],
            outputs=[
                selected_seq_id_state,
                global_rama,
                inspector_md,
                inspector_rama,
                inspector_shifts,
            ],
        )

        rama_bg_toggle.change(
            fn=on_bg_toggle,
            inputs=[result_state, selected_seq_id_state, rama_bg_toggle],
            outputs=[global_rama],
        )

        prev_btn.click(
            fn=lambda r, s, sid, bg: navigate_residue(r, s, sid, bg, -1),
            inputs=[
                result_state,
                shift_list_state,
                selected_seq_id_state,
                rama_bg_toggle,
            ],
            outputs=[
                selected_seq_id_state,
                global_rama,
                inspector_md,
                inspector_rama,
                inspector_shifts,
            ],
        )

        next_btn.click(
            fn=lambda r, s, sid, bg: navigate_residue(r, s, sid, bg, 1),
            inputs=[
                result_state,
                shift_list_state,
                selected_seq_id_state,
                rama_bg_toggle,
            ],
            outputs=[
                selected_seq_id_state,
                global_rama,
                inspector_md,
                inspector_rama,
                inspector_shifts,
            ],
        )

        gr.Examples(
            examples=[["examples/ubiquitin.tab", "", ["XPLOR .tbl", "CYANA .aco", "pred.tab"]]],
            inputs=[file_input, bmrb_input, format_choices],
            label="Example",
        )

    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
