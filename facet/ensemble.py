"""Per-residue conformational-ensemble export (Phase 3.5.2).

The retrieval step already collects the 25 most similar training
residues per query, with their phi/psi angles. This module formats
those as a per-residue ensemble usable by downstream IDP modelling
tools (flexible-meccano seeds, ENSEMBLE, BME reweighting, etc.).

Outputs:
  - ``emit_ensemble_csv(result, path)``  — one row per (residue, neighbour)
  - ``emit_ensemble_json(result, path)`` — nested dict by residue

Stretch goal (not in this sprint): flexible-meccano-style Cartesian
build-up as a multi-model PDB.
"""
from __future__ import annotations

import json
from pathlib import Path

from .io.formats import FACETResult


def emit_ensemble_csv(result: FACETResult, path: str | Path) -> Path:
    """Write a flat CSV of retrieved-neighbour conformers per residue.

    Columns: seq_id, comp_id, neighbour_rank, entry_id, neighbour_aa,
    phi, psi, ss, basin, similarity.

    One row per (residue, neighbour) pair, sorted by residue then
    neighbour rank. Rows with no retrieval data (retrieval disabled or
    no neighbours) are omitted.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "seq_id,comp_id,neighbour_rank,entry_id,neighbour_aa,"
        "phi,psi,ss,basin,similarity"
    ]
    for r in result.residues:
        if not r.top_neighbors:
            continue
        for rank, nb in enumerate(r.top_neighbors, start=1):
            lines.append(
                f"{r.seq_id},{r.comp_id},{rank},"
                f"{nb.get('entry_id', '')},{nb.get('aa', '?')},"
                f"{nb.get('phi_deg', 0.0):.2f},{nb.get('psi_deg', 0.0):.2f},"
                f"{nb.get('ss', '?')},{nb.get('basin', '?')},"
                f"{nb.get('similarity', 0.0):.4f}"
            )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def emit_ensemble_json(result: FACETResult, path: str | Path) -> Path:
    """Write a nested JSON of retrieved-neighbour conformers per residue.

    Format:
        {
            "source": <input_source>,
            "n_residues": <N>,
            "residues": [
                {
                    "seq_id": ...,
                    "comp_id": "ALA",
                    "predicted_phi": -65.0,
                    "predicted_psi": -42.0,
                    "neighbours": [
                        {"rank": 1, "entry_id": "...", "aa": "L",
                         "phi_deg": -63.5, "psi_deg": -40.0,
                         "ss": "H", "basin": "alpha_R",
                         "similarity": 0.96},
                        ...
                    ]
                },
                ...
            ]
        }

    Suitable as direct input for BME reweighting or as a seed for
    flexible-meccano conformer generation.
    """
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    residues = []
    for r in result.residues:
        rec = {
            "seq_id": r.seq_id,
            "comp_id": r.comp_id,
            "predicted_phi": round(r.phi, 1),
            "predicted_psi": round(r.psi, 1),
        }
        if r.top_neighbors:
            rec["neighbours"] = [
                {
                    "rank": rank,
                    "entry_id": nb.get("entry_id", ""),
                    "aa": nb.get("aa", "?"),
                    "phi_deg": round(nb.get("phi_deg", 0.0), 2),
                    "psi_deg": round(nb.get("psi_deg", 0.0), 2),
                    "ss": nb.get("ss", "?"),
                    "basin": nb.get("basin", "?"),
                    "similarity": round(nb.get("similarity", 0.0), 4),
                }
                for rank, nb in enumerate(r.top_neighbors, start=1)
            ]
        residues.append(rec)

    doc = {
        "source": result.source,
        "n_residues": result.n_residues,
        "index_version": result.index_version,
        "index_n_residues": result.index_n_residues,
        "residues": residues,
    }
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out
