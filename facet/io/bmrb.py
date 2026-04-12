"""Fetch chemical shifts from the BMRB REST API.

Given a BMRB entry ID, downloads the assignment list and returns a
ShiftList. Uses httpx (or urllib as fallback) — BMRB responses are
small JSON blobs so no streaming is needed.
"""
from __future__ import annotations

import json
from urllib.request import urlopen
from urllib.error import URLError

from .formats import BACKBONE_NUCLEI, Residue, ShiftList

BMRB_API = "https://api.bmrb.io/v2/entry"

# Map BMRB atom labels → FACET canonical nucleus names
_BMRB_ATOM_TO_NUC = {
    "H": "H", "HN": "H",
    "HA": "HA", "HA1": "HA", "HA2": "HA", "HA3": "HA",
    "N": "N",
    "CA": "CA", "CB": "CB",
    "C": "C", "CO": "C", "C'": "C",
}


def fetch_bmrb(
    entry_id: str | int,
    timeout: float = 30.0,
) -> ShiftList:
    """Fetch chemical shifts for a BMRB entry.

    Args:
        entry_id: BMRB accession number (e.g. "4493" or 4493).
        timeout: HTTP timeout in seconds.

    Returns:
        ShiftList with backbone shifts for all assigned residues.

    Raises:
        ValueError: if the entry has no chemical shift data or the
            download fails.
    """
    entry_id = str(entry_id).strip()
    url = f"{BMRB_API}/{entry_id}?saveframe_category=assigned_chemical_shifts"

    try:
        with urlopen(url, timeout=timeout) as resp:
            data = json.load(resp)
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to fetch BMRB entry {entry_id}: {e}") from e

    # Navigate the BMRB JSON structure
    # API returns: {entry_id: {assigned_chemical_shifts: [{loops: [...]}, ...]}}
    entry = data.get(entry_id, {})
    saveframes = entry.get("assigned_chemical_shifts", [])
    if not saveframes and "saveframes" in entry:
        saveframes = entry["saveframes"]  # fallback for older API responses

    shift_rows: list[dict] = []
    for sf in saveframes:
        for loop in sf.get("loops", []):
            if loop.get("category") != "_Atom_chem_shift":
                continue
            tags = loop.get("tags", [])
            tag_index = {tag: i for i, tag in enumerate(tags)}
            for row in loop.get("data", []):
                row_dict = {tag: row[i] if i < len(row) else None
                            for tag, i in tag_index.items()}
                shift_rows.append(row_dict)

    if not shift_rows:
        raise ValueError(
            f"BMRB entry {entry_id} has no assigned chemical shift data"
        )

    # Build residues keyed by (entity_id, seq_id)
    by_key: dict[tuple[str, int], Residue] = {}
    for row in shift_rows:
        try:
            seq_id = int(row.get("Seq_ID") or row.get("Comp_index_ID") or 0)
        except (ValueError, TypeError):
            continue
        if seq_id == 0:
            continue

        comp_id = (row.get("Comp_ID") or "UNK").upper()
        atom_id = row.get("Atom_ID") or ""
        val_str = row.get("Val") or row.get("Value")
        entity = row.get("Entity_assembly_ID") or row.get("Entity_ID") or "1"

        try:
            value = float(val_str) if val_str is not None else None
        except (ValueError, TypeError):
            continue
        if value is None:
            continue

        nuc = _BMRB_ATOM_TO_NUC.get(atom_id)
        if nuc is None or nuc not in BACKBONE_NUCLEI:
            continue

        key = (str(entity), seq_id)
        if key not in by_key:
            by_key[key] = Residue(seq_id=seq_id, comp_id=comp_id)
        by_key[key].shifts[nuc] = value

    if not by_key:
        raise ValueError(
            f"BMRB entry {entry_id}: no backbone nuclei found in shift list"
        )

    residues = [by_key[k] for k in sorted(by_key)]
    return ShiftList(residues=residues, source=f"BMRB:{entry_id}")
