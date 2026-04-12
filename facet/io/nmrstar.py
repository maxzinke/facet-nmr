"""NMR-STAR v3 chemical shift list reader.

NMR-STAR is BMRB's native format — STAR syntax with the
``_Atom_chem_shift`` loop inside an ``assigned_chemical_shifts``
saveframe. This parser is minimal: it finds the loop, reads the tag
order, and extracts one Residue per (Entity_assembly_ID, Seq_ID).

For full STAR parsing (including nested quotes, semicolon-delimited
multi-line values, and the full BMRB dictionary), use the pynmrstar
library instead. This reader handles the common case: well-formed
BMRB depositions with simple whitespace-separated values.
"""
from __future__ import annotations

from pathlib import Path

from .formats import BACKBONE_NUCLEI, Residue, ShiftList

# NMR-STAR atom names → FACET canonical nuclei
_STAR_ATOM_TO_NUC: dict[str, str] = {
    "H": "H", "HN": "H",
    "HA": "HA", "HA1": "HA", "HA2": "HA", "HA3": "HA",
    "N": "N",
    "CA": "CA",
    "CB": "CB",
    "C": "C", "CO": "C", "C'": "C",
}


def _split_star_row(line: str) -> list[str]:
    """Split a STAR data row, handling quoted strings."""
    out: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch.isspace():
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < n and line[j] != quote:
                j += 1
            out.append(line[i + 1 : j])
            i = j + 1
        else:
            j = i
            while j < n and not line[j].isspace():
                j += 1
            out.append(line[i:j])
            i = j
    return out


def _parse_semicolon_block(lines: list[str], start: int) -> tuple[str, int]:
    """Parse a ``;`` multi-line block starting at ``lines[start]``.

    Returns (content, next_index). Called when the first non-space
    character on a line is ``;``.
    """
    content_lines: list[str] = []
    # First line starts with ';'
    first = lines[start].lstrip()
    assert first.startswith(";")
    content_lines.append(first[1:])
    i = start + 1
    while i < len(lines):
        if lines[i].lstrip().startswith(";"):
            return ("\n".join(content_lines).strip(), i + 1)
        content_lines.append(lines[i])
        i += 1
    return ("\n".join(content_lines).strip(), i)


def read_nmrstar(path: str | Path) -> ShiftList:
    """Read an NMR-STAR v3 chemical shift list.

    Scans for the first ``_Atom_chem_shift`` loop and extracts backbone
    shifts keyed by (chain, seq_id). Handles both BMRB depositions
    (category = ``assigned_chemical_shifts``) and standalone shift lists.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_lines = text.splitlines()

    # Strip comments and blank lines but preserve indices-relative semantics
    lines: list[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(line)

    # Find the _Atom_chem_shift loop
    tags: list[str] = []
    in_loop = False
    in_shift_loop = False
    rows: list[dict[str, str]] = []

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped == "loop_":
            in_loop = True
            in_shift_loop = False
            tags = []
            i += 1
            continue

        if in_loop and stripped.startswith("_Atom_chem_shift."):
            tag = stripped.split(".", 1)[1]
            tags.append(tag)
            in_shift_loop = True
            i += 1
            continue

        if in_loop and not in_shift_loop and stripped.startswith("_"):
            # Different loop — reset
            in_loop = False
            tags = []
            i += 1
            continue

        if in_loop and in_shift_loop:
            if stripped == "stop_":
                in_loop = False
                in_shift_loop = False
                # Keep collecting if there are multiple shift loops (unusual)
                i += 1
                continue
            if stripped.startswith("_"):
                # End of data rows for this loop
                i += 1
                continue

            # Data row — may span multiple lines if the line has fewer
            # tokens than expected (NMR-STAR allows multi-line data)
            collected: list[str] = _split_star_row(stripped)
            j = i + 1
            while len(collected) < len(tags) and j < len(lines):
                next_stripped = lines[j].strip()
                if next_stripped == "stop_" or next_stripped.startswith("_") or next_stripped == "loop_":
                    break
                if next_stripped.startswith(";"):
                    block, j_after = _parse_semicolon_block(lines, j)
                    collected.append(block)
                    j = j_after
                else:
                    collected.extend(_split_star_row(next_stripped))
                    j += 1

            if len(collected) >= len(tags):
                row = dict(zip(tags, collected[: len(tags)]))
                rows.append(row)
                i = j
            else:
                i += 1
            continue

        i += 1

    if not rows:
        return ShiftList(residues=[], source=str(path))

    # Group by entity first (multi-chain depositions are common for complexes).
    # Each entity gets its own dict of residues; we pick the largest afterward.
    by_entity: dict[str, dict[int, Residue]] = {}

    for row in rows:
        seq_id_raw = row.get("Seq_ID") or row.get("Comp_index_ID") or ""
        # Handle insertion codes ("10A") by stripping non-digits
        seq_digits = "".join(c for c in str(seq_id_raw) if c.isdigit() or c == "-")
        if not seq_digits:
            continue
        try:
            seq_id = int(seq_digits)
        except (ValueError, TypeError):
            continue

        comp_id = (row.get("Comp_ID") or "UNK").upper()
        atom_id = row.get("Atom_ID") or ""
        val_str = row.get("Val") or row.get("Value") or ""
        entity = row.get("Entity_assembly_ID") or row.get("Entity_ID") or "1"

        try:
            value = float(val_str)
        except (ValueError, TypeError):
            continue

        nuc = _STAR_ATOM_TO_NUC.get(atom_id)
        if nuc is None or nuc not in BACKBONE_NUCLEI:
            continue

        entity_str = str(entity)
        if entity_str not in by_entity:
            by_entity[entity_str] = {}
        if seq_id not in by_entity[entity_str]:
            by_entity[entity_str][seq_id] = Residue(seq_id=seq_id, comp_id=comp_id)
        by_entity[entity_str][seq_id].shifts[nuc] = value

    if not by_entity:
        return ShiftList(residues=[], source=str(path))

    # Pick the largest entity (most residues). Multi-chain BMRB files typically
    # list each chain under a separate Entity_assembly_ID. Taking the largest
    # handles both homodimers (identical chains) and heterocomplexes (pick the
    # main chain; others will be silently dropped).
    best_entity = max(by_entity.keys(), key=lambda e: len(by_entity[e]))
    by_key = by_entity[best_entity]

    if len(by_entity) > 1:
        import logging
        sizes = {e: len(r) for e, r in by_entity.items()}
        logging.getLogger("facet").warning(
            "NMR-STAR file has %d entities %s — using entity %s (%d residues)",
            len(by_entity), sizes, best_entity, len(by_key),
        )

    residues = [by_key[k] for k in sorted(by_key)]
    return ShiftList(residues=residues, source=str(path))
