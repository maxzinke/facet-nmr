"""NEF (NMR Exchange Format) chemical shift list reader.

Parses NEF files (STAR-based) and extracts backbone chemical shifts into
a ShiftList. Handles the canonical NEF atom names plus common variants
(H vs HN, HA2/HA3/HAx/HAy/HA% for glycine, etc.) per the NEF v1.1 spec.

This is intentionally a minimal STAR parser — good enough for NEF shift
lists but not a full NMR-STAR parser. For NMR-STAR files, use the
pynmrstar-based reader (optional dependency).
"""
from __future__ import annotations

from pathlib import Path

from .formats import AA_THREE_TO_ONE, BACKBONE_NUCLEI, Residue, ShiftList

# Map NEF atom names → FACET canonical nucleus names
# Per docs/NEF_ATOM_NAMING_CONVENTIONS.md
_NEF_ATOM_TO_NUC: dict[str, str] = {
    # Amide proton
    "H": "H", "HN": "H",
    # Alpha proton (Glycine HA2/HA3/HAx/HAy collapse to HA via average upstream;
    # for shift input we accept any of them as HA)
    "HA": "HA", "HA2": "HA", "HA3": "HA",
    "HAx": "HA", "HAy": "HA", "HA%": "HA",
    # Backbone nitrogen
    "N": "N",
    # Alpha/beta/carbonyl carbons
    "CA": "CA", "CB": "CB", "C": "C", "CO": "C", "C'": "C",
}


def _split_star_row(line: str) -> list[str]:
    """Split a STAR data row into tokens, handling simple quoted strings."""
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


def _parse_nef_loop(
    lines: list[str], loop_prefix: str
) -> list[dict[str, str]]:
    """Generic minimal reader for a NEF loop matching a given tag prefix.

    Walks the (already stripped) `lines` list, extracts the first loop
    whose tags start with `loop_prefix` (e.g. ``_nef_chemical_shift.`` or
    ``_nef_sequence.``), and returns one dict per data row.
    """
    tags: list[str] = []
    in_loop = False
    in_target_loop = False
    rows: list[dict[str, str]] = []

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        if line == "loop_":
            in_loop = True
            in_target_loop = False
            tags = []
            i += 1
            continue

        if in_loop and line.startswith(loop_prefix):
            tags.append(line.split(".", 1)[1])
            in_target_loop = True
            i += 1
            continue

        if in_loop and not in_target_loop and line.startswith("_"):
            in_loop = False
            tags = []
            i += 1
            continue

        if in_loop and in_target_loop:
            if line == "stop_":
                # First matching loop is enough
                return rows
            if line.startswith("_"):
                i += 1
                continue
            tokens = _split_star_row(line)
            if len(tokens) == len(tags):
                rows.append(dict(zip(tags, tokens)))
            i += 1
            continue

        i += 1

    return rows


def _build_sequence_from_nef_sequence(
    seq_rows: list[dict[str, str]],
) -> tuple[str, int]:
    """Build a dense one-letter sequence from nef_sequence rows.

    Picks the chain with the most residues; unknown positions inside the
    [min, max] range fill with "X". Returns (sequence, seq_id_start).
    """
    if not seq_rows:
        return ("", 1)

    # Group by chain
    by_chain: dict[str, dict[int, str]] = {}
    for row in seq_rows:
        chain = row.get("chain_code", "A") or "A"
        seq_code = row.get("sequence_code", "")
        try:
            seq_id = int(seq_code)
        except (ValueError, TypeError):
            continue
        residue_name = (row.get("residue_name") or "").upper()
        if not residue_name:
            continue
        by_chain.setdefault(chain, {})[seq_id] = residue_name

    if not by_chain:
        return ("", 1)

    # Pick the largest chain (matches the reader's shift-loop convention)
    best_chain = max(by_chain.keys(), key=lambda c: len(by_chain[c]))
    residues_by_id = by_chain[best_chain]

    min_sid = min(residues_by_id)
    max_sid = max(residues_by_id)
    chars = ["X"] * (max_sid - min_sid + 1)
    for sid, three in residues_by_id.items():
        chars[sid - min_sid] = AA_THREE_TO_ONE.get(three, "X")
    return ("".join(chars), min_sid)


def read_nef(path: str | Path) -> ShiftList:
    """Read a NEF chemical shift list into a ShiftList.

    Parses the first ``nef_chemical_shift_list`` saveframe and its
    ``_nef_chemical_shift`` loop. Returns one Residue per unique
    (chain, seq_id) combination.

    Only backbone nuclei (H, HA, N, CA, CB, C) are kept — sidechain
    shifts in the file are silently ignored.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Strip comments
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)

    # Extract chemical shift rows and (if present) the full sequence
    rows = _parse_nef_loop(lines, "_nef_chemical_shift.")
    seq_rows = _parse_nef_loop(lines, "_nef_sequence.")
    sequence, seq_id_start = _build_sequence_from_nef_sequence(seq_rows)

    if not rows:
        return ShiftList(
            residues=[],
            sequence=sequence,
            seq_id_start=seq_id_start,
            source=str(path),
        )

    # Group by (chain_code, sequence_code)
    by_residue: dict[tuple[str, int], Residue] = {}

    for row in rows:
        seq_code_str = row.get("sequence_code", "")
        try:
            seq_id = int(seq_code_str)
        except (ValueError, TypeError):
            continue

        chain = row.get("chain_code", "A")
        residue_name = row.get("residue_name", "UNK")
        atom_name = row.get("atom_name", "")
        value_str = row.get("value", "")

        try:
            value = float(value_str)
        except (ValueError, TypeError):
            continue

        nuc = _NEF_ATOM_TO_NUC.get(atom_name)
        if nuc is None or nuc not in BACKBONE_NUCLEI:
            continue

        key = (chain, seq_id)
        if key not in by_residue:
            by_residue[key] = Residue(seq_id=seq_id, comp_id=residue_name.upper())
        by_residue[key].shifts[nuc] = value

    # Sort by chain then seq_id
    residues = [by_residue[k] for k in sorted(by_residue)]
    return ShiftList(
        residues=residues,
        sequence=sequence,
        seq_id_start=seq_id_start,
        source=str(path),
    )
