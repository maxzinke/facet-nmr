"""Input format readers for FACET.

Supported formats:
  - NMRPipe .tab shift list
  - Simple CSV (ResID, AA, H, HA, N, CA, CB, C)
  - Auto-detection from file content
"""
from __future__ import annotations

from pathlib import Path

from .formats import AA_ONE_TO_THREE, BACKBONE_NUCLEI, Residue, ShiftList
from .nef import read_nef
from .nmrstar import read_nmrstar


def read_tab(path: str | Path) -> ShiftList:
    """Read a .tab shift list (NMRPipe convention).

    Expected format::

        VARS   RESID RESNAME ATOMNAME SHIFT
        FORMAT %4d %1s %4s %8.3f

        1 M H 8.421
        1 M HA 4.312
        1 M N 120.5
        ...

    Also handles the condensed input format::

        VARS   RESID RESNAME C CA CB HA H N
        FORMAT %4d %s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f

        1 M 176.2 55.4 32.9 4.31 8.42 120.5
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    # Parse VARS header to determine format
    vars_line = ""
    data_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("REMARK") or stripped.startswith("DATA"):
            continue
        if stripped.startswith("VARS"):
            vars_line = stripped
            continue
        if stripped.startswith("FORMAT"):
            continue
        data_lines.append(stripped)

    if not data_lines:
        return ShiftList(residues=[], source=str(path))

    # Detect format: condensed (RESID RESNAME C CA CB HA H N) or
    # long-form (RESID RESNAME ATOMNAME SHIFT)
    vars_fields = vars_line.split()[1:] if vars_line else []

    residues_dict: dict[int, Residue] = {}

    if "ATOMNAME" in vars_fields or "SHIFT" in vars_fields:
        # Long-form: one row per (residue, atom)
        for line in data_lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                seq_id = int(parts[0])
                aa_one = parts[1]
                atom_name = parts[2]
                shift = float(parts[3])
            except (ValueError, IndexError):
                continue
            if seq_id not in residues_dict:
                aa3 = AA_ONE_TO_THREE.get(aa_one, aa_one)
                if len(aa3) == 1:
                    aa3 = AA_ONE_TO_THREE.get(aa3, aa3)
                residues_dict[seq_id] = Residue(seq_id=seq_id, comp_id=aa3)
            # Map atom name to canonical nucleus
            nuc = _normalize_atom_name(atom_name)
            if nuc and nuc in BACKBONE_NUCLEI:
                residues_dict[seq_id].shifts[nuc] = shift
    else:
        # Condensed: one row per residue, columns are nuclei
        # Determine column mapping from VARS
        if vars_fields:
            col_map = {name: i for i, name in enumerate(vars_fields)}
        else:
            # Guess: RESID RESNAME C CA CB HA H N (canonical column order)
            col_map = {"RESID": 0, "RESNAME": 1, "C": 2, "CA": 3,
                        "CB": 4, "HA": 5, "H": 6, "N": 7}

        for line in data_lines:
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                seq_id = int(parts[col_map.get("RESID", 0)])
                aa_one = parts[col_map.get("RESNAME", 1)]
            except (ValueError, IndexError, KeyError):
                continue
            aa3 = AA_ONE_TO_THREE.get(aa_one, aa_one)
            if len(aa3) == 1:
                aa3 = AA_ONE_TO_THREE.get(aa3, aa3)
            res = Residue(seq_id=seq_id, comp_id=aa3)
            for nuc in BACKBONE_NUCLEI:
                idx = col_map.get(nuc)
                if idx is not None and idx < len(parts):
                    try:
                        val = float(parts[idx])
                        if val != 9999.0:  # 9999.0 sentinel for missing
                            res.shifts[nuc] = val
                    except ValueError:
                        pass
            residues_dict[seq_id] = res

    residues = [residues_dict[k] for k in sorted(residues_dict)]
    return ShiftList(residues=residues, source=str(path))


def read_csv(path: str | Path) -> ShiftList:
    """Read a simple CSV shift list.

    Expected header: ResID,AA,H,HA,N,CA,CB,C
    Missing values: empty or 'nan' or '.'
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines:
        return ShiftList(residues=[], source=str(path))

    # Parse header
    header = [h.strip().upper() for h in lines[0].split(",")]

    residues: list[Residue] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            seq_id = int(parts[header.index("RESID")] if "RESID" in header else parts[0])
        except (ValueError, IndexError):
            continue
        aa_col = "AA" if "AA" in header else "RESNAME" if "RESNAME" in header else None
        if aa_col:
            aa = parts[header.index(aa_col)]
        else:
            aa = parts[1]
        aa3 = AA_ONE_TO_THREE.get(aa, aa)

        res = Residue(seq_id=seq_id, comp_id=aa3)
        for nuc in BACKBONE_NUCLEI:
            if nuc in header:
                idx = header.index(nuc)
                if idx < len(parts):
                    val = parts[idx]
                    if val and val not in (".", "nan", "NaN", ""):
                        try:
                            res.shifts[nuc] = float(val)
                        except ValueError:
                            pass
        residues.append(res)

    return ShiftList(residues=residues, source=str(path))


def read_auto(path: str | Path) -> ShiftList:
    """Auto-detect format and read a shift list.

    Detection heuristic:
      - .csv → CSV
      - .tab → NMRPipe tab
      - .nef → NEF
      - .str → NMR-STAR (if pynmrstar available)
      - Otherwise: sniff content for format markers
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".tab":
        return read_tab(path)
    if suffix == ".nef":
        return read_nef(path)
    if suffix in (".str", ".bmrb"):
        return read_nmrstar(path)

    # Sniff content — read full file for STAR format detection since the
    # _Atom_chem_shift loop may be deep in the file
    text = path.read_text(errors="replace")
    head = text[:2000]

    # STAR markers (NEF and NMR-STAR share the data_/save_/loop_ syntax)
    if "nef_chemical_shift" in text:
        return read_nef(path)
    if "_Atom_chem_shift" in text:
        return read_nmrstar(path)
    # Generic STAR format with data_ header → probably NMR-STAR
    if head.lstrip().startswith("data_"):
        return read_nmrstar(path)

    if "VARS" in head or "FORMAT" in head:
        return read_tab(path)
    if "," in head.split("\n")[0]:
        return read_csv(path)

    # Default to tab
    return read_tab(path)


def _normalize_atom_name(name: str) -> str | None:
    """Map common atom name variants to FACET's canonical names."""
    name = name.strip().upper()
    mapping = {
        "H": "H", "HN": "H", "NH": "H",
        "HA": "HA", "HA2": "HA",  # GLY HA2
        "N": "N", "N15": "N",
        "CA": "CA", "C13A": "CA",
        "CB": "CB", "C13B": "CB",
        "C": "C", "CO": "C", "C'": "C", "C13": "C",
    }
    return mapping.get(name)
