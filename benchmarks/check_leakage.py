#!/usr/bin/env python
"""Prove that no benchmark protein is inside the shipped reference data.

FACET predicts by looking up neighbours in two reference files:

  * ``facet_retrieval_index.entries.json`` — the BMRB entry each row of the
    219,713-residue embedding index came from
  * ``facet_shift_reference.npz`` — the BMRB entry of each row of the mask-safe
    shift reference (``entry_ids`` array)

If a test protein were in either file, its own residues could be retrieved as
neighbours and the benchmark would be measuring memorisation. This script reads
the ID lists under ``benchmarks/data/`` and checks the intersection is empty.

    python benchmarks/check_leakage.py

Exit status is non-zero on any overlap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


def read_ids(name: str) -> set[str]:
    return {line.strip() for line in (HERE / "data" / name).read_text().splitlines() if line.strip()}


def main() -> int:
    from facet.assets import resolve  # local checkout first, then ~/.facet, then download

    index_entries = set(json.load(open(resolve("facet_retrieval_index.entries.json"))))
    ref = np.load(resolve("facet_shift_reference.npz"))
    ref_entries = {str(x) for x in ref["entry_ids"]}
    print(f"retrieval index : {len(index_entries):,} distinct BMRB entries")
    print(f"shift reference : {len(ref_entries):,} distinct BMRB entries")

    failed = False
    for name in ("test_set_745.txt", "ablation_set_39.txt"):
        ids = read_ids(name)
        a = ids & index_entries
        b = ids & ref_entries
        print(f"\n{name}: {len(ids)} entries")
        print(f"  in retrieval index : {len(a)} / {len(ids)}" + (f"  -> {sorted(a)[:10]}" if a else ""))
        print(f"  in shift reference : {len(b)} / {len(ids)}" + (f"  -> {sorted(b)[:10]}" if b else ""))
        failed |= bool(a or b)

    # sanity: the train list should be what the index was built from
    train = read_ids("train_set.txt")
    print(f"\ntrain_set.txt: {len(train)} entries; "
          f"{len(index_entries - train)} index entries are NOT in the train list, "
          f"{len(train - index_entries)} train entries have no index rows")

    print("\nRESULT:", "LEAK DETECTED" if failed else "no overlap — benchmark proteins are absent from both reference files")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
