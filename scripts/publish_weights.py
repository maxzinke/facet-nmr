"""Publish the model weights to the HuggingFace Hub repository FACET downloads from.

The wheel does not contain these files: ``facet_retrieval_index.npz`` alone is 133 MB
and PyPI rejects anything over 100 MB, so ``facet/assets.py`` fetches them on first
use instead. This script is what puts them where it looks.

WHAT IT CHECKS BEFORE UPLOADING
-------------------------------
Every file is hashed and compared against the manifest in ``facet.assets.ASSETS``. A
mismatch aborts the upload rather than publishing it, because the manifest is what
installed copies verify against: shipping a file whose hash differs would make every
download fail its integrity check and read as a corrupt release. If you have
deliberately rebuilt an artifact, update the manifest first and let the mismatch be
the reminder.

The upload is tagged, not pushed to a moving branch. ``assets.py`` pins a revision so
a later upload cannot silently change what an already-installed copy resolves.

USAGE
-----
    hf auth login                       # needs a WRITE token
    python scripts/publish_weights.py --dry-run
    python scripts/publish_weights.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from facet.assets import ASSETS, HF_REPO, HF_REVISION  # noqa: E402

WEIGHTS_DIR = REPO_ROOT / "facet" / "weights"

README = f"""---
license: cc-by-4.0
tags:
  - protein-nmr
  - chemical-shifts
  - torsion-angles
---

# FACET model weights

Model parameters and reference data for [FACET](https://github.com/maxzinke/facet-nmr),
which predicts protein backbone phi/psi torsion angles from NMR chemical shifts.

These files are **downloaded automatically on first use** — you do not need to fetch
them by hand. `facet/assets.py` resolves them into `~/.facet/` and verifies each
against a pinned SHA-256.

They live here rather than in the Python wheel because `facet_retrieval_index.npz` is
133 MB, past PyPI's 100 MB per-file limit; a bundled wheel could not be uploaded at
all. Hosting them separately also means a corrected BMRB entry can reach users without
a new package release.

## Contents

| File | Size | What it is |
|---|---|---|
| `facet_v3.pt` | 5.2 MB | Encoder weights (PyTorch) |
| `facet_v3.onnx` | 5.2 MB | The same encoder, ONNX |
| `facet_retrieval_index.npz` | 132.6 MB | 254K residue embeddings with phi/psi and labels |
| `facet_retrieval_index.entries.json` | 2.0 MB | Per-row source identifiers |
| `facet_shift_reference.npz` | 9.6 MB | Mask-safe retrieval reference (optional; absence degrades to the parametric head) |

## Provenance and licence

These files are licensed **CC BY 4.0**: use them for anything, including commercially,
provided you credit the project (see `CITATION.cff` in the source repository).

Structural data (phi/psi, secondary structure) come from the **Protein Data Bank** and
chemical-shift data from the **BMRB** — both released under **CC0 1.0**, a
public-domain dedication with no conditions. Trained parameters are the work of this
project. See `DATA_PROVENANCE.md` in the source repository.

No deposition from the 745-entry benchmark test set is present in the retrieval index
or the shift reference (`benchmarks/check_leakage.py` in the source repository).

## Citation

Zinke, M. ([ORCID 0000-0002-0541-5139](https://orcid.org/0000-0002-0541-5139)).
*FACET: backbone torsion angle prediction from NMR chemical shifts* (2026).
Software: <https://github.com/maxzinke/facet-nmr> — DOI [10.5281/zenodo.22190034](https://doi.org/10.5281/zenodo.22190034).

Citing BMRB is appreciated:
Hoch *et al.*, *Nucleic Acids Research* **51**, D368 (2023), doi:10.1093/nar/gkac1050.
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--revision", default=HF_REVISION, help="tag to create for this upload")
    ap.add_argument("--dry-run", action="store_true", help="verify only, upload nothing")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    args = ap.parse_args()

    print(f"repo     {args.repo}")
    print(f"revision {args.revision}")
    print(f"source   {WEIGHTS_DIR}\n")

    problems, total = [], 0
    for name, asset in ASSETS.items():
        path = WEIGHTS_DIR / name
        if not path.exists():
            problems.append(f"{name}: missing from {WEIGHTS_DIR}")
            continue
        size = path.stat().st_size
        digest = sha256(path)
        ok = size == asset.size and digest == asset.sha256
        total += size
        print(f"  {'OK  ' if ok else 'FAIL'} {name:38} {size/1e6:7.1f} MB")
        if not ok:
            problems.append(
                f"{name}: manifest says {asset.size} bytes / {asset.sha256[:16]}..., "
                f"file is {size} bytes / {digest[:16]}..."
            )

    if problems:
        print("\nRefusing to upload — the files do not match facet.assets.ASSETS:\n")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nInstalled copies verify downloads against that manifest, so publishing a "
            "mismatched file would make every download fail its integrity check.\n"
            "If an artifact was rebuilt on purpose, update ASSETS first."
        )
        return 1

    print(f"\nall {len(ASSETS)} files match the manifest ({total/1e6:.1f} MB total)")
    if args.dry_run:
        print("dry run — nothing uploaded")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        api.whoami()
    except Exception:
        print("\nNot authenticated. Run `hf auth login` with a WRITE token first.")
        return 1

    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=args.private)
    readme = WEIGHTS_DIR.parent.parent / "_hf_weights_README.md"
    readme.write_text(README, encoding="utf-8")
    try:
        api.upload_file(path_or_fileobj=str(readme), path_in_repo="README.md",
                        repo_id=args.repo, repo_type="model")
        for name in ASSETS:
            print(f"uploading {name} ...", flush=True)
            api.upload_file(path_or_fileobj=str(WEIGHTS_DIR / name), path_in_repo=name,
                            repo_id=args.repo, repo_type="model")
    finally:
        readme.unlink(missing_ok=True)

    # Tag it. assets.py pins this revision, so a later upload to main cannot change
    # what an already-installed copy resolves.
    try:
        api.create_tag(args.repo, tag=args.revision, repo_type="model")
        print(f"\ntagged {args.revision}")
    except Exception as exc:
        print(f"\nuploaded, but tagging {args.revision} failed: {exc}")
        print("Create the tag by hand — assets.py resolves that revision, not main.")
        return 1

    print(f"\ndone: https://huggingface.co/{args.repo}/tree/{args.revision}")
    print("verify a fresh install with:  FACET_HOME=/tmp/facet-check python -m facet.assets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
