# Data provenance

What FACET ships or downloads, where each file came from, and what may be done with it.
For how the data were curated and split see [docs/DATA.md](docs/DATA.md); for the
benchmark sets see [benchmarks/](benchmarks/).

## Files

| File | Where | What it is | Source | Licence |
|---|---|---|---|---|
| `facet_v3.pt`, `facet_v3.onnx` | HF `SiXa18/facet-weights` | Trained encoder + heads (1.29 M parameters) | Trained by this project on BMRB + PDB | CC BY 4.0 |
| `facet_retrieval_index.npz` | HF | 253,573 × 128-d learned embeddings with φ/ψ, secondary-structure state, basin and residue type, from 3,458 training-split depositions | Embeddings from this project's encoder; φ/ψ from the PDB | CC BY 4.0 |
| `facet_retrieval_index.entries.json` | HF | BMRB deposition ID for every index row | BMRB | CC0 (identifiers) |
| `facet_shift_reference.npz` | HF (optional) | 310,923 per-residue secondary-shift rows with deposition and residue IDs, from 3,470 depositions; backs the mask-safe fallback | BMRB | CC0 (data) — packaged file CC BY 4.0 |
| `facet/data/ss_popn_params.npz` | in the wheel | Per-residue-type Gaussian parameters of the d2D-style engine, fit on 49,262 curated residues | Fitted by this project | CC BY 4.0 |

Sizes and SHA-256 of the downloadable files are pinned in `facet/assets.py`; a
downloaded file that does not match is rejected. The HuggingFace repository is
addressed by tag (`v0.4.0`, commit `a359356`), so a later upload cannot change what an
installed copy resolves.

## Citing the weights

The model repository is archived with its own DOI:
[10.57967/hf/10248](https://doi.org/10.57967/hf/10248).

```bibtex
@misc{zinke_facet_weights_2026,
  author    = {Zinke, Maximilian},
  title     = {facet-weights},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/10248},
  url       = {https://huggingface.co/SiXa18/facet-weights}
}
```

That DOI is registered against repository revision `d35fe56` (branch `main`), which
carries byte-identical weight files to the pinned tag `v0.4.0` — the two revisions
differ only in the model card. Cite the software itself with the Zenodo concept DOI in
[CITATION.cff](CITATION.cff).

## Upstream sources

**Protein Data Bank.** Structural data (φ/ψ torsions, DSSP secondary structure) derive
from PDB entries paired to BMRB depositions. The PDB is released under **CC0 1.0**.

**Biological Magnetic Resonance Data Bank.** All chemical-shift information derives
from BMRB depositions. BMRB is a wwPDB member and its data are released under
**CC0 1.0** — a public-domain dedication with no attribution requirement and no
restriction on redistribution or commercial use. Confirmed directly with BMRB on
2026-08-06; BMRB's only comment was that fetching on first use is preferable to
bundling, because depositors submit corrections and a bundled copy silently goes
stale. That is why the large files are downloaded rather than shipped in the wheel.

`facet_shift_reference.npz` is a *redistribution* of BMRB data, not a derived
statistic: secondary shifts are observed minus a random-coil constant, and the
constant ships with the package (`facet/masked_retrieval.py::RANDOM_COIL_SHIFTS`), so
raw ppm are recoverable. Under CC0 that is permitted.

Citing BMRB is appreciated:

> Hoch, J. C. *et al.* "Biological Magnetic Resonance Data Bank."
> *Nucleic Acids Research* **51**(D1), D368–D376 (2023).
> <https://doi.org/10.1093/nar/gkac1050>

## Test-set separation

No deposition in the 745-entry benchmark test set (`benchmarks/data/test_set_745.txt`)
appears in the retrieval index or in the shift reference. `benchmarks/check_leakage.py`
verifies this against the shipped files.

## History

Earlier revisions of this file recorded the BMRB redistribution status as unresolved
and described a defect in the mask-safe retrieval scorer and a units bug in an
abandoned "centroid" reference builder. The licensing question is settled above; the
scorer fix and its effect on the reported numbers are recorded in
[CHANGELOG.md](CHANGELOG.md) (0.4.0) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md).
