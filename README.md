# FACET — backbone torsion angles from NMR chemical shifts

[![CI](https://github.com/maxzinke/facet-nmr/actions/workflows/ci.yml/badge.svg)](https://github.com/maxzinke/facet-nmr/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/facet-nmr.svg)](https://pypi.org/project/facet-nmr/)
[![Try it](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/SiXa18/facet)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

**FACET** (Fold And Conformation Estimation Tool) predicts per-residue backbone
**φ/ψ torsion angles** with a confidence tier, **secondary structure**, **χ1
rotamers** and **Ramachandran basin populations** from backbone chemical shifts
(H, HA, N, CA, CB, C), and writes dihedral restraints for XPLOR-NIH / CNS / HADDOCK /
ARIA, CYANA and NEF.

A 1.29 M-parameter local-biased transformer embeds each residue from a pentapeptide
window of secondary shifts, masking whatever is unassigned; φ/ψ are inferred by
kNN + DBSCAN retrieval over a 220 K-residue reference index.

## Install and run

```bash
pip install facet-nmr
facet predict shifts.tab            # writes shifts_facet.tbl, .aco, .predtab
facet predict shifts.tab --all      # + .nef, .csv, .json
facet predict --bmrb 4493           # straight from a BMRB entry
facet predict shifts.tab --include-medium   # broaden restraints to the Medium tier
```

Inputs: NMRPipe/TALOS `.tab`, CSV (`ResID, AA, H, HA, N, CA, CB, C`), NEF, NMR-STAR —
auto-detected. Weights and the retrieval index (~125 MB) are downloaded on first use
into `~/.facet` (override with `$FACET_HOME`; `python -m facet.assets` pre-fetches;
`FACET_NO_DOWNLOAD=1` never fetches). Every file is verified against a pinned SHA-256.

```python
from facet import predict

result = predict("shifts.tab")
for r in result.residues:
    print(r.seq_id, r.comp_id, f"{r.phi:.0f} {r.psi:.0f}", r.ss, r.confidence_class)
result.to_tbl("restraints.tbl")     # High tier only, by default
result.to_nef("restraints.nef")
print(result.summary())
```

Or use the web app: <https://huggingface.co/spaces/SiXa18/facet>.

## Accuracy

Leak-safe benchmark of 745 proteins held out by sequence-similarity clustering,
scored against the deposited structures. Figures are over the **53,876 residues both
methods predict** (724 proteins; FACET emits a prediction for 99.3 % of the 55,036
scored residues, TALOS-N for 98.6 %). Differences are FACET − TALOS-N with a 95 %
protein-level paired-bootstrap interval.

| | TALOS-N | **FACET** | difference |
|---|---|---|---|
| Median φ/ψ error | 11.51° | **11.03°** | −0.48° [−0.64, −0.37] |
| Residues with error > 25° | 20.1 % | **18.4 %** | −1.7 pt [−2.1, −1.4] |
| Mean | 21.2° | **20.5°** | |
| p90 | 45.3° | **41.8°** | |
| Helix median (n = 24,099) | 7.4° | **7.1°** | −0.30° [−0.39, −0.20] |
| Strand median (n = 11,831) | 13.5° | **12.9°** | −0.68° [−0.91, −0.45] |
| Coil median (n = 17,946) | 19.2° | **18.3°** | −0.92° [−1.26, −0.59] |
| Head-to-head win rate | 47.0 % | **53.0 %** | [52.4, 53.7] |

Error is `sqrt((Δφ² + Δψ²) / 2)`, each difference wrapped to [0°, 180°], against the
circular-mean angles over the models of the deposited structure. The table is the
benchmark of record with a validated correction to the ground truth of 63 X-ray
entries (see the note below); every number regenerates from
`benchmarks/results/talosn_clean/per_residue_corrected.csv` with
`python benchmarks/rescore.py --csv … --bootstrap`.

**Confidence tiers** (retrieval mode, same benchmark):

| Tier | Share | Median error | > 25° | Written to restraints |
|---|---|---|---|---|
| **High** | 76.3 % | 9.3° | 10.1 % | yes (±20°) |
| **Medium** | 18.1 % | 19.3° | 39.0 % | with `--include-medium` (±35°) |
| **Low** | 3.5 % | 52.7° | 66.5 % | no — multi-modal |
| **Flexible** | 2.1 % | — | — | no — no coherent cluster |

*Flexible* is not a failure state: it flags residues whose retrieved neighbours do
not agree, which is what conformational averaging looks like in shift space. The
basin populations still describe those residues.

**Missing shifts.** 49 of the benchmark proteins with NMR-ensemble truth have no HA
shifts at all (perdeuterated-style assignments); on their 4,133 residues FACET scores
9.6° median against TALOS-N's 10.1°, so incomplete assignments are handled by the
default path. An opt-in shift-space fallback (`--mask-safe-fallback`) exists for
inputs where the encoder path fails; it was the default in 0.3.1 and cost accuracy on
real data — see [docs/BENCHMARKS.md](docs/BENCHMARKS.md) §6.

> **What you should know before quoting these numbers.** While preparing this release
> we found that every single-model (X-ray) structure in the data pipeline had its φ/ψ
> converted to radians twice. Consequences, all documented in
> [docs/BENCHMARKS.md](docs/BENCHMARKS.md) §7 and [docs/LIMITATIONS.md](docs/LIMITATIONS.md):
> the benchmark truth for 63 entries (6,978 residues) was wrong and has been corrected
> and validated against the PDB files (median deviation 0.02°) — the table above uses
> the corrected truth, whereas the uncorrected record read 12.65° vs 13.57°; and
> **11.7 % of the training targets, 1.1 % of the shipped retrieval index and 11.7 % of
> the shift reference carry the same defect.** The model still reaches the numbers
> above, FACET is slightly *behind* TALOS-N on the corrected X-ray entries, and a
> retrained release on the fixed pipeline is the next step. Running the released
> package on the same 745 inputs reproduces the recorded per-residue errors on 96.5 %
> of residues (11.18° vs 11.58° on the corrected truth when scored on its own).

## Where the details are

| Question | Read |
|---|---|
| How does it work? | [docs/METHOD.md](docs/METHOD.md) — featurisation, architecture, training, retrieval, tiers, the mask-aware fallback, the population readouts |
| What was it trained on? | [docs/DATA.md](docs/DATA.md) — BMRB + PDB sources, curation, labels, the homology split, exact sizes |
| How was it benchmarked? | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [benchmarks/README.md](benchmarks/README.md) — a ten-minute read, then the files |
| When should I not trust it? | [docs/LIMITATIONS.md](docs/LIMITATIONS.md) |
| What ships and under which licence? | [DATA_PROVENANCE.md](DATA_PROVENANCE.md) |
| What changed between versions? | [CHANGELOG.md](CHANGELOG.md) |

## Things to know before you use it

- **Referencing.** FACET expects DSS-referenced shifts (the BMRB convention). A
  composition-adaptive per-nucleus check warns about systematic offsets and suggests
  a correction (`auto_reference=True` applies it). It is weakest on CA-only offsets;
  for publication-quality restraints run LACS first.
- **Perdeuterated samples.** Pass `--deuteration perdeut-exchanged` (or
  `perdeut-unexchanged`, `ilv-methyl`) to apply the analytical ¹³C isotope correction.
  Missing HA shifts are masked; `--mask-safe-fallback` switches on the shift-space
  fallback if the default path misbehaves on your sample.
- **Disordered proteins.** FACET reports three population readouts that answer
  different questions: *geometric* basin populations (`basin_populations`, share of
  retrieved neighbours per Ramachandran basin), *structural* populations
  (`structural_populations`, kernel-weighted posterior over the reference structures'
  helix/strand/loop states — over-estimates helix on
  pure IDPs), and the retrieval-free d2D-style engine (`facet.predict_ss_populations`,
  the one to report for IDPs and to compare with d2D / CheSPI). See
  [docs/METHOD.md](docs/METHOD.md) (§7–8) and
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md).
- **Ensemble seeds.** The 25 retrieved neighbours per residue can be exported
  (`result.to_ensemble_csv()` / `to_ensemble_json()`) as seed conformers for BME or
  ENSEMBLE-style reweighting.

## Output formats

| Format | Used by |
|---|---|
| `.tbl` | XPLOR-NIH, CNS, HADDOCK, ARIA dihedral restraints |
| `.aco` | CYANA angle constraints |
| `.nef` | NEF 1.1 dihedral restraints (wwPDB, CCPN) |
| `.predtab` | per-residue summary (TALOS-style) |
| `.csv`, `.json` | everything, including basin populations, alternative clusters, RCI S², retrieved neighbours |

Every output records the package version and the retrieval-index version in its
header.

## Licence

Code: [MIT](LICENSE). Model weights, retrieval index, shift reference and fitted
parameters: [CC BY 4.0](LICENSE-WEIGHTS). They derive from the PDB and the BMRB, both CC0 —
please also cite BMRB: Hoch *et al.*, *Nucleic Acids Res.* **51**, D368 (2023),
[10.1093/nar/gkac1050](https://doi.org/10.1093/nar/gkac1050).

## Citation

```
Zinke, M. FACET: backbone torsion angle prediction from NMR chemical shifts.
Software, version 0.4.0 (2026). https://github.com/maxzinke/facet-nmr
```

A machine-readable citation is in [CITATION.cff](CITATION.cff) (GitHub's "Cite this
repository" button). The archived-release DOI and the preprint reference will be
added there on release.

FACET builds on published methods; if you use the corresponding outputs, cite them
too:

- **RCI S²** (`rci_s2`): Berjanskii & Wishart, *J. Biomol. NMR* 40, 31 (2008),
  [10.1007/s10858-007-9208-0](https://doi.org/10.1007/s10858-007-9208-0). Our
  implementation follows Eq. 2 and 3 without the per-subset weight optimisation,
  neighbour corrections, REFCOR or end-effect steps.
- **Random-coil shifts**: Wishart *et al.*, *J. Biomol. NMR* 5, 67 (1995);
  Schwarzinger *et al.*, *JACS* 123, 2970 (2001).
- **Referencing check**: adapted from LACS — Wang & Markley, *J. Biomol. NMR* 44, 95
  (2009).
- **d2D likelihood** (`predict_ss_populations`): Camilloni, De Simone, Vranken &
  Vendruscolo, *Biochemistry* 51, 2224 (2012) — same per-residue-type Gaussian
  likelihood, all parameters refit on our own data.
- **CheSPI** (recommended IDP cross-check): Nielsen & Mulder, *J. Biomol. NMR* 75,
  273 (2021), [10.1007/s10858-021-00374-w](https://doi.org/10.1007/s10858-021-00374-w).
- **Deuterium isotope corrections**: Venters *et al.*, *JACS* 118, 8985 (1996);
  Hansen, *Prog. NMR Spectrosc.* 20, 207 (1988).
- **TALOS-N** (benchmark reference): Shen & Bax, *J. Biomol. NMR* 56, 227 (2013).
