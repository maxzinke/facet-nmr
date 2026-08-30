# FACET — Data

What FACET was trained, indexed and evaluated on; how it was curated; and how it was
split. The numbers here were measured directly on the dataset cache and the shipped
asset files while writing this document (2026-08-30), not copied from earlier notes.
Where the README says something different, the value here is the measured one.

The curation code lives in the training repository (not yet public). Entry-ID lists
for every split are published under `benchmarks/data/` so the sets can be checked
without it.

## 1. Sources and licences

| Source | Used for | Licence |
|---|---|---|
| **BMRB** (Biological Magnetic Resonance Data Bank, <https://bmrb.io>) | assigned backbone chemical shifts, polymer sequences | CC0 1.0 |
| **wwPDB** (mmCIF coordinate files) | φ/ψ/χ1 angles, secondary-structure annotations, ensemble statistics | CC0 1.0 |

Both are public-domain dedications; see [DATA_PROVENANCE.md](../DATA_PROVENANCE.md)
for what each shipped file contains and how BMRB asks to be cited. The candidate list
was the set of BMRB entries that declare a related PDB entry, retrieved from the BMRB
API on 2026-03-28: **7,641** entries.

## 2. Pairing shifts with structures

For each candidate BMRB entry:

1. The entry must contain a `polypeptide(L)` entity and assigned shifts.
2. Every PDB id linked from the entry is tried in order. A structure is accepted when a
   Needleman–Wunsch alignment of the BMRB sequence to a chain reaches **≥ 90 %
   identity** and the chain carries secondary-structure annotation. The first
   acceptable structure is used; the matched chain is remembered.
3. Per residue, the six backbone shifts (`H, HA, N, CA, CB, C`) are collected; residues
   with fewer than **3** observed backbone nuclei are dropped.
4. BMRB residue numbers are mapped to PDB residue numbers through the alignment.
5. Entries are **de-duplicated by sequence**: for each distinct sequence the deposition
   with the most backbone shifts is kept.

## 3. Shift quality control

Applied per deposition before anything is trained (`quality_control.py` in the training
repository):

* **Referencing offsets.** Mean secondary shift per nucleus (against the same
  random-coil table the model uses). Flags: |Cα| > 1.5, |Cβ| > 2.0, |N| > 5.0 ppm, and a
  same-sign Cα/Cβ offset > 1.0 ppm (a ¹³C carrier error).
* **LACS-style diagnostic.** Regression of Δδ(Cα) on Δδ(Cα) − Δδ(Cβ); intercept and R².
* **Outlier residues.** Any nucleus with |Δδ| more than 4 expected standard deviations
  (H 0.6, Hα 0.5, N 4.5, Cα 2.5, Cβ 4.5, C′ 3.1 ppm) or beyond a hard limit
  (3, 2, 15, 10, 15, 8 ppm). An entry with more than max(3, 10 %) outlier residues is
  flagged as having assignment errors.
* **Quality score** in [0, 1]: 1 minus penalties for offsets (0.15 × offset/threshold,
  capped), outlier fraction (× 2), sparse observation (< 4 nuclei per residue) and poor
  LACS correlation (R² < 0.2).
* **Correction and rejection.** Flagged ¹³C offsets are corrected jointly on Cα, Cβ and
  C′ by the mean of the three offsets when it exceeds 0.5 ppm; flagged ¹⁵N offsets are
  subtracted. Entries with quality score < 0.5 are rejected. Of 5,132 depositions that
  reached this stage, **162** were rejected and 1,484 were flagged for a Cα offset
  (14 for N).
* **Deuterium correction.** Depositions whose sample description indicates
  deuteration receive the isotope correction described in [METHOD.md §2](METHOD.md)
  so the model sees protonated-equivalent shifts.

The result is the working dataset: **4,970 proteins, 448,819 residues**. Mean observed
nuclei per residue 5.21; per-nucleus coverage H 93.0 %, Hα 79.8 %, N 93.0 %,
Cα 97.4 %, Cβ 88.2 %, C′ 69.9 %; 50.8 % of residues have all six. Sequence lengths:
median 101 residues (min 4, max 1,515). 87.0 % of residues come from multi-model NMR
structures, 13.0 % from single-model (X-ray or single-model NMR) structures.

## 4. Structural labels

**Secondary structure (H/E/C).** Taken from the PDB-deposited helix and sheet records
of the matched chain (`_struct_conf`, `_struct_sheet_range`, read with gemmi); every
other residue is C. These are the depositors' annotations, not a fresh DSSP run,
although code comments call them "DSSP" labels. Working-set composition: H 38.2 %,
E 19.6 %, C 42.2 %.

**φ/ψ from ensembles.** For NMR entries φ and ψ are computed in *every* model and
summarised by the circular mean and circular variance (1 − R̄). A residue is
**well-defined** when both variances are < 0.20 (≈ ±25° spread); 88.1 % of residues
are. Single-model structures are well-defined by construction.

**X-ray consensus.** Structures with > 95 % sequence identity to the BMRB sequence
(RCSB search) contribute an X-ray φ/ψ consensus and dispersion, excluding residues
within ±3 of any sequence mismatch.

**Geometry class** — which angle, if any, is a trustworthy target:

| Class | Rule | Target | Residues |
|---|---|---|---|
| XG (X-ray gold) | tight X-ray consensus (dispersion < 0.10) and NMR agrees or is absent | X-ray | 79,856 |
| NG (NMR good) | no usable X-ray; NMR not disordered (variance ≤ 0.40) | NMR mean | 298,895 |
| XU (X-ray usable) | X-ray available but NMR disordered / loose consensus | X-ray | 6,509 |
| D (disordered) | NMR variance > 0.40, nothing better | none | 35,580 |
| C (conflict) | well-defined NMR differs from tight X-ray by > 40° | NMR, flagged | 27,979 |

**Training targets** are NG and XG residues that are also well-defined. Everything else
still contributes its shifts as *context* in the window; it just carries no angle loss.

**χ1.** N–Cα–Cβ–Xγ dihedral from the structure, averaged over models; classes
g+ (0°, 120°], g− (−120°, 0°], trans otherwise; undefined for Gly, Ala and residues
without the γ atom.

**Ramachandran basin** (used for retrieval populations and state labels): see
[METHOD.md §5](METHOD.md).

## 5. Splitting

Homology-aware, so that no test protein has a close relative in training:

1. Every protein sequence is turned into a normalised 8,000-dimensional 3-mer
   frequency vector.
2. Proteins with cosine similarity > **0.5** are connected; clusters are the
   **connected components** of that graph (true single linkage — no two proteins in
   different clusters exceed the threshold). Working set: **3,741** clusters.
3. Clusters are shuffled (seed 42) and assigned whole to test until 15 % of proteins
   are covered, then to validation until another 15 %, remainder to train.
4. An all-pairs check between train and test, and train and validation, asserts that
   no pair exceeds the threshold before training starts (0 violations).

| Split | Proteins | Residues | H / E / C | well-defined | NG+XG | all 6 nuclei | median length |
|---|---|---|---|---|---|---|---|
| train | 3,480 | 316,445 | 37.9 / 20.0 / 42.1 % | 88.2 % | 84.3 % | 50.8 % | 102 |
| validation | 745 | 63,750 | 38.5 / 18.5 / 43.0 % | 87.3 % | 84.8 % | 53.4 % | 95 |
| test | 745 | 68,624 | 39.3 / 19.0 / 41.7 % | 88.4 % | 84.4 % | 50.2 % | 105 |

The test split is the 745-protein benchmark set of the README; 88.1 % of its residues
come from multi-model NMR structures. Only the validation split was used for model
selection and tier calibration.

Entry-ID lists: `benchmarks/data/train_set.txt`, `benchmarks/data/val_set.txt`,
`benchmarks/data/test_set_745.txt`, and `benchmarks/data/ablation_set_39.txt` for the
coverage-ablation subset (see [BENCHMARKS.md](BENCHMARKS.md)).

## 6. What each shipped asset contains

| File | Built from | Rows | Depositions | Notes |
|---|---|---|---|---|
| `facet_v3.pt` / `facet_v3.onnx` | training split | — | — | encoder + heads, 1,293,378 parameters |
| `facet_retrieval_index.npz` + `.entries.json` | training-split NG/XG well-defined residues | 219,713 | 3,439 | 128-d embeddings, φ, ψ, SS, basin, residue type; entry id per row in the sidecar |
| `facet_shift_reference.npz` | training-split residues with a defined angle | 310,923 | 3,470 | 18-d secondary-shift triplets with masks, φ, ψ, SS, residue-type triplet, BMRB entry and residue number |
| `facet/data/ss_popn_params.npz` | 49,262 curated training-split residues (1,075 depositions) | — | — | per-(type, state, nucleus) means, σ, correlation, neighbour-context corrections |

None of the 745 test entries appears in the retrieval index or the shift reference
(checked by `benchmarks/check_leakage.py`).

## 7. Exact figures behind rounded statements

* "Trained on ~5,000 proteins" describes the working set after quality control:
  **4,970 proteins / 448,819 residues**, of which the training split is
  **3,480 proteins / 316,445 residues** (validation 745 / 63,750; test 745 / 68,624).
* The "220K-residue reference index" is 219,713 rows.
* The 3,470 depositions in the shift reference versus 3,439 in the embedding index
  differ because the index keeps only NG/XG well-defined residues while the reference
  keeps every residue with a defined angle.
