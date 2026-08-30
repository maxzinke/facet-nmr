# FACET — Limitations

Everything we know FACET does badly, cannot do, or has not been tested on, in one
place. Each item names the code that enforces it or the measurement behind it.
See [METHOD.md](METHOD.md) for how the pieces work and [BENCHMARKS.md](BENCHMARKS.md)
for how the numbers were obtained.

## Input requirements and hard failures

* **At least 3 standard residues** are needed to form a window
  (`facet/inference.py`, `_MIN_RESIDUES`). Peptides shorter than that are rejected.
* **Non-canonical residue types are skipped**, not predicted (MSE, PTR, CGU, …). Their
  shifts are removed before windowing, so the neighbours of a skipped residue see a
  gap. A warning lists what was dropped.
* **Proton-only assignments are rejected.** If more than half of the residues have no
  N/Cα/Cβ/C′ shift the run fails; ¹H^N and ¹Hα alone do not determine backbone geometry.
* **More than 10 % of shifts outside physical ranges is a hard error.** This catches
  swapped columns and unit mistakes but will also refuse a genuinely unusual sample.
* **One chain at a time.** The readers pick a single entity/chain: the largest for BMRB
  fetches and NMR-STAR files, the chain with the most residues for NEF
  (`facet/io/bmrb.py`, `facet/io/nmrstar.py`, `facet/io/nef.py`). Complexes and
  homo-oligomers are treated as their largest chain; inter-chain context is ignored.
* **Sequence numbering must be contiguous** for neighbours to be recognised. Windows
  are built by residue number, so renumbered or gapped tables silently lose context.
* The minimal NMR-STAR parser handles well-formed, whitespace-separated depositions
  only; quoted multi-line values are not supported.

## Accuracy and coverage

* **Hα-missing inputs are a measured, not a solved, case.** On the coverage-ablation
  benchmark (Hα stripped from every residue of every query) the parametric head
  degrades badly and the opt-in mask-safe fallback recovers full-coverage accuracy
  (13.6° vs 13.6° median). On the real benchmark, however, the default
  embedding-retrieval path is *better* than the fallback on Hα-missing residues,
  including on the 49 clean-truth proteins with no Hα at all (9.6° median vs 10.7°).
  The fallback is therefore off by default (`mask_safe_fallback=False`); it is a
  different, simpler predictor (shift-space kNN with greedy clustering, no learned
  representation) and should be switched on only when the default path visibly
  fails on a sample.
* **Losing Cβ costs ~0.5° median** on the same benchmark; losing C′ was not ablated
  separately.
* **Coil is where FACET is least accurate** (20.9° median vs 8.0° helix and 14.6°
  strand on the benchmark). The Low and Flexible tiers exist to keep those residues out
  of restraint files; do not override them with `--include-all` for structure
  calculation.
* **The tier calibration table was measured on the 745-protein test split**, which is
  88 % NMR-ensemble-derived, folded, BMRB-deposited proteins of median length 105.
  The tiers have not been calibrated on membrane proteins, very large systems,
  or intrinsically disordered proteins.
* **The expected-error head is trained only on residues with a well-defined target.**
  On residues that are genuinely disordered its output is an extrapolation; the
  retrieval tier, not the parametric confidence, is what the public tier reports in
  the default mode.

## Referencing

* **Cα-only offsets are partly invisible to the referencing check.** The check infers
  composition from Δδ(Cα) − Δδ(Cβ), so an error confined to the Cα column shifts the
  inferred composition and the expected mean together; offsets below ~2 ppm can pass.
  N, Hα, H^N, C′ and Cβ offsets are caught reliably (`facet/referencing.py`).
* The check needs ≥ 20 residues and ≥ 20 observations per nucleus; small peptides get
  no check.
* `auto_reference` applies constant per-nucleus corrections. It is a convenience, not a
  replacement for LACS or a re-referencing server before publication-quality work.

## Perdeuterated samples

* The isotope correction is **first-order and residue-type-independent**: one-bond
  −0.30 ppm and two-bond −0.10 ppm coefficients, no three-bond terms, no dependence on
  temperature, solvent, or the exact labelling pattern. Residual errors of a few
  tenths of a ppm on ¹³C remain.
* Because of that residual, **perdeuterated inputs land in the Medium/Low/Flexible
  tiers more often** — on the order of 15–25 % more residues than an equivalent
  protonated sample (observation from use, not a benchmark figure). SS prediction is
  largely unaffected.
* When Hα is also missing (the common TROSY case) the default path still predicts
  φ/ψ from the remaining nuclei through input masking; on the benchmark's Hα-free
  proteins this is the more accurate option. The opt-in fallback of METHOD.md §6 is
  available if a sample behaves differently.

## Population readouts

* **Basin populations are sampling fractions of retrieved precedents**, not ensemble
  populations. For a rigid residue they are trivially 100 % in one basin; for a
  disordered residue they say which regions similar-shift residues in the PDB occupy.
* **Structural (kernel-weighted) populations over-estimate helix on IDPs** — about 20 %
  mean helix on tau K18 (BMRB 19253) where δ2D gives ~1 % and CheSPI ~4 % — because
  index residues carry the secondary-structure context of their folded parent
  proteins. Use this readout on folded or partly folded proteins.
* **The d2D-style engine (FACET-D2) has wider σ than δ2D** (Ala Cα pooled 1.29 ppm vs
  0.90 ppm), a consequence of our state-label granularity. It therefore does not
  reproduce δ2D's near-zero helix on pure IDPs (6.6 % mean helix on tau K18) and sits
  closer to CheSPI's 4-state helical definition. It over-calls PPII relative to δ2D
  (broad PPII window). It has no sequence-based prior and no HMM smoothing, only a
  5-residue triangular filter.
* Neither population readout has been validated against experimental ensembles
  (PED); the comparisons are tool-to-tool on a small IDP panel (tau K18, α-synuclein)
  plus folded controls.
* The d2D refit used only 1,558 of 5,132 depositions (those never re-referenced), so
  some (residue type, state, nucleus) cells are supported by few observations; 4 % of
  cells fall below the 30-observation minimum and are backstopped by pooled σ.

## RCI S²

* Implemented with the paper's all-six-nuclei weights only. Missing nuclei contribute
  zero rather than switching to the subset-specific weights, which biases RCI up and
  S² down on incomplete residues. No sequential (i±1) correction, no REFCOR
  re-referencing, no end-effect correction. For values comparable to the Wishart
  server, use the server.

## χ1

* Three-class rotamer label only; no χ1 for Gly and Ala. Overall accuracy on the
  training-run test set was 58.7 % with class-weighted training (g+ 56 %, g− 55 %,
  trans 66 %) — useful as a prior, not as a restraint.

## Scope of the benchmark evidence

* The head-to-head comparison is against **TALOS-N alone**, on **BMRB-deposited
  proteins with PDB structures**. No comparison has been made on proteins without a
  deposited structure, on unassigned or partially assigned spectra, or against other
  predictors.
* Ground truth is the ensemble circular mean over deposited models; for residues with
  a broad ensemble the "true" angle is itself uncertain, and both methods are scored
  against the same convention.
* All 745 benchmark proteins are absent from the training split, the retrieval index
  and the shift reference (checked), but they were drawn from the same BMRB/PDB
  population as the training data. Performance on out-of-distribution samples
  (unusual conditions, non-natural sequences) is unmeasured.

## Software

* **PyTorch is a hard dependency** (`torch>=2.0`), even though inference needs only the
  encoder forward pass and the shipped ONNX file exists; an ONNX-only install path is
  not yet available.
* Model weights and reference data (~125 MB) are downloaded on first use from the
  Hugging Face Hub and verified by SHA-256; an air-gapped install must pre-populate
  `$FACET_HOME` (`python -m facet.assets`). With `FACET_NO_DOWNLOAD=1` and no cached
  files, prediction fails (index) or degrades to the parametric head (shift
  reference).
* Inference is single-process NumPy/PyTorch; the kernel-weighted populations cost a
  full pass over the index per residue and dominate run time on long proteins.
* The retrieval index is a snapshot (version `v0.2.1`); BMRB corrections deposited
  after it was built are not reflected until the index is rebuilt.
