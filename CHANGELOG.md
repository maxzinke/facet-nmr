# Changelog

All notable changes to FACET. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/).

## [0.4.0] — 2026-08-30

First public release.

### Changed
- **The mask-safe shift-retrieval fallback is now opt-in** (`mask_safe_fallback=False`,
  CLI `--mask-safe-fallback`). Since 0.3.1 every residue without an HA shift — 21 % of
  the benchmark's residues — was routed to it. Re-running the 745-protein benchmark
  through the public `facet.predict()` path showed that this cost accuracy on every
  slice, including the proteins with no HA at all (9.6° median with the default
  embedding retrieval vs 10.7° with the fallback; High-tier share 65 % vs 38 %), and
  that the 0.3.1 default did not reproduce the recorded benchmark (13.03° vs 12.65°).
  With the fallback off the public path matches the record on 97.5 % of residues.
  See docs/BENCHMARKS.md §6.
- **Mask-safe retrieval ranks candidates by query coverage before distance.** The
  scorer used a mean squared-z over the columns shared with the query, floored at two
  shared columns, so a reference row overlapping the query on only two atoms could
  win the arg-min far too often (88 % of `-H-HA` queries were won by a row covering
  under 40 % of the query; rank-1 distance carried no error information, ρ = +0.007).
  Candidates are now ranked by coverage first. On the coverage-ablation benchmark the
  median error moves from 15.99° → 13.61° (complete shifts) and 23.87° → 13.42°
  (H and HA absent); accuracy no longer depends materially on how many atoms are
  present. The numbers quoted for missing-HA samples in earlier READMEs (~18°) are
  superseded.
- Weights and reference data are downloaded on first use from the HuggingFace model
  repository `SiXa18/facet-weights`, pinned to tag `v0.4.0` and verified by SHA-256,
  instead of being shipped in the wheel or tracked in the git repository.
- Weights, retrieval index, shift reference and fitted parameters are now licensed
  CC BY 4.0 (code stays MIT); see `LICENSE` and `DATA_PROVENANCE.md`.
- Every output file records the package version and the retrieval-index version in
  its header; `facet --version` prints the package version.

### Added
- `docs/METHOD.md`, `docs/DATA.md`, `docs/BENCHMARKS.md`, `docs/LIMITATIONS.md`.
- `benchmarks/`: test-set and split ID lists, the 745 benchmark input shift tables,
  the per-residue comparison table, `rescore.py` (reproduces every README table),
  `check_leakage.py` (verifies the test set is absent from the shipped index and
  reference), `run_talosn_comparison.py`, figures and a worked example.
- `CITATION.cff`, `.zenodo.json`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, CI.
- `dev` extra (`pip install -e ".[dev]"`).

### Removed
- Internal analysis scripts, an abandoned "centroid" shift reference
  (`facet_shift_reference_centroids.npz`), stray example outputs and a stale copy of
  the HuggingFace Space from the repository. The Space now installs the package from
  PyPI instead of vendoring it.

### Fixed
- `predict()` raised `NameError` instead of degrading to parametric mode when the
  retrieval index could not be found (the fresh-install path).
- `$FACET_V3_PT`-style per-asset environment overrides were derived with a doubled
  prefix and never matched.
- Example outputs regenerated with the current tier vocabulary.
- Test suite brought back to green; tests that need the downloaded assets are marked
  and skipped when they are absent.

### Known issues
- Every single-model (X-ray) structure in the data pipeline had its φ/ψ converted to
  radians twice. The benchmark truth for 63 entries (6,978 residues) has been
  corrected and validated against the PDB files (`benchmarks/build_corrected_truth.py`;
  the README table uses the corrected truth, 11.03° vs 11.51°, where the uncorrected
  record read 12.65° vs 13.57°). The same defect affects 11.7 % of the training
  targets, 1.1 % of the shipped retrieval index and 11.7 % of the shift reference;
  see docs/LIMITATIONS.md. A retrained release on the fixed pipeline will follow.
- 16 of the 740 benchmark entries have no TALOS-N prediction because TALOS-N 4.21
  crashes on them (15 contain non-standard residue codes); they count against
  TALOS-N's coverage and are excluded from the paired set.

## [0.3.1] — 2026-05-29

### Added
- Mask-safe shift-retrieval fallback for residues with missing HA (or other atoms):
  detects HA-missing residues and routes them to a mask-aware kNN over a 310,923-row
  secondary-shift reference instead of the parametric head, which collapses toward
  0° without HA.

## [0.3.0] — 2026-04-21

### Added
- Retrieval-free d2D-style Gaussian secondary-structure population engine
  (`facet.predict_ss_populations`), parameters refit on 49,262 curated residues.
- Kernel-weighted Bayesian "structural" populations over DSSP states using the
  learned embedding as similarity kernel; per-residue neighbour ensemble export
  (`to_ensemble_csv`, `to_ensemble_json`).
- Stacked-bar residual-SS figure; renamed residual-SS tabs and titles.
- Composition-adaptive per-nucleus referencing check with `auto_reference`.
- RCI S² order parameter following Berjanskii & Wishart (2008) Eq. 2 and 3.
- Deuteration flag with analytical ¹³C isotope-shift correction; χ1 in summaries;
  graceful handling of non-canonical residues.

## [0.2.0] — 2026-04-19

### Added
- Retrieval-augmented inference: kNN (k = 25) + DBSCAN (eps 30°) over a 220K-residue
  reference index of learned embeddings; per-residue basin populations and
  alternative clusters.
- Unified tier vocabulary High / Medium / Low / Flexible across retrieval and
  parametric modes.

### Fixed
- 33 K corrupt (φ = ψ = 0) index entries removed; spread clusters from DBSCAN
  chain-merging rejected; tier-ordering bug.

## [0.1.0] — 2026-04-12

### Added
- Initial package: FACET v3 local-biased transformer, NMRPipe `.tab` / CSV / NEF /
  NMR-STAR readers, BMRB fetch, XPLOR/CNS `.tbl`, CYANA `.aco`, NEF, `pred.tab`,
  CSV and JSON writers, ONNX export, sequence + secondary-structure figure,
  HuggingFace Space.

[0.4.0]: https://github.com/maxzinke/facet-nmr/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/maxzinke/facet-nmr/releases/tag/v0.3.1
