# The FACET benchmark, in ten minutes

This directory holds everything behind the numbers in the top-level README: the
proteins, the exact inputs both methods saw, the per-residue results, and scripts that
recompute every table from those results. Nothing here needs the model, the weights or
a network connection except the two scripts that re-run FACET.

If you only read one thing: run `python benchmarks/rescore.py` and compare its output
with the README table. They must agree; if they do not, the README is wrong.

## The question

Given a protein's backbone chemical shifts (H, HA, N, CA, CB, C'), how close are the
predicted backbone torsion angles φ/ψ to the angles in the deposited structure — and
how does that compare with TALOS-N, the standard tool for the job?

## The data

**Where it comes from.** 4,970 BMRB chemical-shift depositions paired with their PDB
structures (both CC0), curated as described in `../docs/DATA.md`. Every residue has
six shifts (some missing), a helix/strand/coil label from the structure's own
annotations, and a φ/ψ from the
structure — for NMR ensembles the mean over models, with a flag saying whether the
models agree (`well_defined`).

**How it is split.** Proteins are clustered by sequence similarity (cosine similarity
of 3-mer frequency vectors, threshold 0.5, connected components) and *whole clusters*
are assigned to train / validation / test at roughly 70 / 15 / 15 %. A test protein
therefore has no close homologue in training. The result is 3,480 / 745 / 745 entries;
the ID lists are in `data/`.

**Why "clean".** Until 2026-04-17 the clustering used complete linkage, which let
chains of similar sequences leak across the split. The leakage gate caught it, the
linkage was switched to connected components, and every number in the README was
re-measured on the new split. Numbers from before that date are not comparable and
are not used anywhere in this repository.

**Proof there is no leakage.** FACET predicts by retrieving neighbours from two
reference files that ship with the package. `check_leakage.py` intersects the test-set
IDs with the BMRB entries in both files:

```
test_set_745.txt: 745 entries
  in retrieval index : 0 / 745
  in shift reference : 0 / 745
ablation_set_39.txt: 39 entries
  in retrieval index : 0 / 39
  in shift reference : 0 / 39
```

Run it yourself: `python benchmarks/check_leakage.py`.

## What is compared, and on what

Both methods received the **same input file** per protein (`data/inputs/<id>.tab`,
NMRPipe format, exported from the curated shifts). TALOS-N 4.21 was run with
`TALOSN -in <id>.tab -noauto -noverb`; FACET with its default retrieval mode.

For each residue the error is one number:

```
error = sqrt( (Δφ² + Δψ²) / 2 )       Δ = |predicted − true|, wrapped into [0°, 180°]
```

Residues are scored only where the structure gives a usable truth (finite φ/ψ and
`well_defined`). That leaves **55,032 residues from 739 proteins**. FACET emits an
angle for every one of them (it tiers instead of abstaining) and TALOS-N for 98.6 %;
all head-to-head numbers are computed on the **54,260 residues both methods predict**,
so neither method is rewarded for abstaining.

## The result

`python benchmarks/rescore.py --bootstrap` prints (54,260 paired residues in 724
proteins; differences are FACET − TALOS-N with a 95 % protein-level paired-bootstrap
interval). The per-residue table it reads was produced by the released package itself
— `run_talosn_comparison.py` calling `facet.predict()` with default settings on
`data/inputs/` — so this table and "what you get from `pip install facet-nmr`" are
the same thing:

| Metric | TALOS-N | FACET | difference |
|---|---|---|---|
| All-residue median | 11.58° | 10.91° | −0.67° [−0.78, −0.56] |
| fail25 (share of residues > 25°) | 20.3 % | 18.0 % | −2.4 pt [−2.7, −2.0] |
| Mean | 21.3° | 19.9° | |
| p90 | 46.1° | 40.3° | |
| Coil median (n = 18,221) | 19.3° | 18.0° | −1.30° [−1.57, −0.98] |
| Helix median (n = 24,157) | 7.4° | 7.0° | −0.35° [−0.44, −0.26] |
| Strand median (n = 11,882) | 13.6° | 12.8° | −0.81° [−0.99, −0.56] |
| Rigid residues (n = 47,177) | 10.7° | 10.1° | |
| Flexible residues (n = 7,083) | 20.5° | 18.9° | |
| Head-to-head: lower error on … | 46.2 % | 53.8 % of residues (4 exact ties) | [53.3, 54.4] |

and the tier calibration — how much to trust each confidence label FACET attaches:

| Tier | Share of scored residues | Median error | fail25 |
|---|---|---|---|
| High | 76.3 % | 9.3° | 9.9 % |
| Medium | 15.8 % | 19.9° | 40.2 % |
| Low | 3.2 % | 59.8° | 67.8 % |
| Flexible | 4.6 % | 20.8° | 43.0 % |

The ground truth carries a validated correction for the 63 X-ray entries whose stored
angles had been converted to radians twice (see the caveats). The 0.3-era tables —
including the uncorrected 12.65° vs 13.57° that earlier versions of this repository
quoted — are preserved under `results/talosn_clean/archive/`.

## How to read one row of `per_residue.csv`

```
bmrb_id,seq_id,residue,ss,flexible,well_defined,n_models,truth_units_suspect,phi_true,psi_true,phi_true_spread,psi_true_spread,phi_talosn,psi_talosn,talosn_class,talosn_has_pred,talosn_err,facet_err_source,phi_facet,psi_facet,facet_phi_err,facet_psi_err,facet_tier,facet_tier_public,facet_has_pred,facet_err,truth_corrected
10034,8,ASP,H,1,1,20,0,-85.221,-41.208,17.654,3.374,-66.349,-39.447,Strong,1,13.402,record,-67.968,-29.507,11.033,17.506,Generous,Medium,1,14.741,1
```

BMRB entry 10034, residue 8 (Asp), helix in the deposited structure. The 20-model NMR
ensemble puts φ/ψ at (−85°, −41°) — the circular mean over models — with a φ spread of
17.7° (`phi_true_spread`, derived from the circular variance across models; see
`../docs/BENCHMARKS.md` §2), so the residue counts as *flexible* (`flexible = 1`: the
spread of φ or ψ exceeds 10°).
FACET predicted (−68.0°, −29.5°) with per-angle error bars of 11.0°/17.5° and the
`Generous` tier (shown to users as `Medium`); its error is 14.7°. TALOS-N predicted
(−66.3°, −39.4°), called it `Strong`, and was 13.4° off — one residue in TALOS-N's
win column. `truth_units_suspect` is 0 (NMR truth, never affected by the X-ray
defect); `truth_corrected = 1` marks that the table as a whole carries the corrected
truth; `facet_err_source` is a legacy column from the truth file and can be ignored.

`per_protein.csv` has one row per entry: residue counts, both medians, the share of
residues on which FACET is closer, and the tier and SS composition — use it to find
the proteins where FACET does badly and go look at them.

## One table, and the archive

`per_residue.csv` is the only current table: released package, default settings,
corrected truth. Everything else that was measured on the way here — the 0.3-era
harness record (uncorrected and corrected), the 0.3-era public-path re-runs, the
0.3.1 fallback-on run, and the 0.4.0 fallback check — is gzipped under
`results/talosn_clean/archive/` with its own README. Those files document history;
none of them describes the released model.

## Re-running things

```
python benchmarks/rescore.py                      # the tables above, no model needed
python benchmarks/rescore.py --bootstrap          # + 95 % CIs and residue-level tests
python benchmarks/rescore.py --figures            # + figures/*.png
python benchmarks/check_leakage.py                # 0/745, 0/39 or a non-zero exit
python benchmarks/walkthrough.py                  # one protein end to end (WALKTHROUGH.md)
python benchmarks/run_talosn_comparison.py --ids 10034 10040 10046     # 3-entry smoke test
python benchmarks/run_talosn_comparison.py --out my_rerun.csv          # all 740 entries, ~30 min CPU
```

`run_talosn_comparison.py` carries the stored TALOS-N columns and the corrected truth
forward and re-runs only FACET, so a full re-run needs no TALOS-N install; pass
`--talosn-dir <dir>` with `<dir>/<id>/pred.tab` if you have TALOS-N outputs to
re-parse, and `--mask-safe-fallback` to reproduce the fallback experiment.
`build_corrected_truth.py` documents how the archived 0.3-era corrected tables were
made from the archived record.

## Caveats, stated plainly

* **The ground truth of 63 entries needed repair.** Every X-ray-truth residue in the
  set (6,978 of 55,032) had a stored truth that was a radian value converted to
  radians a second time. It was repaired by multiplying by 180/π, and the repair was
  verified against φ/ψ recomputed from the deposited mmCIF files (median deviation
  0.02°). The same defect had reached ~12 % of the training targets; the shipped
  0.4.0 model is retrained on the fixed pipeline, and this table is measured on the
  corrected truth. The affected rows stay flagged (`truth_units_suspect = 1`);
  `../docs/BENCHMARKS.md` §7 has the full account, and the uncorrected 0.3-era tables
  are in `archive/`.
* **The margin is modest and statistically solid.** Δ median −0.67° [−0.78, −0.56],
  FACET closer on 53.8 % of residues [53.3, 54.4], every per-class interval excludes
  zero (Wilcoxon p ≈ 10⁻¹⁰⁵). Read it as "consistently better", not "much better".
* **TALOS-N is run as distributed** (v4.21, default parameters, its identical-sequence
  auto-exclusion on); 9 of the 745 test proteins are in its own reference database
  (the methods tie there — 10.05° vs 10.17°; removing them: 10.92° vs 11.60°), NMR
  ground-truth structures may have been refined with TALOS restraints, and TALOS-N
  crashed on 15 entries (525 residues, 14 of 15 sequences containing `X`), which
  count as abstentions. `../docs/BENCHMARKS.md` §4.3.
* **TALOS-N's raw output files are not redistributed.** The TALOS-N licence restricts
  redistribution of the *software*; its output is not addressed, but the safe course is
  to ship what we derived (angles, class and error per residue, all in
  `per_residue.csv`) together with the exact command line, not the files themselves.
* **739, not 745.** Six test entries contribute no scorable residue: three (16449,
  25078, 51836) have no finite φ/ψ ground truth, two (30337, 30344) are 6- and
  8-residue peptides that yield no scored window, and one (6858) has a single
  assigned residue, which the package cannot window. They are kept in
  `test_set_745.txt` because the split is defined by clusters, not by what happened
  to be scorable.
* **The missing-atom ablation is a different experiment.** It uses a different error
  (`(|Δφ| + |Δψ|)/2`, not the RMS form), a 39-entry subset (`data/ablation_set_39.txt`,
  the alphabetically-first test entries up to 4,000 residues) and asks a different
  question — what happens when HA, H or CB are deleted from the input. Its numbers must
  not be placed next to the table above. See `../docs/BENCHMARKS.md` §5.
* **Ground truth is a structure, not the truth.** A deposited NMR ensemble or crystal
  structure is itself a model. Residues where the ensemble models disagree are flagged
  (`flexible`), and residues without a well-defined conformation are excluded, but the
  remaining "truth" still carries its own error, which is why sub-5° differences between
  methods should not be over-read.
* **The 4 exact ties** are residues where both methods returned the same angles to the
  reported precision.

## Files

```
data/test_set_745.txt        745 BMRB IDs — the held-out proteins
data/ablation_set_39.txt     the 39-entry subset used by the missing-atom ablation
data/s3_fallback_check_ids.txt  75 HA-free + 60 control entries for the fallback check
data/train_set.txt           3,480 BMRB IDs the model and the retrieval index were built from
data/val_set.txt             745 BMRB IDs used for model selection only
data/inputs/<id>.tab         the exact shift table each method received (745 files)
data/talosn_database_bmrb_ids.txt  the 498 BMRB entries in TALOS-N 4.21's reference database
results/talosn_clean/per_residue.csv   the benchmark: released package, default settings, corrected truth
results/talosn_clean/per_protein.csv   one row per entry
results/talosn_clean/summary.json      run parameters (weights, index v0.3.0, k=25, DBSCAN eps 30°)
results/talosn_clean/archive/          0.3-era tables and the fallback experiment (gzipped, own README)
results/coverage_ablation/             the missing-atom ablation (results.json, summary.md)
figures/                               generated by rescore.py --figures and walkthrough.py
rescore.py · check_leakage.py · run_talosn_comparison.py · build_corrected_truth.py · walkthrough.py
```
