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
`well_defined`). That leaves **55,036 residues from 740 proteins**. FACET emits an angle
for 99.2 % of them and TALOS-N for 98.6 %; all head-to-head numbers are computed on the
**53,841 residues both methods predict**, so neither method is rewarded for abstaining.

## The result

`python benchmarks/rescore.py --csv benchmarks/results/talosn_clean/per_residue_corrected.csv --bootstrap`
prints (53,876 paired residues in 724 proteins; differences are FACET − TALOS-N with a
95 % protein-level paired-bootstrap interval):

| Metric | TALOS-N | FACET | difference |
|---|---|---|---|
| All-residue median | 11.51° | 11.03° | −0.48° [−0.64, −0.37] |
| fail25 (share of residues > 25°) | 20.1 % | 18.4 % | −1.7 pt [−2.1, −1.4] |
| Mean | 21.2° | 20.5° | |
| p90 | 45.3° | 41.8° | |
| Coil median (n = 17,946) | 19.2° | 18.3° | −0.92° [−1.26, −0.59] |
| Helix median (n = 24,099) | 7.4° | 7.1° | −0.30° [−0.39, −0.20] |
| Strand median (n = 11,831) | 13.5° | 12.9° | −0.68° [−0.91, −0.45] |
| Rigid residues (n = 46,937) | 10.6° | 10.2° | |
| Flexible residues (n = 6,939) | 20.5° | 18.7° | |
| Head-to-head: lower error on … | 47.0 % | 53.0 % of residues (7 exact ties) | [52.4, 53.7] |

and the tier calibration — how much to trust each confidence label FACET attaches:

| Tier | Share of scored residues | Median error | fail25 |
|---|---|---|---|
| High | 76.3 % | 9.3° | 10.1 % |
| Medium | 18.1 % | 19.3° | 39.0 % |
| Low | 3.5 % | 52.7° | 66.5 % |
| Flexible | 2.1 % | 104.4° | 87.5 % |

This is the benchmark of record with the ground truth of 63 X-ray entries corrected
(see the caveats: the stored values had been converted to radians twice; the
correction is validated against the PDB files). The uncorrected record,
`per_residue.csv`, reads 12.65° vs 13.57° — the same comparison, inflated on both
sides by the wrong truth. `python benchmarks/rescore.py` with no arguments prints
that uncorrected table.

## How to read one row of `per_residue.csv`

```
bmrb_id,seq_id,residue,ss,flexible,well_defined,n_models,truth_units_suspect,phi_true,psi_true,phi_true_spread,psi_true_spread,facet_tier,facet_tier_public,facet_has_pred,facet_err,phi_talosn,psi_talosn,talosn_class,talosn_has_pred,talosn_err
10034,8,ASP,H,1,1,20,0,-85.221,-41.208,17.654,3.374,Generous,Medium,1,12.199,-66.349,-39.447,Strong,1,13.402
```

BMRB entry 10034, residue 8 (Asp), helix in the deposited structure. The 20-model NMR
ensemble puts φ/ψ at (−85°, −41°) — the circular mean over models — with a φ spread of
17.7° (`phi_true_spread`, derived from the circular variance across models; see
`../docs/BENCHMARKS.md` §2), so the residue counts as *flexible* (`flexible = 1`: the
spread of φ or ψ exceeds 10°).
FACET's retrieval tier was `Generous` (shown to users as `Medium`) and its error was
12.2°; TALOS-N called it `Strong` and was 13.4° off. The TALOS-N angles are stored;
FACET's are not in this file (see the next section for why) but are in
`per_residue_rerun.csv`. `truth_units_suspect` is 0 here; see the caveats below for
the rows where it is 1.

`per_protein.csv` has one row per entry: residue counts, both medians, the share of
residues on which FACET is closer, and the tier and SS composition — use it to find
the proteins where FACET does badly and go look at them.

## Two files, one benchmark

* **`per_residue.csv` is the benchmark of record.** It was produced inside the training
  harness (batched over the curated shift arrays) with the same weights and the same
  retrieval index that ship with the package. It stores each residue's error and tier
  but not FACET's angles.
* **`per_residue_rerun.csv` is the same benchmark re-run through the public
  `facet.predict()` path** — file parsing, secondary-shift conversion, the package's
  defaults — by `run_talosn_comparison.py`. It adds FACET's φ/ψ and error bars.
  96.5 % of residues land within 0.5° of the recorded error (98.1 % within 5°) with the
  same tier on 96.5 %; scored on its own the public path gives FACET 12.80° vs TALOS-N
  13.65°, win rate 52.5 % (the record: 12.65° vs 13.57°, 53.1 %). If you install the
  package and run it, the re-run is what you will get.
* **`per_residue_rerun_fallback_on.csv` is the same re-run with the 0.3.1 default**,
  which routed every HA-missing residue to the mask-safe fallback. It reproduced the
  record on only 79 % of residues and scored 13.03° vs 13.65°; the loss was largest on
  exactly the proteins the fallback was meant for. That is why the fallback is opt-in
  from 0.4.0. `../docs/BENCHMARKS.md` §6 has the slice-by-slice numbers.

Score either with `rescore.py` (`--csv <file> --rerun` for the re-run;
`--corrected-truth` additionally repairs the flagged ground truth, see the caveats).

## Re-running things

```
python benchmarks/rescore.py --figures            # tables + figures/*.png, no model needed
python benchmarks/rescore.py --bootstrap          # 95 % CIs from a paired bootstrap over proteins
python benchmarks/build_corrected_truth.py        # writes the corrected-truth tables (see caveats)
python benchmarks/rescore.py --csv benchmarks/results/talosn_clean/per_residue_corrected.csv --bootstrap
python benchmarks/check_leakage.py                # 0/745, 0/39 or a non-zero exit
python benchmarks/walkthrough.py                  # one protein end to end (WALKTHROUGH.md)
python benchmarks/run_talosn_comparison.py --ids 10034 10040 10046     # 3-entry smoke test
python benchmarks/run_talosn_comparison.py        # all 740 entries, ~15 min on a laptop CPU
```

`run_talosn_comparison.py` takes `--talosn-dir <dir>` with `<dir>/<id>/pred.tab` if
you have TALOS-N outputs to re-parse; otherwise it carries the stored TALOS-N columns
forward.

## Caveats, stated plainly

* **6,978 of the 55,036 scored residues (63 entries, every X-ray-truth residue in the
  set) have a ground truth that is not in degrees** — a radian value converted to
  radians twice. Multiplying by 180/π repairs it; that was verified against φ/ψ
  recomputed from the deposited mmCIF files for six entries (median deviation 0.02°).
  Both methods were scored against the same wrong values, so the comparison survives,
  but the absolute errors in the table above are inflated and FACET's margin is
  overstated: with the truth corrected the record reads **FACET 11.03° vs TALOS-N
  11.51°** (fail25 18.4 % vs 20.1 %, win rate 53.0 %; `per_residue_corrected.csv`), and
  the released package reproduces 11.18° vs 11.58° (`per_residue_rerun_corrected.csv`).
  Every bootstrap interval still excludes zero. The rows are flagged
  `truth_units_suspect = 1`; `../docs/BENCHMARKS.md` §7 has the validation, the
  extent (the same defect touched ~12 % of the training targets) and the tables.
* **The margin is small and statistically solid.** `rescore.py --bootstrap` resamples
  proteins: Δ median −0.92° [−1.07, −0.71] on the record, −0.48° [−0.64, −0.37]
  corrected; FACET wins 53 % of residues either way, CI ±0.6 points. Read it as
  "consistently a little better", not "much better".
* **TALOS-N is run as distributed** (v4.21, default parameters, its identical-sequence
  auto-exclusion on); 9 of the 745 test proteins are in its own reference database
  (no measurable effect), NMR ground-truth structures may have been refined with
  TALOS restraints, and TALOS-N crashed on 16 entries (532 residues, mostly sequences
  containing `X`), which count as abstentions. `../docs/BENCHMARKS.md` §4.3.
* **TALOS-N's raw output files are not redistributed.** The TALOS-N licence restricts
  redistribution of the *software*; its output is not addressed, but the safe course is
  to ship what we derived (angles, class and error per residue, all in
  `per_residue.csv`) together with the exact command line, not the files themselves.
* **740, not 745.** Five test entries contribute no scorable residue: three
  (16449, 25078, 51836) have no finite φ/ψ ground truth at all, and two (30337, 30344)
  are 6- and 8-residue peptides for which the evaluation produced no scored window.
  They are kept in `test_set_745.txt` because the split is defined by clusters, not by
  what happened to be scorable.
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
* **The 6 exact ties** are residues where both methods returned the same angles to the
  reported precision.

## Files

```
data/test_set_745.txt        745 BMRB IDs — the held-out proteins
data/ablation_set_39.txt     the 39-entry subset used by the missing-atom ablation
data/train_set.txt           3,480 BMRB IDs the model and the retrieval index were built from
data/val_set.txt             745 BMRB IDs used for model selection only
data/inputs/<id>.tab         the exact shift table each method received (745 files)
results/talosn_clean/per_residue.csv        benchmark of record, one row per scored residue
results/talosn_clean/per_residue_rerun.csv  the same, re-run through facet.predict (0.4.0 default), with FACET angles
results/talosn_clean/per_residue_rerun_fallback_on.csv  the re-run with the 0.3.1 default (mask-safe fallback on)
results/talosn_clean/per_residue_corrected.csv          record with the X-ray ground truth repaired (facet_err_source column)
results/talosn_clean/per_residue_rerun_corrected.csv    public-path re-run with the ground truth repaired
data/talosn_database_bmrb_ids.txt  the 498 BMRB entries in TALOS-N 4.21's reference database
results/talosn_clean/per_protein.csv        one row per entry
results/talosn_clean/summary.json           run parameters (k=25, DBSCAN eps 30°, min size 3)
results/coverage_ablation/                  the missing-atom ablation (results.json, summary.md)
figures/                                    generated by rescore.py --figures and walkthrough.py
rescore.py · check_leakage.py · run_talosn_comparison.py · build_corrected_truth.py · walkthrough.py
```
