# Benchmarks: protocol, definitions and sources

This document is the reference behind every number in the README. For a plain-language
tour start with [`benchmarks/README.md`](../benchmarks/README.md); for one protein end
to end see [`benchmarks/WALKTHROUGH.md`](../benchmarks/WALKTHROUGH.md). Everything here
can be recomputed with `python benchmarks/rescore.py` from
`benchmarks/results/talosn_clean/per_residue.csv`.

Contents

1. What was measured
2. Definitions, with formulas
3. The test set and the leak-safe split
4. The FACET-vs-TALOS-N benchmark (main table)
5. The missing-atom ablation
6. Reproducing the record through the public package
7. A known defect in the ground truth, and the numbers without it
8. Every README table and where it comes from
9. Glossary

---

## 1. What was measured

Backbone torsion angles φ/ψ predicted from backbone chemical shifts, compared per
residue with the angles in the deposited structure, on proteins the model never saw.
Two predictors: **FACET** (this package, retrieval mode, default settings) and
**TALOS-N** (Shen & Bax 2013, version 4.21 Rev 2016.343.11.31). Both were given the
same input files.

Settings of record (`benchmarks/results/talosn_clean/summary.json`):

| item | value |
|---|---|
| FACET weights | `facet_v3.pt`, SHA-256 `72efc688…` — the shipped checkpoint |
| Retrieval index | `facet_retrieval_index.npz`, `index_version v0.2.1`, 219,713 residues — the shipped index |
| Retrieval | k = 25 nearest neighbours by cosine similarity of the 128-d embedding |
| Clustering | DBSCAN over neighbour φ/ψ, eps = 30°, min cluster size = 3 |
| TALOS-N invocation | `TALOSN -in <id>.tab -talosnDir $TALOSN_DIR -noauto -noverb` |
| Input | `benchmarks/data/inputs/<id>.tab` (NMRPipe table: H, HA, N, CA, CB, C where assigned) |

## 2. Definitions, with formulas

**Angular error (main benchmark).** For a residue with predicted (φ̂, ψ̂) and true (φ, ψ),
in degrees:

```
Δφ = |φ̂ − φ| wrapped into [0°, 180°]        (i.e. min(d, 360° − d))
Δψ = |ψ̂ − ψ| wrapped into [0°, 180°]
error = sqrt( (Δφ² + Δψ²) / 2 )
```

This is the root-mean-square of the two wrapped deltas. For a residue that is wrong in
one angle only, error = Δ/√2; for a residue equally wrong in both, error = Δ.

**Angular error (missing-atom ablation only).** `error = (Δφ + Δψ) / 2`, the arithmetic
mean of the same two deltas. The two formulas agree when Δφ = Δψ and differ by at most
√2 otherwise; ablation numbers are therefore **not** comparable with main-benchmark
numbers and are never placed in the same table.

**fail25.** Share of scored residues with error > 25°. Chosen because ±20–25° is the
width of a dihedral restraint a structure calculation would accept from a High-tier
prediction.

**Median / mean / p90.** Over the paired residues (below). Medians are the headline
because ~13 % of residues have errors above 90° that are dominated by ground-truth
problems (§7), and a mean is hostage to them.

**Coverage.** Share of scored residues for which a method emits any φ/ψ at all. TALOS-N
returns 9999 for residues it declines (`None` class); FACET always returns an angle but
tiers it.

**Paired residues.** Residues for which *both* methods emit a prediction. Every
head-to-head number — medians, fail25, per-SS, rigid/flexible, win rate — is computed
on this set, so abstaining on hard residues cannot improve a method's score.

**Win rate.** Share of paired residues on which one method's error is strictly lower
than the other's. The two win rates plus ties sum to 100 %.

**Ground truth.** φ/ψ computed from the deposited PDB coordinates paired with the BMRB
entry. For multi-model NMR ensembles the per-residue *circular mean* over models is
used (`atan2` of the mean sine and cosine), and the *circular variance*
`V = 1 − R` (R = mean resultant length, V ∈ [0, 1]) is stored per angle. A residue is
`well_defined` when V ≤ 0.2 for both angles (≈ ±25° spread); a residue is scored only
if its truth is finite and `well_defined` — all 55,036 scored residues satisfy this.

**Spread and rigid / flexible.** The CSV reports `phi_true_spread` and
`psi_true_spread` = `degrees(sqrt(V))`, the convention the benchmark used as a spread
proxy (it treats V as if it were a variance in rad²; for small V it is close to the
circular standard deviation). A residue is *flexible* when this spread exceeds 10° for
φ **or** ψ — equivalently V > 0.0305 — otherwise *rigid*. Single-model entries have
V = 0 and are always rigid. 6,939 of the paired residues are flexible.

**Secondary structure.** Three-state labels taken from the helix and sheet annotations
in the deposited mmCIF (the structure authors' `HELIX`/`SHEET`-equivalent records, read
with gemmi from the first model): H for residues inside an annotated helix, E for
residues inside an annotated strand, C for everything else. These are not DSSP
assignments. Stored per residue as `ss` in the CSV.

**Tiers.** FACET's confidence label per residue, from retrieval cluster agreement:

| internal name | public name | rule (k = 25 neighbours) |
|---|---|---|
| `Strong` | **High** | one cluster holding ≥ k − 2 = 23 neighbours |
| `Generous` | **Medium** | largest cluster holds ≥ 10 neighbours and is ≥ 2× the second |
| `Ambiguous` | **Low** | two clusters, the largest < 2× the second |
| `None` | **Flexible** | no cluster of size ≥ 10 |

The Ambiguous test runs first, so two competing clusters can never be reported as
Medium. Both names appear in the CSV (`facet_tier`, `facet_tier_public`).

**TALOS-N classes.** `Strong`, `Generous` (consensus among database matches, usable),
`Warn` (no consensus), `Dyn` (RCI-S² says the residue is dynamic), `None` (no
prediction). Only `None` affects coverage; all other classes are scored.

## 3. The test set and the leak-safe split

**Population.** 4,970 BMRB entries with a paired PDB structure, 448,819 residues after
curation (`docs/DATA.md`).

**Clustering.** Each entry's sequence is turned into a normalised 8,000-dimensional
3-mer frequency vector; two entries are linked if the cosine similarity of these vectors
is ≥ 0.5; clusters are the **connected components** of that graph (3,741 clusters).

**Assignment.** Clusters are shuffled (seed 42) and assigned whole to test until the
test set reaches 15 % of entries, then to validation until 15 %, the rest to train:
**3,480 train / 745 validation / 745 test entries.** The lists are in
`benchmarks/data/`. The retrieval index (219,713 rows from 3,439 train entries; 41
train entries contribute no rows after quality filtering) and the mask-safe shift
reference (3,470 entries) are built from the train split only.

**The 2026-04-17 correction.** The split originally used complete-linkage clustering,
which cuts a chain of pairwise-similar sequences into several clusters that can land on
both sides of the split. A leakage gate that checks test entries against the index
caught it; the linkage was changed to connected components, the model was retrained,
the index rebuilt and TALOS-N re-run on the new test set. Every number in this
repository is from after that correction. (Earlier internal numbers — e.g. a 12.15°
FACET median — were measured on the leaky split and must not be cited.)

**Verification.** `python benchmarks/check_leakage.py` intersects the test and
ablation ID lists with the BMRB entries recorded in both shipped reference files:
0 / 745 and 0 / 39 in each.

## 4. The FACET-vs-TALOS-N benchmark

**Scored residues.** 55,036 from 740 of the 745 test entries (three entries have no
finite ground truth; two are 6–8-residue peptides that yielded no scored window).
FACET emitted a prediction for 54,610 (99.2 %), TALOS-N for 54,260 (98.6 %); **53,841
are paired**.

**Result** (`python benchmarks/rescore.py`):

| Metric | TALOS-N | FACET |
|---|---|---|
| All-residue median | 13.57° | 12.65° (−0.92°) |
| fail25 | 29.6 % | 27.4 % |
| Mean | 28.9° | 27.4° |
| p90 | 102.7° | 96.4° |
| Coil median (n = 17,917) | 22.8° | 20.9° (−1.92°) |
| Helix median (n = 24,099) | 8.5° | 8.0° (−0.46°) |
| Strand median (n = 11,825) | 15.3° | 14.6° (−0.72°) |
| Rigid (n = 46,902) | 12.6° | 11.8° |
| Flexible (n = 6,939) | 20.5° | 18.7° |
| Win rate | 46.9 % | 53.1 % (6 ties) |

**Tier calibration** (median/fail25 over paired residues in the tier; coverage over
all scored residues):

| Tier | Coverage | Median | fail25 |
|---|---|---|---|
| High | 76.4 % (42,039) | 10.7° | 19.6 % |
| Medium | 18.3 % (10,099) | 24.5° | 49.3 % |
| Low | 3.7 % (2,056) | 65.0° | 70.8 % |
| Flexible | 1.5 % (842, of which 416 carry an angle) | 80.4° | 82.5 % |

The README rounds a few of these differently (TALOS-N fail25 29.65 → 29.7 %, FACET p90
96.45 → 96.5°, High median 10.65 → 10.6°, Low fail25 70.79 → 71.0 %) and quotes the
Flexible row as 0.8 % coverage — the share of Flexible residues that still carry an
angle. The underlying values are identical; `rescore.py` prints them unrounded with
`--json`.

**Per protein.** `per_protein.csv` gives one row per entry. On the 708 entries with at
least ten paired residues, FACET's median is lower on 68 %
(`benchmarks/figures/per_protein_scatter.png`).

**Which residues are paired.** TALOS-N's `Strong`, `Generous`, `Warn` and `Dyn` classes
all carry an angle and are all scored (45,492 / 2,005 / 4,818 / 1,945 of the paired
residues); only `None` (776 residues) is a non-prediction. FACET's `None` tier is scored
whenever an angle was emitted (416 residues) and counted as a non-prediction when the
harness left it blank (426). 16 of the 740 entries (532 residues) have `None` on every
residue because TALOS-N produced no output file for them — re-running one of them
(BMRB 16038) reproduces a crash during "ANN prediction" — so the paired set spans
**724 entries**. Those 532 residues are counted against TALOS-N's coverage (98.6 %);
its genuine per-residue refusals are 244 (0.4 %). Fifteen of the 16 sequences contain
non-standard residues (`X`), which FACET skips; 51 of the 745 inputs contain one, and
TALOS-N handled the other 35.

### 4.1 Statistical uncertainty

`rescore.py --bootstrap` resamples **proteins** with replacement (2,000 draws, seed 0),
carrying each protein's residues together, so the intervals respect the fact that
residues within a protein are not independent. FACET minus TALOS-N, benchmark of
record:

| Quantity | Point | 95 % CI |
|---|---|---|
| Δ all-residue median | −0.92° | [−1.07, −0.71] |
| Δ fail25 | −2.27 points | [−2.60, −1.95] |
| Δ helix median | −0.46° | [−0.57, −0.33] |
| Δ strand median | −0.72° | [−0.95, −0.49] |
| Δ coil median | −1.92° | [−2.49, −1.40] |
| FACET win rate | 53.1 % | [52.6, 53.7] |

Residue-level: FACET is closer on 28,606 of 53,835 non-tied residues (sign test
p = 5 × 10⁻⁴⁸; Wilcoxon signed-rank p = 3 × 10⁻⁷⁶). The mean paired difference is
−1.53°, the median paired difference −0.30°: the advantage is small per residue and
consistent across proteins rather than large anywhere. Intervals for the corrected
benchmark are in §7.

### 4.2 Restricting both methods to their confident classes

A TALOS-N user takes `Strong` (and sometimes `Generous`) residues and discards the rest.
The equivalent FACET set is High (+ Medium). Restricting **both** methods to those
classes on the same residues, benchmark of record:

| Residues | n | TALOS-N median / fail25 | FACET median / fail25 | FACET wins |
|---|---|---|---|---|
| TALOS-N Strong+Generous ∩ FACET High+Medium | 46,299 | 11.95° / 23.7 % | 11.40° / 22.4 % | 52.9 % |
| TALOS-N Strong ∩ FACET High | 38,975 | 10.82° / 19.4 % | 10.30° / 18.5 % | 53.2 % |

Each method on its *own* confident subset (different residues, so not a head-to-head):
TALOS-N Strong covers 84.1 % of paired residues at 11.84° / 23.4 % fail25; FACET High
covers 77.2 % at 10.65° / 19.6 %. TALOS-N Strong+Generous covers 87.7 % at 12.14° /
24.6 %; FACET High+Medium covers 95.5 % at 12.17° / 25.3 %. FACET's High tier is
tighter than TALOS-N's Strong class; its Medium tier is looser than TALOS-N's
Generous class and covers more.

### 4.3 Fairness to TALOS-N

* **Version and invocation.** TALOS-N 4.21 (Rev 2016.343.11.31), Linux x64 binary,
  run once per protein with
  `TALOSN.linux9_x64 -in <entry>.tab -talosnDir /opt/talosn -noauto -noverb` and no
  other options. `-noauto` only disables TALOS-N's input-format auto-detection (the
  inputs are already NMRPipe tables); its default `-autoExcl` — exclude database
  proteins whose sequence is identical to the input — was left on. Chemical shifts were
  given to both methods as deposited, without re-referencing; FACET's referencing check
  was likewise off (`auto_reference=False`) for the record.
* **TALOS-N's database.** The distribution's `talos.obsCS.tab` lists its reference
  proteins by BMRB accession: 498 entries
  (`benchmarks/data/talosn_database_bmrb_ids.txt`). **9 of the 745 test proteins are in
  it** (5868, 6589, 10092, 10137, 15013, 15038, 15303, 15541, 15913; 651 paired
  residues, 1.2 %); they favour TALOS-N unless `-autoExcl` removed them. On those 651
  residues the two methods tie (clean truth: TALOS-N 12.52° vs FACET 12.94°, FACET wins
  52.0 %); removing them leaves the headline at 12.57° vs 13.50°. 47 train and 9
  validation entries are also in TALOS-N's database, which does not affect the
  comparison. The test set was never selected to be held out from TALOS-N's database;
  the split is homology-aware with respect to FACET's training data only.
* **Ground truth provenance.** 12.8 % of paired residues (62 entries) have a
  single-model X-ray structure as truth; the remainder are NMR ensembles, some of which
  will have been refined with TALOS-family dihedral restraints. That favours TALOS-N on
  those entries by an amount that cannot be quantified from the deposition records
  alone.
* **Failures counted as abstentions.** The 16 entries above are counted as TALOS-N
  non-predictions rather than dropped; dropping them instead changes no head-to-head
  number, because unpaired residues are not scored.

## 5. The missing-atom ablation

**Question.** When HA, H or CB are unassigned — perdeuterated samples, incomplete
assignments — how well does the mask-safe shift-retrieval fallback (opt-in since 0.4.0,
`mask_safe_fallback=True`; see §6 for why it is no longer the default) predict φ/ψ?

**Design** (`benchmarks/results/coverage_ablation/results.json`):

* Reference database: 316,445 residues from the train split, full coverage.
* Queries: the alphabetically first test entries up to 4,000 residues — **39 entries,
  4,019 residues, 3,404 scored** (finite and `well_defined` truth). List in
  `benchmarks/data/ablation_set_39.txt`.
* Six scenarios: `full`, `−HA`, `−H`, `−H−HA`, `−CB`, `−H−HA−CB`. In each, the named
  columns are zeroed *in the query only* (value and mask); the database is untouched.
* Two predictors: the training harness's reference retrieval (hierarchical database,
  merged query, k = 25) and the shipped `facet.masked_retrieval` module loaded from
  `facet_shift_reference.npz`.
* Error: `(Δφ + Δψ)/2` (§2) — **not** the RMS form of §4.

**Result** — shipped module (`masked_module` rows):

| available shifts | median | fail25 |
|---|---|---|
| all backbone shifts | 13.6° | 22.2 % |
| HA absent | 13.6° | 23.6 % |
| H absent | 13.6° | 23.3 % |
| H and HA absent | 13.4° | 22.8 % |
| CB absent | 14.1° | 23.7 % |
| H, HA and CB absent | 14.1° | 25.7 % |

**The scorer fix, told once.** The shipped module originally ranked candidates by mean
squared z-distance over the shared atom columns with a floor of two shared columns, so
a candidate sharing two columns with the query was judged on two numbers and won far
too often. Measured at `−H−HA`, 88 % of queries were won by a candidate covering under
40 % of the query, and the Spearman correlation between rank-1 distance and rank-1 error
was +0.007. Ranking by query coverage first, then distance, gave:

| scenario | before fix | after fix |
|---|---|---|
| all shifts | 15.99° | 13.61° |
| −HA | 18.15° | 13.64° |
| −H | 18.00° | 13.57° |
| −H−HA | 23.87° | 13.42° |
| −CB | 17.73° | 14.05° |
| −H−HA−CB | 24.29° | 14.08° |

The `retrieval` rows in `results.json` (15.6°, 17.8°, 17.9°, 22.9°, 17.4°, 24.2°) are
the training-harness reference implementation, which still uses the old scorer; they
are kept as the baseline the fix was measured against. A separate, never-tuned
validation split ordered the scenarios monotonically before and after, and every paired
bootstrap confidence interval on the improvement excluded zero. `DATA_PROVENANCE.md`
records the same episode from the data side.

## 6. Reproducing the record through the public package

`per_residue.csv` was produced inside the training harness: the same weights and index
as shipped, but batched over the curated shift arrays. `benchmarks/run_talosn_comparison.py`
re-runs the whole benchmark through `facet.predict()` on the `.tab` inputs and writes
`per_residue_rerun.csv` with FACET's angles. This section records how close the two
are — and the one thing that had to change in the package to make them close.

### 6.1 The 0.3.1 default did not reproduce the record

Version 0.3.1 introduced the mask-safe shift-retrieval fallback (§5) and routed
**every residue without an HA shift** to it. 21 % of the benchmark's residues have no
HA. Re-running the benchmark with that default (`per_residue_rerun_fallback_on.csv`)
gave FACET 13.03° vs TALOS-N 13.65° (record: 12.65° vs 13.57°), only 79 % of residues
within 0.5° of the recorded error, and 7 points fewer High-tier residues. Splitting the
residues by whether HA is present shows where the loss came from (clean-truth, paired
residues; "record" is the harness run, which predates the fallback and therefore never
used it):

| residues | n | record: median / fail25 / High share | fallback on: median / fail25 / High share |
|---|---|---|---|
| all | 47,356 | 10.88° / 17.2 % / 0.78 | 11.18° / 19.0 % / 0.72 |
| HA present | 40,318 | 10.81° / 16.0 % / 0.82 | 10.97° / 17.3 % / 0.80 |
| HA missing | 7,038 | 11.53° / 24.1 % / 0.60 | 12.91° / 28.9 % / 0.31 |
| HA missing, whole protein (49 entries) | 4,138 | **9.55°** / 18.0 % / 0.65 | 10.68° / 22.7 % / 0.38 |
| HA missing, sporadic gaps | 2,900 | 15.21° / 33.0 % / 0.54 | 17.45° / 37.8 % / 0.21 |

The default embedding-retrieval path is better than the fallback on every slice —
including the 49 proteins that have no HA at all, the case the fallback was built for
(on those, TALOS-N scores 10.14° and FACET's default path wins 54.9 % of residues). The
ablation of §5 strips HA from *every query residue* and compares against the parametric
head alone; real assignments with missing HA are handled by the encoder's input
masking, and the retrieval over embeddings does not collapse. The fallback is
therefore **opt-in from 0.4.0** (`mask_safe_fallback=False`; CLI `--mask-safe-fallback`).

### 6.2 Agreement with the record under the 0.4.0 default

`per_residue_rerun.csv` (fallback off; 54,594 residues with a recorded FACET error —
one single-residue entry, 6858, cannot be run through the package):

| FACET error, public path vs record | share of residues |
|---|---|
| within 0.5° | 96.5 % |
| within 2° | 97.3 % |
| within 5° | 98.1 % |
| differs by more than 20° | 1.3 % |
| identical tier | 96.5 % |

High-tier counts: 41,043 (public path) vs 42,039 (record). The remaining 3.5 % of
residues differ because the public path parses the `.tab` file and converts shifts
itself, and because it always emits an angle (the harness left 426 residues without
one, and the public path labels 2,740 residues Flexible against the record's 842). This
residual is small but real, and is left as an open item.

### 6.3 The public path scored as its own benchmark

`rescore.py --csv per_residue_rerun.csv --rerun`. The public path always emits an
angle, so the paired set is TALOS-N's 54,260 residues rather than the record's 53,841:

| Metric | TALOS-N | FACET (public path, 0.4.0) | FACET (record, §4) |
|---|---|---|---|
| All-residue median | 13.65° | 12.80° (−0.85°) | 12.65° (−0.92°) |
| fail25 | 29.9 % | 28.3 % | 27.4 % |
| Mean | 29.1° | 28.5° | 27.4° |
| p90 | 103.0° | 102.1° | 96.4° |
| Coil / Helix / Strand median | 23.0 / 8.5 / 15.4° | 21.3 / 8.1 / 14.7° | 20.9 / 8.0 / 14.6° |
| Win rate | 47.5 % | 52.5 % | 53.1 % |
| High tier: coverage, median, fail25 | — | 74.6 %, 10.6°, 19.4 % | 76.4 %, 10.7°, 19.6 % |

Users of the package get the public-path numbers. The margin over TALOS-N is
preserved (−0.85° vs −0.92° on the median, every per-SS median, the head-to-head
count); the tail is heavier because the public path emits an angle for residues the
harness left blank.

## 7. The ground-truth units defect, its extent, and the corrected benchmark

**What is wrong.** 6,978 scored residues in 63 entries have φ/ψ "degrees" within ±3.15°
of zero — exactly π·π/180, a radian value converted to radians a second time. All 63
are single-model (X-ray) structures, and **every** single-model residue in the test set
is affected (6,869 of 6,869 paired residues with X-ray truth).

**Validation of the correction.** Multiplying the stored values by 180/π was checked
against φ/ψ recomputed independently with gemmi from the deposited mmCIF files of six
of the affected entries (1,648 residues, all their residues; PDB 3P62, 4JCC, 1GUB,
1ECE, 1RTC, 3OOI): median absolute deviation **0.02°**, 100 % of residues within 1°,
against 82–120° for the values as stored. The correction is right.

**How far it reaches.** The same double conversion sits in the single-model path of the
data pipeline, so it affects every X-ray-truth entry in the dataset, not only the test
set: 442 entries, 50,983 residues; **36,237 training residues (11.7 % of the finite
training targets, all marked well-defined)**, 7,671 validation residues, 105 of the
3,404 residues scored in the missing-atom ablation (§5), 2,334 of the 219,713 rows of
the shipped retrieval index (1.1 %; the 0.2.0 clean-up removed the rows at exactly
zero but not these), and 36,237 of the 310,923 rows of the shipped shift reference
(11.7 %). The consequences for the *model* — training on ~12 % near-(0°, 0°) targets,
and an opt-in fallback whose reference is 12 % corrupt — are recorded in
`docs/LIMITATIONS.md`; the pipeline fix, retraining and rebuilt assets are the next
release. This section deals with the *benchmark*.

**Corrected tables.** `benchmarks/build_corrected_truth.py` writes two files:

* `per_residue_corrected.csv` — the benchmark of record with the flagged truth
  rescaled. TALOS-N's error on those rows is recomputed from its stored angles. The
  record stores no FACET angles, so FACET's error on the flagged rows comes from the
  public-path re-run (§6; 96.5 % agreement with the record elsewhere); the column
  `facet_err_source` marks each row `record` or `rerun`. Unflagged rows are untouched.
* `per_residue_rerun_corrected.csv` — the public-path re-run with the flagged truth
  rescaled and both errors recomputed. Single-source; what the released package
  reproduces.

`rescore.py --csv <file> [--rerun] --bootstrap`:

| Metric | TALOS-N | FACET, corrected record | FACET, corrected public path |
|---|---|---|---|
| Paired residues | | 53,876 | 54,260 |
| All-residue median | 11.51° / 11.58° | 11.03° (−0.48°) | 11.18° (−0.40°) |
| fail25 | 20.1 % / 20.3 % | 18.4 % | 19.6 % |
| Mean | 21.2° / 21.3° | 20.5° | 21.8° |
| p90 | 45.3° / 46.1° | 41.8° | 52.5° |
| Coil / Helix / Strand median | 19.2 / 7.4 / 13.5° | 18.3 / 7.1 / 12.9° | 18.7 / 7.1 / 12.9° |
| Win rate | 47.0 % / 47.8 % | 53.0 % (7 ties) | 52.2 % (7 ties) |
| High tier: coverage, median, fail25 | — | 76.3 %, 9.3°, 10.1 % | 74.6 %, 9.2°, 9.9 % |
| Medium tier | — | 18.1 %, 19.3°, 39.0 % | 17.2 %, 19.3°, 38.8 % |

(TALOS-N's two columns differ only in the paired set.) Bootstrap over proteins, 95 %
CIs, FACET minus TALOS-N:

| Quantity | corrected record | corrected public path | NMR-truth only (`--exclude-suspect`) |
|---|---|---|---|
| Δ median | −0.48° [−0.64, −0.37] | −0.40° [−0.54, −0.27] | −0.62° [−0.74, −0.50] |
| Δ fail25 | −1.71 [−2.08, −1.37] | −0.73 [−1.10, −0.40] | −2.20 [−2.54, −1.85] |
| Δ helix / strand / coil median | −0.30 / −0.68 / −0.92° | −0.27 / −0.64 / −0.62° | −0.32 / −0.73 / −1.23° |
| FACET win rate | 53.0 % [52.4, 53.7] | 52.2 % [51.6, 52.9] | 53.8 % [53.2, 54.3] |

Every interval excludes zero. On the 63 corrected entries alone (X-ray truth) FACET is
*behind*: 12.19° vs 11.65°, win rate 47.8 % (public path). Two readings are consistent
with that: the model's X-ray training targets were corrupt (above), and solution
shifts against crystal coordinates are a slightly different task from solution shifts
against a solution ensemble. Either way, the corrected margin over TALOS-N is about
half the record's — roughly −0.4 to −0.5° on the median, 1–2 points of fail25, 52–53 %
of residues — and it is statistically unambiguous.

**What to cite.** The record (§4) is what was measured, and is kept because the
per-residue rows are what every other number here is derived from; its absolute errors
are inflated by the defect and its margin over TALOS-N overstated by ~0.4°. The
corrected record is the honest single table; the corrected public path is what
`pip install facet-nmr` reproduces. Do not cite the record's absolute medians without
the caveat.

## 8. Every README table and where it comes from

| README table | source file | command |
|---|---|---|
| "Clean 745-protein φ/ψ test benchmark" (median, fail25, mean, p90, per-SS, rigid/flexible, win rate) | `benchmarks/results/talosn_clean/per_residue.csv` | `rescore.py` |
| Coverage 99.2 % / 98.6 %, 53,841 paired | same | `rescore.py` (first line) |
| Tier calibration (High/Medium/Low/Flexible) | same | `rescore.py` |
| Missing-atoms table (13.6°, 13.6°, 13.6°, 13.4°, 14.1°, 14.1°) | `benchmarks/results/coverage_ablation/results.json`, `masked_module.overall.median` | `rescore.py` (last table) |
| "49 proteins with no HA: 9.6° vs 10.1°" | `per_residue.csv` restricted to entries whose inputs have no HA column, clean truth | §6.1 |
| Ground-truth caveat (10.88° vs 11.50°; 11.18° vs 11.58°) | `per_residue.csv` with `--exclude-suspect`; `per_residue_rerun.csv` with `--rerun --corrected-truth` | `rescore.py` |
| "Public path reproduces the record on 96.5 % of residues; 12.80° vs 13.65°" | `per_residue_rerun.csv` | `rescore.py --rerun`; §6.2 |
| Bootstrap CIs, sign / Wilcoxon tests | any of the above | `rescore.py --bootstrap`; §4.1, §7 |
| Corrected-truth tables (11.03° vs 11.51°; 11.18° vs 11.58°) | `per_residue_corrected.csv`, `per_residue_rerun_corrected.csv` | `build_corrected_truth.py`, then `rescore.py`; §7 |
| Confidence-restricted comparison (11.40° vs 11.95° on S+G ∩ H+M) | `per_residue.csv`, columns `talosn_class`, `facet_tier` | §4.2 |
| TALOS-N database overlap (9 / 745) | `benchmarks/data/talosn_database_bmrb_ids.txt` | §4.3 |
| Before/after scorer-fix table in `DATA_PROVENANCE.md` | §5 above; `after` = the same `results.json` | — |
| Test-set identity | `benchmarks/data/test_set_745.txt` | `check_leakage.py` |

## 9. Glossary

**Secondary shift.** Observed chemical shift minus the random-coil value for that
amino acid (Wishart/Schwarzinger tables). The model's input; sign and size encode
secondary structure.

**Well-defined residue.** In an NMR ensemble, a residue whose φ/ψ agree across models
closely enough that a single "true" value is meaningful (circular variance ≤ 0.2 for
both angles); only these are scored.

**Ensemble mean / spread.** Circular mean of φ (and ψ) across the models of a
deposited ensemble, and `degrees(sqrt(circular variance))` as the spread proxy; the
spread defines *flexible* (§2).

**Geometry classes (NG / XG / D / XU / C).** Labels from the data curation describing
how the structure was determined and how trustworthy the local geometry is
(NMR-good, X-ray-good, disordered, X-ray-unclear, conflicting). FACET v3 was trained on
NG + XG residues; see `docs/DATA.md`.

**Retrieval index.** 219,713 train-split residues, each stored as the encoder's 128-d
embedding plus its true φ/ψ, three-state secondary-structure label and Ramachandran basin. A query residue's
embedding is compared with all of them by cosine similarity.

**DBSCAN cluster agreement.** The φ/ψ of the 25 nearest neighbours are clustered
(DBSCAN, eps 30°, min 3). One big cluster = the neighbours agree = High tier; two
competing clusters = Low; none = Flexible. The prediction is the circular mean of the
top cluster.

**Tier.** FACET's per-residue confidence label (High / Medium / Low / Flexible), §2.

**Basin.** One of four Ramachandran regions: α_R, β, PPII, other. Neighbour counts per
basin give the *basin populations* reported for each residue.

**PPII.** Polyproline-II, the left-handed extended conformation around (−75°, +145°);
common in disordered regions and distinguished from β by φ.

**fail25.** Share of residues with error > 25° (§2).

**Paired residues.** Residues both methods predict (§2).
