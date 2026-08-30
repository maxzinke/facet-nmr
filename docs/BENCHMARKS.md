# Benchmarks: protocol, definitions and sources

This document is the reference behind every number in the README. For a plain-language
tour start with [`benchmarks/README.md`](../benchmarks/README.md); for one protein end
to end see [`benchmarks/WALKTHROUGH.md`](../benchmarks/WALKTHROUGH.md). Everything here
can be recomputed with `python benchmarks/rescore.py` from
`benchmarks/results/talosn_clean/per_residue.csv` — the released package's own
predictions on the corrected ground truth.

Contents

1. What was measured
2. Definitions, with formulas
3. The test set and the leak-safe split
4. The FACET-vs-TALOS-N benchmark (main table)
5. The missing-atom ablation
6. The record is the released package; the 0.3.1 fallback history
7. The ground-truth units defect and how it was fixed
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
| FACET weights | `facet_v3.pt`, SHA-256 `d117fd39…` — the shipped checkpoint (retrained 2026-08-30 on the repaired truth) |
| Retrieval index | `facet_retrieval_index.npz`, `index_version v0.3.0`, 253,573 residues — the shipped index |
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
because ~6 % of residues have errors above 90° — loops and turns where the deposited
conformation and the solution shifts genuinely disagree — and a mean is hostage to
them.

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
if its truth is finite and `well_defined` — all 55,032 scored residues satisfy this.
The truth of the 63 X-ray entries carries the validated ×180/π correction of §7
(`truth_corrected = 1` in the CSV).

**Spread and rigid / flexible.** The CSV reports `phi_true_spread` and
`psi_true_spread` = `degrees(sqrt(V))`, the convention the benchmark used as a spread
proxy (it treats V as if it were a variance in rad²; for small V it is close to the
circular standard deviation). A residue is *flexible* when this spread exceeds 10° for
φ **or** ψ — equivalently V > 0.0305 — otherwise *rigid*. Single-model entries have
V = 0 and are always rigid. 7,083 of the paired residues are flexible.

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
`benchmarks/data/`. The retrieval index (253,573 rows from 3,458 train entries; 22
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

**Scored residues.** 55,032 from 739 of the 745 test entries (three entries have no
finite ground truth, two are 6–8-residue peptides that yield no scored window, and one
is a single assigned residue, which the package cannot window). FACET emits a
prediction for every scored residue (100 % coverage — it tiers instead of abstaining);
TALOS-N for 54,260 (98.6 %), so **54,260 residues are paired**.

**Result** (`python benchmarks/rescore.py`; produced by running the released package
with default settings on `benchmarks/data/inputs/`, scored against the corrected
truth):

| Metric | TALOS-N | FACET |
|---|---|---|
| All-residue median | 11.58° | 10.91° (−0.67°) |
| fail25 | 20.3 % | 18.0 % |
| Mean | 21.3° | 19.9° |
| p90 | 46.1° | 40.3° |
| Coil median (n = 18,221) | 19.3° | 18.0° (−1.30°) |
| Helix median (n = 24,157) | 7.4° | 7.0° (−0.35°) |
| Strand median (n = 11,882) | 13.6° | 12.8° (−0.81°) |
| Rigid (n = 47,177) | 10.7° | 10.1° |
| Flexible (n = 7,083) | 20.5° | 18.9° |
| Win rate | 46.2 % | 53.8 % (4 ties) |

**Tier calibration** (median/fail25 over paired residues in the tier; coverage over
all scored residues):

| Tier | Coverage | Median | fail25 |
|---|---|---|---|
| High | 76.3 % | 9.3° | 9.9 % |
| Medium | 15.8 % | 19.9° | 40.2 % |
| Low | 3.2 % | 59.8° | 67.8 % |
| Flexible | 4.6 % | 20.8° | 43.0 % |

The Flexible tier's median is lower than Low's: it holds residues with *no* coherent
neighbour cluster, many of which are genuinely averaging conformations near a basin
centre, whereas Low holds residues caught between two competing conformations. Both
stay out of restraint files.

**Per protein.** `per_protein.csv` gives one row per entry. On the 708 entries with at
least ten paired residues, FACET's median is lower on 69 %
(`benchmarks/figures/per_protein_scatter.png`).

**Which residues are paired.** TALOS-N's `Strong`, `Generous`, `Warn` and `Dyn` classes
all carry an angle and are all scored (45,492 / 2,005 / 4,818 / 1,945 of the paired
residues); only `None` (772 residues) is a non-prediction. 15 of the 739 entries
(525 residues) have `None` on every residue because TALOS-N produced no output file
for them — re-running one (BMRB 16038) reproduces a crash during "ANN prediction",
and 14 of the 15 sequences contain non-standard residues (`X`), which FACET skips —
so the paired set spans **724 entries**. Those 525 residues count against TALOS-N's
coverage; its genuine per-residue refusals are 247 (0.4 %). FACET abstains nowhere:
2,513 paired residues carry its Flexible tier and are scored like any others.

### 4.1 Statistical uncertainty

`rescore.py --bootstrap` resamples **proteins** with replacement (2,000 draws, seed 0),
carrying each protein's residues together, so the intervals respect the fact that
residues within a protein are not independent. FACET minus TALOS-N:

| Quantity | Point | 95 % CI |
|---|---|---|
| Δ all-residue median | −0.67° | [−0.78, −0.56] |
| Δ fail25 | −2.35 points | [−2.66, −2.03] |
| Δ helix median | −0.35° | [−0.44, −0.26] |
| Δ strand median | −0.81° | [−0.99, −0.56] |
| Δ coil median | −1.30° | [−1.57, −0.98] |
| FACET win rate | 53.8 % | [53.3, 54.4] |

Every interval excludes zero. Residue-level: FACET is closer on 29,190 of 54,256
non-tied residues (sign test p = 4 × 10⁻⁷⁰; Wilcoxon signed-rank p = 4 × 10⁻¹⁰⁵). The
advantage is small per residue and consistent across proteins rather than large
anywhere.

### 4.2 Restricting both methods to their confident classes

A TALOS-N user takes `Strong` (and sometimes `Generous`) residues and discards the rest.
The equivalent FACET set is High (+ Medium). Restricting **both** methods to those
classes on the same residues:

| Residues | n | TALOS-N median / fail25 | FACET median / fail25 | FACET wins |
|---|---|---|---|---|
| TALOS-N Strong+Generous ∩ FACET High+Medium | 45,013 | 10.17° / 13.4 % | 9.69° / 11.9 % | 53.8 % |
| TALOS-N Strong ∩ FACET High | 38,984 | 9.35° / 9.9 % | 9.00° / 8.7 % | 53.8 % |

Each method on its *own* confident subset (different residues, so not a head-to-head):
TALOS-N Strong covers 83.8 % of paired residues at 10.20° / 13.5 % fail25; FACET High
covers 76.6 % at 9.29° / 9.9 %. TALOS-N Strong+Generous covers 87.5 % at 10.47° /
14.6 %; FACET High+Medium covers 92.2 % at 10.35° / 15.0 % — more coverage at the
same accuracy.

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
  it** (5868, 6589, 10092, 10137, 15013, 15038, 15303, 15541, 15913; 656 paired
  residues, 1.2 %); they favour TALOS-N unless `-autoExcl` removed them. On those 656
  residues the two methods tie (TALOS-N 10.05° vs FACET 10.17°, FACET wins 52.4 %);
  removing them leaves the headline at 10.92° vs 11.60°. 47 train and 9 validation
  entries are also in TALOS-N's database, which does not affect the comparison. The
  test set was never selected to be held out from TALOS-N's database; the split is
  homology-aware with respect to FACET's training data only.
* **Ground truth provenance.** 12.7 % of paired residues (6,904, 62 entries) have a
  single-model X-ray structure as truth (FACET 11.07° vs TALOS-N 11.65° on them); the
  remainder are NMR ensembles, some of which will have been refined with TALOS-family
  dihedral restraints. That favours TALOS-N on those entries by an amount that cannot
  be quantified from the deposition records alone.
* **Failures counted as abstentions.** The 15 crashed entries above are counted as
  TALOS-N non-predictions rather than dropped; dropping them instead changes no
  head-to-head number, because unpaired residues are not scored.

## 5. The missing-atom ablation

**Question.** When HA, H or CB are unassigned — perdeuterated samples, incomplete
assignments — how well do the parametric model and the mask-safe shift-retrieval
fallback (opt-in since 0.4.0, `mask_safe_fallback=True`; see §6 for why it is not the
default) predict φ/ψ?

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

**Result** (re-run 2026-08-31 with the retrained model, rebuilt reference and
repaired truth; median / fail25 per scenario):

| available shifts | parametric model | mask-safe fallback |
|---|---|---|
| all backbone shifts | 12.5° / 16.6 % | 13.0° / 19.1 % |
| HA absent | 12.5° / 17.1 % | 13.0° / 19.5 % |
| H absent | 12.6° / 18.0 % | 12.9° / 20.6 % |
| H and HA absent | 12.7° / 18.2 % | 12.8° / 19.4 % |
| CB absent | 12.9° / 18.5 % | 13.4° / 21.0 % |
| H, HA and CB absent | 13.2° / 21.6 % | 13.5° / 22.6 % |

**The collapse is gone.** The 0.3-era parametric model degraded badly when HA was
stripped from every residue — the observation that motivated building the fallback and
making it the 0.3.1 default. Retrained on the repaired targets, the parametric model
loses only ~0.2° from full coverage to −H−HA and outperforms the shift-space fallback
in every scenario: the apparent fragility was largely a product of the corrupt
training targets (§7), not of the architecture. The fallback is kept as an opt-in
independent cross-check.

**The scorer fix, told once.** The shipped module originally ranked candidates by mean
squared z-distance over the shared atom columns with a floor of two shared columns, so
a candidate sharing two columns with the query was judged on two numbers and won far
too often. Measured at `−H−HA`, 88 % of queries were won by a candidate covering under
40 % of the query, and the Spearman correlation between rank-1 distance and rank-1 error
was +0.007. Ranking by query coverage first, then distance, gave:

(Both columns of this historical table were measured against the pre-repair truth of
§7; the 105 affected residues inflate the absolute level slightly, evenly on both
sides.)

| scenario | before fix | after fix |
|---|---|---|
| all shifts | 15.99° | 13.61° |
| −HA | 18.15° | 13.64° |
| −H | 18.00° | 13.57° |
| −H−HA | 23.87° | 13.42° |
| −CB | 17.73° | 14.05° |
| −H−HA−CB | 24.29° | 14.08° |

The `retrieval` rows in `results.json` (14.0°, 15.9°, 15.4°, 19.5°, 15.0°, 21.3° in
the 2026-08-31 re-run) are the training-harness reference implementation, which still
uses the old scorer; they
are kept as the baseline the fix was measured against. A separate, never-tuned
validation split ordered the scenarios monotonically before and after, and every paired
bootstrap confidence interval on the improvement excluded zero. `DATA_PROVENANCE.md`
records the same episode from the data side.

## 6. The record is the released package; the 0.3.1 fallback history

Since 0.4.0 there is nothing to bridge between "the benchmark" and "what the package
does": `per_residue.csv` **is** the output of `benchmarks/run_talosn_comparison.py`,
which calls `facet.predict()` with default settings on the published input files —
the same code path as `pip install facet-nmr` + `facet predict`. Re-running it
reproduces the table (the walkthrough protein reproduces its 96 recorded errors to
within 0.5°, 96 of 96).

That was not true of 0.3.x, and the discrepancy is why the mask-safe fallback became
opt-in.

### 6.1 The 0.3.1 fallback default, and why it was retired

Version 0.3.1 introduced the mask-safe shift-retrieval fallback (§5 of METHOD.md) and
routed **every residue without an HA shift** to it — 21 % of the benchmark's residues.
Re-running the 0.3-era benchmark through the then-public path
(`archive/per_residue_rerun_v031_fallback_on.csv.gz`) showed the fallback lost
accuracy on every slice, most on exactly the proteins it was built for; the
0.3-era slice table is preserved in the archive. Re-verified with the released 0.4.0
model and the rebuilt reference on the 75 test proteins that have **no HA at all**
plus 60 sampled controls (`archive/per_residue_v040_s3_fallback.csv.gz`,
`data/s3_fallback_check_ids.txt`), entry-wide HA-missing residues (n = 7,824):

| path | median | High-tier share |
|---|---|---|
| default (embedding retrieval, fallback off) | **9.63°** | 0.73 |
| `--mask-safe-fallback` | 11.03° | 0.45 |
| TALOS-N on the same residues | 10.29° | — |

Sporadic HA gaps (n = 218): 13.43° vs 15.09°. The encoder's input masking handles
missing HA; the ablation of §5 (HA stripped from every query, compared against the
parametric head alone) is a harder case than real data presents. The fallback is
therefore **opt-in** (`mask_safe_fallback=False`; CLI `--mask-safe-fallback`).

### 6.2 The 0.3-era record (historical)

The 0.3-era benchmark of record was produced inside the training harness rather than
through the package. Its tables are preserved in
`benchmarks/results/talosn_clean/archive/` with a README: the uncorrected record
(12.65° vs 13.57° — inflated by the §7 defect on both sides), the corrected record
(11.03° vs 11.51°), and the 0.3-model public-path runs (96.5 % of residues within
0.5° of the harness with the fallback off; 79 % with the 0.3.1 default). None of
those numbers describe the released model; they document the path here.

## 7. The ground-truth units defect and how it was fixed

**What was wrong.** Every single-model (X-ray) structure in the data pipeline had its
φ/ψ converted to radians twice, leaving values within ±3.15° of zero — exactly
π·π/180. In the test set that was 6,978 scored residues in 63 entries (every residue
with X-ray truth); across the pipeline it reached 442 entries and 50,983 residues,
including **36,237 training residues (11.7 % of the finite training targets, all
marked well-defined)**, 7,671 validation residues, 105 of the 3,404 ablation residues,
1.1 % of the 0.3.x retrieval index and 11.7 % of the 0.3.x shift reference. (The
"33,770 sentinel rows" stripped from the index in 0.2.0 were the same defect seen
from the other side.)

**Validation of the correction.** Multiplying the stored values by 180/π was checked
against φ/ψ recomputed independently with gemmi from the deposited mmCIF files of six
affected entries (1,648 residues; PDB 3P62, 4JCC, 1GUB, 1ECE, 1RTC, 3OOI): median
absolute deviation **0.02°**, 100 % of residues within 1°, against 82–120° for the
values as stored. A second, independent in-place repair of the pipeline cache agreed
with this correction to < 0.03° on every benchmark row.

**What was done for 0.4.0.** The truth was repaired at the source, the geometry
labels re-derived (50,981 spuriously well-defined near-origin targets → 0), the model
retrained with the unchanged recipe, the retrieval index rebuilt (253,573 rows,
`index_version v0.3.0` — the stripped X-ray rows are valid again), and the shift
reference rebuilt (near-origin rows 36,239 → 2). The d2D-style SS-population
parameters were refit and came out bit-identical — that fit never ingested the
corrupt rows. Retraining moved the released package's benchmark from 11.18° vs 11.58°
(0.3-era model on corrected truth, archive) to **10.91° vs 11.58°**, and turned the
X-ray-truth entries from FACET's weakest slice (12.19° vs 11.65°, behind) into a
typical one (11.07° vs 11.65°, ahead).

**Residues remain flagged.** The CSV keeps `truth_units_suspect = 1` on the 63
entries' rows so the provenance stays visible; their `phi_true`/`psi_true` are the
corrected values (`truth_corrected = 1` on every row). `--exclude-suspect` scores the
NMR-truth subset alone; do **not** combine `--corrected-truth` with the current table
— that flag exists for the archived, uncorrected files and would rescale already
corrected values.

## 8. Every README table and where it comes from

| README statement | source file | command |
|---|---|---|
| Headline table (median, fail25, mean, p90, per-SS, rigid/flexible, win rate) | `benchmarks/results/talosn_clean/per_residue.csv` | `rescore.py` |
| Coverage 100 % / 98.6 %, 54,260 paired | same | `rescore.py` (first line) |
| Tier calibration (High/Medium/Low/Flexible) | same | `rescore.py` |
| Bootstrap CIs, sign / Wilcoxon tests | same | `rescore.py --bootstrap`; §4.1 |
| Missing-atoms ablation table | `benchmarks/results/coverage_ablation/results.json`, `masked_module.overall.median` | `rescore.py` (last table) |
| "75 proteins with no HA: 9.6° vs 10.3°; fallback 11.0°" | `archive/per_residue_v040_s3_fallback.csv.gz` vs `per_residue.csv` on `data/s3_fallback_check_ids.txt` | §6.1 |
| Confident-class comparison (9.69° vs 10.17° on S+G ∩ H+M) | `per_residue.csv`, columns `talosn_class`, `facet_tier` | §4.2 |
| TALOS-N database overlap (9 / 745) | `benchmarks/data/talosn_database_bmrb_ids.txt` | §4.3 |
| Ground-truth defect history (12.65° → 11.03° → 10.91°) | `archive/` tables + `per_residue.csv` | §6.2, §7 |
| Before/after scorer-fix table in `DATA_PROVENANCE.md` | §5; `after` = the ablation `results.json` | — |
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

**Retrieval index.** 253,573 train-split residues, each stored as the encoder's 128-d
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
