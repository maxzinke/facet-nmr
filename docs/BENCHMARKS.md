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

## 7. A known defect in the ground truth, and the numbers without it

While exporting the per-residue table it became apparent that **6,978 scored residues
in 63 entries have a ground truth that is not in degrees.** Their φ/ψ "degrees" all lie
within ±3.15° of zero — exactly π·π/180, what a radian value becomes when converted to
radians a second time. No real residue has both φ and ψ within 3° of zero. All affected
entries are single-model structures; rescaling their truth by 180/π returns ordinary
Ramachandran values, and TALOS-N's median error on them drops from ~69° to ~12°, so
the residues are ordinary and the stored truth is wrong, not the proteins.

Both methods are scored against the same wrong truth, so the *comparison* survives —
they fail identically on those residues — but the absolute numbers in §4 are inflated
by them. The CSV flags each such residue (`truth_units_suspect = 1`), and
`rescore.py --exclude-suspect` gives the benchmark without them:

| Metric (46,972 paired residues) | TALOS-N | FACET |
|---|---|---|
| All-residue median | 11.50° | 10.88° (−0.62°) |
| fail25 | 19.4 % | 17.2 % |
| Mean | 20.6° | 19.3° |
| p90 | 41.6° | 36.8° |
| Coil median (n = 15,371) | 19.0° | 17.8° |
| Helix median (n = 21,419) | 7.4° | 7.1° |
| Strand median (n = 10,182) | 13.6° | 12.8° |
| Win rate | 46.2 % | 53.8 % |
| High tier (78.2 % of residues) | — | 9.4°, fail25 9.9 % |
| Medium tier (17.1 %) | — | 19.3°, fail25 38.7 % |
| Low tier (3.2 %) | — | 57.2°, fail25 67.2 % |

Excluding them changes the *absolute* level a lot and the *comparison* a little: the
FACET margin goes from −0.92° to −0.62° on the median and the win rate rises from
53.1 % to 53.8 %.

**Scoring the flagged residues properly instead of dropping them.** The record stores
only errors, so its FACET angles cannot be re-scored; the public-path re-run (§6)
stores angles for both methods. `rescore.py --csv per_residue_rerun.csv --corrected-truth`
rescales the flagged truth by 180/π and recomputes both errors from the stored angles:

| Metric (54,260 paired residues, public path, 0.4.0 default) | TALOS-N | FACET |
|---|---|---|
| All-residue median | 11.58° | 11.18° (−0.40°) |
| fail25 | 20.3 % | 19.6 % |
| Mean | 21.3° | 21.8° |
| p90 | 46.1° | 52.5° |
| Coil / Helix / Strand median | 19.3 / 7.4 / 13.6° | 18.7 / 7.1 / 12.9° |
| Win rate | 47.8 % | 52.2 % |

Excluding the flagged residues from the public-path run instead
(`--rerun --exclude-suspect`, 47,356 paired) gives 11.07° vs 11.57° (−0.50°), fail25
18.6 % vs 19.6 %, win rate 52.9 %.

On the corrected entries alone (6,904 paired residues in 62 entries, all single-model
structures) FACET is *behind*: median 12.19° vs 11.65°, win rate 47.8 %, fail25 26.5 %
vs 25.0 %. So the corrected comparison is closer than either the record or the
exclusion suggests: FACET keeps an edge on every median and on the head-to-head
count, and TALOS-N has the lighter tail.

The README's headline table is the benchmark of record and is left as measured; this
section exists so that a reader who notices the ±3° cluster in the CSV knows it was
noticed too, and knows what the benchmark looks like without it. The fix belongs in the
ground-truth builder (and then in a re-measured record), not in the scoring script.

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
