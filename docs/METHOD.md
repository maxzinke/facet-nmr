# FACET — Method

This is the Methods section for FACET, written for a reader who wants to know exactly
what the software computes. Every number below was read from the shipped code, the
shipped weights, or the training records; where a value is quoted from the training
run it is marked as such. File references are relative to this repository. The
training and data-curation code lives in the training repository (not yet public);
[DATA.md](DATA.md) documents what it produced, and [BENCHMARKS.md](BENCHMARKS.md)
documents how the numbers in the README were measured.

## 1. Overview

FACET predicts backbone φ/ψ torsion angles, a three-state secondary-structure label
(H/E/C), a χ1 rotamer class, and several per-residue population readouts from assigned
backbone chemical shifts (¹H^N, ¹Hα, ¹⁵N, ¹³Cα, ¹³Cβ, ¹³C′).

The pipeline has one learned component and several non-parametric components built on
top of it:

1. **A local-biased transformer encoder** (1.29 M parameters) maps a five-residue window
   of secondary shifts to a 128-dimensional embedding of the central residue
   (`facet/model.py`).
2. **Parametric heads** on that embedding predict a 36×36 Ramachandran distribution
   plus a within-bin residual, the SS class, the χ1 class, and an expected error.
3. **Retrieval-augmented φ/ψ** (the default): the embedding is used as a query
   against an index of 219,713 training-residue embeddings; the neighbours' φ/ψ are
   clustered and the dominant cluster becomes the prediction, its agreement becomes
   the confidence tier (`facet/retrieval.py`).
4. **A mask-safe shift-retrieval fallback** (opt-in, `mask_safe_fallback=True`) can
   replace step 3 for residues whose Hα is unassigned, matching directly in
   secondary-shift space with the missing atom dropped from the distance
   (`facet/masked_retrieval.py`). It is off by default — see §6.
5. **Population readouts**: geometric basin populations from the retrieved neighbours,
   kernel-weighted "structural" state populations over the whole index
   (`facet/structural.py`), and a retrieval-free d2D-style Gaussian engine
   (`facet/ss_populations.py`).
6. **Auxiliary physics**: a referencing sanity check (`facet/referencing.py`), a
   deuterium isotope-shift correction (`facet/deuteration.py`) and a Random Coil
   Index S² (`facet/rci.py`).

## 2. Input featurisation

**Nuclei.** Six backbone nuclei in the fixed order `H, HA, N, CA, CB, C`
(`facet/io/formats.py::BACKBONE_NUCLEI`). Every reader (NMRPipe `.tab`, CSV, NEF,
NMR-STAR, BMRB fetch) produces an `(N, 6)` array of ppm values and an `(N, 6)`
observation mask.

**Secondary shifts.** The model never sees raw ppm. Each observed value is converted
to a secondary shift, Δδ = δ_obs − δ_rc(residue type, nucleus), using the
random-coil table in `facet/random_coil.py` (Wishart *et al.* 1995, J. Biomol. NMR 5,
67; Schwarzinger *et al.* 2001, JACS 123, 2970; DSS-referenced, pH 5–7, 25 °C).
Unobserved nuclei are set to 0 and their mask bit to 0.

**Window.** Each residue is predicted from a five-residue window (±2, `half_window=2`)
centred on it. Neighbours are looked up by sequence number, so a gap in the assignment
table becomes a missing neighbour rather than a shifted one. Missing neighbours are
zero-filled and flagged.

**Per-residue features (50 values).** `6` secondary shifts + `6` mask bits + a `32`-d
learned embedding of the residue type (21-token vocabulary: 20 amino acids + padding)
+ `6` binary flags: Gly, Pro, pre-Pro (next residue is Pro), aromatic neighbour
(i±1 ∈ {Phe, Trp, Tyr, His}), aromatic, missing neighbour.

**Input checks before featurisation** (`facet/inference.py::predict`):

* at least 3 residues;
* physical range check per nucleus (H 5–12, Hα 1.5–6.5, N 95–140, Cα 40–75,
  Cβ 15–80, C′ 168–185 ppm). More than 10 % of observed values out of range is a
  hard error (usually a units or column-order mistake); fewer are logged as outliers;
* non-canonical residue types are dropped with a warning and prediction proceeds on the
  standard residues; if fewer than 3 remain the run fails;
* at least one heavy-atom shift (N/Cα/Cβ/C′) is required on more than half of the
  residues — proton-only lists are rejected.

**Referencing check** (`facet/referencing.py`). Run on every input with ≥ 20 residues.
A provisional H/E/C composition is estimated from the chemical-shift-index difference
CSI = Δδ(Cα) − Δδ(Cβ) (helix if CSI > +2 ppm, strand if CSI < −2 ppm), which is
invariant to a common ¹³C offset. The expected mean secondary shift per nucleus is the
composition-weighted sum of canonical per-state means (Cα +3.10/−1.50, Cβ −0.40/+1.90,
C′ +1.80/−1.00, N −2.00/+2.00, Hα −0.30/+0.30 ppm for H/E; coil 0). Nuclei whose
observed mean deviates from that expectation by more than a tolerance
(Cα 0.8, Cβ 0.8, C′ 0.6, N 1.5, Hα 0.35, H 0.35 ppm; ≥ 20 observations required) are
reported with a suggested additive correction. With `auto_reference=True` the
out-of-tolerance corrections are applied before prediction and the check is re-run.
The CSI step makes the check blind to an offset applied to Cα alone (see
[LIMITATIONS.md](LIMITATIONS.md)).

**Deuterium isotope correction** (`facet/deuteration.py`). Applied when
`deuteration` is set. A sample is described by three deuteration fractions
(amide, α, side chain); presets: `protonated` (0, 0, 0), `perdeut_exchanged`
(0, 0.97, 0.97), `perdeut_unexchanged` (0.97, 0.97, 0.97), `ilv_methyl`
(0, 0.97, 0.70). The correction added to each ¹³C shift undoes a one-bond effect of
−0.30 ppm per bonded ²H and a two-bond effect of −0.10 ppm per two-bond ²H (Venters
*et al.* 1996, JACS 118; Hansen 1988, Prog. NMR Spectrosc. 20), counting bonded
hydrogens per residue type (Gly Cα has two; Ala Cβ three; Ile/Thr/Val Cβ one). For
`perdeut_exchanged` this is +0.29 ppm on Cα, +0.10 ppm on C′, and +0.68 ppm on a CH₂
Cβ. The training data received the same correction at build time, so the model always
sees protonated-equivalent shifts.

## 3. Encoder architecture

`facet/model.py::FACETv3`, configuration `FACETv3Config` with `hidden_dim=128`.

| Stage | Specification |
|---|---|
| Input projection | Linear 50 → 128 |
| Local stem | 3 × `ConvBlock`: pre-LayerNorm → two parallel Conv1d(128→128, kernel 5, dilation *d*) combined as `conv(x) · σ(gate(x))` (GLU gating) → Conv1d 1×1 → dropout 0.1 → residual. Dilations 1, 2, 4. |
| Global layers | 2 × [pre-LN multi-head self-attention (4 heads, head dim 32) with rotary position embedding on q and k, dropout 0.1, residual] followed by [pre-LN SwiGLU feed-forward 128 → 512 → 128, residual] |
| Output | LayerNorm; the token at the window centre is the residue embedding **h** ∈ ℝ¹²⁸ |

Rotary embeddings (`RotaryPositionalEmbedding`, base 10 000) give the attention layers
relative position information within the window. Attention is unmasked (bidirectional)
across the five positions.

**Heads** (all operate on **h** unless stated):

| Head | Layers | Output |
|---|---|---|
| SS | Linear 128→64, GELU, Linear 64→3 | H/E/C logits |
| SS gate | Linear 3→32, GELU, Linear 32→128, sigmoid | gate *g* ∈ (0,1)¹²⁸; the torsion head sees **h**_cond = **h** ⊙ (0.5 + *g*) |
| Coarse Ramachandran | Linear 128→128, GELU, Linear 128→1296 (on **h**_cond) | logits over a 36×36 grid of 10°×10° bins |
| Fine residual | Linear 128→6 (on **h**_cond) | (sin, cos) pairs → atan2 → Δφ, Δψ clamped to ±5°; two von Mises concentrations κ = softplus(·) + 0.1, capped at 50 |
| χ1 | Linear 128→64, GELU, Linear 64→3 | g+ / g− / trans logits |
| Expected error | Linear 128→64, GELU, Linear 64→1 | log(1 + expected angular error in degrees); confidence = −(this value) |
| Order (legacy) | Linear 128→1 | trained but unused at inference |
| Angle auxiliary | Linear 128→4 | (sin φ, cos φ, sin ψ, cos ψ) regression, training-only |

**Parameter count**, computed by instantiating the shipped configuration and summing
tensor sizes, is **1,293,378** (encoder 1,078,720; coarse+fine torsion head 184,470;
SS head 8,451; χ1 head 8,451; expected-error head 8,321; SS gate 4,352; the rest
< 1 k). The training record reports the same number. The shipped checkpoint
`facet_v3.pt` contains exactly the 85 tensors of this configuration.

**Parametric decoding** (`FACETv3.predict`): the arg-max coarse bin gives a centre
(φ_c, ψ_c); the fine residual is added; angles are wrapped to [−180°, 180°). SS and
χ1 are arg-max over their logits. This path is used on its own only when the retrieval
index is unavailable (`use_retrieval=False`); its SS and χ1 outputs are used in every
mode.

## 4. Training

Training was run in the training repository with `scripts/train_facet_v3.py`. The
shipped weights are byte-identical (SHA-256) to the checkpoint
`facet_v3_ng_xg_conf_chi1_best.pt` from that run. The values below are taken from the
script defaults and the run record; flags are inferred from the checkpoint name and its
contents (it contains the expected-error head; χ1 class weights are recorded in the
training notes).

| Item | Value |
|---|---|
| Training targets | residues of geometry class NG or XG with a well-defined ensemble angle ([DATA.md §4](DATA.md)); φ/ψ = ensemble circular mean |
| Split | homology-clustered, 3,480 / 745 / 745 proteins train / val / test ([DATA.md §5](DATA.md)) |
| Optimiser | AdamW, lr 3 × 10⁻⁴, weight decay 10⁻⁴ |
| Schedule | cosine annealing over 40 epochs to 10⁻⁶ |
| Batch | 512 windows |
| Gradient clipping | global norm 1.0; batches with non-finite gradients are skipped |
| Dropout | 0.1 in every block |
| Atom dropout | each of the 6 nuclei zeroed with p = 0.10 per window (≥ 2 kept) |
| Context dropout | each non-centre window position zeroed with p = 0.15 |
| Model selection | lowest validation coarse-grid cross-entropy (3.680 for the shipped model) |

**Objective** (`facet/model.py::facet_v3_loss`), summed over valid residues:

```
L = CE_coarse(1296-way, label smoothing 0.01)
  + 0.5 · NLL_fine            (von Mises on the within-bin residual, per angle)
  + 0.3 · CE_SS
  + 0.2 · MSE(sin/cos angle auxiliary)
  + 0.1 · BCE_order            (legacy head)
  + 0.15 · CE_χ1               (inverse-frequency class weights 1.859 / 0.677 / 1.016 for g+ / g− / trans)
  + 0.1 · SmoothL1(expected-error head vs. log(1 + decoded error), target detached)
```

The expected-error target is the RMS angular error of the *decoded* (coarse + fine)
prediction, computed with the gradient detached so the confidence head cannot leak into
the torsion head.

**Held-out result of the parametric model** (training-run record, 55,036 test
residues, RMS-of-(Δφ,Δψ) metric used inside the training script — not the benchmark
metric): median 13.81°, fail25 36.0 %, SS Q3 86.3 %. The retrieval path described next
is what the README numbers refer to; see [BENCHMARKS.md](BENCHMARKS.md) for the metric
definitions and the like-for-like comparison.

## 5. Retrieval-augmented φ/ψ (default mode)

`facet/retrieval.py::FACETRetrieval`.

**Index construction.** For every training-split residue with a defined target angle
(the same NG/XG, well-defined set used as training targets), the encoder embedding
**h** — taken *before* SS conditioning — is stored with the residue's φ, ψ, SS label,
Ramachandran basin, residue type and source entry. The shipped index
(`facet_retrieval_index.npz`, version `v0.2.1`) holds **219,713** rows from
**3,439** BMRB entries: 95,149 H / 50,209 E / 74,355 C by SS label; 113,057 α_R /
62,554 β / 26,393 PPII / 17,709 other by basin. No validation or test protein is in the
index (verified: 0 of the 745 benchmark entries; see BENCHMARKS.md).

**Query.** The query residue's embedding is L2-normalised and compared to all index
rows by cosine similarity; the top *k* = 25 are retrieved.

**Clustering.** The neighbours' (φ, ψ) are clustered with a DBSCAN-style pass:
two points are connected if their angular distance
√(Δφ² + Δψ²) (each difference wrapped to ≤ 180°) is ≤ 30°; connected components with
≥ 3 members are clusters. A cluster whose circular standard deviation exceeds 30° in
either angle is discarded (its members become noise) — such "chained" clusters have a
circular mean near (0°, 0°) that is not a physical conformation. Clusters are sorted by
size; the largest gives the prediction as the circular means of its members' φ and ψ.

**Confidence tier** (checked in this order):

| Condition | Internal label | Public tier |
|---|---|---|
| ≥ 2 clusters and the largest has fewer than twice the members of the second | Ambiguous | Low |
| exactly one cluster with ≥ *k* − 2 = 23 members | Strong | High |
| largest cluster has ≥ 10 members | Generous | Medium |
| otherwise | None | Flexible |

**Consistency guard** (`facet/inference.py`). The encoder's SS prediction is used to
veto a top cluster in an implausible basin: an H-predicted residue must land in α_R,
an E-predicted residue in β or PPII, C accepts any basin. A top cluster within 10° of
the origin in the "other" basin is also vetoed. A vetoed residue falls back to the
parametric arg-max angles and is tiered Flexible, with the clusters reported as
alternatives.

**Per-residue uncertainty.** φ_err and ψ_err are the top cluster's circular standard
deviations, clipped to [3°, 25°]. Residues without an accepted cluster receive a
tier-based bound (see §10).

**Basin populations.** Each index row carries a basin label from `assign_basin`:
α_R φ ∈ [−180, −30), ψ ∈ [−100, 30); β φ ∈ [−180, −90), ψ ∈ [90, 180] ∪ [−180, −150);
PPII φ ∈ [−90, −30), ψ ∈ [90, 180]; everything else "other". The reported
`basin_populations` are the fractions of the *clustered* neighbours in each basin
(all 25 neighbours if no cluster survived). They describe which Ramachandran regions
the retrieved precedents occupy, not a thermodynamic population.

**Alternative clusters and neighbours.** All clusters beyond the first are returned as
(φ, ψ, weight = size/*k*); the five most similar neighbours are returned with entry id,
residue type, φ/ψ, SS, basin and cosine similarity, and all 25 can be exported with
`to_ensemble_csv` / `to_ensemble_json`.

## 6. Mask-safe shift-retrieval fallback (Hα missing)

`facet/masked_retrieval.py`, **opt-in** (`mask_safe_fallback=True`, CLI
`--mask-safe-fallback`) and applied only to residues whose Hα is unobserved.

*Why it exists.* The encoder was trained with Hα present ~80 % of the time. On the
coverage-ablation benchmark, where Hα is stripped from every query, the parametric
head degrades badly and this shift-space retrieval holds full-coverage accuracy
(BENCHMARKS.md §5).

*Why it is off by default.* On the real 745-protein benchmark the default
embedding-retrieval path (§5) is already more accurate on Hα-missing residues than
the fallback — including on the 49 clean-truth proteins that have **no Hα shifts at
all**, where the default path scores 9.6° median (TALOS-N 10.1°) against 10.7° with the
fallback, and keeps 65 % of residues in the High tier against 38 %. Sporadic Hα gaps
inside an otherwise assigned window are handled by the encoder's input masking; the
ablation's whole-protein stripping is a harder case than real data presents. Routing
every Hα-missing residue to the fallback, as version 0.3.1 did, therefore cost
accuracy and High-tier coverage (BENCHMARKS.md §6). The fallback remains available
for inputs where the encoder path visibly fails.

**Reference.** `facet_shift_reference.npz` holds one row per training-split residue with
a defined angle: **310,923** rows from **3,470** depositions. Each row is the
18-dimensional vector of secondary shifts of residues (i−1, i, i+1) with an 18-bit mask,
the three residue types, φ, ψ, SS label, and per-column standard deviations σ_j computed
over observed values.

**Distance.** For a query with mask *m*_q and a candidate with mask *m*_c, the shared
columns are S = {j : m_q,j · m_c,j = 1} and

d = (1 / |S|) · Σ_{j∈S} ((x_q,j − x_c,j) / σ_j)².

Candidates are first restricted to the same central residue type (when at least *k*
exist). Because *d* is a mean over shared columns, rows that share only a few columns
are judged on few samples and win far too often; the query is therefore ranked by
**coverage first**: only candidates whose |S| is at least the coverage of the
(5*k*)-th best-covering candidate (never below 2) are eligible, relaxing to |S| ≥ 2 if
that leaves fewer than *k*. Candidates matching both flanking residue types have *d*
multiplied by 0.6. At most one hit per deposition is kept; *k* = 25 hits are returned.

**Prediction.** Hits are weighted 1/(d + 10⁻⁸), greedily clustered at 30° in (φ, ψ),
and the highest-weight cluster's weighted circular mean is the prediction, with its
circular standard deviations as φ_err/ψ_err (clipped to [3°, 25°]). Confidence:
`strong` (≥ 7 hits, SS agreement > 0.8, both angle std < 30°, mean d < 1.5),
`good` (≥ 5, > 0.6, mean d < 3.0), `weak` (≥ 3, > 0.4), else `insufficient`; these map to
High / Medium / Low / Flexible.

When the reference file is unavailable the residue keeps its encoder prediction and a
warning is logged.

## 7. Structural state populations (kernel-weighted retrieval)

`facet/structural.py::compute_structural_populations`, reported as
`structural_populations` in retrieval mode.

Each index row is assigned a d2D-style state: **helix** if SS = H, **beta** if SS = E,
**PPII** if SS = C and basin = PPII, **coil** otherwise. For a query embedding
**h**_q of residue type *a*,

P(s | **h**_q, a) = Σ_i K(**h**_q, **h**_i) · 1[state_i = s, aa_i = a] / Σ_i K(**h**_q, **h**_i) · 1[aa_i = a],

K(**h**_q, **h**_i) = exp(β · cos(**h**_q, **h**_i)), β = 15 (`structural_beta`).

The sum runs over the 2,000 most similar index rows (which carry > 99 % of the kernel
mass at this β) restricted to the query's residue type; if none share the type, all
2,000 are used. The effective sample size exp(H(w)) of the normalised weights is
returned as `structural_populations_eff_n`.

## 8. Retrieval-free SS populations (d2D-style Gaussian engine)

`facet/ss_populations.py::predict_ss_populations`, available as
`facet.predict_ss_populations`. The inference algorithm is that of δ2D (Camilloni
*et al.* 2012, Biochemistry 51, 2224); every parameter was refit on this project's data.

**Model.** For residue type *a* and state *s* ∈ {H, E, P, C}, the predicted shift
vector over the observed nuclei is μ_{a,s} + Σ_{o∈{−2,−1,+1,+2}} c_{o,s}(neighbour type
at offset *o*). With the observed vector *x*,

log L_s = −½ (x − μ_{a,s})ᵀ Σ_a⁻¹ (x − μ_{a,s}),  Σ_a = diag(σ_a) · R · diag(σ_a),

where σ_a is the per-residue-type, state-pooled standard deviation per nucleus and *R*
the 6×6 inter-nucleus residual correlation. Populations are L_s / Σ_s′ L_s′. Nuclei are
dropped from the vector and the covariance when unobserved or when the H or E cell mean
is unsupported; residues with fewer than `min_nuclei = 3` usable nuclei return NaN.
A 5-residue triangular smoothing (weights 0.25, 0.5, 1, 0.5, 0.25) is applied by default.

**Refit** (`facet/data/ss_popn_params.npz`; fitting script in the training repository).
Only depositions never touched by the LACS-style re-referencing step and with small
residual offsets (|Cα|, |Cβ| ≤ 0.8 ppm, |N| ≤ 2.5 ppm) were used — 1,558 of 5,132
depositions — because re-referencing compresses the separation between state means.
State labels were curated: helix = interior residues (2 trimmed from each end) of
helix segments ≥ 8 long with φ ∈ [−80°, −45°], ψ ∈ [−60°, −20°]; strand = interior
residues (1 trimmed) of strand segments ≥ 3 long with φ ∈ [−150°, −90°],
ψ ∈ [100°, 160°]; PPII = coil residues in the PPII basin; coil = other coil residues in
coil segments ≥ 4 long. From the training split this yields **49,262** residues from
1,075 depositions (H 11,253 / E 8,947 / P 6,504 / C 22,558; reproduced from the cache
for this document). A cell mean requires ≥ 30 observations (96 % of the 20×4×6 cells are
supported); a context correction requires ≥ 20; σ is re-estimated on residuals after the
context corrections.

## 9. Random Coil Index S²

`facet/rci.py::compute_rci_s2`, reported as `rci_s2`. Following Berjanskii & Wishart
2008 (J. Biomol. NMR 40, 31): |Δδ| per nucleus is averaged over residues i−1, i, i+1;
the weighted sum with the paper's all-six-nuclei weights (Cα 0.74, C′ 0.72, Cβ 0.13,
N 0.38, H^N 0.15, Hα 0.91) gives RCI = 1 / (6 · Σ w_j |Δδ_j|) and
S² = 1 − 0.4 · ln(1 + 17.7 · RCI), clipped to [0, 1]. Residues with fewer than three
observed nuclei get no value. Not implemented: the per-nucleus-subset weight sets,
sequential corrections, REFCOR re-referencing and end-effect correction of the original.

## 10. Outputs and restraint export

`facet/io/writers.py`. The restraint writers (`.tbl` for XPLOR/CNS/HADDOCK/ARIA, `.aco`
for CYANA, `.nef`) emit, by default, only **High**-tier residues; `--include-medium`
adds Medium; `--include-all` writes every residue. Each restraint is centred on the
predicted angle with half-width **2 × err**, where err is the per-residue 1σ from §5/§6.
For residues without a retrieval cluster the 1σ is half of a tier bound derived from the
parametric confidence: 16° (High), 22° (Medium), 26° (Low), 40° (Flexible). φ needs
residue i−1 and ψ needs residue i+1 to be present in the input, otherwise that restraint
is skipped. The per-residue table (`.predtab`), CSV and JSON contain all residues with
tier, angles, errors, SS, χ1 (and its probabilities), RCI S², basin populations,
structural populations, alternative clusters and top neighbours.

## 11. What FACET does not do

* It does not use the 3D structure, homology models, or sequence-profile features; the
  only inputs are the assigned backbone shifts and the residue types.
* It does not use any other torsion predictor's output as a feature; TALOS-N appears
  only as the benchmark comparator.
* It does not ensemble models or use test-time augmentation; one encoder, one index.
* It does not predict side-chain torsions beyond a three-class χ1 label, and gives no
  χ1 for Gly and Ala.
* It does not re-reference shifts unless asked (`auto_reference=True`), and its check is
  not a substitute for LACS-type re-referencing.
* It does not model temperature, pH or solvent effects on shifts; it inherits whatever
  conditions the BMRB depositions had.
* The population readouts are not calibrated against experimental ensembles; their
  validation is described in [LIMITATIONS.md](LIMITATIONS.md) and the README.
