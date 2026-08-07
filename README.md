# FACET: Backbone Torsion Angle Prediction from NMR Chemical Shifts

**FACET** (Fold And Conformation Estimation Tool) predicts backbone phi/psi torsion angles, secondary structure, chi1 rotamers, and per-residue Ramachandran basin populations from NMR backbone chemical shifts (H, HA, N, CA, CB, C).

**v0.3.0 — retrieval-augmented + retrieval-free SS populations.** Phi/psi inference is backed by a kNN + DBSCAN lookup over a bundled 220K-residue reference index, producing multi-modal predictions with per-residue FACET tiers (High / Medium / Low / Flexible) plus α / β / PPII / other basin populations. Alongside, a retrieval-free d2D-style per-AA Gaussian engine (`facet.predict_ss_populations`) reports canonical (H / E / PPII / C) populations fit on 49K LACS-free curated residues — use this on IDPs where retrieval-derived populations inherit folded-neighbor DSSP bias. Clean 745-protein φ/ψ test benchmark (53,841 paired residues):

| Metric | Reference baseline | **FACET v0.2** |
|---|---|---|
| All-residue median | 13.57° | **12.65°** (−0.92°) |
| fail25 rate | 29.6% | **27.4%** |
| Coil median | 22.8° | **20.9°** (−1.92°) |
| Helix median | 8.5° | **8.0°** (−0.46°) |
| Strand median | 15.3° | **14.6°** (−0.72°) |
| High-tier residues (76%) | — | **10.6° median / 19.6% fail25** |
| Coverage | 98.6% | **99.2%** |
| Head-to-head win rate | 45.9% | **52.1%** |

FACET wins on every SS class and on both rigid (−0.77°) and flexible (−1.76°) subsets. The **High tier — 76% of residues — carries 10.6° median error at 19.6% failure rate**, making it directly usable for structure calculation restraints.

## Data and licensing

Code and bundled data are MIT (`LICENSE`). Model weights and fitted parameters are
produced by this project. Structural data derive from the PDB and chemical-shift data
from the BMRB — both released under **CC0 1.0**, a public-domain dedication with no
conditions attached.

If you use this tool, citing BMRB is appreciated:
Hoch *et al.*, *Nucleic Acids Res.* **51**, D368 (2023), doi:10.1093/nar/gkac1050.

See [DATA_PROVENANCE.md](DATA_PROVENANCE.md) for what each bundled file contains.

## Model weights

Weights and reference data are **downloaded on first use** into `~/.facet/` (override
with `$FACET_HOME`), not shipped in the wheel. The retrieval index alone is 106 MB,
past PyPI's 100 MB per-file limit, so a bundled wheel could not be published at all.
Downloading also means a corrected BMRB entry can reach you without a new release.

Files come from a HuggingFace model repository
([SiXa18/facet-weights](https://huggingface.co/SiXa18/facet-weights)), pinned to a
revision so a later upload cannot change what an installed copy resolves. Each is
verified against a pinned SHA-256. To prepare an offline or container install ahead of
time:

```
python -m facet.assets          # fetch everything now
FACET_NO_DOWNLOAD=1 facet ...   # never fetch; fail or degrade instead
```

Without the mask-safe shift reference, residues missing HA fall back to the parametric
head with a warning rather than failing.

## Quick Start

```bash
pip install facet-nmr

# Predict from a .tab shift list
facet predict shifts.tab

# Outputs:
#   shifts_facet.tbl      XPLOR/CNS dihedral restraints (High tier only)
#   shifts_facet.aco      CYANA angle constraints
#   shifts_facet.predtab  per-residue summary table

# All formats (+ NEF, CSV, JSON)
facet predict shifts.tab --all

# Specific format
facet predict shifts.tab -o restraints.nef --format nef

# Broaden restraint output to Medium tier
facet predict shifts.tab --include-medium
```

## Python API

```python
from facet import predict

# Default: retrieval-augmented inference (uses bundled reference index)
result = predict("shifts.tab")

# Per-residue predictions
for r in result.residues:
    print(f"{r.seq_id} {r.comp_id}: phi={r.phi:.1f} psi={r.psi:.1f} "
          f"SS={r.ss} tier={r.confidence_class}")
    if r.basin_populations is not None:
        a, b, p, o = r.basin_populations
        print(f"    basin populations: alpha={a:.0%} beta={b:.0%} "
              f"PPII={p:.0%} other={o:.0%}")
    if r.alt_clusters:
        for phi, psi, weight in r.alt_clusters:
            print(f"    alt cluster: ({phi:.1f}, {psi:.1f}) weight {weight:.2f}")

# Disable retrieval (parametric argmax only — v0.1-compatible)
result_parametric = predict("shifts.tab", use_retrieval=False)

# Export restraints
result.to_tbl("restraints.tbl")     # XPLOR / CNS / HADDOCK / ARIA
result.to_aco("restraints.aco")     # CYANA
result.to_nef("restraints.nef")     # NEF (wwPDB standard)
result.to_predtab("pred.tab")       # per-residue summary table

# Print summary
print(result.summary())

# Retrieval-free d2D-style SS populations (for IDPs, or as cross-check)
from facet import predict_ss_populations
from facet.io.readers import read_auto
from facet.io.bmrb import fetch_bmrb
sl = read_auto("shifts.tab")             # auto-detects .tab / .csv / .nef / NMR-STAR
# …or fetch from BMRB directly:
# sl = fetch_bmrb("19253")
ss = predict_ss_populations(sl)
# ss.populations[i] is (helix, beta, ppii, coil) for residue i
# ss.mean_populations gives the sequence-average 4-vector
```

## Input Formats

| Format | Extension | Description |
|---|---|---|
| NMRPipe .tab | `.tab` | Standard backbone chemical shift table |
| CSV | `.csv` | ResID, AA, H, HA, N, CA, CB, C columns |

Auto-detected from file content. NEF and NMR-STAR readers coming soon.

### Minimum shift coverage

FACET was trained on the full backbone (H, HA, N, CA, CB, C). Missing shifts are treated as masked inputs at inference — the model still runs but more residues will fall into Medium/Low/Flexible tiers.

| Available shifts | Expected behaviour |
|---|---|
| Full backbone (H, HA, N, CA, CB, C) | Recommended. Paper-reported tier distribution. |
| Missing HA | Auto-routed to the mask-safe shift-retrieval fallback (the parametric head alone collapses without HA). ~18° median φ/ψ error vs ~16° on the full backbone. |
| Missing C' or CB | Noticeable drop on β / PPII discrimination. |
| Only H, N, CA (e.g. TROSY-HNCA minimal set) | SS prediction workable; phi/psi largely Medium/Low tier. |
| Only H, N (e.g. 15N HSQC only) | Not recommended. Most residues will be Flexible. |

Formal per-shift ablation numbers will accompany the preprint.

### Chemical shift referencing

FACET was trained on BMRB-deposited shifts referenced against **DSS** (per the IUPAC/BMRB convention — 1H directly, 13C and 15N indirectly via frequency ratios). Before prediction, FACET runs a composition-adaptive sanity check: per-nucleus secondary-shift means are compared to what the protein's apparent secondary-structure composition would predict, and warnings surface if any systematic offset exceeds the tolerance (roughly ±0.4 ppm on 13C, ±1.5 ppm on 15N). When warnings fire the output includes a suggested additive correction per nucleus.

The check catches mis-referencing on N, HA, H, C', and CB reliably. It is weakest on CA-only offsets (the built-in SS classifier is driven by CA-CB, so a pure CA offset partially masks itself). For rigorous pre-publication work, users should run an external tool such as **LACS** (Wang & Markley 2009) before FACET prediction.

### Perdeuterated samples

For samples with deuterium labeling, set the `--deuteration` flag (or the Gradio dropdown) to `protonated` / `perdeut-exchanged` / `perdeut-unexchanged` / `ilv-methyl`. FACET applies an analytical 13C isotope correction (Venters et al. 1996; Hansen 1988) — roughly +0.29 ppm to CA, +0.68 ppm to CB, +0.10 ppm to C' for standard perdeut-exchanged samples — to recover protonated-equivalent shifts.

**Missing HA (common in perdeuterated samples).** When the α-proton is unobserved, the parametric torsion head — and the embedding retrieval, which shares the encoder — collapse toward ~0° because the model was trained never to lose HA. As of v0.3.1, FACET detects HA-missing residues and routes them to a **mask-safe shift-retrieval fallback** that matches on the observed atoms only (the missing channel drops out of the distance), recovering **~18° median φ/ψ error versus ~53° for the raw model** on held-out HA-stripped data. These residues are tiered from retrieval-cluster agreement and carry a real per-residue spread. Disable with `mask_safe_fallback=False`.

**The correction is a first-order average and not perfect.** It uses residue-type-independent coefficients, ignores three-bond effects, and doesn't handle temperature / solvent / labelling-scheme variation. As a result, **perdeuterated samples show systematically more Flexible-tier assignments than equivalent protonated samples** — the encoder's embedding drifts into sparser regions of retrieval space, and FACET honestly flags residues where it can't find a tight cluster instead of emitting confidently-wrong phi/psi.

Typical impact on a folded, perdeuterated protein: expect ~15-25% more residues in Medium/Low/Flexible tiers than on a protonated equivalent. SS prediction is largely unaffected; basin populations still carry signal on Flexible-tier residues. A domain-adapted encoder (Phase 2.1.5) will address this properly in a future release.

## Output Formats

| Format | Used by | Description |
|---|---|---|
| `.tbl` | XPLOR-NIH, CNS, HADDOCK, ARIA | Dihedral restraints |
| `.aco` | CYANA | Angle constraints |
| `.nef` | wwPDB, CCPN Analysis | NEF dihedral restraints (v1.1) |
| `pred.tab` | General | Per-residue summary with tier labels |
| `.csv` | Custom pipelines | Comma-separated values |
| `.json` | Programmatic access | Machine-readable |

## Confidence Tiers

FACET assigns each residue to one of four tiers. In retrieval mode (default) the tier comes from DBSCAN cluster agreement among the top-25 retrieved neighbors; in parametric mode it comes from the entropy of the coarse Ramachandran head. The vocabulary — **High / Medium / Low / Flexible** — is shared across both modes.

Calibration on the 745-protein clean benchmark (53,841 paired residues, retrieval mode):

| Tier | Coverage | Median err | fail25 | Restraint bound | Use for restraints? |
|---|---|---|---|---|---|
| **High** | 76.4% | 10.6° | 19.6% | ±20° | **Yes** (default) |
| **Medium** | 18.3% | 24.5° | 49.3% | ±35° | With `--include-medium` (use cautiously) |
| **Low** | 3.7% | 65.0° | 71.0% | — | No — multi-modal |
| **Flexible** | 0.8% | — | — | — | No — disordered / no cluster |

**Flexible is not a failure state.** It flags residues whose retrieved neighbors do not form a coherent cluster — the chemical shifts are consistent with conformational averaging (flexible loops, disordered tails). These residues should be excluded from structure-calculation restraints, but the Flexible label itself carries biological meaning; the per-residue basin populations give you the alpha / beta / PPII / other mix directly.

By default, only **High** residues are written to restraint files (`.tbl`, `.aco`, `.nef`). Use `--include-medium` to add Medium-tier residues (extends coverage to ~95% at the cost of wider bounds), or `--include-all` to emit every residue regardless of tier.

## Basin populations

For each residue FACET reports the fraction of retrieved neighbors that fall in each of four Ramachandran basins (α_R / β / PPII / other). These are **geometric sampling fractions**: they tell you which Ramachandran region the retrieved neighbors occupy, not the canonical H/E/C state populations of the ensemble.

| Basin | phi range | psi range | Canonical conformation |
|---|---|---|---|
| **α_R** | [−180°, −30°] | [−100°, +30°] | Right-handed α-helix |
| **β** | [−180°, −90°] | [+90°, +180°] ∪ [−180°, −150°] | Extended β-strand |
| **PPII** | [−90°, −30°] | [+90°, +180°] | Polyproline II / left-twisted extended |
| **other** | everything else | | Left-handed helix (α_L, mostly GLY), bridges, γ-turns |

A folded α-helix residue with tight retrieval typically shows `α100/β0/P0/o0`. A β-strand: `α0/β90/P10/o0`. An IDR residue sampling multiple geometries might show `α30/β15/P40/o15`.

### Interpretation on disordered samples (IDRs / IDPs)

FACET provides **three** complementary SS-population readouts, each answering a slightly different question:

| readout | API | what it measures | best for |
|---|---|---|---|
| **Geometric** | `ResiduePrediction.basin_populations` | fraction of retrieval neighbors in each Ramachandran basin (α_R / β / PPII / other) | spotting transient-structure regions, quick IDR fingerprints |
| **Structural (retrieval)** | `ResiduePrediction.structural_populations` | kernel-weighted Bayesian posterior over DSSP states using the embedding as similarity kernel | folded proteins, ensemble-reweighting priors |
| **SS populations (d2D-style)** | `facet.predict_ss_populations(...)` | d2D-style per-AA multivariate Gaussian likelihood on raw 6 backbone shifts — **no retrieval** | IDPs, cross-check against d2D / CheSPI, retrieval-free baseline |

For a rigid folded residue all three coincide. For a disordered residue they diverge — measured on tau K18 (BMRB 19253, 123 residues aligned across all three readouts):

| readout | mean helix population |
|---|---|
| Geometric (α-basin sampling) | 0.52 |
| Structural — retrieval | 0.21 |
| FACET-D2 | 0.07 |
| (d2D reference) | 0.01 |
| (CheSPI 4-state reference) | 0.04 |

Report whichever answers the question you're asking — sampling fraction (geometric), cooperative-SS from retrieval, or cooperative-SS from the retrieval-free Gaussian engine.

#### Structural (retrieval) mode — kernel-weighted Bayesian retrieval

Structural populations in the main `predict()` output come from **non-parametric Bayesian inference** over the full 220K-residue retrieval index, using the learned embedding as a similarity kernel:

```
P(state = s | query, AA) = [Σᵢ K(h_query, hᵢ) · 1[stateᵢ = s ∧ AAᵢ = AA]]
                         / [Σᵢ K(h_query, hᵢ) · 1[AAᵢ = AA]]

K(h_q, hᵢ) = exp(β · cos_sim(h_q, hᵢ))       (β = 15 by default)
```

State labels for index residues come from DSSP (H, E) plus a phi/psi-region override for PPII, matching d2D's 4-state convention:

| state | condition |
|---|---|
| **Helix** | DSSP = H |
| **Beta** | DSSP = E |
| **PPII** | DSSP = C (loop) AND phi/psi in PPII region |
| **Coil** | DSSP = C AND not in PPII region |

Conceptually this is *"non-parametric d2D using FACET's learned shift embeddings instead of hand-crafted per-AA Gaussians"*. Validation on folded proteins shows per-residue mean populations: helix residues → 0.90 H / 0.00 E / 0.02 P / 0.08 C; strand → 0.00 / 0.86 / 0.01 / 0.13. **On pure IDPs this mode over-estimates helix** (~20% on tau K18) because neighbor DSSP labels carry their crystal-structure context — which is why we also ship the d2D-style Gaussian engine below.

#### SS populations — d2D-style Gaussian engine, refit on our data

For IDPs and cross-tool cross-checks, use the retrieval-free d2D-style engine:

```python
from facet import predict_ss_populations
from facet.io.bmrb import fetch_bmrb             # or read_auto(path) for files

sl = fetch_bmrb("19253")                         # tau K18 (canonical IDP)
res = predict_ss_populations(sl, smooth=True, min_nuclei=3)

for sid, pops in zip(res.seq_ids, res.populations):
    h, e, p, c = pops
    print(f"{sid}: H={h:.2f} E={e:.2f} P={p:.2f} C={c:.2f}")

# Sequence-average 4-vector:
print(res.mean_populations)   # (H, E, P, C)
```

Per-residue inference:

```
v_s = observed_shifts - (μ[aa, s] + Σ_neighbor context[offset, aa_nbr, s])
L_s = N(0 | v_s, Σ)  =  (2π)^(-n/2) · |Σ|^(-1/2) · exp(-½ · v_s^T Σ^(-1) v_s)
P[s | shifts, aa, neighbors] = L_s / Σ_s' L_s'
```

Engine = Camilloni et al. 2012 (d2D). **Parameters = our own refit on 49,262 carefully curated BMRB residues** — no reliance on d2D's 2012 tables:
- LACS-corrected entries excluded (LACS compresses state-mean separation, biasing absolute per-state means)
- Helix state restricted to canonical α-helix phi/psi window (core residues only, ≥ 8-residue segments, 2-residue edge trim)
- Strand state restricted to canonical β phi/psi window, edge-trimmed
- PPII state from Ramachandran basin (d2D used fragment matching)
- Coil state from DSSP=C segments ≥ 4 residues long
- Neighbor-context corrections fit per (state, nucleus, offset, neighbor-AA)

### Honest comparison vs d2D and CheSPI

We compared FACET's d2D-style engine (FACET-D2) to [d2D (Camilloni 2012)](https://github.com/carlocamilloni/d2D) and [CheSPI (Nielsen & Mulder 2021)](https://github.com/protein-nmr/CheSPI) on a canonical IDP panel (tau K18, α-synuclein) plus folded controls (ubiquitin, GB1). Mean helix populations on tau K18 (BMRB 19253):

| method | mean helix | max helix | localisation (vs FACET-D2 r) |
|---|---|---|---|
| FACET retrieval mode | 0.206 | 1.000 | baseline |
| **FACET-D2 (this package)** | **0.066** | 0.237 | — |
| d2D | 0.010 | 0.061 | r = +0.46 |
| CheSPI 4-state (H+G+I) | 0.044 | 0.127 | r = +0.23 |
| CheSPI probs3 | 0.000 | 0.011 | r ≈ 0 |

**Where FACET-D2 is better**:
- No 2012 table dependency — parameters refit on 49K modern, LACS-free, state-curated residues
- Removes the retrieval-mode helix blowup (20.6% → 6.6% on tau K18)
- Ships with FACET so you get φ/ψ + SS populations in one pass; d2D and CheSPI are separate tools

**Where FACET-D2 is worse, honestly**:
- Our σ is ~40% wider than d2D's (ALA CA pooled: 1.29 vs 0.90 ppm). Intrinsic to our state-label granularity — d2D's 2012 library was hand-curated with protocols we can't fully replicate. Consequence: FACET-D2 does not reproduce d2D's crisp "≈ 0% helix" on pure IDPs
- FACET-D2 lands closer to **CheSPI 4-state** (DSSP H+G+I combined) than to d2D's strict α-only H state. This means FACET-D2's helix score includes low-level transient helical sampling (3₁₀ / π / edge) that d2D filters away
- No sequence-based SS prior (CheSPI uses a RaptorX-style NN); no HMM Viterbi cooperativity smoothing (CheSPI adds this). FACET-D2 uses only a 5-residue triangular smoothing
- On folded proteins FACET-D2 over-calls PPII vs d2D (tau K18: 0.37 vs 0.17; ubiquitin: 0.33 vs 0.05) — PPII basin boundaries in our training (broad phi/psi window) are more inclusive than d2D's fragment-matching approach

**Recommended cross-check**: For IDP manuscript figures or structure calculation, run d2D or CheSPI alongside FACET-D2. When all three agree on transient-structure locations (which they do within their r values), cite the magnitudes you'd report from whichever tool your reviewer prefers. FACET-D2 adds value as a retrieval-free, retrievable-inside-FACET, modern-data baseline.

#### Per-residue ensemble export

The 25 retrieved neighbours per residue (entry id, AA, phi/psi, DSSP, similarity) can also be exported directly as a seed ensemble for downstream tools (BME reweighting, ENSEMBLE, flexible-meccano):

```python
result.to_ensemble_csv("ensemble.csv")
result.to_ensemble_json("ensemble.json")
```

#### Practical recommendation for IDP / IDR analysis

- Use **dihedral sampling** (`basin_populations`) as a Ramachandran-region fingerprint: identify residues with elevated α, β, or PPII propensity, spot transient-structure regions. This is the honest per-residue φ/ψ-sampling claim, not a cooperative-structure number.
- Use **FACET-D2** (`facet.predict_ss_populations`) for canonical (H / E / PPII / C) populations suitable for ensemble reweighting, direct citation alongside d2D / CheSPI, or paper methods sections. This is the retrieval-free engine and doesn't inherit folded-neighbor DSSP bias.
- Treat **retrieval-based structural mode** (`structural_populations`) as a secondary readout: fine on folded / partially-folded samples, but over-estimates helix on pure IDPs (~20% on tau K18). Good for cross-checking FACET-D2 on structured regions.
- Use the **ensemble export** (`to_ensemble_csv` / `to_ensemble_json`) as seed conformers for BME / ENSEMBLE pipelines — no separate neighbour-retrieval step needed downstream.
- For rigorous cross-check, run **d2D** (Camilloni et al. 2012, https://github.com/carlocamilloni/d2D) or **CheSPI** (Nielsen & Mulder 2021, https://github.com/protein-nmr/CheSPI) alongside FACET-D2. All three agree well on transient-structure *locations*; the magnitudes you'd cite come from whichever tool your reviewer prefers.

## Model

FACET v3 is a 1.29M-parameter local-biased transformer:
- **Input**: pentapeptide window (±2 residues), 6 backbone secondary shifts
- **Encoder**: 3 dilated conv1d layers (GLU gating) + 2 RoPE self-attention layers
- **Torsion head**: 36×36 coarse Ramachandran grid + circular fine residual
- **SS head**: 3-class (H/E/C) with soft conditioning into torsion
- **Chi1 head**: 3-class rotamer (gauche+/gauche-/trans)
- **Retrieval (v0.2)**: kNN + DBSCAN over 220K reference residue embeddings
- Trained on 5,000 proteins (443K residues) from BMRB + PDB

## License

MIT

## References

FACET builds on several established methods for chemical-shift-based protein analysis. If you use any of the following outputs, please cite the underlying paper in addition to FACET itself:

- **Per-residue RCI S² order parameter** (reported in CSV / JSON / inspector as `rci_s2`): Berjanskii, M. V. & Wishart, D. S. *Application of the random coil index to studying protein flexibility.* **J. Biomol. NMR** 40, 31–48 (2008). DOI: [10.1007/s10858-007-9208-0](https://doi.org/10.1007/s10858-007-9208-0). Our implementation follows the paper's Eq. 2 (all-6-nuclei weighting coefficients from Supplemental Table 1) and Eq. 3 (S² = 1 − 0.4 · ln(1 + 17.7 · RCI)) but is **slightly simplified** — we do not apply the per-combination weight optimisation for subsets of observed nuclei, the sequential i±1 neighbour corrections, the REFCOR re-referencing step, or the end-effect correction. For exact parity with the published RCI server, use [the Wishart group's web tool](http://wishart.biology.ualberta.ca/rci).

- **Random-coil reference shifts** used to compute secondary shifts: Wishart, D. S. *et al.* J. Biomol. NMR 5, 67–81 (1995); Schwarzinger, S. *et al.* JACS 123, 2970 (2001).

- **Shift referencing sanity check**: adapted from the LACS approach — Wang, L. & Markley, J. L. *A simple method to predict protein chemical shifts from backbone amide 1H, 15N and 13C nuclei.* **J. Biomol. NMR** 44, 95 (2009). Users seeking rigorous re-referencing before publication-quality structure calculation should run LACS directly.

- **d2D engine** (the inference algorithm used by `facet.predict_ss_populations`): Camilloni, C., De Simone, A., Vranken, W. F. & Vendruscolo, M. *Determination of Secondary Structure Populations in Disordered States of Proteins Using Nuclear Magnetic Resonance Chemical Shifts.* **Biochemistry** 51, 2224–2231 (2012). Source at [github.com/carlocamilloni/d2D](https://github.com/carlocamilloni/d2D). FACET uses the same per-AA multivariate Gaussian likelihood as d2D, but with **all parameters refit** on our own 49 K-residue modern LACS-free curated training set — no reliance on Camilloni 2012's published parameter tables. See the "Honest comparison vs d2D and CheSPI" section above for validation on IDPs + folded proteins and where FACET-D2 lands relative to d2D / CheSPI.

- **CheSPI** (recommended IDP cross-check): Nielsen, J. T. & Mulder, F. A. A. *CheSPI: chemical shift secondary structure population inference.* **J. Biomol. NMR** 75, 273–291 (2021). DOI: [10.1007/s10858-021-00374-w](https://doi.org/10.1007/s10858-021-00374-w). Source at [github.com/protein-nmr/CheSPI](https://github.com/protein-nmr/CheSPI). CheSPI projects secondary shifts to two principal components, applies a per-state 2D Gaussian likelihood, combines with a RaptorX-style sequence-based prior, and post-processes with a Viterbi-like HMM over DSSP 8-state. FACET-D2's helix magnitudes on IDPs match CheSPI's 4-state "helical" (H+G+I) definition more closely than d2D's strict α-only H.

- **Deuterium isotope shift corrections** applied when `deuteration != "protonated"`: coefficients from Venters, R. A. *et al.* JACS 118, 8985 (1996); Hansen, P. E. Prog. NMR Spectrosc. 20, 207 (1988).

## Citation

If you use FACET in your research, please cite:
```
Zinke, M. FACET: Retrieval-augmented backbone torsion angle prediction
from NMR chemical shifts. Manuscript in preparation (2026).
https://github.com/bluegems661/facet-nmr
```
