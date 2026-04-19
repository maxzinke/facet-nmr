# FACET: Backbone Torsion Angle Prediction from NMR Chemical Shifts

**FACET** (Fold And Conformation Estimation Tool) predicts backbone phi/psi torsion angles, secondary structure, chi1 rotamers, and per-residue Ramachandran basin populations from NMR backbone chemical shifts (H, HA, N, CA, CB, C).

**v0.2.0 — retrieval-augmented.** Backs inference with a kNN + DBSCAN lookup over a bundled 253K-residue reference index, emitting multi-modal predictions with TALOS-N-style Strong / Generous / Ambiguous / None tiers plus per-residue alpha/beta/PPII/other populations. Clean 745-protein test benchmark (54,259 paired residues):

| Metric | TALOS-N | **FACET v0.2** |
|---|---|---|
| All-residue median | 13.65° | **12.15°** (+1.50°) |
| fail25 rate | 29.9% | **26.9%** |
| Coil median | 23.0° | **19.5°** (+3.47°) |
| Strong tier | — | **9.6° median / 14.5% fail25** on 50% of residues |
| Coverage | 98.6% | **100%** |
| Head-to-head | 46.6% wins | **53.4% wins** |

FACET beats TALOS-N on every SS class (helix +0.58°, strand +1.57°, coil +3.47°) and on both rigid (+1.45°) and flexible (+1.02°) subsets. The Strong tier — half the residues — carries 9.6° median error at 14.5% failure rate, making it directly usable for structure calculation restraints.

## Quick Start

```bash
pip install facet-nmr

# Predict from a TALOS-N format shift list
facet predict shifts.tab

# Outputs:
#   shifts_facet.tbl      XPLOR/CNS dihedral restraints
#   shifts_facet.aco      CYANA angle constraints
#   shifts_facet.predtab  TALOS-N-style summary

# All formats (+ NEF, CSV, JSON)
facet predict shifts.tab --all

# Specific format
facet predict shifts.tab -o restraints.nef --format nef
```

## Python API

```python
from facet import predict

# Default: retrieval-augmented inference (uses bundled reference index)
result = predict("shifts.tab")

# Per-residue predictions
for r in result.residues:
    print(f"{r.seq_id} {r.comp_id}: phi={r.phi:.1f} psi={r.psi:.1f} "
          f"SS={r.ss} tier={r.retrieval_tier}")
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
result.to_tbl("restraints.tbl")     # XPLOR/CNS/HADDOCK/ARIA
result.to_aco("restraints.aco")     # CYANA
result.to_nef("restraints.nef")     # NEF (wwPDB standard)
result.to_predtab("pred.tab")       # TALOS-N-style summary

# Print summary
print(result.summary())
```

## Input Formats

| Format | Extension | Description |
|---|---|---|
| TALOS-N tab | `.tab` | NMRPipe convention (drop-in TALOS-N replacement) |
| CSV | `.csv` | ResID, AA, H, HA, N, CA, CB, C columns |

Auto-detected from file content. NEF and NMR-STAR readers coming soon.

## Output Formats

| Format | Used by | Description |
|---|---|---|
| `.tbl` | XPLOR-NIH, CNS, HADDOCK, ARIA | Dihedral restraints |
| `.aco` | CYANA | Angle constraints |
| `.nef` | wwPDB, CCPN Analysis | NEF dihedral restraints (v1.1) |
| `pred.tab` | TALOS-N users | Per-residue summary with confidence classes |
| `.csv` | Custom pipelines | Comma-separated values |
| `.json` | Programmatic access | Machine-readable |

## Confidence Tiers

FACET assigns each residue to one of four tiers based on the entropy of
its coarse Ramachandran distribution. Thresholds are calibrated from
the v3 risk-coverage curve on a 26,944-residue held-out test set.

| Tier | Coverage | Fail25 | Restraint bound | Use for restraints? |
|---|---|---|---|---|
| **High** | top 30% | 8% | ±16° | Yes |
| **Medium** | 30–60% | 15% | ±22° | Yes |
| **Low** | 60–85% | 22% | ±26° | Interpret cautiously |
| **Flexible** | bottom 15% | — | — | Biologically flexible / disordered |

**Flexible is not a failure state** — it flags residues where the
model correctly refuses to assign rigid phi/psi because the chemical
shifts are consistent with conformational averaging (flexible loops,
disordered tails, etc.). These residues should be excluded from
structure-calculation restraints but their Flexible label is itself a
meaningful biological signal.

By default, only **High** and **Medium** residues are written to
restraint files (`.tbl`, `.aco`, `.nef`). Use `--include-all` to
include Low-tier residues as well.

## Model

FACET v3 is a 1.29M-parameter local-biased transformer:
- **Input**: pentapeptide window (±2 residues), 6 backbone secondary shifts
- **Encoder**: 3 dilated conv1d layers (GLU gating) + 2 RoPE self-attention layers
- **Torsion head**: 36×36 coarse Ramachandran grid + circular fine residual
- **SS head**: 3-class (H/E/C) with soft conditioning into torsion
- **Chi1 head**: 3-class rotamer (gauche+/gauche-/trans)
- Trained on 5,000 proteins (443K residues) from BMRB + PDB

## License

MIT

## Citation

If you use FACET in your research, please cite:
```
[citation pending]
```
