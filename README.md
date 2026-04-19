# FACET: Backbone Torsion Angle Prediction from NMR Chemical Shifts

**FACET** (Fold And Conformation Estimation Tool) predicts backbone phi/psi torsion angles, secondary structure, chi1 rotamers, and per-residue Ramachandran basin populations from NMR backbone chemical shifts (H, HA, N, CA, CB, C).

**v0.2.0 — retrieval-augmented.** Inference is backed by a kNN + DBSCAN lookup over a bundled 253K-residue reference index, producing multi-modal predictions with per-residue FACET tiers (High / Medium / Low / Flexible) plus alpha / beta / PPII / other basin populations. Clean 745-protein test benchmark (54,259 paired residues):

| Metric | Reference baseline | **FACET v0.2** |
|---|---|---|
| All-residue median | 13.65° | **12.15°** (−1.50°) |
| fail25 rate | 29.9% | **26.9%** |
| Coil median | 23.0° | **19.5°** (−3.47°) |
| High-tier residues (~50%) | — | **9.6° median / 14.5% fail25** |
| Coverage | 98.6% | **100%** |
| Head-to-head win rate | 46.6% | **53.4%** |

FACET wins on every SS class (helix −0.58°, strand −1.57°, coil −3.47°) and on both rigid (−1.45°) and flexible (−1.02°) subsets. The High tier — half the residues — carries 9.6° median error at 14.5% failure rate, making it directly usable for structure calculation restraints.

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

# Broaden restraint output to Medium tier (reaches ~98% coverage)
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
```

## Input Formats

| Format | Extension | Description |
|---|---|---|
| NMRPipe .tab | `.tab` | Standard backbone chemical shift table |
| CSV | `.csv` | ResID, AA, H, HA, N, CA, CB, C columns |

Auto-detected from file content. NEF and NMR-STAR readers coming soon.

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

Calibration on the 745-protein clean benchmark (54,259 paired residues, retrieval mode):

| Tier | Coverage | Median err | fail25 | Restraint bound | Use for restraints? |
|---|---|---|---|---|---|
| **High** | 50.5% | 9.6° | 14.5% | ±16° | **Yes** (default) |
| **Medium** | 47.5% | 16.5° | 38.2% | ±24° | With `--include-medium` |
| **Low** | 1.7% | 73.8° | 76.5% | — | No — multi-modal |
| **Flexible** | 0.3% | 65.2° | — | — | No — disordered |

**Flexible is not a failure state.** It flags residues whose retrieved neighbors do not form a coherent cluster — the chemical shifts are consistent with conformational averaging (flexible loops, disordered tails). These residues should be excluded from structure-calculation restraints, but the Flexible label itself carries biological meaning; the per-residue basin populations give you the alpha / beta / PPII / other mix directly.

By default, only **High** residues are written to restraint files (`.tbl`, `.aco`, `.nef`). Use `--include-medium` to add Medium-tier residues (brings coverage to ~98%), or `--include-all` to emit every residue regardless of tier.

## Model

FACET v3 is a 1.29M-parameter local-biased transformer:
- **Input**: pentapeptide window (±2 residues), 6 backbone secondary shifts
- **Encoder**: 3 dilated conv1d layers (GLU gating) + 2 RoPE self-attention layers
- **Torsion head**: 36×36 coarse Ramachandran grid + circular fine residual
- **SS head**: 3-class (H/E/C) with soft conditioning into torsion
- **Chi1 head**: 3-class rotamer (gauche+/gauche-/trans)
- **Retrieval (v0.2)**: kNN + DBSCAN over 253K reference residue embeddings
- Trained on 5,000 proteins (443K residues) from BMRB + PDB

## License

MIT

## Citation

If you use FACET in your research, please cite:
```
[citation pending]
```
