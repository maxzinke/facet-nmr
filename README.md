# FACET: Backbone Torsion Angle Prediction from NMR Chemical Shifts

**FACET** (Fold And Conformation Estimation Tool) predicts backbone phi/psi torsion angles, secondary structure, and chi1 rotamers from NMR backbone chemical shifts (H, HA, N, CA, CB, C).

FACET v3 surpasses TALOS-N on 27,461 matched residues with identical ground truth:

| Metric | TALOS-N | FACET v3 |
|---|---|---|
| All-residue median | 14.4° | **13.4°** |
| <15° fraction | 51.7% | **55.0%** |
| Coil median | 23.1° | **21.1°** |
| SS Q3 | ~88% | 86.6% |

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

result = predict("shifts.tab")

# Per-residue predictions
for r in result.residues:
    print(f"{r.seq_id} {r.comp_id}: phi={r.phi:.1f} psi={r.psi:.1f} "
          f"SS={r.ss} conf={r.confidence_class}")

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

## Confidence Classes

FACET maps its internal confidence score (negative entropy of the coarse Ramachandran distribution) to TALOS-N-compatible labels:

| Class | Meaning | Use for restraints? |
|---|---|---|
| **Strong** | High confidence, narrow distribution | Yes |
| **Good** | Moderate confidence | Yes |
| **Warn** | Low confidence, interpret with caution | No (by default) |
| **Dynamic** | Likely disordered or ambiguous | No |

By default, only Strong and Good residues are included in restraint files (.tbl, .aco, .nef). Use `--include-all` to override.

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
