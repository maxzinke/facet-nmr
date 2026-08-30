# FACET partial-coverage ablation

- Ground truth: `crystalline_all_phipsi_ensemble.npz` (well_defined-only scoring: True)
- DB (train) residues: 316,445; test residues: 4,019 from 39 entries; scored (valid): 3,404
- Retrieval k=25, query_mode=merged

Each cell = **median angular error (deg) / fail25 (% > 25 deg)**.

| scenario | predictor | overall | helix | strand | coil |
|---|---|---|---|---|---|
| `full` | retrieval | 15.6/32% | 10.6/18% | 14.4/22% | 26.6/52% |
| `-HA` | retrieval | 17.8/38% | 11.7/23% | 16.5/30% | 29.6/57% |
| `-H` | retrieval | 17.9/38% | 11.7/22% | 17.4/33% | 28.7/55% |
| `-H-HA` | retrieval | 22.9/47% | 13.8/31% | 25.3/50% | 34.5/59% |
| `-CB` | retrieval | 17.4/38% | 11.2/23% | 16.4/31% | 29.5/56% |
| `-H-HA-CB` | retrieval | 24.1/49% | 13.2/28% | 27.7/55% | 37.3/62% |

## Retrieval abstention (valid residues by confidence tier)

| scenario | strong | good | weak | insufficient |
|---|---|---|---|---|
| `full` | 401 | 2475 | 522 | 6 |
| `-HA` | 226 | 2415 | 744 | 19 |
| `-H` | 294 | 2386 | 710 | 14 |
| `-H-HA` | 120 | 2259 | 1000 | 25 |
| `-CB` | 317 | 2465 | 609 | 13 |
| `-H-HA-CB` | 167 | 2244 | 966 | 27 |

## Read
- **Go signal**: retrieval `-HA` / `-H-HA` median close to its `full` row and much lower than parametric `-HA`.
- Coil column matters most (partial coverage hurts coil first).
- Rising `insufficient` count = retrieval honestly abstaining as coverage drops.
