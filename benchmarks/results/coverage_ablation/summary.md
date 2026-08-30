# FACET partial-coverage ablation

- Ground truth: `crystalline_all_phipsi_ensemble.npz` (well_defined-only scoring: True)
- DB (train) residues: 316,445; test residues: 4,019 from 39 entries; scored (valid): 3,404
- Retrieval k=25, query_mode=merged

Each cell = **median angular error (deg) / fail25 (% > 25 deg)**.

| scenario | predictor | overall | helix | strand | coil |
|---|---|---|---|---|---|
| `full` | retrieval | 14.0/25% | 9.5/10% | 13.8/16% | 22.5/46% |
| `full` | parametric | 12.4/17% | 9.0/6% | 12.6/10% | 17.4/31% |
| `-HA` | retrieval | 15.8/32% | 10.4/16% | 15.1/24% | 26.7/53% |
| `-HA` | parametric | 12.5/17% | 9.0/6% | 12.6/10% | 17.6/33% |
| `-H` | retrieval | 15.4/30% | 10.3/13% | 15.1/23% | 24.4/49% |
| `-H` | parametric | 12.6/18% | 9.0/5% | 12.8/12% | 18.1/34% |
| `-H-HA` | retrieval | 19.5/41% | 12.2/22% | 21.0/44% | 29.7/55% |
| `-H-HA` | parametric | 12.7/18% | 8.9/6% | 13.1/12% | 18.2/34% |
| `-CB` | retrieval | 15.0/30% | 9.9/13% | 14.8/22% | 25.4/50% |
| `-CB` | parametric | 12.9/18% | 9.1/6% | 13.4/12% | 18.0/34% |
| `-H-HA-CB` | retrieval | 21.3/44% | 11.9/22% | 24.7/49% | 33.0/59% |
| `-H-HA-CB` | parametric | 13.2/22% | 8.9/6% | 14.2/17% | 18.8/39% |

## Retrieval abstention (valid residues by confidence tier)

| scenario | strong | good | weak | insufficient |
|---|---|---|---|---|
| `full` | 838 | 2038 | 522 | 6 |
| `-HA` | 377 | 2264 | 744 | 19 |
| `-H` | 580 | 2100 | 710 | 14 |
| `-H-HA` | 158 | 2221 | 1000 | 25 |
| `-CB` | 578 | 2204 | 609 | 13 |
| `-H-HA-CB` | 185 | 2226 | 966 | 27 |

## Read
- **Go signal**: retrieval `-HA` / `-H-HA` median close to its `full` row and much lower than parametric `-HA`.
- Coil column matters most (partial coverage hurts coil first).
- Rising `insufficient` count = retrieval honestly abstaining as coverage drops.
