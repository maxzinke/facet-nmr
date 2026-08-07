# Data provenance

What ships inside `facet/`, where it came from, and what may be done with it.

## Summary

| Bundled file | What it is | Source | Contains records? |
|---|---|---|---|
| `weights/facet_v3.pt`, `facet_v3.onnx` | Trained model parameters | Trained by this project | No |
| `weights/facet_retrieval_index.npz` | 128-d learned embeddings + φ/ψ + labels | Embeddings from this project's encoder; φ/ψ from the PDB | No |
| `weights/facet_shift_reference.npz` | Per-residue secondary shifts, with deposition and residue identifiers | BMRB (**CC0**) | Yes, and that is fine — see below |
| `data/ss_popn_params.npz` | Fitted per-residue Gaussian parameters | Fitted by this project | No |

Structural data (φ/ψ torsion angles, secondary-structure labels) derive from the
**Protein Data Bank**, released under **CC0 1.0** — a public-domain dedication that
imposes no conditions. Recorded for reproducibility rather than obligation.

Chemical-shift information derives from the **Biological Magnetic Resonance Data
Bank** (<https://bmrb.io>), gratefully acknowledged as the source.

## The shift reference: BMRB records, redistributed under CC0

`weights/facet_shift_reference.npz` backs the mask-safe retrieval fallback — the path
that recovers φ/ψ for residues whose HA or other atoms are unassigned.

It holds 310,923 rows of per-residue secondary shifts with the BMRB deposition ID and
residue number attached: about 4.8 million values from 3,470 depositions. Secondary
shifts are observed minus a random-coil constant, and this package ships that constant
(`facet/masked_retrieval.py::RANDOM_COIL_SHIFTS`), so raw ppm are recoverable. It is a
redistribution of BMRB data, not a derived statistic.

**That is permitted.** BMRB is part of the wwPDB and its data are released under
**CC0 1.0** — a public-domain dedication carrying no conditions, no attribution
requirement and no restriction on commercial use or redistribution. Confirmed directly
with BMRB (2026-08-06).

Citation is therefore courtesy rather than obligation, and BMRB asks for:

> Hoch, J. C. *et al.* "Biological Magnetic Resonance Data Bank."
> *Nucleic Acids Research* **51**(D1), D368–D376 (2023).
> <https://doi.org/10.1093/nar/gkac1050>

### A freshness argument for fetching rather than bundling

BMRB noted that downloading on first use would be preferable — not for any legal
reason, but because depositors submit corrections, and a bundled copy silently goes
stale. That is a real argument for fetch-on-first-use as a future improvement, on
data-quality grounds rather than licensing ones.

### An earlier version of this document was wrong

It recorded the redistribution status as unresolved, on the reasoning that BMRB's
aims-and-policies page promises free *access* without stating a licence grant. Absence
of a statement on one page is not absence of a licence. The CC0 position follows from
wwPDB membership, which the same document already noted for the PDB.

## Aggregation was tried and rejected on measurement

The obvious fix — replace records with aggregates — does not work. It is recorded
here because the reasoning is sound and someone will propose it again.

`scripts/build_centroid_reference.py` compresses the reference 88×, to 3,522 rows,
each a mean over ~88 observations grouped by residue type, secondary structure and
shift-space cluster. It carries no deposition identifiers and no individual
measurement, so it would have resolved the redistribution question outright.

Measured on the project's held-out coverage-ablation benchmark
(`benchmarks/facet_coverage_ablation.py` in the CRYSTALLINE-FID tree; leak-safe split
by sequence similarity, 3,404 scored residues from 39 held-out entries; median
angular error in degrees):

| index | rows | complete | −HA | −H | −H−HA | −CB | −H−HA−CB |
|---|---|---|---|---|---|---|---|
| full records | 310,923 | **15.99** | **18.15** | **18.00** | 23.87 | **17.73** | 24.29 |
| centroids, 14× | 22,128 | 19.73 | 19.99 | 19.95 | **20.18** | 20.21 | **21.75** |
| centroids, 88× | 3,522 | 22.95 | 23.21 | 22.76 | 23.16 | 22.83 | 23.25 |

Compression was measured as a curve rather than a single setting, because the first
attempt used 88× arbitrarily. Halving the loss requires dropping to 14×, and the
trend implies parity needs essentially no compression at all.

**Why:** accuracy tracks point density in shift space. For real held-out queries the
nearest neighbour sits at standardized distance 83.8 in the full index and 270.7 in
the 88× index — **3.2× further**. Retrieval needs a close neighbour, and the
information that provides one is the fine detail of the records themselves. Compressing
it removes exactly the thing being used. This is an information-theoretic limit, not a
clustering-quality problem: cluster conformational purity was verified high (median φ
spread 0.7°) and mask usability was unaffected (100% both).

### A useful side-finding: aggregation helps when coverage is poor

The compressed indices are **flatter** across scenarios, and they cross over:

* complete input — full index better by 3.7°
* −H−HA — **aggregated better by 3.7°** (20.18 vs 23.87)
* −H−HA−CB — **aggregated better by 2.5°** (21.75 vs 24.29)

Averaging denoises, and denoising matters more when there is less signal. A
coverage-dependent index choice — full records when atoms are present, aggregated when
they are sparse — would improve the hardest cases by ~2.5–3.7° over what ships today.
That is a prediction-quality opportunity, not a licensing fix, since the full index
would still ship.

### Why an earlier evaluation said the opposite

A self-consistency evaluation — querying with rows drawn from the reference
distribution itself — showed the aggregated index equal or better in every
missing-atom scenario (−0.02° to −0.13°). That measurement was real but answered the
wrong question: clean, complete, same-pipeline queries are served well by a smoothed
index; real proteins carrying referencing offsets and partial assignments are not.

Its absolute errors were ~0.35° against a true ~16° — a 45× discrepancy that should
have been treated as disqualifying the whole evaluation rather than noted as a
caveat alongside a recommendation.

**Only the held-out benchmark decides.** Anything else ranks methods on a question
nobody asked.
