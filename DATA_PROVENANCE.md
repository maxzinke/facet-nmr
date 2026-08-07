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

## The retrieval scorer had a defect; aggregation was a symptom, not a cure

An aggregated ("centroid") reference was built and benchmarked, appeared to beat the
full record set when atoms were missing, and was then abandoned when the full
benchmark said otherwise. Both readings were wrong, for the same underlying reason.

### The real defect

`masked_retrieval.py::_query` scored candidates as a **mean** squared z over the
columns shared with the query, floored only at `n_shared >= 2`. A row sharing two
columns is judged on two samples and wins the argmin far too often — a winner's curse.
Measured at `-H-HA`: **88% of queries were won by a row covering under 40% of the
query**, and the Spearman correlation between rank-1 distance and rank-1 error was
**+0.007**. The distance carried no quality information at all.

The fix ranks candidates by query coverage before ranking by distance, reusing the
`k*5` candidate cap already present. Median angular error, on the project's
coverage-ablation benchmark (leak-safe split, 3,404 scored residues), independently
reproduced twice:

| scenario | before | after | gain |
|---|---|---|---|
| complete | 15.99 | **13.61** | −2.38 |
| −HA | 18.15 | **13.64** | −4.51 |
| −H | 18.00 | **13.57** | −4.43 |
| −H−HA | 23.87 | **13.42** | **−10.45** |
| −CB | 17.73 | **14.05** | −3.68 |
| −H−HA−CB | 24.29 | **14.08** | **−10.21** |

Coverage sensitivity collapses from +8.3° to +0.66°: after the fix, accuracy barely
depends on how many atoms are present. Confirmed on a never-tuned validation split
(14.72→20.64 becomes 12.50→12.72), with every paired bootstrap CI excluding zero.

### Why aggregation appeared to help

The centroid index filled its masks by majority vote, giving rows near-complete
coverage (occupancy 0.957 against 0.856). That *accidentally* did what the coverage
requirement now does deliberately. Three independent checks confirm averaging was
never the active ingredient: a raw random subset of the same size is flatter and
better than the shipped centroid index; replacing the averaged *label* with one
member's angles improves it by ~7°, while replacing the averaged *shift vector*
changes nothing; and the flatness effect reproduces across index sizes with no
averaging at all.

With the scorer fixed there is no crossover left — the full record index is best at
every coverage level, so there is nothing for an adaptive scheme to switch between.

### A units bug in the builder

`scripts/build_centroid_reference.py` called `np.radians()` on `phi`/`psi` values that
are **already radians** in the reference file, compressing them onto a 1/57th arc. The
aggregated numbers that motivated abandoning the approach were inflated by it, and the
same mistake in the evaluation scripts reported radians as degrees, making errors look
~57× smaller than they were. Both are corrected; the episode is recorded because the
misleading numbers looked plausible for a long time.

