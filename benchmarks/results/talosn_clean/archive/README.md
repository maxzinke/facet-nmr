# Archived benchmark tables (0.3-era)

The canonical benchmark is `../per_residue.csv` — the released package's own
predictions (defaults, retrained 0.4.0 model) scored against the corrected ground
truth. These archives document how the numbers evolved while the ground-truth defect
(docs/BENCHMARKS.md §7) was found and fixed; gunzip to read.

| File | What it is |
|---|---|
| `per_residue_record_v03.csv.gz` | The original benchmark of record: 0.3-era model, training-harness run, **uncorrected** truth (the "12.65° vs 13.57°" table) |
| `per_residue_record_v03_corrected.csv.gz` | Same record with the 63 X-ray entries' truth corrected; FACET errors for those rows come from the public-path re-run (`facet_err_source`) |
| `per_residue_rerun_v03_default.csv.gz` | 0.3-era model re-run through the public path, fallback off |
| `per_residue_rerun_v031_fallback_on.csv.gz` | 0.3-era model with the 0.3.1 default (mask-safe fallback on) — the run behind §6.1's slice table |
| `per_residue_rerun_v03_corrected.csv.gz` | 0.3-era public path scored on corrected truth |
| `per_residue_v040_raw_run_output.csv.gz` | Raw output of the 0.4.0 run before column canonicalisation |
| `per_protein_v03.csv.gz` | Per-protein summary of the old record |
| `per_residue_v040_s3_fallback.csv.gz` | 0.4.0 model with the opt-in fallback ON, on the 75 HA-free + 60 sampled entries (`../../data/s3_fallback_check_ids.txt`) — the run behind §6's "still worse with the rebuilt reference" check |
