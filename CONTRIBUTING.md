# Contributing

Thanks for taking the time. Bug reports, benchmark results on your own proteins and
documentation fixes are all welcome — open an issue first for anything larger than a
small fix so the approach can be agreed before you invest in it.

## Development setup

```bash
git clone https://github.com/maxzinke/facet-nmr
cd facet-nmr
pip install -e ".[dev]"
python -m facet.assets      # download weights + retrieval index into ~/.facet (~155 MB)
pytest
```

The weights are not in the repository; `facet/assets.py` resolves them from
`facet/weights/` (if present), then `$FACET_HOME` / `~/.facet`, and downloads on first
use. Tests that need them are marked `needs_assets` and skip when they are absent, so
`pytest` passes offline too.

## Before you open a pull request

- `pytest` is green and `ruff check facet --select F821,E9` is clean.
- New behaviour comes with a test; a changed number in the README comes with the
  `benchmarks/rescore.py` output that produced it.
- Do not add the weight files to git. `.gitignore` excludes `facet/weights/`; if you
  rebuilt an artifact, update the manifest in `facet/assets.py` and say so.
- Keep `torch`, `matplotlib`, `httpx` and `onnxruntime` imports function-local outside
  `model.py` so the CLI stays importable without the optional extras.

## Reporting a bad prediction

The most useful report contains the input shift list (or the BMRB ID), the
`pred.tab` / JSON output, the reference structure if you have one, and the package and
index versions from the output header.

## Licence

By contributing you agree that your contribution is licensed under the MIT licence
(code) and CC BY 4.0 (data and weights) like the rest of the project.
