# Deterministic Tafel curve QC

This directory contains the canonical **97 curves / 97 records**. All 97 pass
deterministic QC and model selection; none is excluded.

The reliable onset is the first contiguous positive-current run with at least
12 effective bins, 0.08 V potential span, 0.5 decade log-current span, and a
strictly positive Spearman association. Pre-onset nonpositive points are kept
out of log fitting; no unmeasured detection limit is invented.

- `images/`: one self-contained QC figure per canonical curve.
- `curve_manifest.csv`, `curve_fit_summary.csv`, `model_comparison.csv`: the
  filtered 97-curve outputs.
- `qc_review_queue.csv`: focused visual checks; being listed does not itself
  exclude a curve.

`sample_record` is the canonical 1–97 record ID in every table. Image filenames
use both identifiers:

```text
record_NNN__curve_<curve_id>__qc.png
```

Thus Record 42 retains evidence ID `sr042_v1`, while Record 43 retains evidence
ID `sr042_v2`. The plotted title shows the immutable curve evidence ID.
