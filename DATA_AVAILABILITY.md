# Data Availability

The raw option data used in this project came from OptionsDX.  The raw data are
not redistributed in this repository because they are third-party data and may
be subject to provider licensing terms.

## What Is Included

The repository includes saved paper-input artifacts under:

```text
reproducibility_artifacts/paper_inputs/
```

These are sufficient to regenerate the paper figures and LaTeX tables with:

```powershell
python scripts\make_paper_figures.py
```

## What Is Not Included

The repository does not include:

- raw OptionsDX files;
- cleaned full option panels derived from OptionsDX;
- large temporary walk-forward data folders;
- model checkpoints unless added separately in a release artifact.

## Expected Raw-Data Placement for Layer 2

To rerun the empirical pipeline, place licensed cleaned OptionsDX parquet files
in:

```text
cleaned_data/
```

The scripts expect, at minimum, the following columns:

```text
quote_date
underlying_last
expire_date
strike
c_bid
c_ask
risk_free_rate
dte
```

Some diagnostics also use implied-volatility and path-level columns produced by
the testing pipeline.  Those are present in the saved paper-input CSVs.

## Redistribution Guidance

Do not commit raw provider data to GitHub unless the provider license explicitly
permits redistribution.  The recommended public-release structure is:

1. keep raw data out of the repository;
2. include saved paper-input artifacts for figure/table reproducibility;
3. document the raw-data source and expected local folder layout;
4. provide scripts and command manifests so licensed users can rerun the
   second layer.
