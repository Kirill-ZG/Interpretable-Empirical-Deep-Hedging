# Interpretable Empirical Deep Hedging

This repository contains the code and saved reproducibility inputs for the
paper:

**What Does Deep Hedging Actually Learn? Delta Corrections, Regime Fragility, and Symbolic Distillation**

Repository: https://github.com/Kirill-ZG/Interpretable-Empirical-Deep-Hedging

The repository is organized around two reproducibility layers.

## Reproducibility Layers

### Layer 1: Paper Figures and Tables

This layer is verified.  It regenerates the manuscript figures and LaTeX tables
from saved paper-input artifacts.

```powershell
python scripts\make_paper_figures.py
```

Inputs are stored in:

```text
reproducibility_artifacts/paper_inputs/
```

Generated outputs are written to:

```text
paper/figures/
paper/tables/
```

### Layer 2: Regenerating the Saved Inputs

This layer is documented but not fully verified in the cleaned repository.  It
requires raw option data and, for some diagnostics, trained model checkpoints.
The exact historical shell commands were not logged, but several final-run
settings were recovered from saved manifests.  See:

```text
ARTIFACTS.md
reproducibility_artifacts/COMMANDS.md
reproducibility_artifacts/generation_manifests/
```

## Repository Layout

```text
paper/
  main.tex
  Bibliography.bib
  figures/
  tables/

src/empirical_deep_hedging/
  main.py
  testing.py
  include/

scripts/
  run_walkforward.py
  distill_empirical_agents.py
  build_regime_forensics.py
  build_rho_variance_diagnostics.py
  run_long_horizon.py
  run_hull_white_benchmark.py
  run_haircut_benchmark.py
  run_switching_robustness.py
  make_paper_figures.py

reproducibility_artifacts/
  paper_inputs/
  generation_manifests/
  hall_of_fame_formulas/
  trained_models/
  COMMANDS.md

model/
  final_WF_exp1_k1_test/
  new_seed_final_WF_exp1_k1_test/
  1_seed_final_WF_exp1_k1_test/
```

## Installation

Python 3.10 is recommended for the full workflow.  The first layer needs only
the scientific Python stack and Matplotlib.  The second layer additionally uses
PyTorch, QuantLib, scikit-learn, and PySR.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PySR requires a working Julia installation.  It is only needed for regenerating
symbolic distillation artifacts, not for rerunning `make_paper_figures.py` from
the saved inputs.

## Data

Raw OptionsDX data are not included.  Layer 1 does not need the raw data.  For
Layer 2, place licensed cleaned parquet files in:

```text
cleaned_data/
```

See `DATA_AVAILABILITY.md` for the expected columns and redistribution policy.

## Main Commands

Verified paper-output command:

```powershell
python scripts\make_paper_figures.py
```

Manifest-backed reconstructed commands for regenerating saved inputs are listed
in `reproducibility_artifacts/COMMANDS.md`.

The repository includes only compact first-layer inputs, selected trained
actors/scalers, formula text/equation tables, and provenance manifests.  It
does not include large intermediate Layer-2 outputs such as all bootstrap
paired episodes, all HOF trade-step files, or full training checkpoint grids.

## Runtime Expectations

The verified Layer-1 command is lightweight.  The Layer-2 empirical pipeline is
not: the walk-forward neural training/testing stage is hours-scale, and the
full nine-year symbolic distillation run should be treated as a multi-day
computation on a typical workstation-class setup.

## Citation

Use `CITATION.cff` for citation metadata.

## License

Code is released under the MIT License.  Third-party data are not covered by
this license.
