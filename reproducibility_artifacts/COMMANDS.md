# Reproducibility Commands

This file separates verified commands from reconstructed commands.

## Layer 1: Verified Figure and Table Regeneration

This command has been verified in the cleaned repository:

```powershell
python scripts\make_paper_figures.py
```

It reads saved inputs from:

```text
reproducibility_artifacts/paper_inputs/
```

and writes generated manuscript outputs to:

```text
paper/figures/
paper/tables/
```

## Layer 2: Reconstructed Artifact-Generation Commands

The exact historical shell history was not preserved.  The commands below are
reconstructed from script defaults, saved settings files, output paths, and
available manifests in `reproducibility_artifacts/generation_manifests/`.

They should be treated as the current reproducibility recipe, not as a literal
log of every original terminal command.

### Walk-Forward Neural Agents

The final saved settings use:

- prefix: `final_WF_exp1_k1_test`
- years: 2015-2023
- training start: 2010
- validation year: Y-1
- test year: Y
- episodes: 20000
- validation interval: 1000
- transaction cost: 0
- kappa: 1
- reward exponent: 1
- seeds: 20262442 through 20262450

Reconstructed command:

```powershell
python scripts\run_walkforward.py
```

This command requires licensed raw/cleaned option data in `cleaned_data/`.

### Empirical Symbolic Distillation

The aggregate artifacts and candidate manifest indicate the final
walk-forward distillation for `final_WF_exp1_k1_test`, years 2015-2023, with
the three paper candidates:

- `uniform_bs_delta_residual`
- `smooth_uniform_bs_delta_residual`
- `smooth_focus_bs_delta_residual`

The copied symbolic manifests in
`reproducibility_artifacts/generation_manifests/symbolic_distillation/` record:

- seed: 123
- device: `cpu`
- canonical position: `bs`
- maximum train episodes per year: 20000
- train-support probes: 20000
- general probes: 20000
- PySR samples per candidate/year: 50000
- PySR iterations: 200
- max expression size: 24
- populations: 20
- deterministic search: true
- smoothing rows: 100000
- smoothing neighbors: 256
- smoothing bandwidth scale: 1.0

Manifest-backed reconstructed command:

```powershell
python scripts\distill_empirical_agents.py --model-prefix final_WF_exp1_k1_test --first-test-year 2015 --final-test-year 2023 --seed 123 --device cpu --canonical-position bs --max-train-episodes 20000 --train-support-probes 20000 --general-probes 20000 --formula-candidates uniform_bs_delta_residual smooth_uniform_bs_delta_residual smooth_focus_bs_delta_residual --n-sr-samples 50000 --niterations 200 --maxsize 24 --populations 20 --max-smoothing-rows 100000 --smoothing-neighbors 256 --smoothing-bandwidth-scale 1.0 --n-bootstrap 10000
```

This requires the selected trained neural checkpoints, scalers, selected test
CSVs, and cleaned data.

### Regime-Forensics Tables

Reconstructed command:

```powershell
python scripts\build_regime_forensics.py --prefix final_WF_exp1_k1_test --results-dir results/testing --output-prefix forensic_final
```

### Rho/Variance Diagnostics

The paper input uses the final-theory step-asymmetry artifact.  Reconstructed
command:

```powershell
python scripts\build_rho_variance_diagnostics.py --results-dir results/testing --prefix final_WF_exp1_k1_test --write-prefix rho_variance_audit_all_years_final_theory
```

### Long-Horizon Robustness

The copied long-horizon manifest records:

- prefix: `final_WF_exp1_k1_test`
- first test year: 2015
- final test year: 2023
- bootstrap draws: 10000
- seed: 42
- parallel workers: 3
- skip testing: false
- skip bootstrap: false

Manifest-backed reconstructed command:

```powershell
python scripts\run_long_horizon.py --prefix final_WF_exp1_k1_test --first-test-year 2015 --final-test-year 2023 --n-bootstrap 10000 --seed 42 --parallel-workers 3
```

### Hull-White Benchmark

The copied Hull-White manifest records:

- prefix: `final_WF_exp1_k1_test`
- years: 2015-2023
- bootstrap draws: 10000
- seed: 123
- transaction cost: 0
- ridge alpha: 0
- min absolute dS: 0
- clipped deltas: true
- formula selection: `parsimonious_10pct`
- workers: 3

Manifest-backed reconstructed command:

```powershell
python scripts\run_hull_white_benchmark.py --prefix final_WF_exp1_k1_test --first-test-year 2015 --final-test-year 2023 --n-bootstrap 10000 --seed 123 --workers 3 --transaction-cost 0 --ridge-alpha 0 --min-abs-ds 0 --formula-selection parsimonious_10pct
```

### Scalar Delta-Haircut Benchmark

The copied haircut manifest records:

- prefix: `final_WF_exp1_k1_test`
- years: 2015-2023
- bootstrap draws: 10000
- seed: 123
- transaction cost: 0
- lambda grid: 0.70, 0.75, ..., 1.05
- selection metric: reward
- clipped deltas: true
- formula selection: `parsimonious_10pct`

Manifest-backed reconstructed command:

```powershell
python scripts\run_haircut_benchmark.py --prefix final_WF_exp1_k1_test --first-test-year 2015 --final-test-year 2023 --n-bootstrap 10000 --seed 123 --transaction-cost 0 --lambdas 0.70,0.75,0.80,0.85,0.90,0.95,1.00,1.05 --selection-metric rew --formula-selection parsimonious_10pct
```

### Switching Robustness

The copied switching completion manifest records:

- model: `final_WF_exp1_k1_test2022`
- checkpoint: 20000
- output folder: `results/switching_test_2022`
- policies: two-band and three-band moneyness switching policies

The script defaults fill in the remaining parameters:

- validation year: 2021
- two-band cut: 1.0
- three-band cuts: 0.95, 1.05
- sample allocation: equal
- minimum band rows: 1000
- PySR iterations: 200
- PySR samples: 50000
- bootstrap draws: 10000
- seed: 123

Manifest-backed reconstructed command:

```powershell
python scripts\run_switching_robustness.py --model-prefix final_WF_exp1_k1_test --test-year 2022 --validation-year 2021 --two-band-cuts 1.0 --three-band-cuts 0.95,1.05 --sample-allocation equal --min-band-rows 1000 --niterations 200 --n-sr-samples 50000 --n-bootstrap 10000 --seed 123 --canonical-position bs
```

### Final Paper Outputs

After regenerating the saved inputs, rerun:

```powershell
python scripts\make_paper_figures.py --recompute-summary
python scripts\make_paper_figures.py
```

## Known Limits

The first layer is verified and sufficient for paper figure/table
reproducibility.

The second layer is a strong reconstructed recipe, but it depends on raw
OptionsDX data that are not redistributed in this repository.  Layer 2 should
therefore be described as reproducible for licensed data holders, while Layer 1
is the public reproducibility guarantee.  The selected trained actors and
scalers are included so model-based diagnostics can be inspected without
publishing raw option data.

## Runtime Expectations

Runtime is hardware-dependent.  The empirical walk-forward training/testing
stage is an hours-scale computation.  The full nine-year symbolic distillation
stage is substantially heavier and should be treated as a multi-day computation
on a typical workstation-class setup.  The figure/table regeneration command is
lightweight and runs from saved inputs in seconds to minutes.
