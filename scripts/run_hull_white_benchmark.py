"""
Validation-calibrated Hull-White-style minimum-variance delta benchmark.

This script implements the quadratic Hull-White correction discussed as a
compact robustness benchmark:

    delta_HW = clip(delta_BS + Vega/(S*sqrt(T)) *
                    (a + b*delta_BS + c*delta_BS^2), 0, 1)

For each walk-forward test year Y, the coefficients (a, b, c) are estimated
only on validation year Y-1 from the regression

    dOption - delta_BS*dS =
        Vega*dS/(S*sqrt(T)) * (a + b*delta_BS + c*delta_BS^2) + error.

The frozen validation coefficients are then traded on year Y.  The script does
not rerun training or retest the neural agent/symbolic formulas.  It only reads
the cached walk-forward trade-step and episode-metric files, computes the
Hull-White benchmark, and compares it with BS, the raw agent, and the paper's
parsimonious selected symbolic formula using the same two-stage bootstrap
convention as the rest of the project.

Suggested manual run:

    python scripts/run_hull_white_benchmark.py --prefix final_WF_exp1_k1_test --n-bootstrap 10000

Outputs are written under results/hull_white_check/<prefix>/.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from run_haircut_benchmark import (
    METRICS,
    atomic_to_csv,
    atomic_to_json,
    bootstrap_pair,
    build_pair_table,
    episode_metrics_from_steps,
    load_existing_episode_metrics,
    load_year_contexts,
    reward_from_pnl,
    selected_parsimonious_formulas,
    sort_trade_steps,
    stable_seed,
)


POLICY_NAME = "hull_white_quadratic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="final_WF_exp1_k1_test")
    parser.add_argument("--walkforward-dir", default="results/interpret_real_walkforward")
    parser.add_argument("--output-dir", default="results/hull_white_check")
    parser.add_argument("--first-test-year", type=int, default=2015)
    parser.add_argument("--final-test-year", type=int, default=2023)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help=(
            "Number of test years to process in parallel. Each worker writes "
            "only its own per-year folder; aggregate CSVs are written by the "
            "parent process after all workers complete."
        ),
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0,
        help=(
            "Stock transaction-cost rate used when replaying Hull-White. "
            "The current paper experiments use zero; set explicitly if needed."
        ),
    )
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--reward-exponent", type=float, default=1.0)
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=1e-10,
        help="Numerical tolerance for bootstrap significance flags around zero.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=0.0,
        help=(
            "Optional ridge penalty for the validation regression. Default 0 "
            "uses ordinary least squares, which is the cleanest Hull-White check."
        ),
    )
    parser.add_argument(
        "--min-abs-ds",
        type=float,
        default=0.0,
        help=(
            "Optional validation-row filter on |dS|. Default 0 keeps all finite "
            "rows; zero-move rows have zero regressors and do not affect OLS."
        ),
    )
    parser.add_argument(
        "--no-clip-delta",
        action="store_true",
        help=(
            "Do not clip the corrected delta to [0,1]. By default we clip, "
            "matching the action range available to the learned call hedge."
        ),
    )
    parser.add_argument(
        "--formula-selection",
        choices=["parsimonious_10pct"],
        default="parsimonious_10pct",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cached Hull-White outputs even if a complete manifest exists.",
    )
    return parser.parse_args()


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def time_to_maturity_years(df: pd.DataFrame) -> np.ndarray:
    if "T_years" in df.columns:
        tau = df["T_years"].to_numpy(dtype=float)
    elif "TauYears" in df.columns:
        tau = df["TauYears"].to_numpy(dtype=float)
    elif "T" in df.columns:
        # Env.option["T"] is stored in days in the original environment.
        tau = df["T"].to_numpy(dtype=float) / 365.0
    else:
        raise ValueError("Cannot infer time to maturity; missing T_years/TauYears/T")
    return np.clip(tau, 1e-8, None)


def strike_from_steps(df: pd.DataFrame) -> np.ndarray:
    if "strike" in df.columns:
        return df["strike"].to_numpy(dtype=float)
    if "Strike" in df.columns:
        return df["Strike"].to_numpy(dtype=float)
    if "S/K" in df.columns:
        return df["S-1"].to_numpy(dtype=float) / df["S/K"].to_numpy(dtype=float)
    raise ValueError("Cannot infer strike; missing strike/Strike/S/K")


def normal_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def bs_vega_from_steps(df: pd.DataFrame) -> np.ndarray:
    # The cached trade-step files contain both:
    #   iv : the decision-state implied volatility available before trading;
    #   v  : Env.info["v"], written after the environment advances.
    # Hull-White is a tradable benchmark, so vega must use the start-of-interval
    # IV.  Prefer "iv" and fall back to "v" only for legacy files that do not
    # have the decision-state column.
    iv_col = "iv" if "iv" in df.columns else "v"
    required = ["S-1", iv_col, "r", "q"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot compute BS vega; missing columns: {missing}")
    s = np.clip(df["S-1"].to_numpy(dtype=float), 1e-12, None)
    k = np.clip(strike_from_steps(df), 1e-12, None)
    sigma = np.clip(df[iv_col].to_numpy(dtype=float), 1e-8, None)
    r = df["r"].to_numpy(dtype=float)
    q = df["q"].to_numpy(dtype=float)
    r = np.where(r > 1.0, r / 100.0, r)
    q = np.where(q > 1.0, q / 100.0, q)
    tau = time_to_maturity_years(df)
    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    return s * np.exp(-q * tau) * normal_pdf(d1) * sqrt_tau


def hull_white_design(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build the validation regression y = X beta.

    The regression is written directly in PnL units:
        dC - delta_BS*dS = [Vega*dS/(S*sqrt(T))]*(a+b*delta+c*delta^2).
    """
    steps = sort_trade_steps(df)
    required = ["bs_delta", "S0", "S-1", "P0", "P-1"]
    missing = [col for col in required if col not in steps.columns]
    if missing:
        raise ValueError(f"Trade-step file is missing required columns: {missing}")

    s_start = np.clip(steps["S-1"].to_numpy(dtype=float), 1e-12, None)
    d_s = steps["S0"].to_numpy(dtype=float) - s_start
    d_p = steps["P0"].to_numpy(dtype=float) - steps["P-1"].to_numpy(dtype=float)
    delta = steps["bs_delta"].to_numpy(dtype=float)
    tau = time_to_maturity_years(steps)
    vega = bs_vega_from_steps(steps)
    base = vega * d_s / (s_start * np.sqrt(tau))
    x = np.column_stack([base, base * delta, base * delta * delta])
    y = d_p - delta * d_s
    diag = steps[["episode"]].copy()
    diag["dS"] = d_s
    diag["dP"] = d_p
    diag["bs_delta"] = delta
    diag["tau_years"] = tau
    diag["bs_vega"] = vega
    diag["hw_base_regressor"] = base
    diag["hw_response"] = y
    return x, y, diag


def estimate_hull_white_coefficients(
    validation_steps: pd.DataFrame, args: argparse.Namespace
) -> tuple[dict[str, float], pd.DataFrame]:
    x, y, diag = hull_white_design(validation_steps)
    finite = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if float(args.min_abs_ds) > 0.0:
        finite &= np.abs(diag["dS"].to_numpy(dtype=float)) >= float(args.min_abs_ds)
    x_fit = x[finite]
    y_fit = y[finite]
    if len(y_fit) < 3:
        raise ValueError(f"Not enough finite validation rows for Hull-White fit: {len(y_fit)}")

    ridge_alpha = float(args.ridge_alpha)
    if ridge_alpha > 0.0:
        xtx = x_fit.T @ x_fit
        xty = x_fit.T @ y_fit
        beta = np.linalg.solve(xtx + ridge_alpha * np.eye(3), xty)
        rank = int(np.linalg.matrix_rank(xtx))
        singular_values = np.linalg.svd(x_fit, compute_uv=False)
    else:
        beta, _, rank, singular_values = np.linalg.lstsq(x_fit, y_fit, rcond=None)

    fitted = x_fit @ beta
    resid = y_fit - fitted
    response_std = float(np.std(y_fit, ddof=1)) if len(y_fit) > 1 else np.nan
    coeffs = {
        "a": float(beta[0]),
        "b": float(beta[1]),
        "c": float(beta[2]),
        "n_fit_rows": int(len(y_fit)),
        "n_total_rows": int(len(y)),
        "rank": int(rank),
        "ridge_alpha": ridge_alpha,
        "rmse": float(np.sqrt(np.mean(resid * resid))),
        "mae": float(np.mean(np.abs(resid))),
        "response_std": response_std,
        "r2": float(1.0 - np.sum(resid * resid) / np.sum((y_fit - np.mean(y_fit)) ** 2))
        if np.sum((y_fit - np.mean(y_fit)) ** 2) > 1e-18
        else np.nan,
        "singular_value_1": float(singular_values[0]) if len(singular_values) > 0 else np.nan,
        "singular_value_2": float(singular_values[1]) if len(singular_values) > 1 else np.nan,
        "singular_value_3": float(singular_values[2]) if len(singular_values) > 2 else np.nan,
    }
    diag["used_in_fit"] = finite
    diag["hw_fitted_response"] = np.nan
    diag.loc[finite, "hw_fitted_response"] = fitted
    diag["hw_residual"] = np.nan
    diag.loc[finite, "hw_residual"] = resid
    return coeffs, diag


def corrected_delta_from_coefficients(
    steps: pd.DataFrame, coeffs: dict[str, float], clip_delta: bool
) -> tuple[np.ndarray, pd.DataFrame]:
    df = sort_trade_steps(steps)
    delta_bs = df["bs_delta"].to_numpy(dtype=float)
    s_start = np.clip(df["S-1"].to_numpy(dtype=float), 1e-12, None)
    tau = time_to_maturity_years(df)
    vega = bs_vega_from_steps(df)
    poly = coeffs["a"] + coeffs["b"] * delta_bs + coeffs["c"] * delta_bs * delta_bs
    correction = vega / (s_start * np.sqrt(tau)) * poly
    delta_hw = delta_bs + correction
    if clip_delta:
        delta_hw = np.clip(delta_hw, 0.0, 1.0)

    diagnostics = pd.DataFrame(
        {
            "episode": df["episode"].to_numpy(),
            "bs_delta": delta_bs,
            "tau_years": tau,
            "bs_vega": vega,
            "hw_poly": poly,
            "hw_delta_correction_raw": correction,
            "hw_delta_raw": delta_bs + correction,
            "hw_delta": delta_hw,
            "hw_was_clipped": np.abs(delta_hw - (delta_bs + correction)) > 1e-12,
        }
    )
    return delta_hw, diagnostics


def replay_hull_white_steps(
    source_steps: pd.DataFrame,
    coeffs: dict[str, float],
    args: argparse.Namespace,
    split: str,
    split_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = sort_trade_steps(source_steps)
    delta, diagnostics = corrected_delta_from_coefficients(
        df, coeffs, clip_delta=not bool(args.no_clip_delta)
    )

    prev_pos = np.zeros(len(df), dtype=float)
    for _, idxs in df.groupby("episode", sort=False).indices.items():
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) > 1:
            prev_pos[idxs[1:]] = -delta[idxs[:-1]]

    s_start = df["S-1"].to_numpy(dtype=float)
    d_s = df["S0"].to_numpy(dtype=float) - s_start
    d_p = df["P0"].to_numpy(dtype=float) - df["P-1"].to_numpy(dtype=float)
    tc = -np.abs((-delta) - prev_pos) * s_start * float(args.transaction_cost)
    pnl = -delta * d_s + d_p + tc

    out = df.copy()
    out["policy"] = POLICY_NAME
    out["chosen_delta"] = delta
    out["formula_delta"] = delta
    out["A Pos"] = -delta
    out["A TC"] = tc
    out["A PnL"] = pnl
    out["A PnL - TC"] = pnl - tc
    out["A Reward"] = reward_from_pnl(
        pnl, kappa=float(args.kappa), reward_exponent=float(args.reward_exponent)
    )
    out["hw_a"] = coeffs["a"]
    out["hw_b"] = coeffs["b"]
    out["hw_c"] = coeffs["c"]
    out["split"] = split
    out["split_year"] = int(split_year)

    diagnostics["split"] = split
    diagnostics["split_year"] = int(split_year)
    diagnostics["A PnL"] = pnl
    diagnostics["A Reward"] = out["A Reward"].to_numpy(dtype=float)
    return out, diagnostics


def summarize_pair(
    pair_df: pd.DataFrame,
    comparison: str,
    left_policy: str,
    right_policy: str,
    ctx,
    split: str,
    split_year: int,
    args: argparse.Namespace,
) -> pd.DataFrame:
    boot = bootstrap_pair(
        pair_df,
        n_bootstrap=int(args.n_bootstrap),
        seed=stable_seed(args.seed, comparison, ctx.test_year, split),
        zero_tolerance=float(args.zero_tolerance),
    )
    rows = []
    for metric, stats in boot.items():
        rows.append(
            {
                "comparison": comparison,
                "metric": metric,
                "left_policy": left_policy,
                "right_policy": right_policy,
                "model_name": ctx.model_name,
                "checkpoint": ctx.checkpoint,
                "train_start_year": ctx.train_start_year,
                "train_end_year": ctx.train_end_year,
                "validation_year": ctx.validation_year,
                "test_year": ctx.test_year,
                "split": split,
                "split_year": split_year,
                "n_episodes": int(len(pair_df)),
                "n_clusters": int(pair_df["start_date"].nunique()),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def compare_hull_white(
    ctx,
    selected_formula_policy: str,
    hw_metrics: pd.DataFrame,
    args: argparse.Namespace,
    split: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    split_year = ctx.validation_year if split == "validation" else ctx.test_year
    agent_metrics, hof_metrics = load_existing_episode_metrics(ctx, split)
    comparisons = []
    pairs_to_save: list[pd.DataFrame] = []
    specs = [
        ("hw_vs_bs", hw_metrics, POLICY_NAME, "A", hw_metrics, POLICY_NAME, "B"),
        ("hw_vs_agent", hw_metrics, POLICY_NAME, "A", agent_metrics, "agent", "A"),
        (
            "hw_vs_selected_formula",
            hw_metrics,
            POLICY_NAME,
            "A",
            hof_metrics,
            selected_formula_policy,
            "A",
        ),
    ]
    for label, left_df, left_policy, left_side, right_df, right_policy, right_side in specs:
        pair = build_pair_table(left_df, left_policy, left_side, right_df, right_policy, right_side)
        pair["comparison"] = label
        pair["left_policy"] = left_policy
        pair["right_policy"] = "bs" if label == "hw_vs_bs" else right_policy
        pair["test_year"] = ctx.test_year
        pair["split"] = split
        pair["split_year"] = split_year
        pairs_to_save.append(pair)
        comparisons.append(
            summarize_pair(
                pair,
                comparison=label,
                left_policy=left_policy,
                right_policy="bs" if label == "hw_vs_bs" else right_policy,
                ctx=ctx,
                split=split,
                split_year=split_year,
                args=args,
            )
        )
    return pd.concat(comparisons, ignore_index=True), pairs_to_save


def run_one_year(task: tuple[object, dict, argparse.Namespace, Path]) -> dict:
    ctx, formula_row, args, output_root = task
    year_out = output_root / f"{ctx.test_year}_{ctx.model_name}"
    year_out.mkdir(parents=True, exist_ok=True)
    selected_formula_policy = str(formula_row["policy"])

    coeff_path = year_out / f"{ctx.stem}_hull_white_coefficients.csv"
    fit_diag_path = year_out / f"{ctx.stem}_hull_white_validation_fit_rows.csv"
    validation_steps_path = year_out / f"{ctx.stem}_validation_hull_white_trade_steps.csv"
    test_steps_path = year_out / f"{ctx.stem}_test_hull_white_trade_steps.csv"
    validation_diag_path = year_out / f"{ctx.stem}_validation_hull_white_delta_diagnostics.csv"
    test_diag_path = year_out / f"{ctx.stem}_test_hull_white_delta_diagnostics.csv"
    validation_metrics_path = year_out / f"{ctx.stem}_validation_hull_white_episode_metrics.csv"
    test_metrics_path = year_out / f"{ctx.stem}_test_hull_white_episode_metrics.csv"
    validation_bootstrap_path = year_out / f"{ctx.stem}_validation_hull_white_bootstrap.csv"
    test_bootstrap_path = year_out / f"{ctx.stem}_test_hull_white_bootstrap.csv"
    validation_pairs_path = year_out / f"{ctx.stem}_validation_hull_white_paired_episodes.csv"
    test_pairs_path = year_out / f"{ctx.stem}_test_hull_white_paired_episodes.csv"
    complete_path = year_out / f"{ctx.stem}_hull_white_complete.json"

    result_paths = {
        "test_year": ctx.test_year,
        "year_dir": str(year_out),
        "coefficients_path": str(coeff_path),
        "fit_diagnostics_path": str(fit_diag_path),
        "validation_steps_path": str(validation_steps_path),
        "test_steps_path": str(test_steps_path),
        "validation_delta_diagnostics_path": str(validation_diag_path),
        "test_delta_diagnostics_path": str(test_diag_path),
        "validation_metrics_path": str(validation_metrics_path),
        "test_metrics_path": str(test_metrics_path),
        "validation_bootstrap_path": str(validation_bootstrap_path),
        "test_bootstrap_path": str(test_bootstrap_path),
        "validation_pairs_path": str(validation_pairs_path),
        "test_pairs_path": str(test_pairs_path),
    }
    run_config = {
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "transaction_cost": float(args.transaction_cost),
        "kappa": float(args.kappa),
        "reward_exponent": float(args.reward_exponent),
        "zero_tolerance": float(args.zero_tolerance),
        "ridge_alpha": float(args.ridge_alpha),
        "min_abs_ds": float(args.min_abs_ds),
        "clip_delta": not bool(args.no_clip_delta),
        "vega_iv_column": "iv_preferred_else_v",
        "base_policy": "agent",
        "selected_symbolic_policy": selected_formula_policy,
    }
    required_outputs = [Path(path) for key, path in result_paths.items() if key.endswith("_path")]
    if complete_path.exists() and not args.force:
        try:
            manifest = json.loads(complete_path.read_text(encoding="utf-8"))
            if (
                manifest.get("status") == "completed"
                and manifest.get("run_config") == run_config
                and all(path.exists() and path.stat().st_size > 0 for path in required_outputs)
            ):
                print(f"[hull-white] year {ctx.test_year}: complete cached outputs found")
                return result_paths
        except (OSError, json.JSONDecodeError):
            pass

    print(f"[hull-white] fitting validation coefficients for test year {ctx.test_year}")
    validation_source = pd.read_csv(ctx.validation_steps_path)
    test_source = pd.read_csv(ctx.test_steps_path)
    if "policy" in validation_source.columns:
        validation_source = validation_source[validation_source["policy"] == "agent"].copy()
    if "policy" in test_source.columns:
        test_source = test_source[test_source["policy"] == "agent"].copy()
    if validation_source.empty:
        raise ValueError(f"No raw agent rows found in {ctx.validation_steps_path}")
    if test_source.empty:
        raise ValueError(f"No raw agent rows found in {ctx.test_steps_path}")
    coeffs, fit_diag = estimate_hull_white_coefficients(validation_source, args)
    coeff_row = {
        "model_name": ctx.model_name,
        "checkpoint": ctx.checkpoint,
        "train_start_year": ctx.train_start_year,
        "train_end_year": ctx.train_end_year,
        "validation_year": ctx.validation_year,
        "test_year": ctx.test_year,
        "selected_symbolic_policy": selected_formula_policy,
        "selected_symbolic_candidate": formula_row["candidate"],
        "selected_symbolic_complexity": int(formula_row["complexity"]),
        **coeffs,
    }
    atomic_to_csv(pd.DataFrame([coeff_row]), coeff_path, index=False)
    atomic_to_csv(fit_diag, fit_diag_path, index=False)

    validation_steps, validation_diag = replay_hull_white_steps(
        validation_source, coeffs, args, "validation", ctx.validation_year
    )
    test_steps, test_diag = replay_hull_white_steps(test_source, coeffs, args, "test", ctx.test_year)
    validation_metrics = episode_metrics_from_steps(validation_steps)
    test_metrics = episode_metrics_from_steps(test_steps)

    atomic_to_csv(validation_steps, validation_steps_path, index=False)
    atomic_to_csv(test_steps, test_steps_path, index=False)
    atomic_to_csv(validation_diag, validation_diag_path, index=False)
    atomic_to_csv(test_diag, test_diag_path, index=False)
    atomic_to_csv(validation_metrics, validation_metrics_path, index=False)
    atomic_to_csv(test_metrics, test_metrics_path, index=False)

    val_boot, val_pairs = compare_hull_white(
        ctx, selected_formula_policy, validation_metrics, args, split="validation"
    )
    test_boot, test_pairs = compare_hull_white(
        ctx, selected_formula_policy, test_metrics, args, split="test"
    )
    atomic_to_csv(val_boot, validation_bootstrap_path, index=False)
    atomic_to_csv(test_boot, test_bootstrap_path, index=False)
    atomic_to_csv(pd.concat(val_pairs, ignore_index=True), validation_pairs_path, index=False)
    atomic_to_csv(pd.concat(test_pairs, ignore_index=True), test_pairs_path, index=False)

    atomic_to_json(
        {
            "status": "completed",
            "model_name": ctx.model_name,
            "checkpoint": ctx.checkpoint,
            "validation_year": ctx.validation_year,
            "test_year": ctx.test_year,
            "coefficients": {k: coeffs[k] for k in ["a", "b", "c"]},
            "selected_symbolic_policy": selected_formula_policy,
            "run_config": run_config,
        },
        complete_path,
    )
    return result_paths


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir) / args.prefix
    output_root.mkdir(parents=True, exist_ok=True)
    contexts = load_year_contexts(args)
    selected_formulas = selected_parsimonious_formulas(args)
    atomic_to_csv(selected_formulas, output_root / "selected_symbolic_formulas.csv", index=False)

    years = list(range(args.first_test_year, args.final_test_year + 1))
    tasks = []
    for year in years:
        formula_row = selected_formulas[selected_formulas["test_year"] == year].iloc[0].to_dict()
        tasks.append((contexts[year], formula_row, args, output_root))

    if int(args.workers) <= 1:
        results = [run_one_year(task) for task in tasks]
    else:
        results = []
        max_workers = min(int(args.workers), len(tasks))
        print(f"[hull-white] processing {len(tasks)} years with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_one_year, task): task[0].test_year for task in tasks}
            for future in as_completed(futures):
                year = futures[future]
                result = future.result()
                results.append(result)
                print(f"[hull-white] completed year {year}: {result['year_dir']}")

    results = sorted(results, key=lambda item: int(item["test_year"]))
    coefficients = pd.concat(
        [pd.read_csv(item["coefficients_path"]) for item in results],
        ignore_index=True,
    )
    validation_metrics = pd.concat(
        [pd.read_csv(item["validation_metrics_path"]) for item in results],
        ignore_index=True,
    )
    test_metrics = pd.concat(
        [pd.read_csv(item["test_metrics_path"]) for item in results],
        ignore_index=True,
    )
    validation_bootstrap = pd.concat(
        [pd.read_csv(item["validation_bootstrap_path"]) for item in results],
        ignore_index=True,
    )
    test_bootstrap = pd.concat(
        [pd.read_csv(item["test_bootstrap_path"]) for item in results],
        ignore_index=True,
    )
    validation_pairs = pd.concat(
        [pd.read_csv(item["validation_pairs_path"]) for item in results],
        ignore_index=True,
    )
    test_pairs = pd.concat(
        [pd.read_csv(item["test_pairs_path"]) for item in results],
        ignore_index=True,
    )

    atomic_to_csv(coefficients, output_root / "hull_white_coefficients_all.csv", index=False)
    atomic_to_csv(
        validation_metrics,
        output_root / "hull_white_validation_episode_metrics_all.csv",
        index=False,
    )
    atomic_to_csv(test_metrics, output_root / "hull_white_test_episode_metrics_all.csv", index=False)
    atomic_to_csv(
        validation_bootstrap,
        output_root / "hull_white_validation_bootstrap_all.csv",
        index=False,
    )
    atomic_to_csv(test_bootstrap, output_root / "hull_white_test_bootstrap_all.csv", index=False)
    atomic_to_csv(
        validation_pairs,
        output_root / "hull_white_validation_paired_episodes_all.csv",
        index=False,
    )
    atomic_to_csv(test_pairs, output_root / "hull_white_test_paired_episodes_all.csv", index=False)
    atomic_to_json(
        {
            "prefix": args.prefix,
            "first_test_year": args.first_test_year,
            "final_test_year": args.final_test_year,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "workers": args.workers,
            "transaction_cost": args.transaction_cost,
            "kappa": args.kappa,
            "reward_exponent": args.reward_exponent,
            "zero_tolerance": args.zero_tolerance,
            "ridge_alpha": args.ridge_alpha,
            "min_abs_ds": args.min_abs_ds,
            "clip_delta": not args.no_clip_delta,
            "vega_iv_column": "iv_preferred_else_v",
            "base_policy": "agent",
            "formula_selection": args.formula_selection,
            "policy": POLICY_NAME,
            "outputs": {
                "coefficients": str(output_root / "hull_white_coefficients_all.csv"),
                "test_bootstrap": str(output_root / "hull_white_test_bootstrap_all.csv"),
                "validation_bootstrap": str(output_root / "hull_white_validation_bootstrap_all.csv"),
            },
        },
        output_root / "hull_white_manifest.json",
    )
    print(f"[hull-white] completed. Outputs are in {output_root}")


if __name__ == "__main__":
    run(parse_args())
