"""
Validation-selected Black-Scholes haircut benchmark.

This script implements a deliberately simple non-neural benchmark:

    delta_haircut = clip(lambda * delta_BS, 0, 1)

For each walk-forward test year Y, lambda is selected on validation year Y-1
from the fixed grid {0.70, 0.75, ..., 1.05}.  The selected lambda is then
evaluated out of sample on year Y.  The script never retrains or retests the
neural agent or symbolic formulas.  It only:

1. replays the already cached validation/test trade-step files to compute the
   haircut benchmark;
2. selects lambda on validation only;
3. compares the selected haircut with BS, the raw agent, and the paper's
   selected parsimonious symbolic formula using the same two-stage bootstrap
   convention used elsewhere in the project.

Suggested manual run:

    python scripts/run_haircut_benchmark.py --prefix final_WF_exp1_k1_test --n-bootstrap 10000

Outputs are written under results/haircut_check/<prefix>/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = [
    "mean",
    "std",
    "rew",
    "cvar",
    "log_down_var_ratio",
    "log_var_ratio",
    "antonov",
]
CI_LEVELS = (0.90, 0.95, 0.99)
DEFAULT_LAMBDAS = tuple(round(x, 2) for x in np.arange(0.70, 1.05 + 0.0001, 0.05))
DEFAULT_SELECTION_METRIC = "rew"

# We keep the symbolic-formula comparator aligned with the main paper rule in
# make_paper_figures.py: pool all Hall-of-Fame formulas from the three families
# and select the lowest-complexity formula whose validation MAE to the actually
# traded raw agent is within 10% of the best pooled validation MAE.
PARSIMONIOUS_TOLERANCE = 0.10


@dataclass(frozen=True)
class YearContext:
    test_year: int
    validation_year: int
    model_name: str
    checkpoint: str
    year_dir: Path
    stem: str
    train_start_year: int | None = None
    train_end_year: int | None = None

    @property
    def validation_steps_path(self) -> Path:
        return self.year_dir / f"{self.stem}_validation_trade_steps.csv"

    @property
    def test_steps_path(self) -> Path:
        return self.year_dir / f"{self.stem}_test_trade_steps.csv"

    @property
    def validation_episode_metrics_path(self) -> Path:
        return self.year_dir / f"{self.stem}_validation_episode_metrics.csv"

    @property
    def test_episode_metrics_path(self) -> Path:
        return self.year_dir / f"{self.stem}_test_episode_metrics.csv"

    @property
    def validation_hof_episode_metrics_path(self) -> Path:
        return self.year_dir / f"{self.stem}_validation_hof_episode_metrics.csv"

    @property
    def test_hof_episode_metrics_path(self) -> Path:
        return self.year_dir / f"{self.stem}_test_hof_episode_metrics.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="final_WF_exp1_k1_test")
    parser.add_argument("--walkforward-dir", default="results/interpret_real_walkforward")
    parser.add_argument("--output-dir", default="results/haircut_check")
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
        "--lambdas",
        default=",".join(f"{x:.2f}" for x in DEFAULT_LAMBDAS),
        help="Comma-separated lambda grid. Default is 0.70,0.75,...,1.05.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=METRICS,
        default=DEFAULT_SELECTION_METRIC,
        help=(
            "Validation metric used to choose lambda. Higher is better for "
            "mean, reward, and CVaR; lower is better for std, variance ratios, "
            "and Antonov."
        ),
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0,
        help=(
            "Stock transaction-cost rate used when replaying the haircut. "
            "The current paper experiments use zero; set explicitly if needed."
        ),
    )
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--reward-exponent", type=float, default=1.0)
    parser.add_argument(
        "--zero-tolerance",
        type=float,
        default=1e-10,
        help=(
            "Numerical tolerance used when deciding whether a bootstrap "
            "interval excludes zero. This prevents exact-BS lambda=1.00 "
            "comparisons from being marked significant because of roundoff."
        ),
    )
    parser.add_argument(
        "--no-clip-delta",
        action="store_true",
        help=(
            "Do not clip lambda*BS delta to [0,1]. By default we clip, matching "
            "the action range available to the learned call-hedging policy."
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
        help="Recompute cached haircut episode metrics and bootstrap outputs.",
    )
    return parser.parse_args()


def parse_lambda_grid(text: str) -> list[float]:
    values = [round(float(part.strip()), 10) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("Lambda grid is empty")
    if len(set(values)) != len(values):
        raise ValueError(f"Lambda grid contains duplicates: {values}")
    return values


def lambda_slug(lam: float) -> str:
    return f"{lam:.2f}".replace(".", "p")


def atomic_to_csv(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, **kwargs)
    os.replace(tmp, path)


def atomic_to_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def load_year_contexts(args: argparse.Namespace) -> dict[int, YearContext]:
    root = Path(args.walkforward_dir)
    path = root / "walkforward_test_episode_metrics_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate episode metrics: {path}")

    df = pd.read_csv(path)
    df = df[(df["policy"] == "agent") & (df["model_name"].astype(str).str.startswith(args.prefix))]
    contexts: dict[int, YearContext] = {}
    for year in range(args.first_test_year, args.final_test_year + 1):
        group = df[df["test_year"] == year]
        if group.empty:
            raise FileNotFoundError(
                f"No agent test metrics found for year {year} and prefix {args.prefix}"
            )
        row = group.iloc[0]
        source_file = Path(str(row["source_file"]))
        if not source_file.is_absolute():
            source_file = Path.cwd() / source_file
        year_dir = source_file.parent
        stem = source_file.name.removesuffix("_test_episode_metrics.csv")
        checkpoint = str(row["checkpoint"])
        contexts[year] = YearContext(
            test_year=year,
            validation_year=int(row["validation_year"]),
            model_name=str(row["model_name"]),
            checkpoint=checkpoint,
            year_dir=year_dir,
            stem=stem,
            train_start_year=int(row["train_start_year"])
            if pd.notna(row.get("train_start_year"))
            else None,
            train_end_year=int(row["train_end_year"])
            if pd.notna(row.get("train_end_year"))
            else None,
        )

        required = [
            contexts[year].validation_steps_path,
            contexts[year].test_steps_path,
            contexts[year].validation_episode_metrics_path,
            contexts[year].test_episode_metrics_path,
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Year {year} is missing required cached walk-forward files: {missing}"
            )
    return contexts


def sort_trade_steps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_original_row_order"] = np.arange(len(out))
    keys = ["episode"]
    if "Date" in out.columns:
        keys.append("Date")
    if "step" in out.columns:
        keys.append("step")
    keys.append("_original_row_order")
    return (
        out.sort_values(keys, kind="mergesort")
        .drop(columns=["_original_row_order"])
        .reset_index(drop=True)
    )


def reward_from_pnl(
    pnl: np.ndarray, kappa: float = 1.0, reward_exponent: float = 1.0
) -> np.ndarray:
    # This mirrors Env.step: PnL is first converted to percent-like units,
    # then the asymmetric reward is multiplied by 10.
    pnl_scaled = pnl * 100.0
    return (0.03 + pnl_scaled - kappa * np.abs(pnl_scaled) ** reward_exponent) * 10.0


def replay_haircut_steps(
    source_steps: pd.DataFrame,
    lam: float,
    args: argparse.Namespace,
    split: str,
    split_year: int,
) -> pd.DataFrame:
    df = sort_trade_steps(source_steps)
    required = ["episode", "bs_delta", "S0", "S-1", "P0", "P-1"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Trade-step file is missing required columns: {missing}")

    out = df.copy()
    bs_delta = out["bs_delta"].to_numpy(dtype=float)
    delta = lam * bs_delta
    if not args.no_clip_delta:
        delta = np.clip(delta, 0.0, 1.0)

    prev_pos = np.zeros(len(out), dtype=float)
    for _, idxs in out.groupby("episode", sort=False).indices.items():
        idxs = np.asarray(idxs, dtype=int)
        if len(idxs) > 1:
            prev_pos[idxs[1:]] = -delta[idxs[:-1]]

    s_start = out["S-1"].to_numpy(dtype=float)
    d_s = out["S0"].to_numpy(dtype=float) - s_start
    d_p = out["P0"].to_numpy(dtype=float) - out["P-1"].to_numpy(dtype=float)
    tc = -np.abs((-delta) - prev_pos) * s_start * float(args.transaction_cost)
    pnl = -delta * d_s + d_p + tc

    policy = f"haircut_lambda_{lambda_slug(lam)}"
    out["policy"] = policy
    out["chosen_delta"] = delta
    out["formula_delta"] = delta
    out["A Pos"] = -delta
    out["A TC"] = tc
    out["A PnL"] = pnl
    out["A PnL - TC"] = pnl - tc
    out["A Reward"] = reward_from_pnl(
        pnl, kappa=float(args.kappa), reward_exponent=float(args.reward_exponent)
    )
    out["haircut_lambda"] = float(lam)
    out["split"] = split
    out["split_year"] = int(split_year)
    return out


def calculate_episode_antonov(group: pd.DataFrame) -> pd.Series:
    """
    Calculate Antonov's hedging defect using the corrected interval convention.

    This is intentionally copied into the benchmark script so the haircut CSVs
    are self-contained.  DateEnd-aware CSVs use every interval; legacy CSVs
    without DateEnd fall back to the cached-artifact convention.  The interest
    rate is taken from the row-level r column when available; otherwise the
    visible fallback is 1%.
    """
    df = group.copy()
    df["_original_row_order"] = np.arange(len(df))
    df = df.sort_values(["Date", "_original_row_order"], kind="mergesort").reset_index(drop=True)
    n_steps = len(df)
    if n_steps < 2:
        return pd.Series({"D_policy": np.nan, "D_BS": np.nan})

    p_start = df["P-1"].to_numpy(dtype=float)
    s_start = df["S-1"].to_numpy(dtype=float)
    delta_policy = df["A Pos"].to_numpy(dtype=float)
    delta_bs = df["B Pos"].to_numpy(dtype=float)
    tc_policy = df["A TC"].to_numpy(dtype=float) if "A TC" in df.columns else np.zeros(n_steps)
    tc_bs = df["B TC"].to_numpy(dtype=float) if "B TC" in df.columns else np.zeros(n_steps)

    has_full_interval_dates = (
        "DateEnd" in df.columns
        and df["DateEnd"].notna().all()
        and (df["DateEnd"].astype(str).str.len() >= 10).all()
    )
    if has_full_interval_dates:
        dt_years = (
            pd.to_datetime(df["DateEnd"]) - pd.to_datetime(df["Date"])
        ).dt.days.to_numpy(dtype=float) / 365.0
        h_policy = float(df["P0"].iloc[-1])
        h_bs = float(df["P0"].iloc[-1])
        loop_range: Iterable[int] = range(n_steps - 1, -1, -1)
    else:
        date_series = pd.to_datetime(df["Date"])
        dt_years = (date_series.shift(-1) - date_series).dt.days.to_numpy(dtype=float) / 365.0
        h_policy = float(p_start[-1])
        h_bs = float(p_start[-1])
        loop_range = range(n_steps - 2, -1, -1)

    chi2_policy = 0.0
    chi2_bs = 0.0
    total_t = 0.0
    for t in loop_range:
        dt = float(dt_years[t])
        if not np.isfinite(dt) or dt <= 0.0:
            continue
        if "r" in df.columns and pd.notna(df["r"].iloc[t]):
            r = float(df["r"].iloc[t])
            if r > 1.0:
                r /= 100.0
        else:
            r = 0.01
        discount = float(np.exp(-r * dt))
        s_end = float(df["S0"].iloc[t]) if has_full_interval_dates else float(s_start[t + 1])

        h_policy = discount * h_policy + delta_policy[t] * (
            discount * s_end - s_start[t]
        ) + tc_policy[t]
        h_bs = discount * h_bs + delta_bs[t] * (discount * s_end - s_start[t]) + tc_bs[t]
        chi2_policy += ((h_policy - p_start[t]) ** 2) * dt
        chi2_bs += ((h_bs - p_start[t]) ** 2) * dt
        total_t += dt

    if total_t <= 0.0:
        return pd.Series({"D_policy": np.nan, "D_BS": np.nan})
    return pd.Series(
        {
            "D_policy": float(np.sqrt(chi2_policy / total_t)),
            "D_BS": float(np.sqrt(chi2_bs / total_t)),
        }
    )


def episode_metrics_from_steps(trade_steps: pd.DataFrame) -> pd.DataFrame:
    trade_steps = trade_steps.copy()
    # Older cached files should have transaction-cost columns, but the haircut
    # diagnostic should still be usable on a stripped step file.  The paper
    # experiments use zero transaction costs, so this fallback is conservative.
    for col in ["A TC", "B TC"]:
        if col not in trade_steps.columns:
            trade_steps[col] = 0.0
    antonov = (
        trade_steps.groupby(["policy", "episode"], sort=False)
        .apply(calculate_episode_antonov, include_groups=False)
        .reset_index()
    )
    sums = (
        trade_steps.groupby(["policy", "episode"], sort=False)[
            ["A PnL", "B PnL", "A Reward", "B Reward", "A TC", "B TC"]
        ]
        .sum()
        .reset_index()
    )
    starts = (
        trade_steps.groupby(["policy", "episode"], sort=False)["Date"]
        .first()
        .reset_index()
        .rename(columns={"Date": "start_date"})
    )
    out = sums.merge(starts, on=["policy", "episode"], how="left").merge(
        antonov, on=["policy", "episode"], how="left"
    )
    return out


def table_from_metrics(df: pd.DataFrame, policy: str, side: str) -> pd.DataFrame:
    if side not in {"A", "B"}:
        raise ValueError(f"Unknown side: {side}")
    sub = df[df["policy"] == policy].copy()
    if sub.empty:
        raise ValueError(f"Policy {policy!r} not found in episode metrics")
    if side == "A":
        cols = {"A PnL": "PnL", "A Reward": "Reward", "D_policy": "Antonov"}
    else:
        cols = {"B PnL": "PnL", "B Reward": "Reward", "D_BS": "Antonov"}
    return sub.rename(columns=cols)[["episode", "start_date", "PnL", "Reward", "Antonov"]]


def build_pair_table(
    left_df: pd.DataFrame,
    left_policy: str,
    left_side: str,
    right_df: pd.DataFrame,
    right_policy: str,
    right_side: str,
) -> pd.DataFrame:
    left = table_from_metrics(left_df, left_policy, left_side)
    right = table_from_metrics(right_df, right_policy, right_side)
    pair = left.merge(right, on=["episode", "start_date"], suffixes=("_left", "_right"))
    return pair.dropna(subset=["Antonov_left", "Antonov_right"]).copy()


def downside_second_moment(values: np.ndarray) -> float:
    downside = np.where(values < 0.0, values, 0.0)
    return float(np.sum(downside**2) / len(downside))


def cvar_5(values: np.ndarray) -> float:
    n_tail = max(1, int(len(values) * 0.05))
    return float(np.mean(np.partition(values, n_tail - 1)[:n_tail]))


def metric_point_estimates(pair_df: pd.DataFrame) -> dict[str, float]:
    left = pair_df["PnL_left"].to_numpy(dtype=float) * 100.0
    right = pair_df["PnL_right"].to_numpy(dtype=float) * 100.0
    var_left = float(np.var(left, ddof=1))
    var_right = float(np.var(right, ddof=1))
    down_left = downside_second_moment(left)
    down_right = downside_second_moment(right)
    return {
        "mean": float(np.mean(left - right)),
        "std": float(np.sqrt(var_left) - np.sqrt(var_right)),
        "rew": float(np.mean(pair_df["Reward_left"] - pair_df["Reward_right"])),
        "cvar": cvar_5(left) - cvar_5(right),
        "log_down_var_ratio": (
            float(np.log(down_left / down_right))
            if down_left > 1e-12 and down_right > 1e-12
            else np.nan
        ),
        "log_var_ratio": (
            float(np.log(var_left / var_right))
            if var_left > 1e-12 and var_right > 1e-12
            else np.nan
        ),
        "antonov": float(np.nanmean(pair_df["Antonov_left"] - pair_df["Antonov_right"])),
    }


def metric_levels(pair_df: pd.DataFrame) -> dict[str, float]:
    left = pair_df["PnL_left"].to_numpy(dtype=float) * 100.0
    right = pair_df["PnL_right"].to_numpy(dtype=float) * 100.0
    return {
        "left_mean": float(np.mean(left)),
        "right_mean": float(np.mean(right)),
        "left_std": float(np.std(left, ddof=1)),
        "right_std": float(np.std(right, ddof=1)),
        "left_reward": float(np.mean(pair_df["Reward_left"])),
        "right_reward": float(np.mean(pair_df["Reward_right"])),
        "left_cvar": cvar_5(left),
        "right_cvar": cvar_5(right),
        "left_downside_second_moment": downside_second_moment(left),
        "right_downside_second_moment": downside_second_moment(right),
        "left_variance": float(np.var(left, ddof=1)),
        "right_variance": float(np.var(right, ddof=1)),
        "left_antonov": float(np.nanmean(pair_df["Antonov_left"])),
        "right_antonov": float(np.nanmean(pair_df["Antonov_right"])),
    }


def bootstrap_pair(
    pair_df: pd.DataFrame, n_bootstrap: int, seed: int, zero_tolerance: float
) -> dict[str, dict[str, float]]:
    pair_df = pair_df.dropna(subset=["Antonov_left", "Antonov_right"]).copy()
    if pair_df.empty:
        raise ValueError("Pair table is empty after Antonov dropna")
    clusters = sorted(pair_df["start_date"].unique())
    if not clusters:
        raise ValueError("No bootstrap clusters available")
    cluster_frames = [pair_df[pair_df["start_date"] == cluster] for cluster in clusters]
    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(n_bootstrap, dtype=float) for metric in METRICS}

    for i in range(n_bootstrap):
        sampled_clusters = rng.integers(0, len(cluster_frames), size=len(cluster_frames))
        pieces = []
        for idx in sampled_clusters:
            cluster = cluster_frames[idx]
            sampled_rows = rng.integers(0, len(cluster), size=len(cluster))
            pieces.append(cluster.iloc[sampled_rows])
        boot_sample = pd.concat(pieces, ignore_index=True)
        estimates = metric_point_estimates(boot_sample)
        for metric in METRICS:
            draws[metric][i] = estimates[metric]

    point = metric_point_estimates(pair_df)
    levels = metric_levels(pair_df)
    out: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        values = draws[metric][np.isfinite(draws[metric])]
        row: dict[str, float] = {
            "point_estimate": point[metric],
            "bootstrap_center": float(np.mean(values)) if len(values) else np.nan,
            "bootstrap_std": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
            **levels,
        }
        for level in CI_LEVELS:
            suffix = int(round(level * 100))
            if len(values):
                alpha = (1.0 - level) / 2.0
                low, high = np.quantile(values, [alpha, 1.0 - alpha])
                row[f"ci_low_{suffix}"] = float(low)
                row[f"ci_high_{suffix}"] = float(high)
                row[f"significant_{suffix}"] = int(
                    low > zero_tolerance or high < -zero_tolerance
                )
            else:
                row[f"ci_low_{suffix}"] = np.nan
                row[f"ci_high_{suffix}"] = np.nan
                row[f"significant_{suffix}"] = 0
        out[metric] = row
    return out


def stable_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(text))
    return int((base_seed + offset) % (2**32 - 1))


def summarize_pair(
    pair_df: pd.DataFrame,
    comparison: str,
    left_policy: str,
    right_policy: str,
    ctx: YearContext,
    split: str,
    split_year: int,
    args: argparse.Namespace,
    haircut_lambda: float,
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
                "haircut_lambda": float(haircut_lambda),
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


def lower_is_better(metric: str) -> bool:
    return metric in {"std", "log_down_var_ratio", "log_var_ratio", "antonov"}


def validation_metric_for_lambda(metrics: pd.DataFrame, policy: str, metric: str) -> float:
    pair = build_pair_table(metrics, policy, "A", metrics, policy, "B")
    return float(metric_point_estimates(pair)[metric])


def choose_lambda(validation_summary: pd.DataFrame, metric: str) -> pd.Series:
    df = validation_summary.copy()
    df["distance_to_one"] = np.abs(df["haircut_lambda"] - 1.0)
    ascending = [lower_is_better(metric), True, True]
    selected = df.sort_values(
        [f"validation_{metric}", "distance_to_one", "haircut_lambda"],
        ascending=ascending,
        kind="mergesort",
    ).iloc[0]
    return selected


def compute_or_load_haircut_grid(
    ctx: YearContext,
    lambdas: list[float],
    args: argparse.Namespace,
    out_dir: Path,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_year = ctx.validation_year if split == "validation" else ctx.test_year
    metrics_path = out_dir / f"{ctx.stem}_{split}_haircut_grid_episode_metrics.csv"
    summary_path = out_dir / f"{ctx.stem}_{split}_haircut_grid_summary.csv"
    manifest_path = out_dir / f"{ctx.stem}_{split}_haircut_grid_manifest.json"
    replay_config = {
        "model_name": ctx.model_name,
        "checkpoint": ctx.checkpoint,
        "test_year": int(ctx.test_year),
        "split": split,
        "split_year": int(split_year),
        "base_policy": "agent",
        "lambdas": [float(x) for x in lambdas],
        "transaction_cost": float(args.transaction_cost),
        "kappa": float(args.kappa),
        "reward_exponent": float(args.reward_exponent),
        "clip_delta": not bool(args.no_clip_delta),
    }
    if metrics_path.exists() and summary_path.exists() and manifest_path.exists() and not args.force:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("replay_config") == replay_config:
                print(f"[haircut] loading cached {split} haircut grid for {ctx.test_year}")
                return pd.read_csv(metrics_path), pd.read_csv(summary_path)
        except (OSError, json.JSONDecodeError):
            pass

    steps_path = ctx.validation_steps_path if split == "validation" else ctx.test_steps_path
    print(f"[haircut] replaying {split} haircut grid for test year {ctx.test_year}: {steps_path}")
    source_steps = pd.read_csv(steps_path)
    if "policy" in source_steps.columns:
        source_steps = source_steps[source_steps["policy"] == "agent"].copy()
    if source_steps.empty:
        raise ValueError(f"No raw agent rows found in {steps_path}; cannot replay haircut benchmark")
    metric_frames = []
    summary_rows = []
    for lam in lambdas:
        policy = f"haircut_lambda_{lambda_slug(lam)}"
        steps = replay_haircut_steps(source_steps, lam, args, split, split_year)
        metrics = episode_metrics_from_steps(steps)
        metric_frames.append(metrics)
        pair = build_pair_table(metrics, policy, "A", metrics, policy, "B")
        point = metric_point_estimates(pair)
        summary_rows.append(
            {
                "model_name": ctx.model_name,
                "checkpoint": ctx.checkpoint,
                "validation_year": ctx.validation_year,
                "test_year": ctx.test_year,
                "split": split,
                "split_year": split_year,
                "haircut_lambda": float(lam),
                "policy": policy,
                "n_episodes": int(len(pair)),
                "n_clusters": int(pair["start_date"].nunique()),
                **{f"validation_{k}" if split == "validation" else f"test_{k}": v for k, v in point.items()},
            }
        )

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    atomic_to_csv(all_metrics, metrics_path, index=False)
    atomic_to_csv(summary, summary_path, index=False)
    atomic_to_json(
        {
            "status": "completed",
            "replay_config": replay_config,
            "episode_metrics": str(metrics_path),
            "summary": str(summary_path),
        },
        manifest_path,
    )
    return all_metrics, summary


def selected_parsimonious_formulas(args: argparse.Namespace) -> pd.DataFrame:
    path = Path(args.walkforward_dir) / "walkforward_validation_hof_fidelity_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing HOF validation fidelity file: {path}")
    hof = pd.read_csv(path)
    hof = hof[
        (hof["model_name"].astype(str).str.startswith(args.prefix))
        & (hof["target_name"] == "actual_traded_agent")
        & (hof["test_year"].between(args.first_test_year, args.final_test_year))
    ].copy()
    if hof.empty:
        raise ValueError("No validation HOF fidelity rows found for selected formula rule")

    rows = []
    for year, group in hof.groupby("test_year", sort=True):
        best_mae = float(group["mae"].min())
        eligible = group[group["mae"] <= (1.0 + PARSIMONIOUS_TOLERANCE) * best_mae]
        selected = eligible.sort_values(
            ["complexity", "mae", "hof_index", "candidate"], kind="mergesort"
        ).iloc[0]
        best = group.sort_values(["mae", "complexity", "hof_index", "candidate"]).iloc[0]
        rows.append(
            {
                "test_year": int(year),
                "policy": selected["policy"],
                "candidate": selected["candidate"],
                "hof_index": int(selected["hof_index"]),
                "complexity": int(selected["complexity"]),
                "validation_target_mae": float(selected["mae"]),
                "best_validation_target_mae": best_mae,
                "best_fit_policy": best["policy"],
                "best_fit_candidate": best["candidate"],
                "best_fit_complexity": int(best["complexity"]),
                "equation": selected.get("equation", ""),
                "selection_rule": "parsimonious_10pct_actual_traded_agent",
            }
        )
    selected = pd.DataFrame(rows).sort_values("test_year")
    missing = set(range(args.first_test_year, args.final_test_year + 1)) - set(
        selected["test_year"].astype(int)
    )
    if missing:
        raise ValueError(f"Missing selected symbolic formulas for years: {sorted(missing)}")
    return selected


def load_existing_episode_metrics(ctx: YearContext, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    agent_path = ctx.validation_episode_metrics_path if split == "validation" else ctx.test_episode_metrics_path
    hof_path = (
        ctx.validation_hof_episode_metrics_path
        if split == "validation"
        else ctx.test_hof_episode_metrics_path
    )
    if not agent_path.exists():
        raise FileNotFoundError(f"Missing agent episode metrics: {agent_path}")
    if not hof_path.exists():
        raise FileNotFoundError(f"Missing HOF episode metrics: {hof_path}")
    return pd.read_csv(agent_path), pd.read_csv(hof_path)


def compare_selected_haircut(
    ctx: YearContext,
    selected_lambda: float,
    selected_formula_policy: str,
    haircut_metrics: pd.DataFrame,
    args: argparse.Namespace,
    split: str,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    split_year = ctx.validation_year if split == "validation" else ctx.test_year
    haircut_policy = f"haircut_lambda_{lambda_slug(selected_lambda)}"
    haircut_selected = haircut_metrics[haircut_metrics["policy"] == haircut_policy].copy()
    if haircut_selected.empty:
        raise ValueError(f"Selected haircut policy {haircut_policy} missing for {ctx.test_year} {split}")

    agent_metrics, hof_metrics = load_existing_episode_metrics(ctx, split)
    comparisons = []
    pairs_to_save: list[pd.DataFrame] = []

    specs = [
        (
            "haircut_vs_bs",
            haircut_selected,
            haircut_policy,
            "A",
            haircut_selected,
            haircut_policy,
            "B",
        ),
        (
            "haircut_vs_agent",
            haircut_selected,
            haircut_policy,
            "A",
            agent_metrics,
            "agent",
            "A",
        ),
        (
            "haircut_vs_selected_formula",
            haircut_selected,
            haircut_policy,
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
        pair["right_policy"] = "bs" if label == "haircut_vs_bs" else right_policy
        pair["test_year"] = ctx.test_year
        pair["split"] = split
        pair["split_year"] = split_year
        pairs_to_save.append(pair)
        comparisons.append(
            summarize_pair(
                pair,
                comparison=label,
                left_policy=left_policy,
                right_policy="bs" if label == "haircut_vs_bs" else right_policy,
                ctx=ctx,
                split=split,
                split_year=split_year,
                args=args,
                haircut_lambda=selected_lambda,
            )
        )

    return pd.concat(comparisons, ignore_index=True), pairs_to_save


def run_one_year(task: tuple[YearContext, dict, list[float], argparse.Namespace, Path]) -> dict:
    """
    Process one walk-forward year.

    This function is deliberately top-level so Windows multiprocessing can
    pickle it.  It writes only files inside the year-specific output directory;
    the parent process later concatenates those files into aggregate CSVs.  That
    makes crashes recoverable: completed year folders remain valid and partial
    files are protected by atomic writes.
    """
    ctx, formula_row, lambdas, args, output_root = task
    year_out = output_root / f"{ctx.test_year}_{ctx.model_name}"
    year_out.mkdir(parents=True, exist_ok=True)
    selected_formula_policy = str(formula_row["policy"])
    selected_lambda_path = year_out / f"{ctx.stem}_haircut_selected_lambda.csv"
    validation_bootstrap_path = year_out / f"{ctx.stem}_validation_haircut_bootstrap.csv"
    test_bootstrap_path = year_out / f"{ctx.stem}_test_haircut_bootstrap.csv"
    validation_pairs_path = year_out / f"{ctx.stem}_validation_haircut_paired_episodes.csv"
    test_pairs_path = year_out / f"{ctx.stem}_test_haircut_paired_episodes.csv"
    complete_path = year_out / f"{ctx.stem}_haircut_complete.json"
    result_paths = {
        "test_year": ctx.test_year,
        "year_dir": str(year_out),
        "validation_grid_episode_path": str(year_out / f"{ctx.stem}_validation_haircut_grid_episode_metrics.csv"),
        "test_grid_episode_path": str(year_out / f"{ctx.stem}_test_haircut_grid_episode_metrics.csv"),
        "validation_grid_manifest_path": str(year_out / f"{ctx.stem}_validation_haircut_grid_manifest.json"),
        "test_grid_manifest_path": str(year_out / f"{ctx.stem}_test_haircut_grid_manifest.json"),
        "validation_grid_summary_path": str(year_out / f"{ctx.stem}_validation_haircut_grid_summary.csv"),
        "test_grid_summary_path": str(year_out / f"{ctx.stem}_test_haircut_grid_summary.csv"),
        "selected_lambda_path": str(selected_lambda_path),
        "validation_bootstrap_path": str(validation_bootstrap_path),
        "test_bootstrap_path": str(test_bootstrap_path),
        "validation_pairs_path": str(validation_pairs_path),
        "test_pairs_path": str(test_pairs_path),
    }
    run_config = {
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "lambdas": [float(x) for x in lambdas],
        "selection_metric": str(args.selection_metric),
        "transaction_cost": float(args.transaction_cost),
        "kappa": float(args.kappa),
        "reward_exponent": float(args.reward_exponent),
        "clip_delta": not bool(args.no_clip_delta),
        "zero_tolerance": float(args.zero_tolerance),
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
                print(f"[haircut] year {ctx.test_year}: complete cached outputs found")
                return result_paths
        except (OSError, json.JSONDecodeError):
            pass

    val_metrics, val_summary = compute_or_load_haircut_grid(
        ctx, lambdas, args, year_out, "validation"
    )
    test_metrics, test_summary = compute_or_load_haircut_grid(
        ctx, lambdas, args, year_out, "test"
    )

    selected = choose_lambda(val_summary, args.selection_metric)
    selected_lambda = float(selected["haircut_lambda"])
    selected_lambda_row = {
        "model_name": ctx.model_name,
        "checkpoint": ctx.checkpoint,
        "train_start_year": ctx.train_start_year,
        "train_end_year": ctx.train_end_year,
        "validation_year": ctx.validation_year,
        "test_year": ctx.test_year,
        "selection_metric": args.selection_metric,
        "selected_lambda": selected_lambda,
        "selected_policy": f"haircut_lambda_{lambda_slug(selected_lambda)}",
        "selected_symbolic_policy": selected_formula_policy,
        "selected_symbolic_candidate": formula_row["candidate"],
        "selected_symbolic_complexity": int(formula_row["complexity"]),
        **{key: selected[key] for key in selected.index if str(key).startswith("validation_")},
    }
    atomic_to_csv(pd.DataFrame([selected_lambda_row]), selected_lambda_path, index=False)
    print(
        f"[haircut] test year {ctx.test_year}: selected lambda={selected_lambda:.2f} "
        f"on validation {ctx.validation_year} by {args.selection_metric}"
    )

    val_boot, val_pairs = compare_selected_haircut(
        ctx,
        selected_lambda,
        selected_formula_policy,
        val_metrics,
        args,
        split="validation",
    )
    test_boot, test_pairs = compare_selected_haircut(
        ctx,
        selected_lambda,
        selected_formula_policy,
        test_metrics,
        args,
        split="test",
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
            "selected_lambda": selected_lambda,
            "selected_symbolic_policy": selected_formula_policy,
            "selection_metric": args.selection_metric,
            "run_config": run_config,
        },
        complete_path,
    )

    return result_paths


def run(args: argparse.Namespace) -> None:
    lambdas = parse_lambda_grid(args.lambdas)
    output_root = Path(args.output_dir) / args.prefix
    output_root.mkdir(parents=True, exist_ok=True)

    contexts = load_year_contexts(args)
    selected_formulas = selected_parsimonious_formulas(args)
    atomic_to_csv(selected_formulas, output_root / "selected_symbolic_formulas.csv", index=False)

    years = list(range(args.first_test_year, args.final_test_year + 1))
    tasks = []
    for year in years:
        formula_row = selected_formulas[selected_formulas["test_year"] == year].iloc[0].to_dict()
        tasks.append((contexts[year], formula_row, lambdas, args, output_root))

    if int(args.workers) <= 1:
        results = [run_one_year(task) for task in tasks]
    else:
        results = []
        max_workers = min(int(args.workers), len(tasks))
        print(f"[haircut] processing {len(tasks)} years with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(run_one_year, task): task[0].test_year for task in tasks}
            for future in as_completed(futures):
                year = futures[future]
                result = future.result()
                results.append(result)
                print(f"[haircut] completed year {year}: {result['year_dir']}")

    results = sorted(results, key=lambda item: int(item["test_year"]))

    validation_grid = pd.concat(
        [pd.read_csv(item["validation_grid_summary_path"]) for item in results],
        ignore_index=True,
    )
    test_grid = pd.concat(
        [pd.read_csv(item["test_grid_summary_path"]) for item in results],
        ignore_index=True,
    )
    selected_lambdas = pd.concat(
        [pd.read_csv(item["selected_lambda_path"]) for item in results],
        ignore_index=True,
    ).sort_values("test_year")
    validation_bootstrap = pd.concat(
        [pd.read_csv(item["validation_bootstrap_path"]) for item in results],
        ignore_index=True,
    )
    test_bootstrap = pd.concat(
        [pd.read_csv(item["test_bootstrap_path"]) for item in results],
        ignore_index=True,
    )

    validation_pair_paths = [item["validation_pairs_path"] for item in results]
    test_pair_paths = [item["test_pairs_path"] for item in results]

    atomic_to_csv(validation_grid, output_root / "haircut_validation_grid_summary_all.csv", index=False)
    atomic_to_csv(test_grid, output_root / "haircut_test_grid_summary_all.csv", index=False)
    atomic_to_csv(selected_lambdas, output_root / "haircut_selected_lambdas.csv", index=False)
    atomic_to_csv(validation_bootstrap, output_root / "haircut_validation_bootstrap_all.csv", index=False)
    atomic_to_csv(test_bootstrap, output_root / "haircut_test_bootstrap_all.csv", index=False)
    atomic_to_csv(
        pd.concat([pd.read_csv(path) for path in validation_pair_paths], ignore_index=True),
        output_root / "haircut_validation_paired_episodes_all.csv",
        index=False,
    )
    atomic_to_csv(
        pd.concat([pd.read_csv(path) for path in test_pair_paths], ignore_index=True),
        output_root / "haircut_test_paired_episodes_all.csv",
        index=False,
    )
    atomic_to_json(
        {
            "prefix": args.prefix,
            "first_test_year": args.first_test_year,
            "final_test_year": args.final_test_year,
            "lambdas": lambdas,
            "selection_metric": args.selection_metric,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "transaction_cost": args.transaction_cost,
            "kappa": args.kappa,
            "reward_exponent": args.reward_exponent,
            "clip_delta": not args.no_clip_delta,
            "zero_tolerance": args.zero_tolerance,
            "base_policy": "agent",
            "formula_selection": args.formula_selection,
            "outputs": {
                "selected_lambdas": str(output_root / "haircut_selected_lambdas.csv"),
                "test_bootstrap": str(output_root / "haircut_test_bootstrap_all.csv"),
                "validation_bootstrap": str(output_root / "haircut_validation_bootstrap_all.csv"),
            },
        },
        output_root / "haircut_manifest.json",
    )

    print(f"[haircut] completed. Outputs are in {output_root}")


if __name__ == "__main__":
    run(parse_args())
