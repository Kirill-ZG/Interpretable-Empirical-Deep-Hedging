"""
Publication-grade long-horizon retesting for empirical walk-forward models.

Purpose
-------
For a model class such as ``final_WF_exp1_k1_test``, the ordinary walk-forward
experiment tests the model selected for year Y only on year Y.  This script
asks a different robustness question:

    If the model first tested in year Y were kept fixed, how would it trade in
    years Y, Y+1, ..., final_year?

The script does not train anything.  It only:

1. finds already-trained checkpoints from ``results/testing/{prefix}{Y}_*.csv``;
2. builds yearly DATA_DIR folders from ``cleaned_data``;
3. retests each frozen actor on the requested future test years;
4. calculates the same paired metrics used in the paper-figure workflow;
5. writes two-stage bootstrap confidence intervals at 90%, 95%, and 99%.

The script never mutates the repository-level ``data`` folder.  Each worker
gets its own DATA_DIR through the environment, and all outputs are written
under ``results/long_horizon/<prefix>/``.

Example
-------
    python scripts/run_long_horizon.py --prefix final_WF_exp1_k1_test

The default runs three testing workers.  Use ``--skip-testing`` to recompute
only metrics/bootstrap from cached long-horizon result CSVs.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from empirical_deep_hedging.include.actor_critic import ActorCritic
from empirical_deep_hedging.include.env import Env
from empirical_deep_hedging.include.settings import getSettings, setSettings
from empirical_deep_hedging.include.utility import StatePrepare, maybe_make_dirs
from empirical_deep_hedging.testing import set_random_seeds, test_run


DEFAULT_PREFIX = "final_WF_exp1_k1_test"
DEFAULT_FIRST_TEST_YEAR = 2015
DEFAULT_FINAL_TEST_YEAR = 2023
DEFAULT_CLEANED_DATA_DIR = Path("cleaned_data")
DEFAULT_RESULTS_DIR = Path("results/testing")
DEFAULT_OUTPUT_ROOT = Path("results/long_horizon")
DEFAULT_DATA_ROOT = Path("data_long_horizon")

CI_LEVELS = (0.90, 0.95, 0.99)
BOOTSTRAP_METRICS = (
    "mean",
    "std",
    "tc",
    "rew",
    "cvar",
    "log_down_var_ratio",
    "log_var_ratio",
    "antonov",
)


@dataclass(frozen=True)
class RetestTask:
    """One frozen model/year evaluation task."""

    prefix: str
    model_year: int
    target_year: int
    model_name: str
    checkpoint: str
    data_dir: str
    raw_output_path: str
    force_retest: bool
    seed: int
    torch_num_threads: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--first-test-year", type=int, default=DEFAULT_FIRST_TEST_YEAR)
    parser.add_argument("--final-test-year", type=int, default=DEFAULT_FINAL_TEST_YEAR)
    parser.add_argument("--cleaned-data-dir", type=Path, default=DEFAULT_CLEANED_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parallel-workers", type=int, default=3)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=1,
        help=(
            "Torch threads per retesting worker.  Keeping this at 1 avoids CPU "
            "oversubscription when --parallel-workers is greater than one."
        ),
    )
    parser.add_argument(
        "--force-retest",
        action="store_true",
        help="Retest even if the long-horizon raw CSV already exists.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild cached yearly DATA_DIR folders from cleaned_data.",
    )
    parser.add_argument(
        "--skip-testing",
        action="store_true",
        help="Skip actor evaluation and recompute metrics/bootstrap from cached raw CSVs.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Retest and write episode metrics, but skip bootstrap summaries.",
    )
    return parser.parse_args()


def seed_script(seed: int) -> None:
    """Seed lightweight Python-side work done by the driver."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_cleaned_panel(cleaned_data_dir: Path) -> pd.DataFrame:
    """
    Load and normalize the OptionsDX panel used to build test DATA_DIR folders.

    
    We keep this transformation aligned with correction_antonov.py and the
    current preprocessing convention.  The long-horizon script only needs test
    data, but Env/DataKeeper also require train.csv and validation.csv to exist,
    so those are filled with a small deterministic dummy sample.
    """

    parquet_files = sorted(cleaned_data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {cleaned_data_dir}")

    needed_cols = [
        "quote_date",
        "underlying_last",
        "expire_date",
        "strike",
        "c_bid",
        "c_ask",
        "risk_free_rate",
        "dte",
    ]
    frames = [pd.read_parquet(path, columns=needed_cols) for path in parquet_files]
    df = pd.concat(frames, ignore_index=True)

    df = df.dropna(
        subset=[
            "quote_date",
            "underlying_last",
            "expire_date",
            "strike",
            "c_bid",
            "c_ask",
            "risk_free_rate",
        ]
    )
    df = df[(df["c_bid"] > 0) & (df["c_ask"] > df["c_bid"])]
    intrinsic_value = df["underlying_last"] - df["strike"]
    df = df[df["c_bid"] >= intrinsic_value]
    df = df[df["dte"] > 0]

    df = df.rename(
        columns={
            "quote_date": "quote_datetime",
            "expire_date": "expiration",
            "c_bid": "bid",
            "c_ask": "ask",
        }
    )
    df["underlying_bid"] = df["underlying_last"]
    df["underlying_ask"] = df["underlying_last"]
    df["ticker"] = "SPX"
    df["quote_datetime"] = df["quote_datetime"].astype(str).str.slice(0, 10)
    df["expiration"] = df["expiration"].astype(str).str.slice(0, 10)
    df["year"] = df["quote_datetime"].str.slice(0, 4).astype(int)
    return df


def process_env_csv(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a filtered OptionsDX frame into the CSV layout expected by DataKeeper.

    DataKeeper later slices ``loc[start:start+steps]``.  Sorting by option id and
    date, then resetting the index, guarantees that a selected start row is
    followed by consecutive observations of the same option contract.
    """

    keep_cols = [
        "quote_datetime",
        "expiration",
        "strike",
        "underlying_bid",
        "underlying_ask",
        "bid",
        "ask",
        "ticker",
    ]
    out = dataset[keep_cols].copy()
    out["option_id"] = out["expiration"] + "_" + out["strike"].astype(str)
    out = out.sort_values(["option_id", "quote_datetime"]).reset_index(drop=True)
    out["nbr_next_steps"] = out.groupby("option_id").cumcount(ascending=False)
    return out.drop(columns=["option_id"])


def ensure_yearly_data_dirs(args: argparse.Namespace) -> dict[int, Path]:
    """
    Build/reuse one DATA_DIR per target year.

    The folder depends only on the target year, not on the model year, because
    retesting never draws from train.csv or validation.csv.  Small dummy train
    and validation files are still written to satisfy Env/DataKeeper
    initialization.  This avoids copying a large expanding train window for
    every model-year/target-year pair.
    """

    prefix_data_root = args.data_root / args.prefix
    prefix_data_root.mkdir(parents=True, exist_ok=True)
    required = ["train.csv", "validation.csv", "test.csv", "1yr_treasury.csv"]

    target_years = range(args.first_test_year, args.final_test_year + 1)
    existing = {
        year: prefix_data_root / f"test_{year}"
        for year in target_years
        if all((prefix_data_root / f"test_{year}" / name).exists() for name in required)
    }
    if len(existing) == len(list(target_years)) and not args.rebuild_data:
        return existing

    print("[long_horizon] building/reusing yearly DATA_DIR folders...")
    df = read_cleaned_panel(args.cleaned_data_dir)
    treasury = (
        df[["quote_datetime", "risk_free_rate"]]
        .drop_duplicates(subset=["quote_datetime"])
        .rename(columns={"quote_datetime": "Date", "risk_free_rate": "1y"})
        .sort_values("Date")
    )
    dummy = process_env_csv(df.head(5000))

    out: dict[int, Path] = {}
    for year in target_years:
        data_dir = prefix_data_root / f"test_{year}"
        out[year] = data_dir
        if all((data_dir / name).exists() for name in required) and not args.rebuild_data:
            continue
        data_dir.mkdir(parents=True, exist_ok=True)
        year_data = df[df["year"] == year].copy()
        if year_data.empty:
            raise ValueError(f"No cleaned_data rows found for target year {year}")

        dummy.to_csv(data_dir / "train.csv", index=False)
        dummy.to_csv(data_dir / "validation.csv", index=False)
        process_env_csv(year_data).to_csv(data_dir / "test.csv", index=False)
        treasury.to_csv(data_dir / "1yr_treasury.csv", index=False)

        manifest = {
            "target_year": year,
            "cleaned_data_dir": str(args.cleaned_data_dir),
            "purpose": "long_horizon_retesting",
            "dummy_train_validation_rows": int(len(dummy)),
            "test_rows": int(len(year_data)),
        }
        (data_dir / "long_horizon_data_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    return out


def find_selected_checkpoints(args: argparse.Namespace) -> dict[int, tuple[str, str, Path]]:
    """Return {test_year: (model_name, checkpoint, original_result_csv)}."""

    selected: dict[int, tuple[str, str, Path]] = {}
    for year in range(args.first_test_year, args.final_test_year + 1):
        matches = sorted(args.results_dir.glob(f"{args.prefix}{year}_*.csv"))
        if not matches:
            raise FileNotFoundError(
                f"No original testing result found for {args.prefix}{year}_*.csv"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Multiple selected checkpoints found for {args.prefix}{year}: "
                f"{[m.name for m in matches]}. Keep one CSV per year or use a unique prefix."
            )

        stem = matches[0].stem
        match = re.fullmatch(rf"({re.escape(args.prefix)}{year})_(\d+)", stem)
        if not match:
            raise ValueError(f"Could not parse model/checkpoint from {matches[0].name}")
        selected[year] = (match.group(1), match.group(2), matches[0])
    return selected


def build_tasks(args: argparse.Namespace, data_dirs: dict[int, Path]) -> list[RetestTask]:
    selected = find_selected_checkpoints(args)
    raw_dir = args.output_root / args.prefix / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[RetestTask] = []
    for model_year, (model_name, checkpoint, _) in selected.items():
        for target_year in range(model_year, args.final_test_year + 1):
            raw_path = (
                raw_dir
                / f"{model_name}_{checkpoint}_tested_{target_year}.csv"
            )
            tasks.append(
                RetestTask(
                    prefix=args.prefix,
                    model_year=model_year,
                    target_year=target_year,
                    model_name=model_name,
                    checkpoint=checkpoint,
                    data_dir=str(data_dirs[target_year]),
                    raw_output_path=str(raw_path),
                    force_retest=bool(args.force_retest),
                    seed=int(args.seed),
                    torch_num_threads=int(args.torch_num_threads),
                )
            )
    return tasks


def retest_one_task(task: RetestTask) -> dict[str, object]:
    """
    Retest one actor on one target year and write a raw hedge-interval CSV.

    
    This mirrors the package testing entry point's ``test_load`` path, but
    returns status metadata and writes to a long-horizon output folder.  It
    uses DATA_DIR instead of renaming the repository's global ``data``
    directory, making parallel runs safe.
    """

    output_path = Path(task.raw_output_path)
    if output_path.exists() and not task.force_retest:
        return {
            "status": "skipped_existing",
            "model_year": task.model_year,
            "target_year": task.target_year,
            "raw_output_path": str(output_path),
        }

    if task.torch_num_threads > 0:
        torch.set_num_threads(task.torch_num_threads)

    maybe_make_dirs()
    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = task.data_dir
    try:
        setSettings(task.model_name)
        settings = getSettings()
        set_random_seeds(settings.get("seed", task.seed))
        env = Env(settings)
        empirical = settings["process"] == "Real"
        if not empirical:
            raise RuntimeError(f"{task.model_name} is not an empirical Real-process model")

        scaler = StatePrepare(env, 1, task.model_name)
        scaler.load(task.model_name)
        state_size = scaler.state_size

        actor_critic = ActorCritic(state_size)
        actor_critic.load(f"model/{task.model_name}_{task.checkpoint}")

        env.data_keeper.switch_to_test()
        env.data_keeper.reset(soft=False)
        set_count = env.data_keeper.set_count

        info_df = None
        a_rewards = 0.0
        b_rewards = 0.0
        episode = 0
        while not env.data_keeper.no_more_sets:
            stats, _, t_info = test_run(
                env, actor_critic, scaler, state_size, episode, empirical
            )
            a_rewards += float(np.sum(stats["rewards"]))
            b_rewards += float(np.sum(stats["b rewards"]))
            info_df = t_info if info_df is None else pd.concat(
                [info_df, t_info], ignore_index=True
            )
            episode += 1

        if info_df is None or info_df.empty:
            raise RuntimeError(
                f"No test episodes generated for {task.model_name} on {task.target_year}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
        info_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, output_path)
        return {
            "status": "completed",
            "model_year": task.model_year,
            "target_year": task.target_year,
            "episodes": int(info_df["episode"].nunique()),
            "rows": int(len(info_df)),
            "a_rewards": a_rewards,
            "b_rewards": b_rewards,
            "raw_output_path": str(output_path),
        }
    finally:
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir


def run_retesting(tasks: list[RetestTask], workers: int) -> list[dict[str, object]]:
    """Run retesting tasks serially or in a ProcessPool."""

    if workers <= 1:
        results = []
        for idx, task in enumerate(tasks, start=1):
            print(
                f"[long_horizon] retesting {idx}/{len(tasks)}: "
                f"model {task.model_year} on {task.target_year}"
            )
            results.append(retest_one_task(task))
        return results

    results = []
    print(f"[long_horizon] running {len(tasks)} retests with {workers} workers...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {executor.submit(retest_one_task, task): task for task in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[long_horizon] {result['status']}: "
                    f"model {task.model_year} on {task.target_year}"
                )
            except Exception as exc:
                print(
                    f"[long_horizon] FAILED: model {task.model_year} on "
                    f"{task.target_year}: {exc!r}"
                )
                raise
    return sorted(results, key=lambda r: (int(r["model_year"]), int(r["target_year"])))


def build_rate_cache(cleaned_data_dir: Path) -> dict[str, float]:
    """Fallback rate cache for legacy CSVs that do not store interval r."""

    parquet_files = sorted(cleaned_data_dir.glob("*.parquet"))
    if not parquet_files:
        return {}
    frames = [
        pd.read_parquet(path, columns=["quote_date", "risk_free_rate"])
        for path in parquet_files
    ]
    rates = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["quote_date"])
    rates["quote_date"] = rates["quote_date"].astype(str).str.slice(0, 10)
    rates["r_clean"] = np.where(
        rates["risk_free_rate"] > 1.0,
        rates["risk_free_rate"] / 100.0,
        rates["risk_free_rate"],
    )
    return dict(zip(rates["quote_date"], rates["r_clean"]))


def get_rate_for_date(date: str, rate_cache: dict[str, float], missing_dates: set[str]) -> float:
    """Visible 1% fallback matching the paper-metric convention."""

    key = str(date)[:10]
    if key in rate_cache:
        return float(rate_cache[key])
    missing_dates.add(key)
    return 0.01


def calculate_episode_antonov(
    group: pd.DataFrame, rate_cache: dict[str, float], missing_dates: set[str]
) -> pd.Series:
    """
    Calculate Antonov's hedging-defect metric for one episode.

    This is aligned with the paper-metric implementation: hedge
    positions are applied to their own intervals, DateEnd is used when present,
    and transaction costs are included.  In the current experiments transaction
    costs are zero, so including them is a consistency check rather than a
    source of numerical change.
    """

    df = group.copy()
    df["_original_row_order"] = np.arange(len(df))
    df = df.sort_values(["Date", "_original_row_order"], kind="mergesort").reset_index(drop=True)
    n_steps = len(df)
    if n_steps < 2:
        return pd.Series({"D_Ag": np.nan, "D_BS": np.nan})

    p_start = df["P-1"].to_numpy(dtype=float)
    s_start = df["S-1"].to_numpy(dtype=float)
    delta_ag = df["A Pos"].to_numpy(dtype=float)
    delta_bs = df["B Pos"].to_numpy(dtype=float)
    tc_ag = df["A TC"].to_numpy(dtype=float) if "A TC" in df.columns else np.zeros(n_steps)
    tc_bs = df["B TC"].to_numpy(dtype=float) if "B TC" in df.columns else np.zeros(n_steps)
    dates = df["Date"].astype(str).str.slice(0, 10).to_numpy()

    has_full_interval_dates = (
        "DateEnd" in df.columns
        and df["DateEnd"].notna().all()
        and (df["DateEnd"].astype(str).str.len() >= 10).all()
    )
    if has_full_interval_dates:
        dt_years = (
            pd.to_datetime(df["DateEnd"]) - pd.to_datetime(df["Date"])
        ).dt.days.to_numpy(dtype=float) / 365.0
        h_ag = float(df["P0"].iloc[-1])
        h_bs = float(df["P0"].iloc[-1])
        loop_range: Iterable[int] = range(n_steps - 1, -1, -1)
    else:
        date_series = pd.to_datetime(df["Date"])
        dt_years = (date_series.shift(-1) - date_series).dt.days.to_numpy(dtype=float) / 365.0
        h_ag = float(p_start[-1])
        h_bs = float(p_start[-1])
        loop_range = range(n_steps - 2, -1, -1)

    chi2_ag = 0.0
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
            r = get_rate_for_date(dates[t], rate_cache, missing_dates)

        discount = float(np.exp(-r * dt))
        s_end = float(df["S0"].iloc[t]) if has_full_interval_dates else float(s_start[t + 1])

        h_ag = discount * h_ag + delta_ag[t] * (discount * s_end - s_start[t]) + tc_ag[t]
        h_bs = discount * h_bs + delta_bs[t] * (discount * s_end - s_start[t]) + tc_bs[t]
        chi2_ag += ((h_ag - p_start[t]) ** 2) * dt
        chi2_bs += ((h_bs - p_start[t]) ** 2) * dt
        total_t += dt

    if total_t <= 0.0:
        return pd.Series({"D_Ag": np.nan, "D_BS": np.nan})
    return pd.Series(
        {
            "D_Ag": float(np.sqrt(chi2_ag / total_t)),
            "D_BS": float(np.sqrt(chi2_bs / total_t)),
        }
    )


def downside_second_moment(values: np.ndarray) -> float:
    downside = np.where(values < 0.0, values, 0.0)
    return float(np.sum(downside**2) / len(downside))


def cvar_5(values: np.ndarray) -> float:
    n_tail = max(1, int(len(values) * 0.05))
    return float(np.mean(np.partition(values, n_tail - 1)[:n_tail]))


def raw_to_episode_metrics(
    task: RetestTask, rate_cache: dict[str, float], missing_dates: set[str]
) -> pd.DataFrame:
    """Aggregate one raw hedge-interval CSV to episode-level metrics."""

    raw = pd.read_csv(task.raw_output_path)
    if raw.empty:
        raise ValueError(f"{task.raw_output_path} is empty")

    antonov = (
        raw.groupby("episode")
        .apply(
            lambda g: calculate_episode_antonov(g, rate_cache, missing_dates),
            include_groups=False,
        )
        .reset_index()
    )
    ep = (
        raw.groupby("episode")
        .agg(
            a_pnl=("A PnL", "sum"),
            b_pnl=("B PnL", "sum"),
            a_reward=("A Reward", "sum"),
            b_reward=("B Reward", "sum"),
            a_tc=("A TC", "sum"),
            b_tc=("B TC", "sum"),
            start_date=("Date", "min"),
            end_date=("DateEnd", "max") if "DateEnd" in raw.columns else ("Date", "max"),
        )
        .reset_index()
        .merge(antonov, on="episode", how="left")
    )
    ep["a_pnl"] *= 100.0
    ep["b_pnl"] *= 100.0
    ep["a_tc"] *= 100.0
    ep["b_tc"] *= 100.0
    ep["model_year"] = task.model_year
    ep["target_year"] = task.target_year
    ep["model_name"] = task.model_name
    ep["checkpoint"] = task.checkpoint
    ep["raw_output_path"] = task.raw_output_path
    return ep


def metric_point_estimates(sample: pd.DataFrame) -> dict[str, float]:
    """Calculate paired point estimates on episode-level data."""

    a_pnl = sample["a_pnl"].to_numpy(dtype=float)
    b_pnl = sample["b_pnl"].to_numpy(dtype=float)
    diff_pnl = a_pnl - b_pnl
    diff_reward = (sample["a_reward"] - sample["b_reward"]).to_numpy(dtype=float)
    diff_tc = (sample["a_tc"] - sample["b_tc"]).to_numpy(dtype=float)
    diff_antonov = (sample["D_Ag"] - sample["D_BS"]).to_numpy(dtype=float)

    var_a = float(np.var(a_pnl, ddof=1)) if len(a_pnl) > 1 else np.nan
    var_b = float(np.var(b_pnl, ddof=1)) if len(b_pnl) > 1 else np.nan
    down_a = downside_second_moment(a_pnl)
    down_b = downside_second_moment(b_pnl)

    return {
        "mean": float(np.mean(diff_pnl)),
        "std": float(np.sqrt(var_a) - np.sqrt(var_b)) if var_a >= 0 and var_b >= 0 else np.nan,
        "tc": float(np.mean(diff_tc)),
        "rew": float(np.mean(diff_reward)),
        "cvar": cvar_5(a_pnl) - cvar_5(b_pnl),
        "log_down_var_ratio": (
            float(np.log(down_a / down_b)) if down_a > 1e-12 and down_b > 1e-12 else np.nan
        ),
        "log_var_ratio": (
            float(np.log(var_a / var_b)) if var_a > 1e-12 and var_b > 1e-12 else np.nan
        ),
        "antonov": float(np.nanmean(diff_antonov)),
    }


def metric_levels(sample: pd.DataFrame) -> dict[str, float]:
    """Store agent and benchmark metric levels alongside paired differences."""

    a_pnl = sample["a_pnl"].to_numpy(dtype=float)
    b_pnl = sample["b_pnl"].to_numpy(dtype=float)
    return {
        "a_mean": float(np.mean(a_pnl)),
        "b_mean": float(np.mean(b_pnl)),
        "a_std": float(np.std(a_pnl, ddof=1)),
        "b_std": float(np.std(b_pnl, ddof=1)),
        "a_reward": float(np.mean(sample["a_reward"])),
        "b_reward": float(np.mean(sample["b_reward"])),
        "a_tc": float(np.mean(sample["a_tc"])),
        "b_tc": float(np.mean(sample["b_tc"])),
        "a_cvar": cvar_5(a_pnl),
        "b_cvar": cvar_5(b_pnl),
        "a_downside_second_moment": downside_second_moment(a_pnl),
        "b_downside_second_moment": downside_second_moment(b_pnl),
        "a_variance": float(np.var(a_pnl, ddof=1)),
        "b_variance": float(np.var(b_pnl, ddof=1)),
        "a_antonov": float(np.nanmean(sample["D_Ag"])),
        "b_antonov": float(np.nanmean(sample["D_BS"])),
    }


def bootstrap_group(
    sample: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    """Two-stage bootstrap: resample date clusters, then episodes inside clusters."""

    sample = sample.dropna(subset=["D_Ag", "D_BS"]).copy()
    point = metric_point_estimates(sample)
    clusters = sorted(sample["start_date"].unique())
    if not clusters:
        raise ValueError("No bootstrap clusters available")
    cluster_frames = [sample[sample["start_date"] == cluster] for cluster in clusters]

    draws = {metric: np.empty(n_bootstrap, dtype=float) for metric in BOOTSTRAP_METRICS}
    for i in range(n_bootstrap):
        selected_cluster_idx = rng.integers(0, len(cluster_frames), size=len(cluster_frames))
        pieces = []
        for idx in selected_cluster_idx:
            cluster = cluster_frames[idx]
            selected_rows = rng.integers(0, len(cluster), size=len(cluster))
            pieces.append(cluster.iloc[selected_rows])
        boot_sample = pd.concat(pieces, ignore_index=True)
        estimates = metric_point_estimates(boot_sample)
        for metric in BOOTSTRAP_METRICS:
            draws[metric][i] = estimates[metric]

    out: dict[str, dict[str, float]] = {}
    for metric, values in draws.items():
        values = values[np.isfinite(values)]
        if len(values) == 0:
            out[metric] = {"point_estimate": point[metric], "bootstrap_center": np.nan}
            continue
        row = {
            "point_estimate": point[metric],
            "bootstrap_center": float(np.mean(values)),
            "bootstrap_std": float(np.std(values, ddof=1)),
        }
        for level in CI_LEVELS:
            alpha = (1.0 - level) / 2.0
            low, high = np.quantile(values, [alpha, 1.0 - alpha])
            suffix = int(round(level * 100))
            row[f"ci_low_{suffix}"] = float(low)
            row[f"ci_high_{suffix}"] = float(high)
            row[f"significant_{suffix}"] = bool(low > 0.0 or high < 0.0)
        out[metric] = row
    return out


def bootstrap_to_rows(
    sample: pd.DataFrame,
    scope: str,
    model_year: int,
    target_year: int | str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    """Bootstrap one pair/year or one pooled horizon and flatten to rows."""

    boot = bootstrap_group(sample, n_bootstrap, rng)
    levels = metric_levels(sample)
    rows = []
    for metric, stats in boot.items():
        row = {
            "scope": scope,
            "model_year": model_year,
            "target_year": target_year,
            "metric": metric,
            "n_episodes": int(len(sample)),
            "n_clusters": int(sample["start_date"].nunique()),
            **stats,
            **levels,
        }
        rows.append(row)
    return rows


def collect_episode_metrics(tasks: list[RetestTask], cleaned_data_dir: Path) -> pd.DataFrame:
    """Read all cached raw CSVs and return one episode-level table."""

    rate_cache = build_rate_cache(cleaned_data_dir)
    missing_dates: set[str] = set()
    frames = []
    for task in tasks:
        path = Path(task.raw_output_path)
        if not path.exists():
            raise FileNotFoundError(f"Missing raw long-horizon output: {path}")
        frames.append(raw_to_episode_metrics(task, rate_cache, missing_dates))
    if missing_dates:
        print(
            "[long_horizon] WARNING: Antonov fallback r=0.01 used for "
            f"{len(missing_dates)} dates. First dates: {sorted(missing_dates)[:5]}"
        )
    return pd.concat(frames, ignore_index=True)


def run_bootstrap_summaries(
    episode_metrics: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap pair-level and pooled long-horizon summaries."""

    rng = np.random.default_rng(args.seed)
    pair_rows = []
    horizon_rows = []

    grouped = episode_metrics.groupby(["model_year", "target_year"], sort=True)
    total_pairs = len(grouped)
    for idx, ((model_year, target_year), sample) in enumerate(grouped, start=1):
        print(
            f"[long_horizon] bootstrap pair {idx}/{total_pairs}: "
            f"model {model_year} on {target_year}"
        )
        pair_rows.extend(
            bootstrap_to_rows(
                sample,
                "pair",
                int(model_year),
                int(target_year),
                args.n_bootstrap,
                rng,
            )
        )

    for model_year, sample in episode_metrics.groupby("model_year", sort=True):
        print(f"[long_horizon] bootstrap pooled horizon for model {model_year}")
        horizon_rows.extend(
            bootstrap_to_rows(
                sample,
                "pooled_horizon",
                int(model_year),
                f"{int(model_year)}-{args.final_test_year}",
                args.n_bootstrap,
                rng,
            )
        )

    return pd.DataFrame(pair_rows), pd.DataFrame(horizon_rows)


def write_manifest(
    args: argparse.Namespace,
    tasks: list[RetestTask],
    retest_results: list[dict[str, object]] | None,
    runtime_seconds: float,
) -> None:
    output_dir = args.output_root / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "prefix": args.prefix,
        "first_test_year": args.first_test_year,
        "final_test_year": args.final_test_year,
        "n_tasks": len(tasks),
        "n_bootstrap": args.n_bootstrap,
        "ci_levels": list(CI_LEVELS),
        "parallel_workers": args.parallel_workers,
        "skip_testing": args.skip_testing,
        "skip_bootstrap": args.skip_bootstrap,
        "force_retest": args.force_retest,
        "runtime_seconds": runtime_seconds,
        "tasks": [asdict(task) for task in tasks],
        "retest_results": retest_results or [],
    }
    (output_dir / "long_horizon_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def run(args: argparse.Namespace) -> None:
    start = time.time()
    seed_script(args.seed)
    output_dir = args.output_root / args.prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dirs = ensure_yearly_data_dirs(args)
    tasks = build_tasks(args, data_dirs)
    print(
        f"[long_horizon] prepared {len(tasks)} tasks for prefix {args.prefix}; "
        f"outputs under {output_dir}"
    )

    retest_results: list[dict[str, object]] | None = None
    if not args.skip_testing:
        retest_results = run_retesting(tasks, max(1, args.parallel_workers))

    episode_metrics = collect_episode_metrics(tasks, args.cleaned_data_dir)
    episode_path = output_dir / "long_horizon_episode_metrics.csv"
    episode_metrics.to_csv(episode_path, index=False)
    print(f"[long_horizon] wrote {episode_path}")

    if not args.skip_bootstrap:
        pair_summary, horizon_summary = run_bootstrap_summaries(episode_metrics, args)
        pair_path = output_dir / "long_horizon_pair_bootstrap_summary.csv"
        horizon_path = output_dir / "long_horizon_pooled_bootstrap_summary.csv"
        pair_summary.to_csv(pair_path, index=False)
        horizon_summary.to_csv(horizon_path, index=False)
        print(f"[long_horizon] wrote {pair_path}")
        print(f"[long_horizon] wrote {horizon_path}")

    write_manifest(args, tasks, retest_results, time.time() - start)
    print("[long_horizon] complete")


if __name__ == "__main__":
    run(parse_args())
