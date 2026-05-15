"""
Leakage-safe symbolic distillation for empirical deep-hedging agents.

This script is intentionally written as a research diagnostic rather than as
training infrastructure.  It does three things:

1. Builds/uses one leakage-safe walk-forward DATA_DIR per test year.  For a
   test year Y, symbolic regression is fit only on training years through
   Y-2; year Y-1 is validation; year Y is final out-of-sample testing.
2. Fits symbolic policies.  The original/raw-agent grid isolated four target
   geometries under two sampling schemes:
      score                : fit probit/normal score, then trade Phi(score).
      logit                : fit logit(delta), then trade sigmoid(logit).
      delta                : fit delta directly, then clip to [0, 1].
      delta_residual       : fit agent_delta - BS_delta, then trade
                             clip(BS_delta + symbolic_residual, 0, 1).

   The current publication walk-forward default fits exactly three symbolic
   BS-residual formula families for every yearly agent:
      uniform_bs_delta_residual          : raw neural target, uniform sample.
      smooth_uniform_bs_delta_residual   : smoothed target, uniform sample.
      smooth_focus_bs_delta_residual     : smoothed target, focus sample.

   The uniform_bs_delta_residual candidate fits the raw residual target.  The
   other two candidates test whether smoothing, more even state coverage, or
   economically focused state coverage helps.

   Additional raw-agent candidates remain available from the command line.  The
   two supported sampling schemes are:
      uniform              : random training-pool sample.
      gap_weighted         : oversample states where the agent differs from BS.
3. Trades all formulas on both validation and testing episodes and compares
   them with the smoothed agent, raw neural agent, and BS benchmark using the
   same two-stage bootstrap convention as the paper metrics.

Important: no validation or test-year states are used to fit formulas.  They
are only evaluated after each year-family regression is frozen and saved.

Default walk-forward model class
--------------------------------
    final_WF_exp1_k1_test

The default run distills yearly agents final_WF_exp1_k1_test2015 through
final_WF_exp1_k1_test2023.

Suggested manual run
--------------------
    python scripts/distill_empirical_agents.py

Quick diagnostic run
--------------------
    python scripts/distill_empirical_agents.py --max-train-episodes 200 --train-support-probes 2000 ^
        --general-probes 2000 --n-sr-samples 3000 --niterations 40 ^
        --n-bootstrap 1000

The script writes detailed outputs to results/interpret_real_walkforward/,
including per-year subfolders and aggregate CSVs.  Per year it saves:
    *_distillation_pairs.csv
    *_<candidate>_formula.txt
    *_<candidate>_pysr_equations.csv
    *_validation_* and *_test_* fidelity/trading/bootstrap outputs
    *_test_trade_steps.csv
    *_test_episode_metrics.csv
    *_bootstrap_summary.csv
    *_interpret_report.txt
"""

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import erf, erfinv


DEFAULT_MODEL_NAME = "final_WF_exp1_k1_test2023"
DEFAULT_TARGET_YEAR = 2023
DEFAULT_TRAIN_START_YEAR = 2010
DEFAULT_TRAIN_END_YEAR = 2021
DEFAULT_VALIDATION_YEAR = 2022
DEFAULT_OUTPUT_DIR = "results/interpret_real_walkforward"
DEFAULT_DATA_DIR = "data_interpret_real_2023"
DEFAULT_WALKFORWARD_DATA_DIR_TEMPLATE = "data_interpret_real_wf_{year}"
DEFAULT_CLEANED_DATA_DIR = "cleaned_data"
DEFAULT_SEED = 123
METRICS = ["mean", "std", "rew", "cvar", "log_down_var_ratio", "log_var_ratio", "antonov"]
DISTILLATION_POOL_VERSION = "train_pool_v2_score_residual_loss_grid"
DEFAULT_MODEL_PREFIX = "final_WF_exp1_k1_test"
DEFAULT_FIRST_TEST_YEAR = 2015
DEFAULT_FINAL_TEST_YEAR = 2023
DEFAULT_MODEL_NAMES = [
    DEFAULT_MODEL_NAME,
]
DEFAULT_FORMULA_CANDIDATES = [
    "uniform_bs_delta_residual",
    "smooth_uniform_bs_delta_residual",
    "smooth_focus_bs_delta_residual",
]


@dataclass(frozen=True)
class FormulaSpec:
    """
    One bounded symbolic-regression experiment.

    target_kind:
        score:          z_agent = sqrt(2) * erfinv(2*agent_delta - 1),
                        then delta = Phi(z_hat).
        logit:          log(agent_delta / (1-agent_delta)),
                        then delta = sigmoid(logit_hat).
        delta:          raw agent delta, clipped to [0, 1] when traded.
        delta_residual: agent_delta - BS_delta, then
                        delta = clip(BS_delta + residual_hat, 0, 1).

    loss_kind:
        default:     ordinary squared error on the target.
        gap_weighted: squared error weighted by |agent_delta - BS_delta|.
                      Available for non-default candidate grids.

    target_source:
        raw_agent: the original neural actor label.
        smooth_kernel_bs_delta_residual: the same three-feature policy surface
            after local KNN smoothing of agent_delta - BS_delta.  This is the
            default because diagnostics show that this smoother preserves
            trading behavior while removing small actor
            bumps that symbolic regression struggled to fit.

    target_smoothing_bandwidth_scale:
        Optional override for the KNN smoother bandwidth used to create the
        symbolic target.  None means use args.smoothing_bandwidth_scale.  The
        default bandwidth experiment sets this field explicitly, so every
        candidate is tied to a distinct smoothed target and cache file.
    """
    name: str
    sample_kind: str
    target_kind: str
    loss_kind: str
    description: str
    target_source: str = "raw_agent"
    target_smoothing_bandwidth_scale: float | None = None


FORMULA_SPECS = [
    FormulaSpec(
        name="uniform_probit_score",
        sample_kind="uniform",
        target_kind="score",
        loss_kind="default",
        description="Uniform sample; direct agent score; ordinary MSE.",
    ),
    FormulaSpec(
        name="gap_probit_score",
        sample_kind="gap_weighted",
        target_kind="score",
        loss_kind="default",
        description="Agent-BS gap sample; direct agent score; ordinary MSE.",
    ),
    FormulaSpec(
        name="uniform_logit_delta",
        sample_kind="uniform",
        target_kind="logit",
        loss_kind="default",
        description="Uniform sample; direct agent logit(delta); ordinary MSE.",
    ),
    FormulaSpec(
        name="gap_logit_delta",
        sample_kind="gap_weighted",
        target_kind="logit",
        loss_kind="default",
        description="Agent-BS gap sample; direct agent logit(delta); ordinary MSE.",
    ),
    FormulaSpec(
        name="uniform_direct_delta",
        sample_kind="uniform",
        target_kind="delta",
        loss_kind="default",
        description="Uniform sample; direct agent delta; ordinary MSE.",
    ),
    FormulaSpec(
        name="gap_direct_delta",
        sample_kind="gap_weighted",
        target_kind="delta",
        loss_kind="default",
        description="Agent-BS gap sample; direct agent delta; ordinary MSE.",
    ),
    FormulaSpec(
        name="uniform_bs_delta_residual",
        sample_kind="uniform",
        target_kind="delta_residual",
        loss_kind="default",
        description="Uniform sample; symbolic correction to BS delta; ordinary MSE.",
    ),
    FormulaSpec(
        name="gap_bs_delta_residual",
        sample_kind="gap_weighted",
        target_kind="delta_residual",
        loss_kind="default",
        description="Agent-BS gap sample; symbolic correction to BS delta; ordinary MSE.",
    ),
    FormulaSpec(
        name="smooth_uniform_bs_delta_residual",
        sample_kind="uniform",
        target_kind="delta_residual",
        loss_kind="default",
        description=(
            "Uniform sample; symbolic correction to BS delta; ordinary MSE; "
            "target is the smooth_kernel_bs_delta_residual policy."
        ),
        target_source="smooth_kernel_bs_delta_residual",
    ),
    FormulaSpec(
        name="smooth_stratified_bs_delta_residual",
        sample_kind="stratified",
        target_kind="delta_residual",
        loss_kind="default",
        description=(
            "Moneyness/IV/T stratified sample; symbolic correction to BS delta; "
            "target is the smooth_kernel_bs_delta_residual policy."
        ),
        target_source="smooth_kernel_bs_delta_residual",
    ),
    FormulaSpec(
        name="smooth_focus_bs_delta_residual",
        sample_kind="focus",
        target_kind="delta_residual",
        loss_kind="default",
        description=(
            "Hybrid stratified/economic-focus sample; symbolic correction to BS delta; "
            "target is the smooth_kernel_bs_delta_residual policy."
        ),
        target_source="smooth_kernel_bs_delta_residual",
    ),
    FormulaSpec(
        name="smooth_focus_bw075_bs_delta_residual",
        sample_kind="focus",
        target_kind="delta_residual",
        loss_kind="default",
        description=(
            "Bandwidth sensitivity: focus-sampled symbolic correction to BS delta; "
            "target is smooth_kernel_bs_delta_residual with bandwidth scale 0.75."
        ),
        target_source="smooth_kernel_bs_delta_residual",
        target_smoothing_bandwidth_scale=0.75,
    ),
    FormulaSpec(
        name="smooth_focus_bw100_bs_delta_residual",
        sample_kind="focus",
        target_kind="delta_residual",
        loss_kind="default",
        description=(
            "Bandwidth sensitivity: focus-sampled symbolic correction to BS delta; "
            "target is smooth_kernel_bs_delta_residual with bandwidth scale 1.00."
        ),
        target_source="smooth_kernel_bs_delta_residual",
        target_smoothing_bandwidth_scale=1.00,
    ),
    FormulaSpec(
        name="smooth_focus_bw125_bs_delta_residual",
        sample_kind="focus",
        target_kind="delta_residual",
        loss_kind="default",
        description=(
            "Bandwidth sensitivity: focus-sampled symbolic correction to BS delta; "
            "target is smooth_kernel_bs_delta_residual with bandwidth scale 1.25."
        ),
        target_source="smooth_kernel_bs_delta_residual",
        target_smoothing_bandwidth_scale=1.25,
    ),
    FormulaSpec(
        name="smooth_focus_bw150_bs_delta_residual",
        sample_kind="focus",
        target_kind="delta_residual",
        loss_kind="default",
        description=(
            "Bandwidth sensitivity: focus-sampled symbolic correction to BS delta; "
            "target is smooth_kernel_bs_delta_residual with bandwidth scale 1.50."
        ),
        target_source="smooth_kernel_bs_delta_residual",
        target_smoothing_bandwidth_scale=1.50,
    ),
]


def set_random_seeds(seed):
    """Set seeds used by this diagnostic script."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Publication runs should fail loudly rather than silently falling back
        # to nondeterministic kernels.  On CPU this is usually a no-op; on CUDA
        # it guards against hard-to-reproduce actor labels.
        torch.use_deterministic_algorithms(True, warn_only=False)
    except Exception as exc:
        print(f"[seed] Warning: could not force PyTorch deterministic algorithms: {exc}")


def ensure_walkforward_data_dir(args):
    """
    Build a DATA_DIR compatible with the hedging environment.

    The directory contains train, validation, and test files for the normal
    walk-forward split.  Distillation reads only the training file during
    fitting; validation and test files are used only after formulas have been
    selected.
    """
    output_dir = Path(args.target_data_dir)
    cleaned_data_dir = Path(args.cleaned_data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required = ["train.csv", "validation.csv", "test.csv", "1yr_treasury.csv"]
    expected_manifest = {
        "train_years": list(range(args.train_start_year, args.train_end_year + 1)),
        "validation_year": args.validation_year,
        "test_year": args.target_year,
    }
    manifest_path = output_dir / "interpret_manifest.json"
    if all((output_dir / name).exists() for name in required) and not args.rebuild_data_dir:
        if manifest_path.exists():
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if current_manifest == expected_manifest:
                print(f"[data] Reusing DATA_DIR: {output_dir}")
                return str(output_dir)
        else:
            print("[data] Existing DATA_DIR has no manifest; rebuilding for safety.")

    print(f"[data] Building leakage-safe DATA_DIR: {output_dir}")
    parquet_files = sorted(cleaned_data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {cleaned_data_dir}")

    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)

    # Same basic cleaning convention used in run_walkforward.py.
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

    treasury_df = (
        df[["quote_date", "risk_free_rate"]]
        .drop_duplicates(subset=["quote_date"])
        .rename(columns={"quote_date": "Date", "risk_free_rate": "1y"})
        .sort_values("Date")
    )
    treasury_df.to_csv(output_dir / "1yr_treasury.csv", index=False)

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
    df = df[keep_cols].copy()
    df["quote_datetime"] = df["quote_datetime"].astype(str).str.slice(0, 10)
    df["expiration"] = df["expiration"].astype(str).str.slice(0, 10)
    df["year"] = df["quote_datetime"].str.slice(0, 4).astype(int)

    def process_and_save(dataset, filename):
        dataset = dataset.drop(columns=["year"]).copy()
        dataset["option_id"] = dataset["expiration"] + "_" + dataset["strike"].astype(str)
        dataset = dataset.sort_values(["option_id", "quote_datetime"]).reset_index(drop=True)
        dataset["nbr_next_steps"] = dataset.groupby("option_id").cumcount(ascending=False)
        dataset = dataset.drop(columns=["option_id"])
        dataset.to_csv(output_dir / filename, index=False)

    train_years = expected_manifest["train_years"]
    process_and_save(df[df["year"].isin(train_years)], "train.csv")
    process_and_save(df[df["year"] == args.validation_year], "validation.csv")
    process_and_save(df[df["year"] == args.target_year], "test.csv")

    source_heston = Path("data") / "heston_params.csv"
    if source_heston.exists():
        shutil.copy(source_heston, output_dir / "heston_params.csv")
    else:
        pd.DataFrame(
            {"date": [], "v0": [], "kappa": [], "theta": [], "sigma": [], "rho": []}
        ).to_csv(output_dir / "heston_params.csv", index=False)

    manifest_path.write_text(json.dumps(expected_manifest, indent=2), encoding="utf-8")
    return str(output_dir)


def infer_checkpoint(model_name, results_testing_dir):
    """Infer selected checkpoint from results/testing/{model_name}_*.csv."""
    files = sorted(glob.glob(str(Path(results_testing_dir) / f"{model_name}_*.csv")))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one selected testing CSV for {model_name}, "
            f"found {len(files)}: {files}"
        )
    return Path(files[0]).stem.rsplit("_", 1)[1], files[0]


def load_settings_json(model_name):
    """Load the JSON settings for the trained actor."""
    path = Path("settings") / f"{model_name}.json"
    if not path.exists():
        matches = sorted(Path("settings").glob(f"*/{model_name}.json"))
        if len(matches) == 1:
            path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"Missing settings file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_model_artifact(filename):
    """Find a selected model artifact in either flat or seed-family layout."""
    path = Path("model") / filename
    if path.exists():
        return path

    matches = sorted(Path("model").glob(f"*/{filename}"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Missing model artifact: {path}")

    match_list = ", ".join(str(match) for match in matches)
    raise FileNotFoundError(f"Ambiguous model artifact {filename}: {match_list}")


def load_actor_and_scaler(model_name, checkpoint, settings, requested_device="cpu"):
    """
    Load actor and scaler directly.

    StatePrepare is not used here because it would create a fresh environment
    and fit/overwrite state scaling logic.  For distillation we only want the
    already-trained actor and its saved scaler.
    """
    from empirical_deep_hedging.include.network import Actor

    scaler_path = find_model_artifact(f"{model_name}_scaler")
    actor_path = find_model_artifact(f"{model_name}_{checkpoint}_actor")

    scaler = joblib.load(scaler_path)
    state_dim = int(getattr(scaler, "n_features_in_", 4))
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        device = torch.device("cuda")
    elif requested_device == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unknown device request: {requested_device}")
    actor = Actor(state_dim, settings).to(device)
    actor.load_state_dict(torch.load(actor_path, map_location=device))
    actor.eval()
    return actor, scaler, device


def actor_action(actor, scaler, device, raw_state):
    """Query actor on one raw env state and return a delta in [0, 1]."""
    scaled = scaler.transform(np.asarray(raw_state, dtype=float).reshape(1, -1))
    tensor = torch.FloatTensor(scaled).to(device)
    with torch.no_grad():
        raw = actor(tensor).detach().cpu().numpy().reshape(-1)[0]
    return float(np.clip(0.5 * (raw + 1.0), 0.0, 1.0))


def normal_cdf(x):
    """Normal CDF written through erf, matching the Black-Scholes convention."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def bs_delta_from_features(feature_row):
    """
    Reconstruct BS delta from forward moneyness, IV, T, and q.

    With m_fwd = S/K * exp((r-q)T):
        d1 = [log(m_fwd) + 0.5 * iv^2 * T] / [iv * sqrt(T)]

    The spot hedge delta under continuous q is exp(-qT) * N(d1), matching the
    currently patched include/env.py convention.
    """
    m_fwd = max(float(feature_row["forward_moneyness"]), 1e-12)
    tau = max(float(feature_row["T_years"]), 1e-12)
    iv = max(float(feature_row["iv"]), 1e-8)
    q = float(feature_row.get("q", 0.0))
    if not np.isfinite(q):
        q = 0.0
    d1 = (math.log(m_fwd) + 0.5 * iv * iv * tau) / (iv * math.sqrt(tau))
    return float(np.clip(math.exp(-q * tau) * normal_cdf(d1), 0.0, 1.0))


def canonical_stock_position(feature_row, mode):
    """
    Choose the stock-position coordinate used to query the actor's target delta.

    The formula is intentionally three-dimensional because transaction costs are
    zero.  The actor was nevertheless trained with current_stock_position, so we
    must specify the fourth input when asking "what target delta would you hold?"
    """
    if mode == "bs":
        return -bs_delta_from_features(feature_row)
    if mode == "zero":
        return 0.0
    raise ValueError(f"Unknown canonical position mode: {mode}")


def actor_target_delta(actor, scaler, device, feature_row, canonical_position):
    """Query the actor at a feature row and canonical stock position."""
    stock_position = canonical_stock_position(feature_row, canonical_position)
    raw_state = np.array(
        [
            feature_row["forward_moneyness"],
            (feature_row["T_years"] * 365.0) / 30.0,
            stock_position,
            feature_row["iv"],
        ],
        dtype=float,
    )
    return actor_action(actor, scaler, device, raw_state)


def reset_env_on_dataset(env, dataset, start_a=0.0, start_b=0.0):
    """
    Reset Env on a fixed empirical path.

    This deliberately uses Env's private update method so all policies can be
    traded on the exact same option path without asking DataKeeper to advance.
    """
    env.testing = True
    env.t = 0
    env.S = np.zeros(env.steps + 1)
    env.stockOwned = start_a
    env.b_stockOwned = start_b
    env.data_set = dataset.copy().reset_index(drop=True)
    env._Env__update_option()
    return env._Env__concat_state()


def feature_row_from_env(env, raw_state):
    """Readable feature row for symbolic regression and diagnostics."""
    t_years = float(env.option["T"] / env.days_in_year)
    try:
        bs_delta = float(env.get_bs_delta())
    except Exception:
        bs_delta = np.nan
    return {
        "forward_moneyness": float(raw_state[0]),
        "T_years": t_years,
        "T_days": float(env.option["T"]),
        "stock_position": float(raw_state[2]),
        "iv": float(raw_state[3]),
        "bs_delta": bs_delta,
        "spot_moneyness": float(env.option.get("spot_S/K", np.nan)),
        "r": float(getattr(env, "r", np.nan)),
        "q": float(getattr(env, "q", np.nan)),
        "date": str(getattr(env, "cur_date", "")),
        "expiry": str(getattr(env, "expiry", "")),
        "strike": float(getattr(env, "K", np.nan)),
        "spot": float(env.S[env.t]) if len(env.S) > env.t else np.nan,
    }


def normalized_dataset_from_start(train_df, start, steps):
    """
    Recreate DataKeeper.next_train_set for a deterministic training start row.

    This is how we avoid random training-state sampling and can probe all valid
    2010-2021 training starts if requested.
    """
    dataset = train_df.loc[start : start + steps, :].copy().reset_index(drop=True)
    divisor = dataset.loc[0, "underlying_bid"]
    for key in ["underlying_bid", "underlying_ask", "bid", "ask", "strike"]:
        dataset[key] = dataset[key] / divisor
    return dataset


def iter_training_datasets(data_dir, steps, max_episodes, seed):
    """Yield deterministic training episodes from train.csv only."""
    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    starts = train_df.index[train_df["nbr_next_steps"] >= steps].to_numpy()

    if max_episodes and max_episodes > 0 and len(starts) > max_episodes:
        rng = np.random.default_rng(seed)
        starts = np.sort(rng.choice(starts, size=max_episodes, replace=False))

    for local_episode, start in enumerate(starts):
        yield local_episode, int(start), normalized_dataset_from_start(train_df, int(start), steps)


def collect_training_distillation_pairs(env, actor, scaler, device, data_dir, args):
    """
    Collect target-delta labels from training years only.

    This is the core leakage fix.  The 2023 test file is not touched here.
    """
    rows = []
    for episode, start_index, dataset in iter_training_datasets(
        data_dir, env.steps, args.max_train_episodes, args.seed
    ):
        raw_state = reset_env_on_dataset(env, dataset)
        done = False
        step = 0
        while not done:
            feature = feature_row_from_env(env, raw_state)
            actual_delta = actor_action(actor, scaler, device, raw_state)
            target_delta = actor_target_delta(
                actor, scaler, device, feature, args.canonical_position
            )
            feature.update(
                {
                    "sample_source": "train_empirical",
                    "episode": episode,
                    "train_start_index": start_index,
                    "step": step,
                    "actual_agent_delta": actual_delta,
                    "agent_delta": target_delta,
                    "canonical_stock_position": canonical_stock_position(
                        feature, args.canonical_position
                    ),
                }
            )
            rows.append(feature)
            raw_state, _, done, _ = env.step(actual_delta)
            step += 1

        if (episode + 1) % args.progress_every == 0:
            print(f"[distill] collected training episode {episode + 1}")

    return pd.DataFrame(rows)


def support_bounds(df):
    """Robust support bounds from training-year feature coverage."""
    bounds = {}
    for col in ["forward_moneyness", "T_years", "iv"]:
        lo, hi = df[col].quantile([0.005, 0.995]).values
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = df[col].min(), df[col].max()
        pad = 0.05 * (hi - lo)
        bounds[col] = (float(lo - pad), float(hi + pad))
    return bounds


def add_probe_rows(base_df, actor, scaler, device, args):
    """
    Add off-path probes without touching 2023.

    Two probe types are created:
      train_support_probe: uniform samples inside expanded training support.
      general_probe:       a broader pre-declared domain for smoothness checks.

    The general domain is intentionally configurable.  If later evidence shows
    the broad domain creates unstable extrapolation, reduce it rather than using
    test-year states to make the formula look better.
    """
    rng = np.random.default_rng(args.seed + 1000)
    rows = []
    train_bounds = support_bounds(base_df)
    med_r = float(base_df["r"].dropna().median()) if "r" in base_df else 0.0
    med_q = float(base_df["q"].dropna().median()) if "q" in base_df else 0.0
    if not np.isfinite(med_r):
        med_r = 0.0
    if not np.isfinite(med_q):
        med_q = 0.0

    def make_one(source, fwd_m, tau, iv):
        row = {
            "sample_source": source,
            "episode": np.nan,
            "train_start_index": np.nan,
            "step": np.nan,
            "forward_moneyness": float(fwd_m),
            "T_years": float(tau),
            "T_days": float(tau * 365.0),
            "iv": float(iv),
            "r": med_r,
            "q": med_q,
            "spot_moneyness": float(fwd_m * math.exp(-(med_r - med_q) * tau)),
            "date": "",
            "expiry": "",
            "strike": np.nan,
            "spot": np.nan,
        }
        row["bs_delta"] = bs_delta_from_features(row)
        row["stock_position"] = canonical_stock_position(row, args.canonical_position)
        row["canonical_stock_position"] = row["stock_position"]
        row["actual_agent_delta"] = np.nan
        row["agent_delta"] = actor_target_delta(
            actor, scaler, device, row, args.canonical_position
        )
        rows.append(row)

    for _ in range(args.train_support_probes):
        make_one(
            "train_support_probe",
            rng.uniform(*train_bounds["forward_moneyness"]),
            rng.uniform(*train_bounds["T_years"]),
            rng.uniform(*train_bounds["iv"]),
        )

    for _ in range(args.general_probes):
        make_one(
            "general_probe",
            rng.uniform(args.general_m_min, args.general_m_max),
            rng.uniform(args.general_t_days_min / 365.0, args.general_t_days_max / 365.0),
            rng.uniform(args.general_iv_min, args.general_iv_max),
        )

    return pd.DataFrame(rows)


def distillation_pool_config(args, checkpoint):
    """
    Configuration that uniquely defines the reusable distillation pool.

    The symbolic-regression settings are deliberately excluded.  We want to
    reuse the same train-year state/action pool when changing loss functions,
    sampling rules, PySR iterations, or formula candidates.  The pool is
    rebuilt only when the data split, model, canonical actor query, probe
    design, or training-episode cap changes.
    """
    return {
        "pool_version": DISTILLATION_POOL_VERSION,
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "target_year": int(args.target_year),
        "train_start_year": int(args.train_start_year),
        "train_end_year": int(args.train_end_year),
        "validation_year": int(args.validation_year),
        "target_data_dir": str(args.target_data_dir),
        "cleaned_data_dir": str(args.cleaned_data_dir),
        "canonical_position": args.canonical_position,
        # Actor labels can differ at tiny tolerances across CPU/CUDA.  Treat
        # the device as part of the label-generating experiment so a CUDA-built
        # pool is never silently reused by a CPU publication run, or vice versa.
        "device": args.device,
        "max_train_episodes": int(args.max_train_episodes),
        "train_support_probes": int(args.train_support_probes),
        "general_probes": int(args.general_probes),
        "general_domain": {
            "m_min": float(args.general_m_min),
            "m_max": float(args.general_m_max),
            "t_days_min": float(args.general_t_days_min),
            "t_days_max": float(args.general_t_days_max),
            "iv_min": float(args.general_iv_min),
            "iv_max": float(args.general_iv_max),
        },
        "seed": int(args.seed),
    }


def load_or_build_distillation_pool(env, actor, scaler, device, data_dir, args, output_dir, checkpoint):
    """
    Reuse a saved train-year distillation pool when the manifest matches.

    This is a pool cache, not a fitted-formula cache.  The expensive
    empirical/probe collection is a deterministic artifact with a small JSON
    manifest.
    """
    distill_path = output_dir / f"{args.model_name}_{checkpoint}_distillation_pairs.csv"
    manifest_path = output_dir / f"{args.model_name}_{checkpoint}_distillation_manifest.json"
    expected_config = distillation_pool_config(args, checkpoint)

    if not args.rebuild_distillation_pool and distill_path.exists() and manifest_path.exists():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current_manifest = {}
        if current_manifest.get("config") == expected_config:
            print(f"[distill] reusing cached train-year pool: {distill_path}")
            try:
                cached = pd.read_csv(distill_path)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
                print(
                    f"[distill] cached pool could not be read; rebuilding. "
                    f"Reason: {exc}"
                )
            else:
                if cached.empty:
                    print(f"[distill] cached pool is empty; rebuilding: {distill_path}")
                else:
                    return cached, distill_path
        print("[distill] cached pool is not usable for current settings; rebuilding.")
    elif distill_path.exists() and not manifest_path.exists():
        # Cache validation note: the CSV alone is not enough to prove that the
        # pool was built from the current model/split/probe settings.
        print("[distill] cached pool has no manifest; rebuilding for auditability.")

    print("[distill] collecting training-year target-delta pairs")
    train_df = collect_training_distillation_pairs(env, actor, scaler, device, data_dir, args)
    probe_df = add_probe_rows(train_df, actor, scaler, device, args)
    distill_df = pd.concat([train_df, probe_df], ignore_index=True)
    atomic_to_csv(distill_df, distill_path, index=False)

    manifest = {
        "config": expected_config,
        "observations": {
            "n_rows": int(len(distill_df)),
            "sample_sources": {
                str(k): int(v)
                for k, v in distill_df["sample_source"].value_counts(dropna=False).items()
            },
            "n_unique_feature_rows_rounded_8dp": int(
                len(
                    distill_df[["forward_moneyness", "T_years", "iv"]]
                    .round(8)
                    .drop_duplicates()
                )
            ),
        },
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2))
    return distill_df, distill_path


def coverage_diagnostics(df):
    """Summarize state-action support in a readable table."""
    cols = [
        "forward_moneyness",
        "T_years",
        "iv",
        "bs_delta",
        "agent_delta",
        "actual_agent_delta",
        "canonical_stock_position",
    ]
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "variable": col,
                "count": len(s),
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "p01": s.quantile(0.01),
                "p05": s.quantile(0.05),
                "p25": s.quantile(0.25),
                "median": s.quantile(0.50),
                "p75": s.quantile(0.75),
                "p95": s.quantile(0.95),
                "p99": s.quantile(0.99),
                "max": s.max(),
            }
        )
    meta = {
        "rows": len(df),
        "unique_feature_rows_rounded_8dp": len(
            df[["forward_moneyness", "T_years", "iv"]].round(8).drop_duplicates()
        ),
        "sample_sources": df["sample_source"].value_counts(dropna=False).to_dict(),
    }
    return pd.DataFrame(rows), meta


def fit_pysr_model(X, y, args, variable_names, weights=None):
    """Fit one PySR model with the finance-native operator set."""
    try:
        from pysr import PySRRegressor
    except ImportError as exc:
        raise ImportError("PySR is required for distill_empirical_agents.py symbolic fitting.") from exc

    # PySR/SymbolicRegression.jl has two different custom-loss signatures:
    #   - without row weights: loss(prediction, target)
    #   - with row weights:    loss(prediction, target, weight)
    # Robustness note:
    #     We always supplied a two-argument loss, then passed weights for the
    #     weighted candidates.  Julia correctly rejected that configuration
    #     after the first two ordinary-loss regressions had already completed.
    # NEW BEHAVIOR:
    #     Only weighted-loss experiments receive the three-argument loss.
    elementwise_loss = (
        "loss(x, y, w) = w * (x - y)^2"
        if weights is not None
        else "loss(x, y) = (x - y)^2"
    )

    model_kwargs = {
        "niterations": args.niterations,
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sqrt", "log", "exp", "erf"],
        "model_selection": "best",
        "elementwise_loss": elementwise_loss,
        "maxsize": args.maxsize,
        "populations": args.populations,
        "verbosity": 1,
        "random_state": args.seed,
    }
    if args.deterministic_search:
        model_kwargs["deterministic"] = True
        model_kwargs["parallelism"] = "serial"
    model = PySRRegressor(**model_kwargs)
    fit_kwargs = {"variable_names": variable_names}
    if weights is not None:
        fit_kwargs["weights"] = weights
    model.fit(X, y, **fit_kwargs)
    return model


def bandwidth_tag(scale):
    """Stable filename/column suffix for a smoothing bandwidth scale."""
    return f"bw{int(round(float(scale) * 100)):03d}"


def spec_smoothing_bandwidth_scale(spec, args):
    """Bandwidth scale attached to a formula spec, falling back to CLI default."""
    if spec.target_smoothing_bandwidth_scale is not None:
        return float(spec.target_smoothing_bandwidth_scale)
    return float(args.smoothing_bandwidth_scale)


def smooth_column_prefix_for_spec(spec, args):
    """Column prefix for the smoothed target attached to one formula spec."""
    return f"smooth_agent_{bandwidth_tag(spec_smoothing_bandwidth_scale(spec, args))}"


def smooth_policy_name_for_spec(spec, args):
    """Traded-policy name for the smoothed target attached to one formula spec."""
    return f"smooth_target_{bandwidth_tag(spec_smoothing_bandwidth_scale(spec, args))}"


def target_delta_col_for_spec(spec, args):
    """Return the delta-label column used by one formula candidate."""
    if spec.target_source == "raw_agent":
        return "agent_delta"
    if spec.target_source == "smooth_kernel_bs_delta_residual":
        return f"{smooth_column_prefix_for_spec(spec, args)}_delta"
    raise ValueError(f"Unknown target_source: {spec.target_source}")


def target_residual_col_for_spec(spec, args):
    """Return the residual diagnostic column used by one smoothed formula."""
    if spec.target_source != "smooth_kernel_bs_delta_residual":
        return None
    return f"{smooth_column_prefix_for_spec(spec, args)}_bs_residual"


def target_minus_raw_col_for_spec(spec, args):
    """Return the smoother-minus-raw diagnostic column for one formula spec."""
    if spec.target_source != "smooth_kernel_bs_delta_residual":
        return None
    return f"{smooth_column_prefix_for_spec(spec, args)}_minus_raw_agent_delta"


def label_delta_column_for_specs(specs):
    """
    Choose the policy-label column used by candidate sampling diagnostics.

    The raw-agent experiments sampled by |agent_delta - BS_delta|.  For
    the new smoothed-policy experiment, the analogous quantity is
    |smooth_agent_delta - BS_delta|, otherwise focus/gap sampling would still
    be steered by raw actor bumps that we are explicitly trying to smooth out.
    """
    target_sources = {spec.target_source for spec in specs}
    if target_sources == {"smooth_kernel_bs_delta_residual"}:
        return "smooth_agent_delta"
    return "agent_delta"


def add_sampling_columns(df, delta_col="agent_delta"):
    """
    Add training-only sampling weights and bins.

    The focus weight is not fitted to 2023.  It encodes general desiderata:
    preserve states where the learned agent differs from BS, preserve the
    high-gamma/transition region, and do not wash out moderately ITM,
    medium/high-IV states where the policy can have meaningful curvature.
    """
    out = df.copy()
    out["agent_bs_gap"] = (out[delta_col] - out["bs_delta"]).abs()

    # A smooth gamma-like proxy.  It is high near ATM, at shorter tenors, and at
    # lower IV, but remains finite everywhere.  This is only a sampling weight,
    # not a model input.
    tau = out["T_years"].clip(lower=1e-4)
    iv = out["iv"].clip(lower=1e-4)
    m = out["forward_moneyness"].clip(lower=1e-6)
    out["gamma_proxy"] = np.exp(-((np.log(m) / (iv * np.sqrt(tau))) ** 2) / 2.0) / (
        iv * np.sqrt(tau)
    )
    out["gamma_proxy"] = out["gamma_proxy"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # This is an ex ante economic focus region, not a 2023-fitted pocket:
    # moderately ITM calls, nontrivial IV, and medium maturities.  It is where a
    # minimum-variance policy can differ materially from BS.
    out["economic_focus"] = (
        out["forward_moneyness"].between(1.03, 1.15)
        & out["iv"].between(0.20, 0.38)
        & out["T_days"].between(25.0, 90.0)
    ).astype(float)

    return out


def build_sampling_pool(df, args, delta_col="agent_delta"):
    """Prepare a large training-only pool for all formula candidates."""
    feature_cols = ["forward_moneyness", "T_years", "iv"]
    pool = df.dropna(subset=feature_cols + [delta_col, "bs_delta", "T_days"]).copy()
    pool = add_sampling_columns(pool, delta_col=delta_col)

    def normalize(s):
        q = s.quantile(0.95)
        if not np.isfinite(q) or q <= 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s / q).clip(0.0, 3.0)

    gap_n = normalize(pool["agent_bs_gap"])
    gamma_n = normalize(pool["gamma_proxy"])
    pool["gap_weight"] = 1.0 + args.gap_weight_scale * gap_n
    pool["focus_weight"] = (
        1.0
        + args.gap_weight_scale * gap_n
        + args.gamma_weight_scale * gamma_n
        + args.focus_weight_boost * pool["economic_focus"]
    )

    m_edges = np.asarray(args.m_bins, dtype=float)
    iv_edges = np.asarray(args.iv_bins, dtype=float)
    t_edges = np.asarray(args.t_day_bins, dtype=float) / 365.0
    pool["m_bin"] = pd.cut(pool["forward_moneyness"], m_edges, include_lowest=True)
    pool["iv_bin"] = pd.cut(pool["iv"], iv_edges, include_lowest=True)
    pool["t_bin"] = pd.cut(pool["T_years"], t_edges, include_lowest=True)
    pool["cell"] = (
        pool["m_bin"].astype(str)
        + "|"
        + pool["iv_bin"].astype(str)
        + "|"
        + pool["t_bin"].astype(str)
    )
    return pool, feature_cols


def weighted_sample(df, n, weight_col, seed, replace=None):
    """Sample rows with explicit probability weights and stable seed."""
    if len(df) == 0 or n <= 0:
        return df.iloc[0:0].copy()
    if replace is None:
        replace = len(df) < n
    weights = df[weight_col].to_numpy(dtype=float)
    weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 0.0)
    if weights.sum() <= 0:
        weights = None
    else:
        weights = weights / weights.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(df.index.to_numpy(), size=n, replace=replace, p=weights)
    return df.loc[idx].copy()


def stratified_sample(pool, n_total, seed, weight_col=None):
    """
    Sample approximately evenly across moneyness/IV/T cells.

    Empty cells are ignored.  If a cell has fewer rows than requested, all its
    rows are used and the remaining budget is filled from the whole pool.
    """
    rng = np.random.default_rng(seed)
    groups = [(cell, g) for cell, g in pool.groupby("cell", observed=False) if len(g) > 0]
    if not groups:
        return pool.sample(min(n_total, len(pool)), random_state=seed)

    per_cell = max(1, int(math.ceil(n_total / len(groups))))
    pieces = []
    for _, g in groups:
        take = min(per_cell, len(g))
        if weight_col is None:
            pieces.append(g.sample(take, random_state=int(rng.integers(0, 2**31 - 1))))
        else:
            pieces.append(
                weighted_sample(
                    g,
                    take,
                    weight_col,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    replace=False,
                )
            )
    sampled = pd.concat(pieces, ignore_index=False)
    sampled = sampled[~sampled.index.duplicated(keep="first")]

    if len(sampled) < n_total:
        remaining = n_total - len(sampled)
        rest = pool.drop(index=sampled.index, errors="ignore")
        if len(rest) > 0:
            if weight_col is None:
                fill = rest.sample(
                    min(remaining, len(rest)),
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
            else:
                fill = weighted_sample(
                    rest,
                    min(remaining, len(rest)),
                    weight_col,
                    seed=int(rng.integers(0, 2**31 - 1)),
                    replace=False,
                )
            sampled = pd.concat([sampled, fill], ignore_index=False)

    if len(sampled) > n_total:
        sampled = sampled.sample(n_total, random_state=seed)
    return sampled.copy()


def select_fit_rows(pool, spec, args, seed):
    """
    Select rows for one formula candidate.

    This is where we control the experiments.  The rules deliberately avoid
    2023 performance feedback.
    """
    n = min(args.n_sr_samples, len(pool))
    if spec.sample_kind == "uniform":
        return pool.sample(n, random_state=seed).copy()
    if spec.sample_kind == "stratified":
        return stratified_sample(pool, n, seed)
    if spec.sample_kind == "gap_weighted":
        return weighted_sample(pool, n, "gap_weight", seed, replace=False)
    if spec.sample_kind == "focus":
        strat_n = int(round(n * args.focus_stratified_share))
        weighted_n = n - strat_n
        a = stratified_sample(pool, strat_n, seed, weight_col="focus_weight")
        rest = pool.drop(index=a.index, errors="ignore")
        if len(rest) == 0:
            b = weighted_sample(pool, weighted_n, "focus_weight", seed + 17, replace=True)
        else:
            b = weighted_sample(
                rest,
                min(weighted_n, len(rest)),
                "focus_weight",
                seed + 17,
                replace=False,
            )
        out = pd.concat([a, b], ignore_index=False)
        if len(out) < n:
            out = pd.concat(
                [
                    out,
                    weighted_sample(pool, n - len(out), "focus_weight", seed + 29, replace=True),
                ],
                ignore_index=False,
            )
        return out.sample(frac=1.0, random_state=seed + 31).head(n).copy()
    raise ValueError(f"Unknown sample_kind: {spec.sample_kind}")


def delta_to_normal_score(delta, eps):
    """
    Convert a bounded delta into the normal score z where delta = Phi(z).

    This keeps the symbolic regression unconstrained while the traded policy
    stays in [0, 1] after mapping back through the normal CDF.
    """
    y = np.clip(np.asarray(delta, dtype=float), eps, 1.0 - eps)
    return math.sqrt(2.0) * erfinv(2.0 * y - 1.0)


def delta_to_logit(delta, eps):
    """
    Convert bounded deltas to log-odds.

    This tests whether the actor's tanh-like bounded action geometry is easier
    for symbolic regression to mimic than the Black-Scholes/probit geometry.
    """
    y = np.clip(np.asarray(delta, dtype=float), eps, 1.0 - eps)
    return np.log(y / (1.0 - y))


def sigmoid(x):
    """Numerically stable sigmoid for logit-target formulas."""
    x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


class KernelKnnSmoother:
    """
    Local Gaussian nearest-neighbor smoother for policy labels.

    This is approach 1 from the smoothing discussion when fitted to raw
    agent_delta, and approach 4 when fitted to agent_delta - BS_delta.  The
    class is intentionally small and pickle-friendly so every smoother can be
    saved immediately after fitting.
    """

    def __init__(self, n_neighbors=256, bandwidth_scale=1.0):
        self.n_neighbors = int(n_neighbors)
        self.bandwidth_scale = float(bandwidth_scale)

    def fit(self, X, y, seed=123):
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.x_mean_ = X.mean(axis=0)
        self.x_std_ = X.std(axis=0)
        self.x_std_ = np.where(self.x_std_ > 1e-12, self.x_std_, 1.0)
        Xs = (X - self.x_mean_) / self.x_std_
        self.y_ = y
        self.n_neighbors_ = min(self.n_neighbors, len(Xs))
        self.nn_ = NearestNeighbors(n_neighbors=self.n_neighbors_, algorithm="auto")
        self.nn_.fit(Xs)

        # Estimate a stable bandwidth from a deterministic subsample's kth-NN
        # distance.  This avoids hand-tuning a scale in raw moneyness/IV/T units.
        rng = np.random.default_rng(seed)
        sample_n = min(5000, len(Xs))
        sample_idx = rng.choice(len(Xs), size=sample_n, replace=False)
        distances, _ = self.nn_.kneighbors(Xs[sample_idx])
        kth_distance = distances[:, -1]
        bandwidth = np.median(kth_distance[kth_distance > 0.0])
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            bandwidth = 1.0
        self.bandwidth_ = float(bandwidth * self.bandwidth_scale)
        return self

    def predict(self, X, chunk_size=20000):
        X = np.asarray(X, dtype=float)
        Xs = (X - self.x_mean_) / self.x_std_
        out = np.empty(len(Xs), dtype=float)
        for start in range(0, len(Xs), chunk_size):
            stop = min(start + chunk_size, len(Xs))
            distances, idx = self.nn_.kneighbors(Xs[start:stop])
            weights = np.exp(-0.5 * (distances / max(self.bandwidth_, 1e-12)) ** 2)
            weight_sum = weights.sum(axis=1)
            weight_sum = np.where(weight_sum > 1e-12, weight_sum, 1.0)
            out[start:stop] = (weights * self.y_[idx]).sum(axis=1) / weight_sum
        return out


class MonotonicMoneynessSmoother:
    """
    Shape-constrained smoother for the raw delta surface.

    It enforces nondecreasing delta in forward moneyness while letting IV and
    maturity enter flexibly.  This is approach 2 from the smoothing discussion.
    """

    def __init__(self, max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1e-3):
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.l2_regularization = float(l2_regularization)

    def fit(self, X, y, seed=123):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.model_ = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=self.l2_regularization,
            monotonic_cst=[1, 0, 0],
            random_state=seed,
            early_stopping=True,
        )
        self.model_.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def target_for_spec(fit_df, spec, args, eps):
    """Transform the selected policy label and BS deltas into the PySR target."""
    label_col = target_delta_col_for_spec(spec, args)

    if label_col not in fit_df.columns:
        raise ValueError(
            f"Formula {spec.name} requires {label_col}, but the column is absent. "
            "This usually means the smoother was not built before symbolic fitting."
        )

    agent_delta = fit_df[label_col].to_numpy(dtype=float)
    bs_delta = fit_df["bs_delta"].to_numpy(dtype=float)
    if spec.target_kind == "score":
        agent_score = delta_to_normal_score(agent_delta, eps)
        return agent_score
    if spec.target_kind == "logit":
        return delta_to_logit(agent_delta, eps)
    if spec.target_kind == "delta":
        return np.clip(agent_delta, 0.0, 1.0)
    if spec.target_kind == "delta_residual":
        return np.clip(agent_delta, 0.0, 1.0) - np.clip(bs_delta, 0.0, 1.0)
    raise ValueError(f"Unknown target_kind: {spec.target_kind}")


def fit_weights_for_spec(fit_df, spec, args):
    """
    Optional PySR row weights for loss-function experiments.

    This is separate from the row-sampling scheme.  A candidate can be sampled
    uniformly but still give higher loss weight to states where the learned
    policy materially departs from BS.  The normalization is training-only and
    robust to a few extreme gaps.
    """
    if spec.loss_kind == "default":
        return None
    if spec.loss_kind != "gap_weighted":
        raise ValueError(f"Unknown loss_kind: {spec.loss_kind}")

    label_col = (
        target_delta_col_for_spec(spec, args)
    )
    gap = (fit_df[label_col] - fit_df["bs_delta"]).abs().to_numpy(dtype=float)
    finite_gap = gap[np.isfinite(gap)]
    scale = np.quantile(finite_gap, 0.95) if len(finite_gap) else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.ones(len(fit_df), dtype=float)

    normalized_gap = np.clip(gap / scale, 0.0, 3.0)
    weights = 1.0 + float(args.loss_gap_weight_scale) * normalized_gap
    return np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)


def formula_text_from_score(score_text, spec):
    """Return the traded bounded delta formula text for one symbolic output."""
    if spec.target_kind == "score":
        return f"0.5 * (1 + erf(({score_text}) / sqrt(2)))"
    if spec.target_kind == "logit":
        return f"1 / (1 + exp(-({score_text})))"
    if spec.target_kind == "delta":
        return f"clip(({score_text}), 0, 1)"
    if spec.target_kind == "delta_residual":
        return (
            "clip(BS_DELTA(m_fwd, tau, iv, q) + "
            f"({score_text}), 0, 1)"
        )
    raise ValueError(f"Unknown target_kind: {spec.target_kind}")


def symbolic_candidate_config(args, checkpoint, spec):
    """
    Settings that define one symbolic candidate artifact.

    This cache is intentionally stricter than the distillation-pool cache.  If
    you change the loss, sample size, operators, iterations, seed, or formula
    specification, the model is refit instead of silently reusing a stale one.
    """
    return {
        "pool_version": DISTILLATION_POOL_VERSION,
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "candidate": spec.name,
        "sample_kind": spec.sample_kind,
        "target_kind": spec.target_kind,
        "loss_kind": spec.loss_kind,
        "target_source": spec.target_source,
        "target_smoothing_bandwidth_scale": (
            None
            if spec.target_smoothing_bandwidth_scale is None
            else float(spec.target_smoothing_bandwidth_scale)
        ),
        "n_sr_samples": int(args.n_sr_samples),
        "niterations": int(args.niterations),
        "maxsize": int(args.maxsize),
        "populations": int(args.populations),
        "bound_epsilon": float(args.bound_epsilon),
        "gap_weight_scale": float(args.gap_weight_scale),
        "loss_gap_weight_scale": float(args.loss_gap_weight_scale),
        "gamma_weight_scale": float(args.gamma_weight_scale),
        "focus_weight_boost": float(args.focus_weight_boost),
        "focus_stratified_share": float(args.focus_stratified_share),
        "deterministic_search": bool(args.deterministic_search),
        "device": args.device,
        # If the symbolic target is smoothed, the smoother's construction is
        # part of the candidate definition.  Include these fields so changing
        # the smoothing bandwidth/neighbors cannot silently reuse a stale PySR
        # formula trained on a different target surface.
        "max_smoothing_rows": int(args.max_smoothing_rows),
        "smoothing_neighbors": int(args.smoothing_neighbors),
        "smoothing_bandwidth_scale": float(args.smoothing_bandwidth_scale),
        "seed": int(args.seed),
    }


def symbolic_candidate_paths(output_dir, args, checkpoint, candidate_name):
    """File paths for one immediately saved PySR candidate."""
    stem = f"{args.model_name}_{checkpoint}_{candidate_name}"
    return {
        "model": output_dir / f"{stem}_pysr_model.pkl",
        "equations": output_dir / f"{stem}_pysr_equations.csv",
        "symbolic_formula": output_dir / f"{stem}_symbolic_formula.txt",
        "score_formula": output_dir / f"{stem}_score_formula.txt",
        "delta_formula": output_dir / f"{stem}_formula.txt",
        "manifest": output_dir / f"{stem}_fit_manifest.json",
    }


def _replace_with_windows_retries(tmp_path, path, max_attempts=25, sleep_seconds=0.20):
    """
    Replace a completed temporary artifact with retries for Windows/OneDrive.

    The repo is often run from a OneDrive-managed folder on Windows.  Very
    small JSON/CSV files can be scanned or synced exactly when Python calls
    os.replace, producing a transient WinError 5 even though the directory is
    writable.  Retrying keeps the artifact write atomic without treating a
    momentary file lock as a failed research run.
    """
    tmp_path = Path(tmp_path)
    path = Path(path)
    last_exc = None
    for attempt in range(int(max_attempts)):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as exc:
            last_exc = exc
            if attempt + 1 >= int(max_attempts):
                break
            time.sleep(float(sleep_seconds))
    raise last_exc


def atomic_write_text(path, text):
    """
    Write text through a same-directory temporary file and then replace.

    This matters for long overnight PySR runs: if the machine crashes halfway
    through a write, the previous valid artifact should not be left half
    overwritten.
    """
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    _replace_with_windows_retries(tmp_path, path)


def atomic_to_csv(df, path, **kwargs):
    """Atomic same-directory CSV write for cache and audit artifacts."""
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp_path, **kwargs)
    _replace_with_windows_retries(tmp_path, path)


def atomic_pickle_dump(obj, path):
    """Atomic pickle write for PySR models and smoothers."""
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    _replace_with_windows_retries(tmp_path, path)


def write_walkforward_status(output_dir, args, checkpoint, stage, **extra):
    """
    Lightweight progress marker for long parallel runs.

    PySR can spend hours inside one candidate without touching the final output
    files.  This marker makes it clear which year/candidate is currently active
    without changing any research output.
    """
    if output_dir is None or checkpoint is None:
        return
    path = Path(output_dir) / f"{args.model_name}_{checkpoint}_walkforward_status.json"
    payload = {
        "model_name": args.model_name,
        "target_year": int(getattr(args, "target_year", -1)),
        "validation_year": int(getattr(args, "validation_year", -1)),
        "checkpoint": int(checkpoint),
        "stage": stage,
        "pid": os.getpid(),
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
    }
    payload.update(extra)
    try:
        atomic_write_text(path, json.dumps(payload, indent=2))
    except OSError as exc:
        # Status markers are diagnostic only.  A transient OneDrive/Windows
        # lock on this JSON must never invalidate completed formula fits or
        # force a multi-hour year to rerun from diagnostic.
        print(
            f"[status] Warning: could not update {path}; continuing. "
            f"Research artifacts are unaffected. {exc}"
        )


def short_hof_selected_slug(comparison_name, left_policy, right_policy, selection_rank):
    """
    Short deterministic filename slug for validation-selected HOF comparisons.

    Windows paths can fail around the historical 260-character limit.  The full
    comparison name can be extremely long because it embeds the whole HOF policy
    id, so the CSV filename must be short.  The full comparison/left/right names
    are still stored inside the file and in the bootstrap summary.
    """
    digest_source = f"{comparison_name}|{left_policy}|{right_policy}|{selection_rank}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    right = str(right_policy).replace("smooth_target_", "smooth_")
    right = "".join(ch if ch.isalnum() else "_" for ch in right)[:24]
    try:
        rank = int(selection_rank)
    except Exception:
        rank = 0
    return f"hofsel_r{rank:02d}_{right}_{digest}"


def save_symbolic_candidate(model, spec, output_dir, args, checkpoint):
    """
    Save one candidate immediately after fitting.

    Implementation note:
        The script saved all PySR models only after all six regressions had
        finished.  A crash in candidate 3 threw away candidates 1-2.

    NEW BEHAVIOR:
        Each candidate is written as soon as it finishes, with a settings
        manifest that lets a later run reuse it if nothing relevant changed.
    """
    paths = symbolic_candidate_paths(output_dir, args, checkpoint, spec.name)
    atomic_pickle_dump(model, paths["model"])

    atomic_to_csv(model.equations_, paths["equations"], index=False)
    score_formula = str(model.sympy())
    delta_formula = formula_text_from_score(score_formula, spec)
    # Keep *_score_formula.txt for compatibility, but also write the clearer
    # *_symbolic_formula.txt because the raw target may now be logit, delta, or
    # BS-delta residual rather than a probit score.
    atomic_write_text(paths["symbolic_formula"], score_formula + "\n")
    atomic_write_text(paths["score_formula"], score_formula + "\n")
    atomic_write_text(paths["delta_formula"], delta_formula + "\n")

    manifest_entry = {
        "candidate": spec.name,
        "sample_kind": spec.sample_kind,
        "target_kind": spec.target_kind,
        "loss_kind": spec.loss_kind,
        "target_source": spec.target_source,
        "target_smoothing_bandwidth_scale": spec.target_smoothing_bandwidth_scale,
        "description": spec.description,
        "symbolic_formula": score_formula,
        "score_formula": score_formula,
        "delta_formula": delta_formula,
    }
    atomic_write_text(
        paths["manifest"],
        json.dumps(
            {
                "config": symbolic_candidate_config(args, checkpoint, spec),
                "candidate_manifest": manifest_entry,
            },
            indent=2,
        ),
    )
    return manifest_entry


def load_symbolic_candidate_if_current(spec, output_dir, args, checkpoint):
    """Load a finished candidate only if its per-candidate manifest matches."""
    if args.refit_symbolic_models:
        return None

    paths = symbolic_candidate_paths(output_dir, args, checkpoint, spec.name)
    if not paths["model"].exists() or not paths["manifest"].exists():
        return None

    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if manifest.get("config") != symbolic_candidate_config(args, checkpoint, spec):
        return None

    print(f"[fit] {spec.name}: loading cached PySR model")
    try:
        with open(paths["model"], "rb") as f:
            model = pickle.load(f)
    except Exception as exc:
        print(
            f"[fit] {spec.name}: cached PySR pickle could not be read; "
            f"refitting this candidate. Reason: {exc}"
        )
        return None

    # CRASH-SAFETY FIX:
    # The model pickle plus manifest is enough to resume computation, but it is
    # not enough for publication auditability.  Publication auditability requires the
    # entire Hall-of-Fame and formula text.  If a previous run crashed after
    # writing the pickle/manifest but before all sidecar files were present,
    # regenerate those sidecars from the cached PySR model immediately.
    required_sidecars = [
        paths["equations"],
        paths["symbolic_formula"],
        paths["score_formula"],
        paths["delta_formula"],
    ]
    if any(not p.exists() for p in required_sidecars):
        missing = [str(p) for p in required_sidecars if not p.exists()]
        print(
            f"[fit] {spec.name}: cached model is missing sidecar file(s); "
            f"regenerating: {missing}"
        )
        if not hasattr(model, "equations_"):
            raise RuntimeError(
                f"Cached PySR model for {spec.name} has no equations_ attribute; "
                "cannot regenerate the Hall-of-Fame. Use --refit-symbolic-models."
            )
        atomic_to_csv(model.equations_, paths["equations"], index=False)
        score_formula = str(model.sympy())
        delta_formula = formula_text_from_score(score_formula, spec)
        atomic_write_text(paths["symbolic_formula"], score_formula + "\n")
        atomic_write_text(paths["score_formula"], score_formula + "\n")
        atomic_write_text(paths["delta_formula"], delta_formula + "\n")

    return model, manifest["candidate_manifest"]


def fit_symbolic_policies(df, args, output_dir=None, checkpoint=None):
    """Fit all bounded symbolic-regression candidates."""
    selected_specs = [s for s in FORMULA_SPECS if s.name in args.formula_candidates]
    if not selected_specs:
        raise ValueError(f"No formula candidates selected: {args.formula_candidates}")

    fit_source_df = prepare_symbolic_target_labels(
        df, selected_specs, args, output_dir=output_dir, checkpoint=checkpoint
    )

    models = {}
    fit_frames = []
    candidate_manifest = []
    for i, spec in enumerate(selected_specs):
        seed = args.seed + 1009 * (i + 1)
        delta_col = target_delta_col_for_spec(spec, args)
        pool, feature_cols = build_sampling_pool(fit_source_df, args, delta_col=delta_col)
        fit_df = select_fit_rows(pool, spec, args, seed)
        X = fit_df[feature_cols].to_numpy(dtype=float)
        y = target_for_spec(fit_df, spec, args, args.bound_epsilon)
        weights = fit_weights_for_spec(fit_df, spec, args)

        print(
            f"[fit] {spec.name}: rows={len(fit_df)}, "
            f"sample_kind={spec.sample_kind}, target_kind={spec.target_kind}, "
            f"loss_kind={spec.loss_kind}"
        )
        write_walkforward_status(
            output_dir,
            args,
            checkpoint,
            "fit_candidate",
            candidate=spec.name,
            candidate_index=i + 1,
            n_candidates=len(selected_specs),
            fit_rows=int(len(fit_df)),
            cached_candidate=0,
        )
        cached = None
        if output_dir is not None and checkpoint is not None:
            cached = load_symbolic_candidate_if_current(spec, output_dir, args, checkpoint)

        if cached is None:
            model = fit_pysr_model(X, y, args, ["m_fwd", "tau", "iv"], weights=weights)
            if output_dir is not None and checkpoint is not None:
                manifest_entry = save_symbolic_candidate(model, spec, output_dir, args, checkpoint)
            else:
                score_formula = str(model.sympy())
                manifest_entry = {
                    "candidate": spec.name,
                    "sample_kind": spec.sample_kind,
                    "target_kind": spec.target_kind,
                    "loss_kind": spec.loss_kind,
                    "target_source": spec.target_source,
                    "target_smoothing_bandwidth_scale": spec.target_smoothing_bandwidth_scale,
                    "description": spec.description,
                    "symbolic_formula": score_formula,
                    "score_formula": score_formula,
                    "delta_formula": formula_text_from_score(score_formula, spec),
                }
        else:
            model, manifest_entry = cached
            write_walkforward_status(
                output_dir,
                args,
                checkpoint,
                "fit_candidate_loaded_from_cache",
                candidate=spec.name,
                candidate_index=i + 1,
                n_candidates=len(selected_specs),
                fit_rows=int(len(fit_df)),
                cached_candidate=1,
            )

        models[spec.name] = {"model": model, "spec": spec, "bound_epsilon": args.bound_epsilon}
        candidate_manifest.append(manifest_entry)
        write_walkforward_status(
            output_dir,
            args,
            checkpoint,
            "fit_candidate_done",
            candidate=spec.name,
            candidate_index=i + 1,
            n_candidates=len(selected_specs),
            fit_rows=int(len(fit_df)),
            cached_candidate=int(cached is not None),
        )

        info_cols = feature_cols + ["agent_delta", "bs_delta", "sample_source"]
        target_cols = [
            target_delta_col_for_spec(spec, args),
            target_residual_col_for_spec(spec, args),
            target_minus_raw_col_for_spec(spec, args),
        ]
        for col in target_cols:
            if col is not None and col in fit_df.columns and col not in info_cols:
                info_cols.append(col)
        fit_info = fit_df[info_cols].copy()
        fit_info["candidate"] = spec.name
        fit_info["sample_kind"] = spec.sample_kind
        fit_info["target_kind"] = spec.target_kind
        fit_info["loss_kind"] = spec.loss_kind
        fit_info["target_source"] = spec.target_source
        fit_info["loss_weight"] = weights if weights is not None else 1.0
        fit_frames.append(fit_info)

    return models, feature_cols, pd.concat(fit_frames, ignore_index=True), pd.DataFrame(candidate_manifest)


def formula_predict(model_entry, rows, feature_cols):
    """Predict bounded delta from one symbolic formula candidate."""
    model = model_entry["model"]
    spec = model_entry["spec"]
    X = rows[feature_cols].to_numpy(dtype=float)
    raw_symbolic = model.predict(X)
    if spec.target_kind == "score":
        bounded = 0.5 * (1.0 + erf(raw_symbolic / math.sqrt(2.0)))
    elif spec.target_kind == "logit":
        bounded = sigmoid(raw_symbolic)
    elif spec.target_kind == "delta":
        bounded = np.clip(raw_symbolic, 0.0, 1.0)
    elif spec.target_kind == "delta_residual":
        bounded = np.clip(rows["bs_delta"].to_numpy(dtype=float) + raw_symbolic, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown target_kind: {spec.target_kind}")
    return np.clip(bounded, 0.0, 1.0), raw_symbolic


def bounded_delta_from_symbolic_output(raw_symbolic, rows, spec):
    """Map a raw symbolic output to the traded delta for the formula family."""
    if spec.target_kind == "score":
        bounded = 0.5 * (1.0 + erf(raw_symbolic / math.sqrt(2.0)))
    elif spec.target_kind == "logit":
        bounded = sigmoid(raw_symbolic)
    elif spec.target_kind == "delta":
        bounded = np.clip(raw_symbolic, 0.0, 1.0)
    elif spec.target_kind == "delta_residual":
        bounded = np.clip(rows["bs_delta"].to_numpy(dtype=float) + raw_symbolic, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown target_kind: {spec.target_kind}")
    return np.clip(bounded, 0.0, 1.0)


def compile_hof_equation(equation):
    """
    Compile one PySR Hall-of-Fame equation into a NumPy callable.

    This intentionally lives outside the PySR model-selection API.  Our 2015
    diagnostics showed that PySR's selected `model.sympy()` formula can be much
    worse than other Hall-of-Fame equations.  For publication selection we
    therefore evaluate the whole frontier on validation data.
    """
    import sympy as sp

    m_fwd, tau, iv = sp.symbols("m_fwd tau iv")
    local_symbols = {
        "m_fwd": m_fwd,
        "tau": tau,
        "iv": iv,
        "erf": sp.erf,
        "sqrt": sp.sqrt,
        "log": sp.log,
        "exp": sp.exp,
    }
    expr = sp.sympify(str(equation), locals=local_symbols)
    return sp.lambdify(
        (m_fwd, tau, iv),
        expr,
        modules=[{"erf": erf, "sqrt": np.sqrt, "log": np.log, "exp": np.exp}, "numpy"],
    )


def hof_predict_delta(equation, rows, spec):
    """Evaluate one Hall-of-Fame equation on a state table."""
    fn = compile_hof_equation(equation)
    with np.errstate(all="ignore"):
        raw_symbolic = np.asarray(
            fn(
                rows["forward_moneyness"].to_numpy(dtype=float),
                rows["T_years"].to_numpy(dtype=float),
                rows["iv"].to_numpy(dtype=float),
            ),
            dtype=float,
        )
    if raw_symbolic.shape == ():
        raw_symbolic = np.full(len(rows), float(raw_symbolic))
    delta = bounded_delta_from_symbolic_output(raw_symbolic, rows, spec)
    return delta, raw_symbolic


def iter_hof_equations(models, max_equations=0):
    """Yield every equation on every selected candidate's Hall-of-Fame."""
    for candidate_name, model_entry in models.items():
        model = model_entry["model"]
        spec = model_entry["spec"]
        if not hasattr(model, "equations_"):
            continue
        hof = model.equations_.copy().reset_index(drop=True)
        if max_equations and max_equations > 0:
            hof = hof.head(int(max_equations))
        for hof_index, row in hof.iterrows():
            policy_name = (
                f"hof__{candidate_name}__idx{int(hof_index):03d}"
                f"__c{int(row.get('complexity', -1)):03d}"
            )
            yield {
                "policy": policy_name,
                "candidate": candidate_name,
                "spec": spec,
                "hof_index": int(hof_index),
                "complexity": int(row.get("complexity", -1)),
                "loss": float(row.get("loss", np.nan)),
                "score": float(row.get("score", np.nan)),
                "equation": str(row.get("equation")),
            }


def fidelity_table(models, df, feature_cols, label, target_col="agent_delta"):
    """Compute formula-vs-selected-policy target-delta fidelity."""
    rows = []
    pred_df = df.dropna(subset=feature_cols + [target_col]).copy()
    for policy_name, model_entry in models.items():
        clipped, raw_output = formula_predict(model_entry, pred_df, feature_cols)
        target_delta = pred_df[target_col].to_numpy(dtype=float)
        err = clipped - target_delta
        abs_err = np.abs(err)
        spec = model_entry["spec"]
        rows.append(
            {
                "dataset": label,
                "target_delta_col": target_col,
                "policy": policy_name,
                "sample_kind": spec.sample_kind,
                "target_kind": spec.target_kind,
                "target_source": spec.target_source,
                "n": len(pred_df),
                "mae": abs_err.mean(),
                "rmse": np.sqrt(np.mean(err ** 2)),
                "p95_abs_error": np.quantile(abs_err, 0.95),
                "max_abs_error": abs_err.max(),
                "corr": np.corrcoef(target_delta, clipped)[0, 1],
                "mean_target_delta": float(np.mean(target_delta)),
                "mean_agent_delta": pred_df["agent_delta"].mean()
                if "agent_delta" in pred_df.columns
                else np.nan,
                "mean_smooth_delta": pred_df["smooth_agent_delta"].mean()
                if "smooth_agent_delta" in pred_df.columns
                else np.nan,
                "mean_formula_delta": clipped.mean(),
                "raw_output_min": np.min(raw_output),
                "raw_output_max": np.max(raw_output),
                "delta_below_0_count": int(np.sum(clipped < 0.0)),
                "delta_above_1_count": int(np.sum(clipped > 1.0)),
            }
        )
    return pd.DataFrame(rows)


def switch_env_to_split(env, split):
    """Move DataKeeper to the requested evaluation split and reset its pointer."""
    if split == "validation":
        env.data_keeper.switch_to_validation()
    elif split == "test":
        env.data_keeper.switch_to_test()
    else:
        raise ValueError(f"Unknown evaluation split: {split}")
    env.data_keeper.reset(soft=False)


def collect_split_target_states(env, actor, scaler, device, args, split, split_year):
    """
    Collect validation/test states for out-of-sample formula fidelity only.

    This function is called after models are fitted.  It never feeds back into
    symbolic regression.
    """
    switch_env_to_split(env, split)
    rows = []
    for episode, dataset in enumerate(env.data_keeper.good_sets):
        raw_state = reset_env_on_dataset(env, dataset)
        done = False
        step = 0
        while not done:
            feature = feature_row_from_env(env, raw_state)
            actual_delta = actor_action(actor, scaler, device, raw_state)
            target_delta = actor_target_delta(
                actor, scaler, device, feature, args.canonical_position
            )
            feature.update(
                {
                    "sample_source": f"{split}_{split_year}",
                    "split": split,
                    "split_year": split_year,
                    "episode": episode,
                    "step": step,
                    "actual_agent_delta": actual_delta,
                    "agent_delta": target_delta,
                    "canonical_stock_position": canonical_stock_position(
                        feature, args.canonical_position
                    ),
                }
            )
            rows.append(feature)
            raw_state, _, done, _ = env.step(actual_delta)
            step += 1
    return pd.DataFrame(rows)


def collect_test_target_states(env, actor, scaler, device, args):
    """Compatibility wrapper for the single-model 2023 workflow."""
    return collect_split_target_states(
        env, actor, scaler, device, args, split="test", split_year=args.target_year
    )


def formula_delta_for_feature(model_entry, feature, feature_cols):
    """One-row symbolic policy evaluation for trading."""
    row = pd.DataFrame([feature])
    clipped, _ = formula_predict(model_entry, row, feature_cols)
    return float(clipped[0])


def smoother_predict_delta(smoother_entry, feature, feature_cols):
    """One-row smoothed-policy evaluation for trading."""
    row = pd.DataFrame([feature])
    X = row[feature_cols].to_numpy(dtype=float)
    if smoother_entry["kind"] == "kernel_delta":
        delta = smoother_entry["smoother"].predict(X)[0]
    elif smoother_entry["kind"] == "monotonic_delta":
        delta = smoother_entry["smoother"].predict(X)[0]
    elif smoother_entry["kind"] == "kernel_bs_delta_residual":
        residual = smoother_entry["smoother"].predict(X)[0]
        delta = float(row["bs_delta"].iloc[0]) + residual
    else:
        raise ValueError(f"Unknown smoother kind: {smoother_entry['kind']}")
    return float(np.clip(delta, 0.0, 1.0))


def trade_one_policy(
    env,
    dataset,
    policy_name,
    actor,
    scaler,
    device,
    args,
    model_entry=None,
    smoother_entry=None,
    feature_cols=None,
):
    """Trade one fixed 2023 option path with agent or one bounded formula."""
    raw_state = reset_env_on_dataset(env, dataset)
    done = False
    step = 0
    rows = []

    while not done:
        feature = feature_row_from_env(env, raw_state)
        agent_same_state_delta = actor_action(actor, scaler, device, raw_state)
        agent_target = actor_target_delta(actor, scaler, device, feature, args.canonical_position)

        if policy_name == "agent":
            chosen_delta = agent_same_state_delta
            formula_delta = np.nan
        elif smoother_entry is not None:
            chosen_delta = smoother_predict_delta(smoother_entry, feature, feature_cols)
            formula_delta = chosen_delta
        else:
            chosen_delta = formula_delta_for_feature(model_entry, feature, feature_cols)
            formula_delta = chosen_delta

        next_state, reward, done, info = env.step(chosen_delta)

        row = dict(feature)
        row.update(info)
        row.update(
            {
                "policy": policy_name,
                "step": step,
                "chosen_delta": chosen_delta,
                "agent_same_state_delta": agent_same_state_delta,
                "agent_target_delta": agent_target,
                "formula_delta": formula_delta,
                "formula_minus_agent_target": formula_delta - agent_target
                if np.isfinite(formula_delta)
                else np.nan,
            }
        )
        rows.append(row)
        raw_state = next_state
        step += 1

    return pd.DataFrame(rows)


def trade_split_year(
    env,
    actor,
    scaler,
    device,
    models,
    feature_cols,
    args,
    smoothing_entries=None,
    split="test",
    split_year=None,
):
    """
    Trade raw agent, optional smoothed agent, and all formulas on one split.

    For the smoothed-symbolic experiment this puts every relevant policy in
    one episode table, which is crucial: formula-vs-smooth-agent,
    formula-vs-raw-agent, formula-vs-BS, smooth-agent-vs-raw-agent, and
    smooth-agent-vs-BS all use the same paths and the same bootstrap code.
    """
    if split_year is None:
        split_year = args.target_year if split == "test" else args.validation_year
    switch_env_to_split(env, split)
    n_episodes = env.data_keeper.set_count
    if args.max_test_episodes and args.max_test_episodes > 0:
        n_episodes = min(n_episodes, args.max_test_episodes)

    frames = []
    for episode in range(n_episodes):
        dataset = env.data_keeper.good_sets[episode].copy()
        policy_specs = {"agent": None}
        smooth_policy_specs = smoothing_entries or {}
        policy_specs.update(models)

        for policy_name in list(policy_specs.keys()) + list(smooth_policy_specs.keys()):
            model_entry = policy_specs.get(policy_name)
            smoother_entry = smooth_policy_specs.get(policy_name)
            path = trade_one_policy(
                env,
                dataset,
                policy_name,
                actor,
                scaler,
                device,
                args,
                model_entry=model_entry,
                smoother_entry=smoother_entry,
                feature_cols=feature_cols,
            )
            path["episode"] = episode
            path["split"] = split
            path["split_year"] = split_year
            frames.append(path)

        if (episode + 1) % args.progress_every == 0:
            print(f"[trade] completed {split} {split_year} episode {episode + 1}/{n_episodes}")

    return pd.concat(frames, ignore_index=True)


def trade_test_year(
    env,
    actor,
    scaler,
    device,
    models,
    feature_cols,
    args,
    smoothing_entries=None,
):
    """Compatibility wrapper for the single-model 2023 workflow."""
    return trade_split_year(
        env,
        actor,
        scaler,
        device,
        models,
        feature_cols,
        args,
        smoothing_entries=smoothing_entries,
        split="test",
        split_year=args.target_year,
    )


def trade_smoothing_test_year(env, actor, scaler, device, smoothing_entries, feature_cols, args):
    """Trade raw agent and all smoothed policies on every 2023 test episode."""
    env.data_keeper.switch_to_test()
    env.data_keeper.reset(soft=False)
    n_episodes = env.data_keeper.set_count
    if args.max_test_episodes and args.max_test_episodes > 0:
        n_episodes = min(n_episodes, args.max_test_episodes)

    frames = []
    for episode in range(n_episodes):
        dataset = env.data_keeper.good_sets[episode].copy()
        policy_specs = {"agent": None}
        policy_specs.update(smoothing_entries)
        for policy_name, smoother_entry in policy_specs.items():
            path = trade_one_policy(
                env,
                dataset,
                policy_name,
                actor,
                scaler,
                device,
                args,
                smoother_entry=smoother_entry,
                feature_cols=feature_cols,
            )
            path["episode"] = episode
            frames.append(path)

        if (episode + 1) % args.progress_every == 0:
            print(f"[smooth-trade] completed 2023 episode {episode + 1}/{n_episodes}")

    return pd.concat(frames, ignore_index=True)


def build_rate_cache(cleaned_data_dir):
    """Local rate cache for option-pricing calculations in this script."""
    all_files = glob.glob(str(Path(cleaned_data_dir) / "*.parquet"))
    if not all_files:
        return {}
    df_list = [pd.read_parquet(f, columns=["quote_date", "risk_free_rate"]) for f in all_files]
    rates = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=["quote_date"])
    rates["quote_date"] = rates["quote_date"].astype(str).str.slice(0, 10)
    rates["r_clean"] = np.where(
        rates["risk_free_rate"] > 1.0,
        rates["risk_free_rate"] / 100.0,
        rates["risk_free_rate"],
    )
    return dict(zip(rates["quote_date"], rates["r_clean"]))


def calculate_episode_antonov(group, rate_cache):
    """
    Ex-post Antonov metric using the corrected interval convention.

    A Pos is the policy stock position, B Pos is the BS stock position.
    """
    # Result CSV rows are hedge intervals:
    #   Date      = interval start date
    #   S-1/P-1  = start-of-interval spot/option price
    #   S0/P0    = end-of-interval spot/option price
    #   A Pos    = policy stock position held over this row's interval
    #   B Pos    = BS stock position held over this row's interval
    # A Pos/B Pos belong to the row's own interval S-1[t] -> S0[t].
    # When DateEnd is present, every row including the final interval is used.
    # Legacy CSVs without DateEnd use the conservative cached-artifact fallback.
    df = group.copy()
    df["_original_row_order"] = np.arange(len(df))
    df = df.sort_values(["Date", "_original_row_order"], kind="mergesort").reset_index(drop=True)
    n_steps = len(df)
    if n_steps < 2:
        return pd.Series({"D_policy": np.nan, "D_BS": np.nan})

    P_start = df["P-1"].values
    S_start = df["S-1"].values
    delta_policy = df["A Pos"].values
    delta_bs = df["B Pos"].values
    dates = df["Date"].astype(str).str.slice(0, 10).values
    tc_policy = df["A TC"].values if "A TC" in df.columns else np.zeros(n_steps)
    tc_bs = df["B TC"].values if "B TC" in df.columns else np.zeros(n_steps)

    has_full_interval_dates = (
        "DateEnd" in df.columns
        and df["DateEnd"].notna().all()
        and (df["DateEnd"].astype(str).str.len() >= 10).all()
    )

    if has_full_interval_dates:
        # Full fixed convention for newly generated formula-trading CSVs:
        # every row is a complete Date -> DateEnd interval.
        dt_years = (
            pd.to_datetime(df["DateEnd"]) - pd.to_datetime(df["Date"])
        ).dt.days.values / 365.0
        h_policy = df["P0"].values[-1]
        h_bs = df["P0"].values[-1]
        loop_range = range(n_steps - 1, -1, -1)
    else:
        # Legacy fallback for older step files that do not record DateEnd.
        date_series = pd.to_datetime(df["Date"])
        dt_years = (date_series.shift(-1) - date_series).dt.days.values / 365.0
        h_policy = P_start[-1]
        h_bs = P_start[-1]
        loop_range = range(n_steps - 2, -1, -1)

    chi2_policy = 0.0
    chi2_bs = 0.0
    total_T = 0.0

    for t in loop_range:
        dt = dt_years[t]
        if not np.isfinite(dt) or dt <= 0:
            continue
        if "r" in df.columns and pd.notna(df["r"].iloc[t]):
            r = float(df["r"].iloc[t])
            if r > 1.0:
                r /= 100.0
        else:
    # Same visible-but-nonfatal 1% fallback convention as the paper metrics.
            r = rate_cache.get(dates[t], 0.01)
        discount = np.exp(-r * dt)
        S_end = df["S0"].values[t] if has_full_interval_dates else S_start[t + 1]

        profit_policy = delta_policy[t] * ((discount * S_end) - S_start[t]) + tc_policy[t]
        h_policy = (discount * h_policy) + profit_policy

        profit_bs = delta_bs[t] * ((discount * S_end) - S_start[t]) + tc_bs[t]
        h_bs = (discount * h_bs) + profit_bs

        chi2_policy += ((h_policy - P_start[t]) ** 2) * dt
        chi2_bs += ((h_bs - P_start[t]) ** 2) * dt
        total_T += dt

    if total_T <= 0:
        return pd.Series({"D_policy": np.nan, "D_BS": np.nan})
    return pd.Series(
        {
            "D_policy": np.sqrt(chi2_policy / total_T),
            "D_BS": np.sqrt(chi2_bs / total_T),
        }
    )


def episode_metrics_from_steps(trade_steps, rate_cache):
    """Aggregate step-level trading output to episode/policy metrics."""
    antonov = (
        trade_steps.groupby(["policy", "episode"])
        .apply(lambda g: calculate_episode_antonov(g, rate_cache), include_groups=False)
        .reset_index()
    )
    sums = (
        trade_steps.groupby(["policy", "episode"])[
            ["A PnL", "B PnL", "A Reward", "B Reward"]
        ]
        .sum()
        .reset_index()
    )
    starts = (
        trade_steps.groupby(["policy", "episode"])["Date"]
        .first()
        .reset_index()
        .rename(columns={"Date": "start_date"})
    )
    metrics = sums.merge(starts, on=["policy", "episode"]).merge(
        antonov, on=["policy", "episode"]
    )
    return metrics


def policy_table(metrics, policy):
    """Return one policy's episode table with standardized column names."""
    df = metrics[metrics["policy"] == policy].copy()
    return df.rename(
        columns={
            "A PnL": "PnL",
            "A Reward": "Reward",
            "D_policy": "Antonov",
        }
    )[["episode", "start_date", "PnL", "Reward", "Antonov"]]


def bs_table_from_agent(metrics):
    """Use the agent run's BS path as the single BS benchmark table."""
    df = metrics[metrics["policy"] == "agent"].copy()
    return df.rename(
        columns={
            "B PnL": "PnL",
            "B Reward": "Reward",
            "D_BS": "Antonov",
        }
    )[["episode", "start_date", "PnL", "Reward", "Antonov"]].assign(policy="bs")


def paired_episode_table(metrics, left_policy, right_policy):
    """Build pairwise episode table for bootstrap comparison."""
    left = policy_table(metrics, left_policy)
    if right_policy == "bs":
        right = bs_table_from_agent(metrics).drop(columns=["policy"])
    else:
        right = policy_table(metrics, right_policy)

    merged = left.merge(right, on=["episode", "start_date"], suffixes=("_left", "_right"))
    merged = merged.dropna(subset=["Antonov_left", "Antonov_right"])
    return merged


def run_two_stage_bootstrap_pair(pair_df, n_bootstrap, seed):
    """
    Same two-stage bootstrap structure as the paper metrics.

    First sample non-overlapping date clusters with replacement.  Then, inside
    each sampled date cluster, sample episodes with replacement.
    """
    rng = np.random.default_rng(seed)
    ep_data = pair_df.copy()

    # Main plots scale terminal PnLs by 100 before computing PnL metrics.
    ep_data["PnL_left"] *= 100.0
    ep_data["PnL_right"] *= 100.0
    ep_data["Diff_PnL"] = ep_data["PnL_left"] - ep_data["PnL_right"]
    ep_data["Diff_Rew"] = ep_data["Reward_left"] - ep_data["Reward_right"]
    ep_data["Diff_Antonov"] = ep_data["Antonov_left"] - ep_data["Antonov_right"]

    clusters = ep_data["start_date"].unique()
    n_clusters = len(clusters)
    cluster_arrays = [
        ep_data[ep_data["start_date"] == d][
            ["PnL_left", "PnL_right", "Diff_PnL", "Diff_Rew", "Diff_Antonov"]
        ].values
        for d in clusters
    ]

    boot = {k: np.zeros(n_bootstrap) for k in METRICS}
    for i in range(n_bootstrap):
        sampled_cluster_idxs = rng.integers(0, n_clusters, size=n_clusters)
        resampled = []
        for idx in sampled_cluster_idxs:
            c_data = cluster_arrays[idx]
            row_idxs = rng.integers(0, c_data.shape[0], size=c_data.shape[0])
            resampled.append(c_data[row_idxs])
        mat = np.concatenate(resampled, axis=0)

        pnl_left, pnl_right = mat[:, 0], mat[:, 1]
        diff_pnl, diff_rew, diff_ant = mat[:, 2], mat[:, 3], mat[:, 4]
        var_left, var_right = np.var(pnl_left, ddof=1), np.var(pnl_right, ddof=1)
        down_left = np.where(pnl_left < 0.0, pnl_left, 0.0)
        down_right = np.where(pnl_right < 0.0, pnl_right, 0.0)

        boot["mean"][i] = np.mean(diff_pnl)
        boot["std"][i] = np.sqrt(var_left) - np.sqrt(var_right)
        boot["rew"][i] = np.mean(diff_rew)
        boot["antonov"][i] = np.mean(diff_ant)

        var_ratio = var_left / var_right if var_right > 1e-10 else np.nan
        down_denom = np.sum(down_right ** 2)
        down_ratio = (
            (np.sum(down_left ** 2) / len(down_left))
            / (down_denom / len(down_right))
            if down_denom > 1e-10
            else np.nan
        )
        boot["log_var_ratio"][i] = np.log(var_ratio) if var_ratio > 0 else np.nan
        boot["log_down_var_ratio"][i] = np.log(down_ratio) if down_ratio > 0 else np.nan

        c_idx = max(1, int(len(pnl_left) * 0.05))
        boot["cvar"][i] = np.mean(np.partition(pnl_left, c_idx - 1)[:c_idx]) - np.mean(
            np.partition(pnl_right, c_idx - 1)[:c_idx]
        )
    return boot


def point_metrics(pair_df):
    """Point estimates on the original, non-resampled episode table."""
    df = pair_df.copy()
    left = df["PnL_left"].to_numpy(dtype=float) * 100.0
    right = df["PnL_right"].to_numpy(dtype=float) * 100.0
    var_left, var_right = np.var(left, ddof=1), np.var(right, ddof=1)
    down_left = np.where(left < 0.0, left, 0.0)
    down_right = np.where(right < 0.0, right, 0.0)
    c_idx = max(1, int(len(left) * 0.05))
    return {
        "n_episodes": len(df),
        "n_clusters": df["start_date"].nunique(),
        "mean": np.mean(left - right),
        "std": np.sqrt(var_left) - np.sqrt(var_right),
        "rew": np.mean(df["Reward_left"] - df["Reward_right"]),
        "cvar": np.mean(np.partition(left, c_idx - 1)[:c_idx])
        - np.mean(np.partition(right, c_idx - 1)[:c_idx]),
        "log_var_ratio": np.log(var_left / var_right) if var_right > 1e-10 else np.nan,
        "log_down_var_ratio": np.log(
            (np.sum(down_left ** 2) / len(down_left))
            / (np.sum(down_right ** 2) / len(down_right))
        )
        if np.sum(down_right ** 2) > 1e-10
        else np.nan,
        "antonov": np.mean(df["Antonov_left"] - df["Antonov_right"]),
    }


def summarize_bootstrap(pair_name, pair_df, args):
    """Summarize bootstrap distributions into main_plots-style intervals."""
    # Python's built-in hash is intentionally randomized between processes.
    # Use a tiny stable name hash so repeated manual runs are reproducible.
    stable_offset = sum((i + 1) * ord(ch) for i, ch in enumerate(pair_name))
    boot = run_two_stage_bootstrap_pair(pair_df, args.n_bootstrap, args.seed + stable_offset)
    points = point_metrics(pair_df)
    alpha = (1.0 - args.confidence_level) / 2.0
    rows = []
    for metric in METRICS:
        dist = boot[metric][~np.isnan(boot[metric])]
        center = float(np.mean(dist))
        lb, ub = np.percentile(dist, [alpha * 100.0, (1.0 - alpha) * 100.0])
        lb90, ub90 = np.percentile(dist, [5.0, 95.0])
        lb95, ub95 = np.percentile(dist, [2.5, 97.5])
        lb99, ub99 = np.percentile(dist, [0.5, 99.5])
        # Store a two-sided bootstrap sign p-value as an audit field.  The
        # table stars should still be driven by fixed percentile intervals,
        # but this makes later sensitivity checks possible without rerunning.
        p_left = float(np.mean(dist <= 0.0))
        p_right = float(np.mean(dist >= 0.0))
        p_value = min(1.0, 2.0 * min(p_left, p_right))
        rows.append(
            {
                "comparison": pair_name,
                "metric": metric,
                "point_estimate": points[metric],
                "bootstrap_center": center,
                "ci_low": lb,
                "ci_high": ub,
                "err_low": center - lb,
                "err_high": ub - center,
                "significant": int(lb > 0.0 or ub < 0.0),
                "ci_low_90": lb90,
                "ci_high_90": ub90,
                "significant_90": int(lb90 > 0.0 or ub90 < 0.0),
                "ci_low_95": lb95,
                "ci_high_95": ub95,
                "significant_95": int(lb95 > 0.0 or ub95 < 0.0),
                "ci_low_99": lb99,
                "ci_high_99": ub99,
                "significant_99": int(lb99 > 0.0 or ub99 < 0.0),
                "bootstrap_p_value": p_value,
                "n_episodes": points["n_episodes"],
                "n_clusters": points["n_clusters"],
            }
        )
    return pd.DataFrame(rows)


def smoothing_candidate_specs():
    """The three smoothing approaches requested for smoothed-vs-raw diagnostics."""
    return [
        {
            "name": "smooth_kernel_delta",
            "kind": "kernel_delta",
            "description": "Approach 1: local Gaussian KNN smoothing of raw agent delta.",
        },
        {
            "name": "smooth_monotonic_delta",
            "kind": "monotonic_delta",
            "description": "Approach 2: monotone-in-moneyness smooth raw delta model.",
        },
        {
            "name": "smooth_kernel_bs_delta_residual",
            "kind": "kernel_bs_delta_residual",
            "description": "Approach 4: local Gaussian KNN smoothing of agent_delta - BS_delta.",
        },
    ]


def smoothing_candidate_config(args, checkpoint, spec):
    """Settings that define one saved smoothing artifact."""
    return {
        "pool_version": DISTILLATION_POOL_VERSION,
        "model_name": args.model_name,
        "checkpoint": str(checkpoint),
        "smoother": spec["name"],
        "kind": spec["kind"],
        "max_smoothing_rows": int(args.max_smoothing_rows),
        "smoothing_neighbors": int(args.smoothing_neighbors),
        "smoothing_bandwidth_scale": float(args.smoothing_bandwidth_scale),
        "device": args.device,
        "monotonic_max_iter": int(args.monotonic_max_iter),
        "monotonic_learning_rate": float(args.monotonic_learning_rate),
        "monotonic_max_leaf_nodes": int(args.monotonic_max_leaf_nodes),
        "monotonic_l2": float(args.monotonic_l2),
        "seed": int(args.seed),
    }


def smoothing_candidate_paths(output_dir, args, checkpoint, smoother_name):
    """File paths for one immediately saved smoother."""
    stem = f"{args.model_name}_{checkpoint}_{smoother_name}"
    return {
        "model": output_dir / f"{stem}_smoother.pkl",
        "manifest": output_dir / f"{stem}_smoother_manifest.json",
    }


def smoothing_training_frame(distill_df, args):
    """Select a deterministic training-only frame for smoothing diagnostics."""
    feature_cols = ["forward_moneyness", "T_years", "iv"]
    required = feature_cols + ["agent_delta", "bs_delta"]
    pool = distill_df.dropna(subset=required).copy()
    if args.max_smoothing_rows and args.max_smoothing_rows > 0 and len(pool) > args.max_smoothing_rows:
        pool = pool.sample(args.max_smoothing_rows, random_state=args.seed).copy()
    return pool, feature_cols


def build_one_smoother(pool, feature_cols, spec, args):
    """Fit one smoothed target surface from training/probe labels only."""
    X = pool[feature_cols].to_numpy(dtype=float)
    agent_delta = pool["agent_delta"].to_numpy(dtype=float)
    bs_delta = pool["bs_delta"].to_numpy(dtype=float)
    if spec["kind"] == "kernel_delta":
        smoother = KernelKnnSmoother(
            n_neighbors=args.smoothing_neighbors,
            bandwidth_scale=args.smoothing_bandwidth_scale,
        ).fit(X, agent_delta, seed=args.seed)
    elif spec["kind"] == "monotonic_delta":
        smoother = MonotonicMoneynessSmoother(
            max_iter=args.monotonic_max_iter,
            learning_rate=args.monotonic_learning_rate,
            max_leaf_nodes=args.monotonic_max_leaf_nodes,
            l2_regularization=args.monotonic_l2,
        ).fit(X, agent_delta, seed=args.seed)
    elif spec["kind"] == "kernel_bs_delta_residual":
        smoother = KernelKnnSmoother(
            n_neighbors=args.smoothing_neighbors,
            bandwidth_scale=args.smoothing_bandwidth_scale,
        ).fit(X, agent_delta - bs_delta, seed=args.seed)
    else:
        raise ValueError(f"Unknown smoothing kind: {spec['kind']}")
    return smoother


def load_or_build_one_smoother(
    distill_df,
    args,
    output_dir,
    checkpoint,
    smoother_name,
    bandwidth_scale=None,
):
    """
    Build/load exactly one smoothing artifact.

    Symbolic-on-smoothed-agent runs need only the requested smoother.  Loading
    a single smoother keeps the run focused: no extra monotonic or raw-delta
    smoother work before PySR starts.
    """
    specs = {spec["name"]: spec for spec in smoothing_candidate_specs()}
    if smoother_name in specs:
        spec = specs[smoother_name]
    elif smoother_name.startswith("smooth_kernel_bs_delta_residual_"):
        # Bandwidth-sensitivity smoothers are all the same residual KNN
        # architecture, but each has a distinct artifact name.  This avoids
        # overwriting bw075/bw100/bw125/bw150 caches while still reusing the
        # original smoothing code path.
        spec = {
            "name": smoother_name,
            "kind": "kernel_bs_delta_residual",
            "description": (
                "Bandwidth-sensitivity target: local Gaussian KNN smoothing "
                "of agent_delta - BS_delta."
            ),
        }
    else:
        raise ValueError(f"Unknown smoother requested: {smoother_name}")

    local_args = copy.copy(args)
    if bandwidth_scale is not None:
        local_args.smoothing_bandwidth_scale = float(bandwidth_scale)

    pool, feature_cols = smoothing_training_frame(distill_df, args)
    paths = smoothing_candidate_paths(output_dir, args, checkpoint, spec["name"])
    expected_config = smoothing_candidate_config(local_args, checkpoint, spec)
    loaded = False

    if not args.refit_smoothers and paths["model"].exists() and paths["manifest"].exists():
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("config") == expected_config:
            print(f"[smooth] {spec['name']}: loading cached smoother")
            with open(paths["model"], "rb") as f:
                smoother = pickle.load(f)
            loaded = True

    if not loaded:
        print(
            f"[smooth] fitting {spec['name']} on {len(pool)} rows "
            f"(bandwidth_scale={local_args.smoothing_bandwidth_scale})"
        )
        smoother = build_one_smoother(pool, feature_cols, spec, local_args)
        atomic_pickle_dump(smoother, paths["model"])
        atomic_write_text(
            paths["manifest"],
            json.dumps(
                {
                    "config": expected_config,
                    "description": spec["description"],
                    "feature_cols": feature_cols,
                    "training_rows": len(pool),
                },
                indent=2,
            ),
        )

    entry = {
        "kind": spec["kind"],
        "smoother": smoother,
        "description": spec["description"],
        "bandwidth_scale": float(local_args.smoothing_bandwidth_scale),
    }
    manifest_row = {
        "smoother": spec["name"],
        "kind": spec["kind"],
        "description": spec["description"],
        "training_rows": len(pool),
        "bandwidth_scale": float(local_args.smoothing_bandwidth_scale),
    }
    return entry, feature_cols, manifest_row


def load_or_build_smoothers(distill_df, args, output_dir, checkpoint):
    """
    Build/load the three requested smoothed policies.

    Each smoother is saved immediately after fitting, for the same crash-safety
    reason as the PySR models.
    """
    entries = {}
    manifest_rows = []
    for spec in smoothing_candidate_specs():
        entry, feature_cols, manifest_row = load_or_build_one_smoother(
            distill_df, args, output_dir, checkpoint, spec["name"]
        )
        entries[spec["name"]] = entry
        manifest_rows.append(manifest_row)
    return entries, feature_cols, pd.DataFrame(manifest_rows)


def add_smoothed_agent_columns(df, smoother_entry, feature_cols, column_prefix="smooth_agent"):
    """
    Add smoothed policy labels without overwriting the raw neural labels.

    The raw actor's `agent_delta` remains untouched, because we need later
    comparisons formula-vs-raw-agent and smooth-agent-vs-raw-agent.  The new
    columns are:
        {column_prefix}_delta                : clipped smoothed target delta.
        {column_prefix}_bs_residual          : smoothed delta - BS delta.
        {column_prefix}_minus_raw_agent_delta: diagnostic smoothing distortion.
    """
    out = df.copy()
    preds = []
    for start in range(0, len(out), 20000):
        chunk = out.iloc[start : start + 20000]
        X = chunk[feature_cols].to_numpy(dtype=float)
        if smoother_entry["kind"] == "kernel_bs_delta_residual":
            delta = chunk["bs_delta"].to_numpy(dtype=float) + smoother_entry["smoother"].predict(X)
        elif smoother_entry["kind"] in {"kernel_delta", "monotonic_delta"}:
            delta = smoother_entry["smoother"].predict(X)
        else:
            raise ValueError(f"Unknown smoother kind: {smoother_entry['kind']}")
        preds.append(np.clip(delta, 0.0, 1.0))

    delta_col = f"{column_prefix}_delta"
    residual_col = f"{column_prefix}_bs_residual"
    minus_raw_col = f"{column_prefix}_minus_raw_agent_delta"
    out[delta_col] = np.concatenate(preds) if preds else np.asarray([], dtype=float)
    out[residual_col] = out[delta_col] - out["bs_delta"]
    out[minus_raw_col] = out[delta_col] - out["agent_delta"]

    # Compatibility aliases for the single-bandwidth smoothing outputs.  They
    # are written only for the canonical column prefix so existing analysis
    # code and reports keep working.
    if column_prefix == "smooth_agent":
        out["smooth_agent_delta"] = out[delta_col]
        out["smooth_agent_bs_residual"] = out[residual_col]
        out["smooth_minus_raw_agent_delta"] = out[minus_raw_col]
    return out


def selected_specs_need_smooth_target(specs):
    """Return True if any selected PySR candidate is trained on smoothed labels."""
    return any(spec.target_source == "smooth_kernel_bs_delta_residual" for spec in specs)


def unique_smoothing_targets_for_specs(specs, args):
    """Map unique bandwidth tags to smoother metadata required by selected specs."""
    targets = {}
    for spec in specs:
        if spec.target_source != "smooth_kernel_bs_delta_residual":
            continue
        scale = spec_smoothing_bandwidth_scale(spec, args)
        tag = bandwidth_tag(scale)
        targets[tag] = {
            "bandwidth_scale": scale,
            "column_prefix": f"smooth_agent_{tag}",
            "policy_name": f"smooth_target_{tag}",
            "smoother_name": f"smooth_kernel_bs_delta_residual_{tag}",
        }
    return targets


def prepare_symbolic_target_labels(df, selected_specs, args, output_dir=None, checkpoint=None):
    """
    Prepare the label columns used by symbolic regression.

    Sampling note:
        PySR fitted directly to `agent_delta`.

    NEW SMOOTHED-POLICY BEHAVIOR:
        If any selected candidate asks for
        target_source='smooth_kernel_bs_delta_residual', build/load that
        smoother from training-only labels and add `smooth_agent_delta`.
        This is still leakage-safe because the smoother is trained only on the
        cached 2010-2021 training/probe pool, never on 2023 states.
    """
    if not selected_specs_need_smooth_target(selected_specs):
        return df
    if output_dir is None or checkpoint is None:
        raise ValueError("Smoothed symbolic targets require output_dir and checkpoint.")

    out = df.copy()
    for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
        smoother_entry, feature_cols, _ = load_or_build_one_smoother(
            df,
            args,
            Path(output_dir),
            checkpoint,
            target["smoother_name"],
            bandwidth_scale=target["bandwidth_scale"],
        )
        out = add_smoothed_agent_columns(
            out,
            smoother_entry,
            feature_cols,
            column_prefix=target["column_prefix"],
        )
    return out


def smoother_fidelity_table(smoothing_entries, df, feature_cols, label):
    """Compute smoothed-policy-vs-raw-target fidelity on a state table."""
    rows = []
    pred_df = df.dropna(subset=feature_cols + ["agent_delta", "bs_delta"]).copy()
    for name, entry in smoothing_entries.items():
        preds = []
        for start in range(0, len(pred_df), 20000):
            chunk = pred_df.iloc[start : start + 20000]
            X = chunk[feature_cols].to_numpy(dtype=float)
            if entry["kind"] == "kernel_delta":
                delta = entry["smoother"].predict(X)
            elif entry["kind"] == "monotonic_delta":
                delta = entry["smoother"].predict(X)
            elif entry["kind"] == "kernel_bs_delta_residual":
                delta = chunk["bs_delta"].to_numpy(dtype=float) + entry["smoother"].predict(X)
            else:
                raise ValueError(f"Unknown smoothing kind: {entry['kind']}")
            preds.append(np.clip(delta, 0.0, 1.0))
        clipped = np.concatenate(preds)
        err = clipped - pred_df["agent_delta"].to_numpy(dtype=float)
        abs_err = np.abs(err)
        rows.append(
            {
                "dataset": label,
                "policy": name,
                "kind": entry["kind"],
                "n": len(pred_df),
                "mae": abs_err.mean(),
                "rmse": np.sqrt(np.mean(err ** 2)),
                "p95_abs_error": np.quantile(abs_err, 0.95),
                "max_abs_error": abs_err.max(),
                "corr": np.corrcoef(pred_df["agent_delta"], clipped)[0, 1],
                "mean_agent_delta": pred_df["agent_delta"].mean(),
                "mean_smooth_delta": clipped.mean(),
            }
        )
    return pd.DataFrame(rows)


def run_smoothing_diagnostics_for_model(args):
    """Run smoothed-policy-vs-raw-agent 2023 diagnostics for one model."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, selected_csv = infer_checkpoint(args.model_name, args.results_testing_dir)
    settings = load_settings_json(args.model_name)
    data_dir = ensure_walkforward_data_dir(args)

    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(data_dir)
    try:
        from empirical_deep_hedging.include.env import Env

        actor, scaler, device = load_actor_and_scaler(
            args.model_name, checkpoint, settings, args.device
        )
        env = Env(settings)

        distill_df, distill_path = load_or_build_distillation_pool(
            env, actor, scaler, device, data_dir, args, output_dir, checkpoint
        )
        smoothers, feature_cols, smooth_manifest = load_or_build_smoothers(
            distill_df, args, output_dir, checkpoint
        )
        smooth_manifest.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_smoothing_manifest.csv",
            index=False,
        )

        distill_fid = smoother_fidelity_table(smoothers, distill_df, feature_cols, "distillation_train_only")
        distill_fid.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_smoothing_distillation_fidelity.csv",
            index=False,
        )

        print("[smooth-test] collecting 2023 states for smoothing fidelity")
        test_state_df = collect_test_target_states(env, actor, scaler, device, args)
        test_state_df.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_smoothing_test_target_states.csv",
            index=False,
        )
        test_fid = smoother_fidelity_table(smoothers, test_state_df, feature_cols, "test_2023")
        test_fid.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_smoothing_test_fidelity.csv",
            index=False,
        )

        print("[smooth-test] trading raw agent and smoothed policies on 2023")
        trade_steps = trade_smoothing_test_year(env, actor, scaler, device, smoothers, feature_cols, args)
        trade_steps_path = output_dir / f"{args.model_name}_{checkpoint}_smoothing_test_trade_steps.csv"
        trade_steps.to_csv(trade_steps_path, index=False)

        rate_cache = build_rate_cache(args.cleaned_data_dir)
        episode_metrics = episode_metrics_from_steps(trade_steps, rate_cache)
        episode_metrics.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_smoothing_test_episode_metrics.csv",
            index=False,
        )

        print("[smooth-bootstrap] running two-stage bootstrap comparisons")
        boot_frames = []
        comparisons = []
        for smoother_name in smoothers:
            comparisons.append((f"{smoother_name}_vs_agent", smoother_name, "agent"))
            comparisons.append((f"{smoother_name}_vs_bs", smoother_name, "bs"))
        for name, left, right in comparisons:
            pair_df = paired_episode_table(episode_metrics, left, right)
            pair_df.to_csv(
                output_dir / f"{args.model_name}_{checkpoint}_{name}_smoothing_paired_episodes.csv",
                index=False,
            )
            boot_frames.append(summarize_bootstrap(name, pair_df, args))
        boot_summary = pd.concat(boot_frames, ignore_index=True)
        boot_summary.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_smoothing_bootstrap_summary.csv",
            index=False,
        )

        report_path = output_dir / f"{args.model_name}_{checkpoint}_smoothing_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Smoothed-policy diagnostic report\n")
            f.write("=================================\n\n")
            f.write(f"Model: {args.model_name}\n")
            f.write(f"Checkpoint: {checkpoint}\n")
            f.write(f"Original selected test CSV: {selected_csv}\n")
            f.write(f"Distillation pairs: {distill_path}\n")
            f.write(f"Max smoothing rows: {args.max_smoothing_rows}\n")
            f.write(f"KNN neighbors: {args.smoothing_neighbors}\n")
            f.write(f"KNN bandwidth scale: {args.smoothing_bandwidth_scale}\n\n")
            f.write("Smoothers\n---------\n")
            f.write(smooth_manifest.to_string(index=False))
            f.write("\n\nDistillation fidelity\n---------------------\n")
            f.write(distill_fid.to_string(index=False))
            f.write("\n\n2023 fidelity\n-------------\n")
            f.write(test_fid.to_string(index=False))
            f.write("\n\n2023 bootstrap\n--------------\n")
            f.write(boot_summary.to_string(index=False))
            f.write("\n")
        print(f"[smooth-done] Wrote smoothing outputs for {args.model_name}: {output_dir}")

    finally:
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir


def write_report(
    path,
    args,
    checkpoint,
    selected_csv,
    coverage_meta,
    coverage_diag,
    candidate_manifest,
    distill_fid,
    test_fid,
    boot_summary,
):
    """Write a compact human-readable report with the key diagnostics."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("Leakage-safe empirical symbolic distillation report\n")
        f.write("==================================================\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Original selected test CSV: {selected_csv}\n")
        f.write(f"Distillation years: {args.train_start_year}-{args.train_end_year}\n")
        f.write(f"Validation year not used for distillation: {args.validation_year}\n")
        f.write(f"Test year used only after fitting: {args.target_year}\n")
        f.write(f"Canonical stock position: {args.canonical_position}\n")
        f.write(f"Formula candidates: {args.formula_candidates}\n")
        f.write(f"Rows per symbolic fit: {args.n_sr_samples}\n")
        f.write(f"Actor query device: {args.device}\n")
        f.write(f"Deterministic PySR search: {args.deterministic_search}\n")
        f.write(f"Bootstrap draws: {args.n_bootstrap}\n\n")

        f.write("Distillation sample metadata\n")
        f.write("----------------------------\n")
        for key, value in coverage_meta.items():
            f.write(f"{key}: {value}\n")
        f.write("\nCoverage diagnostics\n")
        f.write("--------------------\n")
        f.write(coverage_diag.to_string(index=False))
        f.write("\n\n")

        f.write("Formula candidates\n")
        f.write("------------------\n")
        f.write(
            candidate_manifest[
                [
                    "candidate",
                    "sample_kind",
                    "target_kind",
                    "loss_kind",
                    "target_source",
                    "target_smoothing_bandwidth_scale",
                    "description",
                    "delta_formula",
                ]
            ].to_string(index=False)
        )
        f.write("\n\n")

        f.write("In-sample distillation fidelity, training/probe states only\n")
        f.write("----------------------------------------------------------\n")
        f.write(distill_fid.to_string(index=False))
        f.write("\n\n")

        f.write("Out-of-sample target-delta fidelity, 2023 states only\n")
        f.write("----------------------------------------------------\n")
        f.write(test_fid.to_string(index=False))
        f.write("\n\n")

        f.write("2023 bootstrap comparisons\n")
        f.write("--------------------------\n")
        f.write(boot_summary.to_string(index=False))
        f.write("\n")


def add_walkforward_metadata(df, args, checkpoint, split=None):
    """Attach run identifiers to a saved diagnostic table."""
    out = df.copy()
    out["model_name"] = args.model_name
    out["checkpoint"] = str(checkpoint)
    out["train_start_year"] = int(args.train_start_year)
    out["train_end_year"] = int(args.train_end_year)
    out["validation_year"] = int(args.validation_year)
    out["test_year"] = int(args.target_year)
    if split is not None:
        out["split"] = split
        out["split_year"] = int(args.validation_year if split == "validation" else args.target_year)
    return out


def selected_formula_specs(args):
    """Return selected FormulaSpec objects in the same order as FORMULA_SPECS."""
    selected = [s for s in FORMULA_SPECS if s.name in args.formula_candidates]
    if not selected:
        raise ValueError(f"No formula candidates selected: {args.formula_candidates}")
    return selected


def build_smooth_trade_entries_for_specs(distill_df, selected_specs, feature_cols, args, output_dir, checkpoint):
    """
    Build/load the smoothed target policies needed by selected formula specs.

    These smoothers are trained only on the training/probe distillation pool.
    With the publication default there is exactly one smoother:
    smooth_kernel_bs_delta_residual at bandwidth scale 1.00.  Its tag is
    "bw100"; that is a filename label for scale 1.00, not a separate
    experimental bandwidth unless the CLI bandwidth changes.
    """
    smooth_trade_entries = {}
    distill_eval_df = distill_df
    if not selected_specs_need_smooth_target(selected_specs):
        return smooth_trade_entries, distill_eval_df

    for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
        smoother_entry, smooth_feature_cols, _ = load_or_build_one_smoother(
            distill_df,
            args,
            output_dir,
            checkpoint,
            target["smoother_name"],
            bandwidth_scale=target["bandwidth_scale"],
        )
        if smooth_feature_cols != feature_cols:
            raise ValueError("Smoother and symbolic formula feature columns diverged.")
        smooth_trade_entries[target["policy_name"]] = smoother_entry
        distill_eval_df = add_smoothed_agent_columns(
            distill_eval_df,
            smoother_entry,
            feature_cols,
            column_prefix=target["column_prefix"],
        )
    return smooth_trade_entries, distill_eval_df


def formula_fidelity_frames(models, eval_df, feature_cols, selected_specs, args, label):
    """
    Formula-vs-agent and formula-vs-smoothed-agent fidelity tables.

    IMPORTANT DISTINCTION:
        `agent_delta` is the canonical three-feature policy surface: the actor
        queried at the canonical stock-position input, usually BS delta.

        `actual_agent_delta` is the action the raw neural agent actually takes
        while trading that path, with the path's evolving stock-position input.

    The symbolic formula intentionally ignores stock position, so both
    comparisons are informative and neither should be silently substituted for
    the other.
    """
    frames = [
        fidelity_table(
            models,
            eval_df,
            feature_cols,
            f"{label}_vs_canonical_agent",
            target_col="agent_delta",
        )
    ]
    if "actual_agent_delta" in eval_df.columns and eval_df["actual_agent_delta"].notna().any():
        frames.append(
            fidelity_table(
                models,
                eval_df,
                feature_cols,
                f"{label}_vs_actual_traded_agent",
                target_col="actual_agent_delta",
            )
        )
    for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
        target_col = f"{target['column_prefix']}_delta"
        if target_col in eval_df.columns:
            frames.append(
                fidelity_table(
                    models,
                    eval_df,
                    feature_cols,
                    f"{label}_vs_{target['policy_name']}",
                    target_col=target_col,
                )
            )
    return pd.concat(frames, ignore_index=True)


def hof_fidelity_table(models, eval_df, args, label):
    """Evaluate every Hall-of-Fame equation against every available policy target."""
    targets = [("canonical_agent", "agent_delta")]
    if "actual_agent_delta" in eval_df.columns and eval_df["actual_agent_delta"].notna().any():
        targets.append(("actual_traded_agent", "actual_agent_delta"))
    smooth_cols = [c for c in eval_df.columns if c.startswith("smooth_agent_") and c.endswith("_delta")]
    for col in smooth_cols:
        targets.append((col.replace("_delta", ""), col))

    rows = []
    for hof in iter_hof_equations(models, max_equations=args.hof_max_equations_per_candidate):
        try:
            delta, raw_output = hof_predict_delta(hof["equation"], eval_df, hof["spec"])
        except Exception as exc:
            rows.append(
                {
                    "dataset": label,
                    "policy": hof["policy"],
                    "candidate": hof["candidate"],
                    "hof_index": hof["hof_index"],
                    "complexity": hof["complexity"],
                    "loss": hof["loss"],
                    "score": hof["score"],
                    "equation": hof["equation"],
                    "target_name": "EVAL_ERROR",
                    "target_delta_col": "",
                    "n": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "p95_abs_error": np.nan,
                    "max_abs_error": np.nan,
                    "bias": np.nan,
                    "corr": np.nan,
                    "mean_target_delta": np.nan,
                    "mean_formula_delta": np.nan,
                    "error": repr(exc),
                }
            )
            continue

        for target_name, target_col in targets:
            pred_df = eval_df.dropna(subset=[target_col]).copy()
            idx = pred_df.index.to_numpy()
            target_delta = pred_df[target_col].to_numpy(dtype=float)
            pred_delta = delta[idx]
            mask = np.isfinite(pred_delta) & np.isfinite(target_delta)
            if mask.sum() == 0:
                err = np.asarray([], dtype=float)
                abs_err = np.asarray([], dtype=float)
            else:
                err = pred_delta[mask] - target_delta[mask]
                abs_err = np.abs(err)
            rows.append(
                {
                    "dataset": label,
                    "policy": hof["policy"],
                    "candidate": hof["candidate"],
                    "hof_index": hof["hof_index"],
                    "complexity": hof["complexity"],
                    "loss": hof["loss"],
                    "score": hof["score"],
                    "equation": hof["equation"],
                    "target_name": target_name,
                    "target_delta_col": target_col,
                    "n": int(mask.sum()),
                    "mae": float(abs_err.mean()) if len(abs_err) else np.nan,
                    "rmse": float(np.sqrt(np.mean(err ** 2))) if len(err) else np.nan,
                    "p95_abs_error": float(np.quantile(abs_err, 0.95)) if len(abs_err) else np.nan,
                    "max_abs_error": float(abs_err.max()) if len(abs_err) else np.nan,
                    "bias": float(err.mean()) if len(err) else np.nan,
                    "corr": float(np.corrcoef(target_delta[mask], pred_delta[mask])[0, 1])
                    if mask.sum() > 2 and np.std(target_delta[mask]) > 0 and np.std(pred_delta[mask]) > 0
                    else np.nan,
                    "mean_target_delta": float(target_delta[mask].mean()) if mask.sum() else np.nan,
                    "mean_formula_delta": float(pred_delta[mask].mean()) if mask.sum() else np.nan,
                    "error": "",
                }
            )
    return pd.DataFrame(rows)


def reward_from_pnl(pnl, args):
    """Package reward function used for reconstructed Hall-of-Fame trades."""
    pnl_scaled = np.asarray(pnl, dtype=float) * 100.0
    reward = 0.03 + pnl_scaled - float(args.kappa) * (
        np.abs(pnl_scaled) ** float(args.reward_exponent)
    )
    return reward * 10.0


def reconstruct_hof_trade_steps(base_trade_steps, models, args):
    """
    Reconstruct Hall-of-Fame formula trades on already-saved split paths.

    This is exact for the current zero-transaction-cost experiments and remains
    correct for nonzero linear transaction costs because every HOF formula is a
    three-feature policy independent of the previous stock position; we can
    therefore recompute its position path episode by episode.
    """
    basis = base_trade_steps[base_trade_steps["policy"] == "agent"].copy()
    frames = []
    for hof in iter_hof_equations(models, max_equations=args.hof_max_equations_per_candidate):
        try:
            delta, _ = hof_predict_delta(hof["equation"], basis, hof["spec"])
        except Exception as exc:
            print(f"[hof] skipping {hof['policy']} because equation evaluation failed: {exc}")
            continue
        if not np.isfinite(delta).all():
            print(f"[hof] skipping {hof['policy']} because it produced non-finite deltas")
            continue

        out = basis.copy()
        out["policy"] = hof["policy"]
        out["hof_candidate"] = hof["candidate"]
        out["hof_index"] = hof["hof_index"]
        out["hof_complexity"] = hof["complexity"]
        out["hof_loss"] = hof["loss"]
        out["hof_score"] = hof["score"]
        out["hof_equation"] = hof["equation"]
        out["chosen_delta"] = delta
        out["formula_delta"] = delta
        out["formula_minus_agent_target"] = delta - out["agent_target_delta"].to_numpy(dtype=float)

        pnl_parts = []
        reward_parts = []
        pos_parts = []
        tc_parts = []
        for _, g in out.groupby("episode", sort=False):
            prev_stock_owned = 0.0
            ep_pnl = []
            ep_reward = []
            ep_pos = []
            ep_tc = []
            for row_idx, row in g.iterrows():
                d = float(out.loc[row_idx, "chosen_delta"])
                stock_owned = -d
                s_start = float(row["S-1"])
                s_end = float(row["S0"])
                p_start = float(row["P-1"])
                p_end = float(row["P0"])
                t_cost = -abs(stock_owned - prev_stock_owned) * s_start * float(args.transaction_cost)
                pnl = -d * (s_end - s_start) + (p_end - p_start) + t_cost
                ep_pnl.append(pnl)
                ep_reward.append(float(reward_from_pnl(pnl, args)))
                ep_pos.append(stock_owned)
                ep_tc.append(t_cost)
                prev_stock_owned = stock_owned
            pnl_parts.extend(ep_pnl)
            reward_parts.extend(ep_reward)
            pos_parts.extend(ep_pos)
            tc_parts.extend(ep_tc)

        out["A PnL"] = pnl_parts
        out["A Reward"] = reward_parts
        out["A Pos"] = pos_parts
        out["A TC"] = tc_parts
        out["A PnL - TC"] = out["A PnL"] - out["A TC"]
        frames.append(out)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def hof_point_comparison_table(combined_metrics, smooth_trade_entries):
    """Point estimates for every HOF formula against BS, agent, and smoothed targets."""
    hof_policies = sorted(p for p in combined_metrics["policy"].unique() if str(p).startswith("hof__"))
    comparisons = []
    for policy in hof_policies:
        comparisons.append((f"{policy}_vs_bs", policy, "bs"))
        comparisons.append((f"{policy}_vs_agent", policy, "agent"))
        for smooth_name in smooth_trade_entries:
            comparisons.append((f"{policy}_vs_{smooth_name}", policy, smooth_name))

    rows = []
    for name, left, right in comparisons:
        pair_df = paired_episode_table(combined_metrics, left, right)
        if pair_df.empty:
            continue
        points = point_metrics(pair_df)
        row = {
            "comparison": name,
            "left_policy": left,
            "right_policy": right,
            "n_episodes": points["n_episodes"],
            "n_clusters": points["n_clusters"],
        }
        for metric in METRICS:
            row[metric] = points[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_comparison_specs(models, smooth_trade_entries, args):
    """
    All pairwise comparisons needed for publication diagnostics.

    We compare every formula with BS, the raw neural agent, and every traded
    smoothed target, including comparisons beyond the formula's own training
    target.
    """
    comparisons = [("agent_vs_bs", "agent", "bs")]
    for smooth_name in smooth_trade_entries:
        comparisons.append((f"{smooth_name}_vs_agent", smooth_name, "agent"))
        comparisons.append((f"{smooth_name}_vs_bs", smooth_name, "bs"))
    for candidate_name in models:
        for smooth_name in smooth_trade_entries:
            comparisons.append((f"{candidate_name}_vs_{smooth_name}", candidate_name, smooth_name))
        comparisons.append((f"{candidate_name}_vs_agent", candidate_name, "agent"))
        comparisons.append((f"{candidate_name}_vs_bs", candidate_name, "bs"))
    if args.include_pairwise_formula_bootstrap:
        names = list(models.keys())
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                comparisons.append((f"{left}_vs_{right}", left, right))

    # Preserve order while protecting against accidental duplicate names.
    seen = set()
    unique = []
    for name, left, right in comparisons:
        if name in seen:
            continue
        seen.add(name)
        unique.append((name, left, right))
    return unique


def evaluate_symbolic_split(
    env,
    actor,
    scaler,
    device,
    models,
    feature_cols,
    selected_specs,
    smooth_trade_entries,
    args,
    output_dir,
    checkpoint,
    split,
):
    """
    Save all formula/smoother/raw-agent/BS diagnostics for one split.

    This function is called immediately after a year's symbolic regressions
    are complete, so a 9-hour walk-forward run leaves inspectable artifacts
    year by year instead of waiting until the final year finishes.
    """
    split_year = args.validation_year if split == "validation" else args.target_year
    stem = f"{args.model_name}_{checkpoint}_{split}"

    print(f"[{split}] collecting {split_year} target states for fidelity")
    state_df = collect_split_target_states(
        env, actor, scaler, device, args, split=split, split_year=split_year
    )
    state_path = output_dir / f"{stem}_target_states.csv"
    state_df.to_csv(state_path, index=False)

    eval_df = state_df
    for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
        smoother_entry = smooth_trade_entries[target["policy_name"]]
        eval_df = add_smoothed_agent_columns(
            eval_df,
            smoother_entry,
            feature_cols,
            column_prefix=target["column_prefix"],
        )
    eval_state_path = output_dir / f"{stem}_target_states_with_smooth.csv"
    eval_df.to_csv(eval_state_path, index=False)

    formula_fid = formula_fidelity_frames(
        models, eval_df, feature_cols, selected_specs, args, f"{split}_{split_year}"
    )
    formula_fid = add_walkforward_metadata(formula_fid, args, checkpoint, split=split)
    formula_fid.to_csv(output_dir / f"{stem}_fidelity.csv", index=False)

    smoother_fid = smoother_fidelity_table(
        smooth_trade_entries, eval_df, feature_cols, f"{split}_{split_year}_smooth_vs_raw_agent"
    )
    smoother_fid = add_walkforward_metadata(smoother_fid, args, checkpoint, split=split)
    smoother_fid.to_csv(output_dir / f"{stem}_smoother_fidelity.csv", index=False)

    print(f"[{split}] trading raw agent, smoothed agent, formulas, and BS on {split_year}")
    trade_steps = trade_split_year(
        env,
        actor,
        scaler,
        device,
        models,
        feature_cols,
        args,
        smoothing_entries=smooth_trade_entries,
        split=split,
        split_year=split_year,
    )
    trade_steps_path = output_dir / f"{stem}_trade_steps.csv"
    trade_steps.to_csv(trade_steps_path, index=False)

    rate_cache = build_rate_cache(args.cleaned_data_dir)
    episode_metrics = episode_metrics_from_steps(trade_steps, rate_cache)
    episode_metrics = add_walkforward_metadata(episode_metrics, args, checkpoint, split=split)
    episode_metrics.to_csv(output_dir / f"{stem}_episode_metrics.csv", index=False)

    print(f"[{split}] running two-stage bootstrap comparisons")
    boot_frames = []
    for name, left, right in bootstrap_comparison_specs(models, smooth_trade_entries, args):
        pair_df = paired_episode_table(episode_metrics, left, right)
        pair_df = add_walkforward_metadata(pair_df, args, checkpoint, split=split)
        pair_df.to_csv(output_dir / f"{stem}_{name}_paired_episodes.csv", index=False)
        summary = summarize_bootstrap(name, pair_df, args)
        summary = add_walkforward_metadata(summary, args, checkpoint, split=split)
        boot_frames.append(summary)

    boot_summary = pd.concat(boot_frames, ignore_index=True)
    boot_summary.to_csv(output_dir / f"{stem}_bootstrap_summary.csv", index=False)

    hof_outputs = {}
    if args.evaluate_hof_formulas and int(args.target_year) >= int(args.hof_start_year):
        print(f"[{split}] evaluating Hall-of-Fame formulas on {split_year}")
        hof_fid = hof_fidelity_table(models, eval_df, args, f"{split}_{split_year}")
        hof_fid = add_walkforward_metadata(hof_fid, args, checkpoint, split=split)
        hof_fid.to_csv(output_dir / f"{stem}_hof_fidelity.csv", index=False)

        hof_trade_steps = reconstruct_hof_trade_steps(trade_steps, models, args)
        if not hof_trade_steps.empty:
            hof_trade_steps = add_walkforward_metadata(
                hof_trade_steps, args, checkpoint, split=split
            )
            hof_trade_steps.to_csv(output_dir / f"{stem}_hof_trade_steps.csv", index=False)

            hof_episode_metrics = episode_metrics_from_steps(hof_trade_steps, rate_cache)
            hof_episode_metrics = add_walkforward_metadata(
                hof_episode_metrics, args, checkpoint, split=split
            )
            hof_episode_metrics.to_csv(
                output_dir / f"{stem}_hof_episode_metrics.csv", index=False
            )

            combined_metrics = pd.concat(
                [episode_metrics, hof_episode_metrics], ignore_index=True
            )
            hof_points = hof_point_comparison_table(combined_metrics, smooth_trade_entries)
            hof_points = add_walkforward_metadata(hof_points, args, checkpoint, split=split)
            hof_points.to_csv(output_dir / f"{stem}_hof_point_comparisons.csv", index=False)
        else:
            hof_episode_metrics = pd.DataFrame()
            combined_metrics = episode_metrics
            hof_points = pd.DataFrame()

        hof_outputs = {
            "fidelity": hof_fid,
            "episode_metrics": hof_episode_metrics,
            "combined_metrics": combined_metrics,
            "point_comparisons": hof_points,
        }

    return {
        "state_path": state_path,
        "eval_state_path": eval_state_path,
        "trade_steps_path": trade_steps_path,
        "formula_fidelity": formula_fid,
        "smoother_fidelity": smoother_fid,
        "episode_metrics": episode_metrics,
        "bootstrap_summary": boot_summary,
        "hof": hof_outputs,
    }


def year_args_from_walkforward(args, year):
    """Create an argparse-like Namespace for one walk-forward test year."""
    out = copy.copy(args)
    out.model_name = f"{args.model_prefix}{year}"
    out.target_year = int(year)
    out.validation_year = int(year) - 1
    out.train_start_year = int(args.train_start_year)
    out.train_end_year = int(year) - 2
    out.target_data_dir = str(args.walkforward_data_dir_template).format(year=year)
    out.output_dir = str(Path(args.output_dir) / f"{year}_{out.model_name}")
    return out


def walkforward_year_dir_completed(path):
    """
    True only when a year folder is safe to treat as completed.

    This helper is shared by aggregate refresh and resume/skip logic.  A stale
    report is not enough if a newer in-progress marker exists from a rerun.
    """
    path = Path(path)
    progress_markers = list(path.glob("*_walkforward_in_progress.json"))
    report_markers = list(path.glob("*_walkforward_report.txt"))
    complete_markers = list(path.glob("*_walkforward_complete.json"))
    if progress_markers:
        if complete_markers:
            newest_progress = max(p.stat().st_mtime for p in progress_markers)
            newest_complete = max(p.stat().st_mtime for p in complete_markers)
            return newest_complete > newest_progress
        return False
    return bool(report_markers or complete_markers)


def refresh_walkforward_aggregates(root_dir):
    """
    Rebuild aggregate CSVs from per-year outputs.

    The aggregate files are intentionally derived artifacts.  If the long run
    crashes, rerunning the script regenerates them from whatever yearly folders
    already completed.
    """
    root = Path(root_dir)

    aggregate_specs = [
        ("*_candidate_manifest.csv", "walkforward_candidate_manifest_all.csv"),
        ("*_distillation_fidelity.csv", "walkforward_distillation_fidelity_all.csv"),
        ("*_distillation_smoother_fidelity.csv", "walkforward_distillation_smoother_fidelity_all.csv"),
        ("*_validation_fidelity.csv", "walkforward_validation_fidelity_all.csv"),
        ("*_test_fidelity.csv", "walkforward_test_fidelity_all.csv"),
        ("*_validation_smoother_fidelity.csv", "walkforward_validation_smoother_fidelity_all.csv"),
        ("*_test_smoother_fidelity.csv", "walkforward_test_smoother_fidelity_all.csv"),
        ("*_validation_bootstrap_summary.csv", "walkforward_validation_bootstrap_summary_all.csv"),
        ("*_test_bootstrap_summary.csv", "walkforward_test_bootstrap_summary_all.csv"),
        ("*_validation_episode_metrics.csv", "walkforward_validation_episode_metrics_all.csv"),
        ("*_test_episode_metrics.csv", "walkforward_test_episode_metrics_all.csv"),
        ("*_validation_hof_fidelity.csv", "walkforward_validation_hof_fidelity_all.csv"),
        ("*_test_hof_fidelity.csv", "walkforward_test_hof_fidelity_all.csv"),
        ("*_validation_hof_point_comparisons.csv", "walkforward_validation_hof_point_comparisons_all.csv"),
        ("*_test_hof_point_comparisons.csv", "walkforward_test_hof_point_comparisons_all.csv"),
        ("*_distillation_hof_fidelity.csv", "walkforward_distillation_hof_fidelity_all.csv"),
        ("*_hof_validation_selection.csv", "walkforward_hof_validation_selection_all.csv"),
        ("*_validation_hof_selected_bootstrap_summary.csv", "walkforward_validation_hof_selected_bootstrap_all.csv"),
        ("*_test_hof_selected_bootstrap_summary.csv", "walkforward_test_hof_selected_bootstrap_all.csv"),
        ("*_validation_hof_all_bootstrap_summary.csv", "walkforward_validation_hof_all_bootstrap_all.csv"),
        ("*_test_hof_all_bootstrap_summary.csv", "walkforward_test_hof_all_bootstrap_all.csv"),
    ]
    for pattern, out_name in aggregate_specs:
        frames = []
        for path in sorted(root.glob(f"*/*{pattern}")):
            if not walkforward_year_dir_completed(path.parent):
                continue
            try:
                df = pd.read_csv(path)
            except pd.errors.EmptyDataError:
                continue
            df["source_file"] = str(path)
            frames.append(df)
        if frames:
            out_path = root / out_name
            tmp_path = root / f".{out_name}.{os.getpid()}.tmp"
            pd.concat(frames, ignore_index=True).to_csv(tmp_path, index=False)
            try:
                os.replace(tmp_path, out_path)
            except OSError as exc:
                print(
                    f"[aggregate] Warning: could not refresh {out_path}. "
                    f"Is it open in another program? Keeping per-year outputs intact. {exc}"
                )
        else:
            # Keep aggregate files derived from currently completed year folders.
            # If a rerun marks every relevant folder as in-progress, stale
            # aggregate CSVs should not masquerade as current results.
            stale_path = root / out_name
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except OSError as exc:
                    print(
                        f"[aggregate] Warning: could not remove stale aggregate {stale_path}. "
                        f"Is it open in another program? {exc}"
                    )


def write_walkforward_year_report(
    path,
    args,
    checkpoint,
    selected_csv,
    coverage_meta,
    coverage_diag,
    candidate_manifest,
    distill_fid,
    validation_outputs,
    test_outputs,
):
    """Write a readable per-year report as soon as that year completes."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("Publication walk-forward symbolic distillation report\n")
        f.write("====================================================\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Checkpoint: {checkpoint}\n")
        f.write(f"Original selected test CSV: {selected_csv}\n")
        f.write(f"Training years: {args.train_start_year}-{args.train_end_year}\n")
        f.write(f"Validation year: {args.validation_year}\n")
        f.write(f"Test year: {args.target_year}\n")
        f.write(f"Formula candidates: {args.formula_candidates}\n")
        f.write(f"Rows per symbolic fit: {args.n_sr_samples}\n")
        f.write(f"PySR iterations: {args.niterations}\n")
        f.write(f"Actor query device: {args.device}\n")
        f.write(f"Deterministic PySR search: {args.deterministic_search}\n")
        f.write(f"Smoothing bandwidth scale: {args.smoothing_bandwidth_scale}\n\n")

        f.write("Distillation sample metadata\n")
        f.write("----------------------------\n")
        for key, value in coverage_meta.items():
            f.write(f"{key}: {value}\n")
        f.write("\nCoverage diagnostics\n")
        f.write("--------------------\n")
        f.write(coverage_diag.to_string(index=False))
        f.write("\n\nFormula candidates\n------------------\n")
        f.write(candidate_manifest.to_string(index=False))
        f.write("\n\nDistillation fidelity\n---------------------\n")
        f.write(distill_fid.to_string(index=False))

        for split, outputs in [("validation", validation_outputs), ("test", test_outputs)]:
            f.write(f"\n\n{split.title()} formula fidelity\n")
            f.write("---------------------------\n")
            f.write(outputs["formula_fidelity"].to_string(index=False))
            f.write(f"\n\n{split.title()} smoother-vs-raw fidelity\n")
            f.write("--------------------------------\n")
            f.write(outputs["smoother_fidelity"].to_string(index=False))
            f.write(f"\n\n{split.title()} bootstrap\n")
            f.write("--------------------\n")
            f.write(outputs["bootstrap_summary"].to_string(index=False))
        f.write("\n")


def select_hof_policies_from_validation(validation_hof, args):
    """
    Pre-select HOF equations using validation-only information.

    This deliberately does not look at the test year.  By default we rank by
    MAE to the actual traded agent when available.  That is the cleanest
    distillation criterion; performance metrics can be analyzed afterwards.
    """
    if not validation_hof or "fidelity" not in validation_hof:
        return pd.DataFrame()
    fid = validation_hof["fidelity"].copy()
    if fid.empty:
        return pd.DataFrame()

    target_preference = {
        "actual": "actual_traded_agent",
        "canonical": "canonical_agent",
        "smooth": "smooth_agent_bw100",
    }
    requested_target = target_preference.get(args.hof_selection_target, args.hof_selection_target)
    sub = fid[(fid["target_name"] == requested_target) & fid["mae"].notna()].copy()
    if sub.empty and requested_target == "actual_traded_agent":
        # General probes have no actual traded action.  Validation/test splits
        # should have it, but fall back explicitly rather than failing late.
        sub = fid[(fid["target_name"] == "canonical_agent") & fid["mae"].notna()].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(["mae", "complexity", "loss"], ascending=[True, True, True])
    if not args.hof_bootstrap_all and args.hof_bootstrap_top_k > 0:
        sub = sub.head(int(args.hof_bootstrap_top_k)).copy()
    sub["selection_rank"] = np.arange(1, len(sub) + 1)
    sub["selection_rule"] = (
        f"rank by validation MAE to {requested_target}; "
        f"top_k={args.hof_bootstrap_top_k}; bootstrap_all={args.hof_bootstrap_all}"
    )
    return sub


def bootstrap_selected_hof_policies(split_outputs, selected_hof, args, output_dir, checkpoint, split):
    """Bootstrap validation-selected HOF policies on validation or test metrics."""
    if selected_hof.empty:
        return pd.DataFrame()
    hof_outputs = split_outputs.get("hof", {})
    combined_metrics = hof_outputs.get("combined_metrics", pd.DataFrame())
    if combined_metrics.empty:
        return pd.DataFrame()

    split_year = args.validation_year if split == "validation" else args.target_year
    stem = f"{args.model_name}_{checkpoint}_{split}"
    comparisons = []
    for _, selected_row in selected_hof.drop_duplicates(subset=["policy"]).iterrows():
        policy = selected_row["policy"]
        selection_rank = selected_row.get("selection_rank", 0)
        comparisons.append((f"{policy}_vs_bs", policy, "bs"))
        comparisons.append((f"{policy}_vs_agent", policy, "agent"))
        for smooth_name in [
            p for p in combined_metrics["policy"].dropna().unique() if str(p).startswith("smooth_target_")
        ]:
            comparisons.append((f"{policy}_vs_{smooth_name}", policy, smooth_name))

    boot_frames = []
    for name, left, right in comparisons:
        pair_df = paired_episode_table(combined_metrics, left, right)
        if pair_df.empty:
            continue
        pair_df = add_walkforward_metadata(pair_df, args, checkpoint, split=split)
        rank_values = selected_hof.loc[selected_hof["policy"] == left, "selection_rank"]
        selection_rank = int(rank_values.iloc[0]) if len(rank_values) else 0
        file_slug = short_hof_selected_slug(name, left, right, selection_rank)
        pair_df["comparison"] = name
        pair_df["left_policy"] = left
        pair_df["right_policy"] = right
        pair_df["hof_selected_file_slug"] = file_slug
        atomic_to_csv(
            pair_df,
            output_dir / f"{stem}_{file_slug}_paired.csv",
            index=False,
        )
        summary = summarize_bootstrap(name, pair_df, args)
        summary = add_walkforward_metadata(summary, args, checkpoint, split=split)
        summary["selected_by_validation"] = 1
        summary["hof_selected_file_slug"] = file_slug
        summary["left_policy"] = left
        summary["right_policy"] = right
        boot_frames.append(summary)

    if not boot_frames:
        return pd.DataFrame()
    boot_summary = pd.concat(boot_frames, ignore_index=True)
    atomic_to_csv(
        boot_summary,
        output_dir / f"{stem}_hof_selected_bootstrap_summary.csv",
        index=False,
    )
    print(
        f"[hof] bootstrapped {len(selected_hof)} validation-selected HOF formula(s) "
        f"on {split} {split_year}"
    )
    return boot_summary


def bootstrap_all_hof_policies(split_outputs, args, output_dir, checkpoint, split):
    """
    Bootstrap every Hall-of-Fame equation on a completed split.

    This is intentionally separate from bootstrap_selected_hof_policies.  The
    normal walk-forward run bootstraps only the validation-selected top-K HOF
    equations to keep a long PySR run from exploding in wall-clock time.  The
    publication audit run can afford to be exhaustive because the expensive
    formulas have already been fitted and saved.

    We do not write one paired-episode CSV per HOF comparison by default: the
    combined HOF episode-metrics table is already saved and is sufficient to
    reproduce every pair.  Avoiding hundreds of redundant pair files matters on
    Windows and cloud-synced disks, where free space has already been tight.
    """
    hof_outputs = split_outputs.get("hof", {})
    combined_metrics = hof_outputs.get("combined_metrics", pd.DataFrame())
    if combined_metrics.empty:
        return pd.DataFrame()

    split_year = args.validation_year if split == "validation" else args.target_year
    stem = f"{args.model_name}_{checkpoint}_{split}"
    hof_policies = sorted(
        p for p in combined_metrics["policy"].dropna().unique() if str(p).startswith("hof__")
    )
    if not hof_policies:
        return pd.DataFrame()

    smooth_names = [
        p
        for p in combined_metrics["policy"].dropna().unique()
        if str(p).startswith("smooth_target_")
    ]
    comparisons = []
    for policy in hof_policies:
        comparisons.append((f"{policy}_vs_bs", policy, "bs"))
        comparisons.append((f"{policy}_vs_agent", policy, "agent"))
        for smooth_name in smooth_names:
            comparisons.append((f"{policy}_vs_{smooth_name}", policy, smooth_name))

    boot_frames = []
    total = len(comparisons)
    start_time = time.time()
    for i, (name, left, right) in enumerate(comparisons, start=1):
        if i == 1 or i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            if i > 1:
                seconds_per = elapsed / max(i - 1, 1)
                remaining = seconds_per * (total - i + 1)
                eta_text = f", elapsed={elapsed/60.0:.1f}m, eta={remaining/60.0:.1f}m"
            else:
                eta_text = ""
            print(
                f"[repair-hof] {split} {split_year}: bootstrapping HOF "
                f"comparison {i}/{total}{eta_text}"
            )
        pair_df = paired_episode_table(combined_metrics, left, right)
        if pair_df.empty:
            continue
        summary = summarize_bootstrap(name, pair_df, args)
        summary = add_walkforward_metadata(summary, args, checkpoint, split=split)
        summary["left_policy"] = left
        summary["right_policy"] = right
        summary["hof_bootstrap_scope"] = "all_hof"
        boot_frames.append(summary)

    boot_summary = pd.concat(boot_frames, ignore_index=True) if boot_frames else pd.DataFrame()
    atomic_to_csv(
        boot_summary,
        output_dir / f"{stem}_hof_all_bootstrap_summary.csv",
        index=False,
    )
    print(
        f"[repair-hof] bootstrapped {len(hof_policies)} HOF formula(s) "
        f"on {split} {split_year}"
    )
    return boot_summary


def bootstrap_summary_only_one_walkforward_year(year_args, refresh_aggregates=True):
    """
    Upgrade bootstrap summaries from cached episode metrics only.

    This mode is intentionally narrow: it does not reload the actor, refit
    symbolic formulas, rebuild fidelity tables, or rewrite trade-step files.
    It only overwrites *_bootstrap_summary.csv files so new confidence-level
    fields can be added safely after the expensive walk-forward run.
    """
    set_random_seeds(year_args.seed)
    root_output_dir = Path(year_args.root_output_dir)
    output_dir = Path(year_args.output_dir)
    checkpoint, _ = infer_checkpoint(year_args.model_name, year_args.results_testing_dir)

    print("\n" + "=" * 80)
    print(
        f"[bootstrap-only] {year_args.target_year}: model={year_args.model_name}, "
        f"checkpoint={checkpoint}, n_bootstrap={year_args.n_bootstrap}"
    )

    for split in ["validation", "test"]:
        split_year = year_args.validation_year if split == "validation" else year_args.target_year
        stem = f"{year_args.model_name}_{checkpoint}_{split}"
        episode_path = output_dir / f"{stem}_episode_metrics.csv"
        episode_metrics = load_cached_csv(episode_path, f"{split} episode metrics")

        policies = sorted(str(p) for p in episode_metrics["policy"].dropna().unique())
        smooth_names = [p for p in policies if p.startswith("smooth_target_")]
        formula_names = [
            p
            for p in policies
            if p not in {"agent", "bs"} and not p.startswith("smooth_target_")
        ]

        comparisons = [("agent_vs_bs", "agent", "bs")]
        for smooth_name in smooth_names:
            comparisons.append((f"{smooth_name}_vs_agent", smooth_name, "agent"))
            comparisons.append((f"{smooth_name}_vs_bs", smooth_name, "bs"))
        for formula_name in formula_names:
            for smooth_name in smooth_names:
                comparisons.append((f"{formula_name}_vs_{smooth_name}", formula_name, smooth_name))
            comparisons.append((f"{formula_name}_vs_agent", formula_name, "agent"))
            comparisons.append((f"{formula_name}_vs_bs", formula_name, "bs"))
        if year_args.include_pairwise_formula_bootstrap:
            for i, left in enumerate(formula_names):
                for right in formula_names[i + 1 :]:
                    comparisons.append((f"{left}_vs_{right}", left, right))

        print(
            f"[bootstrap-only] {split} {split_year}: selected-policy "
            f"comparisons={len(comparisons)}"
        )
        boot_frames = []
        for name, left, right in comparisons:
            pair_df = paired_episode_table(episode_metrics, left, right)
            if pair_df.empty:
                continue
            summary = summarize_bootstrap(name, pair_df, year_args)
            summary = add_walkforward_metadata(summary, year_args, checkpoint, split=split)
            boot_frames.append(summary)
        boot_summary = pd.concat(boot_frames, ignore_index=True) if boot_frames else pd.DataFrame()
        atomic_to_csv(boot_summary, output_dir / f"{stem}_bootstrap_summary.csv", index=False)

        hof_episode_path = output_dir / f"{stem}_hof_episode_metrics.csv"
        if not hof_episode_path.exists():
            print(f"[bootstrap-only] {split} {split_year}: no HOF episode metrics found; skipping HOF")
            continue

        hof_episode_metrics = load_cached_csv(hof_episode_path, f"{split} HOF episode metrics")
        combined_metrics = pd.concat([episode_metrics, hof_episode_metrics], ignore_index=True)
        combined_policies = sorted(str(p) for p in combined_metrics["policy"].dropna().unique())
        hof_policies = [p for p in combined_policies if p.startswith("hof__")]
        combined_smooth_names = [p for p in combined_policies if p.startswith("smooth_target_")]

        if year_args.hof_bootstrap_all:
            hof_comparisons = []
            for policy in hof_policies:
                hof_comparisons.append((f"{policy}_vs_bs", policy, "bs"))
                hof_comparisons.append((f"{policy}_vs_agent", policy, "agent"))
                for smooth_name in combined_smooth_names:
                    hof_comparisons.append((f"{policy}_vs_{smooth_name}", policy, smooth_name))

            print(
                f"[bootstrap-only] {split} {split_year}: all-HOF "
                f"comparisons={len(hof_comparisons)}"
            )
            hof_frames = []
            start_time = time.time()
            total = len(hof_comparisons)
            for i, (name, left, right) in enumerate(hof_comparisons, start=1):
                if i == 1 or i % 10 == 0 or i == total:
                    elapsed = time.time() - start_time
                    eta_text = ""
                    if i > 1:
                        seconds_per = elapsed / max(i - 1, 1)
                        eta_text = f", eta={(seconds_per * (total - i + 1))/60.0:.1f}m"
                    print(
                        f"[bootstrap-only-hof] {split} {split_year}: "
                        f"{i}/{total}{eta_text}"
                    )
                pair_df = paired_episode_table(combined_metrics, left, right)
                if pair_df.empty:
                    continue
                summary = summarize_bootstrap(name, pair_df, year_args)
                summary = add_walkforward_metadata(summary, year_args, checkpoint, split=split)
                summary["left_policy"] = left
                summary["right_policy"] = right
                summary["hof_bootstrap_scope"] = "all_hof"
                hof_frames.append(summary)
            hof_summary = pd.concat(hof_frames, ignore_index=True) if hof_frames else pd.DataFrame()
            atomic_to_csv(
                hof_summary,
                output_dir / f"{stem}_hof_all_bootstrap_summary.csv",
                index=False,
            )

        selection_path = output_dir / f"{year_args.model_name}_{checkpoint}_hof_validation_selection.csv"
        if selection_path.exists():
            selected = load_cached_csv(selection_path, "HOF validation selection")
            selected_policies = (
                selected.drop_duplicates(subset=["policy"])
                if "policy" in selected.columns
                else pd.DataFrame()
            )
            selected_comparisons = []
            for _, selected_row in selected_policies.iterrows():
                policy = str(selected_row["policy"])
                selected_comparisons.append((f"{policy}_vs_bs", policy, "bs"))
                selected_comparisons.append((f"{policy}_vs_agent", policy, "agent"))
                for smooth_name in combined_smooth_names:
                    selected_comparisons.append((f"{policy}_vs_{smooth_name}", policy, smooth_name))

            print(
                f"[bootstrap-only] {split} {split_year}: validation-selected HOF "
                f"comparisons={len(selected_comparisons)}"
            )
            selected_frames = []
            for name, left, right in selected_comparisons:
                pair_df = paired_episode_table(combined_metrics, left, right)
                if pair_df.empty:
                    continue
                summary = summarize_bootstrap(name, pair_df, year_args)
                summary = add_walkforward_metadata(summary, year_args, checkpoint, split=split)
                rank_values = selected.loc[selected["policy"] == left, "selection_rank"]
                selection_rank = int(rank_values.iloc[0]) if len(rank_values) else 0
                summary["selected_by_validation"] = 1
                summary["hof_selected_file_slug"] = short_hof_selected_slug(
                    name, left, right, selection_rank
                )
                summary["left_policy"] = left
                summary["right_policy"] = right
                selected_frames.append(summary)
            selected_summary = (
                pd.concat(selected_frames, ignore_index=True)
                if selected_frames
                else pd.DataFrame()
            )
            atomic_to_csv(
                selected_summary,
                output_dir / f"{stem}_hof_selected_bootstrap_summary.csv",
                index=False,
            )

    if refresh_aggregates:
        refresh_walkforward_aggregates(root_output_dir)
    print(f"[bootstrap-only] completed {year_args.target_year}")
    return {
        "year": int(year_args.target_year),
        "model_name": year_args.model_name,
        "checkpoint": checkpoint,
        "output_dir": str(output_dir),
        "status": "bootstrap_only_completed",
    }


def expected_hof_policy_count(models, args):
    """Count HOF equations expected from the saved PySR equation tables."""
    return sum(1 for _ in iter_hof_equations(models, max_equations=args.hof_max_equations_per_candidate))


def hof_trade_steps_complete(path, base_trade_steps, models, args):
    """
    Check whether a cached HOF trade file covers every expected equation/path.

    This protects the audit run from trusting a partial artifact left by a
    crash.  If the check fails, the audit run rebuilds the HOF trades from the
    already-saved base policy paths instead of silently using incomplete data.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        cached = pd.read_csv(path, usecols=["policy"])
    except Exception:
        return False
    expected_policies = expected_hof_policy_count(models, args)
    base_rows = int((base_trade_steps["policy"] == "agent").sum())
    if expected_policies <= 0 or base_rows <= 0:
        return False
    return (
        int(cached["policy"].nunique()) == int(expected_policies)
        and int(len(cached)) == int(base_rows * expected_policies)
    )


def finalize_hof_selection_and_bootstrap(validation_outputs, test_outputs, args, output_dir, checkpoint):
    """
    Save validation HOF selection and bootstrap selected equations on both splits.

    All HOF equations get fidelity and point-trading diagnostics.  To keep the
    walk-forward run from exploding, bootstrap defaults to the validation top-K
    equations unless --hof-bootstrap-all is requested.
    """
    selected = select_hof_policies_from_validation(validation_outputs.get("hof", {}), args)
    if selected.empty:
        return {}
    selected = add_walkforward_metadata(selected, args, checkpoint, split="validation")
    selected.to_csv(
        output_dir / f"{args.model_name}_{checkpoint}_hof_validation_selection.csv",
        index=False,
    )
    val_boot = bootstrap_selected_hof_policies(
        validation_outputs, selected, args, output_dir, checkpoint, "validation"
    )
    test_boot = bootstrap_selected_hof_policies(
        test_outputs, selected, args, output_dir, checkpoint, "test"
    )
    return {
        "selection": selected,
        "validation_bootstrap": val_boot,
        "test_bootstrap": test_boot,
    }


def run_one_walkforward_year(year_args, refresh_aggregates=True):
    """
    Run one independent walk-forward year.

    This is the unit of parallelism.  It writes only the year's own output
    directory while it is working.  Root-level aggregate CSVs are refreshed only
    when `refresh_aggregates=True`; the parallel parent sets this to False for
    children and owns aggregate refreshes itself to avoid concurrent writes.
    """
    set_random_seeds(year_args.seed)
    root_output_dir = Path(year_args.root_output_dir)
    output_dir = Path(year_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    in_progress_marker = output_dir / f"{year_args.model_name}_walkforward_in_progress.json"
    failed_marker = output_dir / f"{year_args.model_name}_walkforward_failed.json"
    atomic_write_text(
        in_progress_marker,
        json.dumps(
            {
                "model_name": year_args.model_name,
                "target_year": int(year_args.target_year),
                "validation_year": int(year_args.validation_year),
                "train_start_year": int(year_args.train_start_year),
                "train_end_year": int(year_args.train_end_year),
                "status": "in_progress",
            },
            indent=2,
        ),
    )

    print("\n" + "=" * 80)
    print(
        f"[walk-forward] {year_args.target_year}: model={year_args.model_name}, "
        f"train={year_args.train_start_year}-{year_args.train_end_year}, "
        f"validation={year_args.validation_year}, test={year_args.target_year}"
    )

    checkpoint, selected_csv = infer_checkpoint(year_args.model_name, year_args.results_testing_dir)
    settings = load_settings_json(year_args.model_name)
    # HOF reconstructed trading uses the exact reward/transaction-cost
    # convention from env.py.  Store these on the per-year args so the
    # diagnostics remain self-contained and deterministic.
    year_args.kappa = float(settings.get("kappa", 1.0))
    year_args.reward_exponent = float(settings.get("reward_exponent", 1.0))
    year_args.transaction_cost = float(settings.get("transaction_cost", 0.0))
    data_dir = ensure_walkforward_data_dir(year_args)

    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(data_dir)
    try:
        from empirical_deep_hedging.include.env import Env

        actor, scaler, device = load_actor_and_scaler(
            year_args.model_name, checkpoint, settings, year_args.device
        )
        env = Env(settings)

        distill_df, distill_path = load_or_build_distillation_pool(
            env, actor, scaler, device, data_dir, year_args, output_dir, checkpoint
        )
        coverage_diag, coverage_meta = coverage_diagnostics(distill_df)
        coverage_diag = add_walkforward_metadata(coverage_diag, year_args, checkpoint)
        coverage_diag.to_csv(
            output_dir / f"{year_args.model_name}_{checkpoint}_distillation_coverage.csv",
            index=False,
        )

        print(f"[fit] fitting three publication formula families for {year_args.target_year}")
        write_walkforward_status(output_dir, year_args, checkpoint, "fit_symbolic_policies_start")
        models, feature_cols, fit_df, candidate_manifest = fit_symbolic_policies(
            distill_df, year_args, output_dir=output_dir, checkpoint=checkpoint
        )
        write_walkforward_status(output_dir, year_args, checkpoint, "fit_symbolic_policies_done")
        fit_df = add_walkforward_metadata(fit_df, year_args, checkpoint)
        fit_df.to_csv(
            output_dir / f"{year_args.model_name}_{checkpoint}_candidate_fit_rows.csv",
            index=False,
        )
        candidate_manifest = add_walkforward_metadata(
            candidate_manifest, year_args, checkpoint
        )
        candidate_manifest.to_csv(
            output_dir / f"{year_args.model_name}_{checkpoint}_candidate_manifest.csv",
            index=False,
        )

        selected_specs = selected_formula_specs(year_args)
        smooth_trade_entries, distill_eval_df = build_smooth_trade_entries_for_specs(
            distill_df, selected_specs, feature_cols, year_args, output_dir, checkpoint
        )
        distill_fid = formula_fidelity_frames(
            models,
            distill_eval_df,
            feature_cols,
            selected_specs,
            year_args,
            "distillation_train_only",
        )
        distill_fid = add_walkforward_metadata(distill_fid, year_args, checkpoint)
        distill_fid.to_csv(
            output_dir / f"{year_args.model_name}_{checkpoint}_distillation_fidelity.csv",
            index=False,
        )
        distill_smoother_fid = smoother_fidelity_table(
            smooth_trade_entries,
            distill_eval_df,
            feature_cols,
            "distillation_train_only_smooth_vs_raw_agent",
        )
        distill_smoother_fid = add_walkforward_metadata(
            distill_smoother_fid, year_args, checkpoint
        )
        distill_smoother_fid.to_csv(
            output_dir / f"{year_args.model_name}_{checkpoint}_distillation_smoother_fidelity.csv",
            index=False,
        )

        write_walkforward_status(output_dir, year_args, checkpoint, "validation_evaluation_start")
        validation_outputs = evaluate_symbolic_split(
            env,
            actor,
            scaler,
            device,
            models,
            feature_cols,
            selected_specs,
            smooth_trade_entries,
            year_args,
            output_dir,
            checkpoint,
            split="validation",
        )
        write_walkforward_status(output_dir, year_args, checkpoint, "validation_evaluation_done")
        write_walkforward_status(output_dir, year_args, checkpoint, "test_evaluation_start")
        test_outputs = evaluate_symbolic_split(
            env,
            actor,
            scaler,
            device,
            models,
            feature_cols,
            selected_specs,
            smooth_trade_entries,
            year_args,
            output_dir,
            checkpoint,
            split="test",
        )
        write_walkforward_status(output_dir, year_args, checkpoint, "test_evaluation_done")
        write_walkforward_status(output_dir, year_args, checkpoint, "hof_selection_bootstrap_start")
        finalize_hof_selection_and_bootstrap(
            validation_outputs, test_outputs, year_args, output_dir, checkpoint
        )
        write_walkforward_status(output_dir, year_args, checkpoint, "hof_selection_bootstrap_done")

        write_walkforward_year_report(
            output_dir / f"{year_args.model_name}_{checkpoint}_walkforward_report.txt",
            year_args,
            checkpoint,
            selected_csv,
            coverage_meta,
            coverage_diag,
            candidate_manifest,
            distill_fid,
            validation_outputs,
            test_outputs,
        )
        complete_marker = output_dir / f"{year_args.model_name}_{checkpoint}_walkforward_complete.json"
        atomic_write_text(
            complete_marker,
            json.dumps(
                {
                    "model_name": year_args.model_name,
                    "target_year": int(year_args.target_year),
                    "validation_year": int(year_args.validation_year),
                    "train_start_year": int(year_args.train_start_year),
                    "train_end_year": int(year_args.train_end_year),
                    "checkpoint": int(checkpoint),
                    "output_dir": str(output_dir),
                    "families": [str(name) for name in models.keys()],
                    "status": "completed",
                },
                indent=2,
            ),
        )
        if in_progress_marker.exists():
            try:
                in_progress_marker.unlink()
            except OSError as exc:
                print(
                    f"[walk-forward] Warning: completed {year_args.target_year}, "
                    f"but could not remove in-progress marker {in_progress_marker}: {exc}"
                )
        if failed_marker.exists():
            try:
                failed_marker.unlink()
            except OSError as exc:
                print(
                    f"[walk-forward] Warning: completed {year_args.target_year}, "
                    f"but could not remove stale failed marker {failed_marker}: {exc}"
                )
        if refresh_aggregates:
            refresh_walkforward_aggregates(root_output_dir)
        print(f"[walk-forward] completed {year_args.target_year}. Outputs are ready in {output_dir}")
        print(f"[walk-forward] distillation pairs: {distill_path}")
        return {
            "year": int(year_args.target_year),
            "model_name": year_args.model_name,
            "checkpoint": checkpoint,
            "output_dir": str(output_dir),
            "status": "completed",
        }

    finally:
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir


def run_one_walkforward_year_worker(year_args):
    """
    ProcessPool worker wrapper.

    Return a structured failure instead of hiding a child traceback inside a
    generic multiprocessing exception.  The parent will print the traceback and
    continue collecting any other years that finished successfully.
    """
    try:
        return run_one_walkforward_year(year_args, refresh_aggregates=False)
    except Exception as exc:
        try:
            output_dir = Path(getattr(year_args, "output_dir", "."))
            output_dir.mkdir(parents=True, exist_ok=True)
            failed_marker = output_dir / f"{getattr(year_args, 'model_name', 'unknown')}_walkforward_failed.json"
            atomic_write_text(
                failed_marker,
                json.dumps(
                    {
                        "model_name": getattr(year_args, "model_name", ""),
                        "target_year": int(getattr(year_args, "target_year", -1)),
                        "status": "failed",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                    indent=2,
                ),
            )
        except Exception:
            pass
        return {
            "year": int(getattr(year_args, "target_year", -1)),
            "model_name": getattr(year_args, "model_name", ""),
            "output_dir": str(getattr(year_args, "output_dir", "")),
            "status": "failed",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def load_cached_symbolic_models_for_repair(year_args, output_dir, checkpoint):
    """
    Load already-fitted symbolic candidates without running PySR.

    The audit run is meant to standardize/evaluate existing work, not refit
    formulas.  If a model/equation/manifest is missing or config-incompatible,
    failing loudly is safer than silently producing a mixed-standard result.
    """
    models = {}
    manifest_rows = []
    selected_specs = selected_formula_specs(year_args)
    for spec in selected_specs:
        cached = load_symbolic_candidate_if_current(spec, output_dir, year_args, checkpoint)
        if cached is None:
            raise FileNotFoundError(
                f"Cannot repair {year_args.model_name}: cached symbolic candidate "
                f"{spec.name} is missing or does not match the current settings. "
                "Finish the formula fit first, or rerun with the same CLI settings "
                "used to create the formula."
            )
        model, manifest_entry = cached
        models[spec.name] = {"model": model, "spec": spec, "bound_epsilon": year_args.bound_epsilon}
        manifest_rows.append(manifest_entry)

    feature_cols = ["forward_moneyness", "T_years", "iv"]
    candidate_manifest = add_walkforward_metadata(
        pd.DataFrame(manifest_rows), year_args, checkpoint
    )
    return models, feature_cols, selected_specs, candidate_manifest


def load_cached_csv(path, description):
    """Read a required repair artifact with a direct, useful error message."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return pd.read_csv(path)


def repair_symbolic_split(
    env,
    actor,
    scaler,
    device,
    models,
    feature_cols,
    selected_specs,
    smooth_trade_entries,
    args,
    output_dir,
    checkpoint,
    split,
):
    """
    Bring one validation/test split to the final publication artifact standard.

    Reuse what is already present.  Missing state/trade artifacts are rebuilt
    from the frozen agent/formulas; missing HOF artifacts are reconstructed from
    saved formula equations and the base split trade paths.
    """
    split_year = args.validation_year if split == "validation" else args.target_year
    stem = f"{args.model_name}_{checkpoint}_{split}"
    output_dir = Path(output_dir)

    state_path = output_dir / f"{stem}_target_states.csv"
    eval_state_path = output_dir / f"{stem}_target_states_with_smooth.csv"
    if eval_state_path.exists():
        eval_df = pd.read_csv(eval_state_path)
    else:
        if state_path.exists():
            eval_df = pd.read_csv(state_path)
        else:
            print(f"[repair] collecting missing {split} {split_year} target states")
            eval_df = collect_split_target_states(
                env, actor, scaler, device, args, split=split, split_year=split_year
            )
            atomic_to_csv(eval_df, state_path, index=False)
        for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
            target_col = f"{target['column_prefix']}_delta"
            if target_col not in eval_df.columns:
                eval_df = add_smoothed_agent_columns(
                    eval_df,
                    smooth_trade_entries[target["policy_name"]],
                    feature_cols,
                    column_prefix=target["column_prefix"],
                )
        atomic_to_csv(eval_df, eval_state_path, index=False)

    formula_fid = formula_fidelity_frames(
        models, eval_df, feature_cols, selected_specs, args, f"{split}_{split_year}"
    )
    formula_fid = add_walkforward_metadata(formula_fid, args, checkpoint, split=split)
    atomic_to_csv(formula_fid, output_dir / f"{stem}_fidelity.csv", index=False)

    smoother_fid = smoother_fidelity_table(
        smooth_trade_entries, eval_df, feature_cols, f"{split}_{split_year}_smooth_vs_raw_agent"
    )
    smoother_fid = add_walkforward_metadata(smoother_fid, args, checkpoint, split=split)
    atomic_to_csv(smoother_fid, output_dir / f"{stem}_smoother_fidelity.csv", index=False)

    trade_steps_path = output_dir / f"{stem}_trade_steps.csv"
    if trade_steps_path.exists():
        trade_steps = pd.read_csv(trade_steps_path)
    else:
        print(f"[repair] trading missing selected policies on {split} {split_year}")
        trade_steps = trade_split_year(
            env,
            actor,
            scaler,
            device,
            models,
            feature_cols,
            args,
            smoothing_entries=smooth_trade_entries,
            split=split,
            split_year=split_year,
        )
        atomic_to_csv(trade_steps, trade_steps_path, index=False)

    rate_cache = build_rate_cache(args.cleaned_data_dir)
    episode_metrics = episode_metrics_from_steps(trade_steps, rate_cache)
    episode_metrics = add_walkforward_metadata(episode_metrics, args, checkpoint, split=split)
    atomic_to_csv(episode_metrics, output_dir / f"{stem}_episode_metrics.csv", index=False)

    print(f"[repair] recomputing selected-policy bootstrap for {split} {split_year}")
    boot_frames = []
    for name, left, right in bootstrap_comparison_specs(models, smooth_trade_entries, args):
        pair_df = paired_episode_table(episode_metrics, left, right)
        pair_df = add_walkforward_metadata(pair_df, args, checkpoint, split=split)
        # Selected formulas have only a small number of comparisons, so keep the
        # paired CSVs for auditability.
        atomic_to_csv(pair_df, output_dir / f"{stem}_{name}_paired_episodes.csv", index=False)
        summary = summarize_bootstrap(name, pair_df, args)
        summary = add_walkforward_metadata(summary, args, checkpoint, split=split)
        boot_frames.append(summary)
    boot_summary = pd.concat(boot_frames, ignore_index=True)
    atomic_to_csv(boot_summary, output_dir / f"{stem}_bootstrap_summary.csv", index=False)

    print(f"[repair] evaluating full Hall-of-Fame on {split} {split_year}")
    hof_fid = hof_fidelity_table(models, eval_df, args, f"{split}_{split_year}")
    hof_fid = add_walkforward_metadata(hof_fid, args, checkpoint, split=split)
    atomic_to_csv(hof_fid, output_dir / f"{stem}_hof_fidelity.csv", index=False)

    hof_trade_steps_path = output_dir / f"{stem}_hof_trade_steps.csv"
    if hof_trade_steps_complete(hof_trade_steps_path, trade_steps, models, args):
        hof_trade_steps = pd.read_csv(hof_trade_steps_path)
    else:
        if hof_trade_steps_path.exists():
            print(
                f"[repair] cached {split} HOF trades are incomplete or unreadable; "
                "reconstructing from base trade paths"
            )
        hof_trade_steps = reconstruct_hof_trade_steps(trade_steps, models, args)
        if not hof_trade_steps.empty:
            hof_trade_steps = add_walkforward_metadata(
                hof_trade_steps, args, checkpoint, split=split
            )
            atomic_to_csv(hof_trade_steps, hof_trade_steps_path, index=False)

    if hof_trade_steps.empty:
        hof_episode_metrics = pd.DataFrame()
        combined_metrics = episode_metrics
        hof_points = pd.DataFrame()
    else:
        hof_episode_metrics = episode_metrics_from_steps(hof_trade_steps, rate_cache)
        hof_episode_metrics = add_walkforward_metadata(
            hof_episode_metrics, args, checkpoint, split=split
        )
        atomic_to_csv(
            hof_episode_metrics,
            output_dir / f"{stem}_hof_episode_metrics.csv",
            index=False,
        )
        combined_metrics = pd.concat([episode_metrics, hof_episode_metrics], ignore_index=True)
        hof_points = hof_point_comparison_table(combined_metrics, smooth_trade_entries)
        hof_points = add_walkforward_metadata(hof_points, args, checkpoint, split=split)
        atomic_to_csv(hof_points, output_dir / f"{stem}_hof_point_comparisons.csv", index=False)

    hof_outputs = {
        "fidelity": hof_fid,
        "episode_metrics": hof_episode_metrics,
        "combined_metrics": combined_metrics,
        "point_comparisons": hof_points,
    }
    split_outputs = {
        "state_path": state_path,
        "eval_state_path": eval_state_path,
        "trade_steps_path": trade_steps_path,
        "formula_fidelity": formula_fid,
        "smoother_fidelity": smoother_fid,
        "episode_metrics": episode_metrics,
        "bootstrap_summary": boot_summary,
        "hof": hof_outputs,
    }

    # For auditability this run keeps the entire Hall-of-Fame.  The flag
    # lets quick/debug repairs remain possible, while the publication command
    # should pass --hof-bootstrap-all.
    if args.hof_bootstrap_all:
        bootstrap_all_hof_policies(split_outputs, args, output_dir, checkpoint, split)
    return split_outputs


def repair_one_walkforward_year(year_args, refresh_aggregates=True):
    """
    Standardize one already-fitted walk-forward year without refitting PySR.

    This is the recovery path after an overnight run was interrupted or produced
    mixed-standard outputs.  It reuses saved formulas, distillation pools, and
    trade paths whenever possible, then regenerates fidelity/bootstrap/HOF
    artifacts with the current publication metric set.
    """
    if getattr(year_args, "bootstrap_summary_only", False):
        return bootstrap_summary_only_one_walkforward_year(
            year_args, refresh_aggregates=refresh_aggregates
        )

    set_random_seeds(year_args.seed)
    root_output_dir = Path(year_args.root_output_dir)
    output_dir = Path(year_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, selected_csv = infer_checkpoint(year_args.model_name, year_args.results_testing_dir)
    settings = load_settings_json(year_args.model_name)
    year_args.kappa = float(settings.get("kappa", 1.0))
    year_args.reward_exponent = float(settings.get("reward_exponent", 1.0))
    year_args.transaction_cost = float(settings.get("transaction_cost", 0.0))
    data_dir = ensure_walkforward_data_dir(year_args)

    print("\n" + "=" * 80)
    print(
        f"[repair] {year_args.target_year}: model={year_args.model_name}, "
        f"validation={year_args.validation_year}, test={year_args.target_year}, "
        f"checkpoint={checkpoint}"
    )

    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(data_dir)
    try:
        from empirical_deep_hedging.include.env import Env

        actor, scaler, device = load_actor_and_scaler(
            year_args.model_name, checkpoint, settings, year_args.device
        )
        env = Env(settings)

        models, feature_cols, selected_specs, candidate_manifest = load_cached_symbolic_models_for_repair(
            year_args, output_dir, checkpoint
        )
        atomic_to_csv(
            candidate_manifest,
            output_dir / f"{year_args.model_name}_{checkpoint}_candidate_manifest.csv",
            index=False,
        )

        distill_path = output_dir / f"{year_args.model_name}_{checkpoint}_distillation_pairs.csv"
        distill_df = load_cached_csv(distill_path, "distillation pool")
        coverage_diag, coverage_meta = coverage_diagnostics(distill_df)
        coverage_diag = add_walkforward_metadata(coverage_diag, year_args, checkpoint)
        atomic_to_csv(
            coverage_diag,
            output_dir / f"{year_args.model_name}_{checkpoint}_distillation_coverage.csv",
            index=False,
        )

        smooth_trade_entries, distill_eval_df = build_smooth_trade_entries_for_specs(
            distill_df, selected_specs, feature_cols, year_args, output_dir, checkpoint
        )

        distill_fid = formula_fidelity_frames(
            models,
            distill_eval_df,
            feature_cols,
            selected_specs,
            year_args,
            "distillation_train_only",
        )
        distill_fid = add_walkforward_metadata(distill_fid, year_args, checkpoint)
        atomic_to_csv(
            distill_fid,
            output_dir / f"{year_args.model_name}_{checkpoint}_distillation_fidelity.csv",
            index=False,
        )

        distill_smoother_fid = smoother_fidelity_table(
            smooth_trade_entries,
            distill_eval_df,
            feature_cols,
            "distillation_train_only_smooth_vs_raw_agent",
        )
        distill_smoother_fid = add_walkforward_metadata(
            distill_smoother_fid, year_args, checkpoint
        )
        atomic_to_csv(
            distill_smoother_fid,
            output_dir / f"{year_args.model_name}_{checkpoint}_distillation_smoother_fidelity.csv",
            index=False,
        )

        distill_hof_fid = hof_fidelity_table(
            models, distill_eval_df, year_args, "distillation_train_only"
        )
        distill_hof_fid = add_walkforward_metadata(distill_hof_fid, year_args, checkpoint)
        atomic_to_csv(
            distill_hof_fid,
            output_dir / f"{year_args.model_name}_{checkpoint}_distillation_hof_fidelity.csv",
            index=False,
        )

        validation_outputs = repair_symbolic_split(
            env,
            actor,
            scaler,
            device,
            models,
            feature_cols,
            selected_specs,
            smooth_trade_entries,
            year_args,
            output_dir,
            checkpoint,
            split="validation",
        )
        test_outputs = repair_symbolic_split(
            env,
            actor,
            scaler,
            device,
            models,
            feature_cols,
            selected_specs,
            smooth_trade_entries,
            year_args,
            output_dir,
            checkpoint,
            split="test",
        )

        # Keep the validation-selected top-K summaries as a separate object from
        # the exhaustive HOF bootstrap.  This lets us later compare selection
        # rules without looking at the test year.
        finalize_hof_selection_and_bootstrap(
            validation_outputs, test_outputs, year_args, output_dir, checkpoint
        )

        write_walkforward_year_report(
            output_dir / f"{year_args.model_name}_{checkpoint}_walkforward_report.txt",
            year_args,
            checkpoint,
            selected_csv,
            coverage_meta,
            coverage_diag,
            candidate_manifest,
            distill_fid,
            validation_outputs,
            test_outputs,
        )

        complete_marker = output_dir / f"{year_args.model_name}_{checkpoint}_walkforward_complete.json"
        atomic_write_text(
            complete_marker,
            json.dumps(
                {
                    "model_name": year_args.model_name,
                    "target_year": int(year_args.target_year),
                    "validation_year": int(year_args.validation_year),
                    "train_start_year": int(year_args.train_start_year),
                    "train_end_year": int(year_args.train_end_year),
                    "checkpoint": int(checkpoint),
                    "output_dir": str(output_dir),
                    "families": [str(name) for name in models.keys()],
                    "status": "repaired_completed",
                    "hof_bootstrap_all": bool(year_args.hof_bootstrap_all),
                },
                indent=2,
            ),
        )
        for marker in output_dir.glob("*_walkforward_in_progress.json"):
            try:
                marker.unlink()
            except OSError as exc:
                print(f"[repair] Warning: could not remove stale marker {marker}: {exc}")
        if refresh_aggregates:
            refresh_walkforward_aggregates(root_output_dir)
        print(f"[repair] completed {year_args.target_year}. Outputs are standardized in {output_dir}")
        return {
            "year": int(year_args.target_year),
            "model_name": year_args.model_name,
            "checkpoint": checkpoint,
            "output_dir": str(output_dir),
            "status": "repaired_completed",
        }
    finally:
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir


def run_one_repair_year_worker(year_args):
    """ProcessPool-compatible wrapper for repair jobs."""
    try:
        return repair_one_walkforward_year(year_args, refresh_aggregates=False)
    except Exception:
        return {
            "year": int(year_args.target_year),
            "model_name": year_args.model_name,
            "traceback": traceback.format_exc(),
        }


def run_repair_walkforward(args):
    """
    Repair/standardize already-fitted walk-forward years.

    This mode is deliberately different from the normal PySR walk-forward run:
    it does not refit formulas and it does not skip folders just because they
    already have reports.  It reloads cached formula/HOF artifacts and rewrites
    train/validation/test fidelity plus validation/test trading/bootstrap
    outputs to the current publication standard.
    """
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(int(args.first_test_year), int(args.final_test_year) + 1))
    print(f"[repair] years={years}, model_prefix={args.model_prefix}")
    print(f"[repair] formula families={args.formula_candidates}")
    print(f"[repair] hof_bootstrap_all={args.hof_bootstrap_all}, n_bootstrap={args.n_bootstrap}")
    if args.bootstrap_summary_only:
        print("[repair] bootstrap_summary_only=True; cached episode metrics will be reused")

    year_args_list = []
    for year in years:
        year_args = year_args_from_walkforward(args, year)
        year_args.root_output_dir = str(root_output_dir)
        year_args_list.append(year_args)

    max_workers = max(1, int(args.parallel_years))
    if max_workers == 1 or len(year_args_list) <= 1:
        failures = []
        for year_args in year_args_list:
            try:
                repair_one_walkforward_year(
                    year_args,
                    refresh_aggregates=bool(args.refresh_aggregates),
                )
            except Exception:
                failures.append(
                    {
                        "year": int(year_args.target_year),
                        "model_name": year_args.model_name,
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[repair] year {year_args.target_year} failed")
                print(failures[-1]["traceback"])
        if args.refresh_aggregates:
            refresh_walkforward_aggregates(root_output_dir)
        if failures:
            raise RuntimeError(f"{len(failures)} repair year(s) failed: {failures}")
        return

    if args.parallel_backend == "subprocess":
        # The subprocess scheduler preserves the original CLI.  Therefore child
        # processes also enter --repair-walkforward mode, but each child is
        # pinned to exactly one year by the appended first/final year override.
        run_walkforward_subprocess_scheduler(args, year_args_list, root_output_dir)
        return

    print(f"[repair] running up to {max_workers} repair worker processes")
    pending_years = list(year_args_list)
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {}
        for _ in range(min(max_workers, len(pending_years))):
            year_args = pending_years.pop(0)
            future_to_year[executor.submit(run_one_repair_year_worker, year_args)] = year_args
        while future_to_year:
            done, _ = wait(future_to_year.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                year_args = future_to_year.pop(fut)
                result = fut.result()
                if "traceback" in result:
                    failures.append(result)
                    print(f"[repair] FAILED year {result['year']}")
                    print(result["traceback"])
                else:
                    print(f"[repair] completed year {result['year']}")
                if pending_years:
                    next_args = pending_years.pop(0)
                    future_to_year[executor.submit(run_one_repair_year_worker, next_args)] = next_args
    if args.refresh_aggregates:
        refresh_walkforward_aggregates(root_output_dir)
    if failures:
        raise RuntimeError(f"{len(failures)} repair year(s) failed: {failures}")


def run_walkforward_subprocess_scheduler(args, year_args_list, root_output_dir):
    """
    Run independent years as explicit child Python processes.

    This replaces the fragile ProcessPool path for long overnight runs.  Each
    year gets its own log file, the parent can always see which slots are
    running, and a stuck child does not hide behind a silent ProcessPool future.
    """
    max_workers = max(1, int(args.parallel_years))
    pending_years = list(year_args_list)
    running = {}
    completed = []
    failures = []
    script_path = str(Path(__file__).resolve())
    base_argv = list(sys.argv[1:])

    def child_argv_for_year(year_args):
        # Preserve the original CLI choices, then override only the
        # orchestration controls.  Argparse uses the final occurrence of these
        # scalar options, so this keeps settings like niterations/samples/model
        # prefix intact without reconstructing the whole Namespace by hand.
        return (
            [sys.executable, script_path]
            + base_argv
            + [
                "--first-test-year",
                str(int(year_args.target_year)),
                "--final-test-year",
                str(int(year_args.target_year)),
                "--parallel-years",
                "1",
                "--no-refresh-aggregates",
                "--no-skip-completed-years",
            ]
        )

    def launch_next():
        if not pending_years:
            return
        year_args = pending_years.pop(0)
        output_dir = Path(year_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / f"{year_args.model_name}_walkforward_subprocess.log"
        log_file = open(log_path, "a", encoding="utf-8", buffering=1)
        log_file.write("\n" + "=" * 100 + "\n")
        log_file.write(
            f"[parent] launching year {year_args.target_year} at "
            f"{pd.Timestamp.utcnow().isoformat()}\n"
        )
        log_file.write("[parent] command: " + " ".join(child_argv_for_year(year_args)) + "\n")
        proc = subprocess.Popen(
            child_argv_for_year(year_args),
            cwd=str(Path.cwd()),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running[proc.pid] = {
            "proc": proc,
            "year_args": year_args,
            "log_file": log_file,
            "log_path": log_path,
            "started": time.time(),
        }
        print(
            f"[walk-forward] launched year {year_args.target_year} as pid={proc.pid}; "
            f"running={len(running)}, queued={len(pending_years)}, log={log_path}"
        )

    for _ in range(min(max_workers, len(pending_years))):
        launch_next()

    while running:
        time.sleep(max(5, int(args.subprocess_poll_seconds)))
        finished_pids = []
        for pid, record in list(running.items()):
            proc = record["proc"]
            returncode = proc.poll()
            if returncode is None:
                continue
            year_args = record["year_args"]
            log_file = record["log_file"]
            log_file.write(
                f"[parent] year {year_args.target_year} exited with code {returncode} "
                f"at {pd.Timestamp.utcnow().isoformat()}\n"
            )
            log_file.close()
            finished_pids.append(pid)

            if returncode == 0 and walkforward_year_dir_completed(year_args.output_dir):
                result = {
                    "year": int(year_args.target_year),
                    "model_name": year_args.model_name,
                    "output_dir": str(year_args.output_dir),
                    "status": "completed",
                    "log": str(record["log_path"]),
                }
                completed.append(result)
                print(
                    f"[walk-forward] completed year {result['year']} pid={pid}; "
                    f"outputs={result['output_dir']}"
                )
                if args.refresh_aggregates:
                    refresh_walkforward_aggregates(root_output_dir)
            else:
                result = {
                    "year": int(year_args.target_year),
                    "model_name": year_args.model_name,
                    "output_dir": str(year_args.output_dir),
                    "status": "failed",
                    "returncode": returncode,
                    "log": str(record["log_path"]),
                }
                failures.append(result)
                print(
                    f"[walk-forward] FAILED year {result['year']} pid={pid}; "
                    f"returncode={returncode}; log={record['log_path']}"
                )

        for pid in finished_pids:
            running.pop(pid, None)
            launch_next()

        if running:
            active = [
                f"{rec['year_args'].target_year}:pid{pid}"
                for pid, rec in sorted(running.items())
            ]
            print(
                f"[walk-forward] active yearly subprocesses: {active}; "
                f"queued={len(pending_years)}"
            )

    if args.refresh_aggregates:
        refresh_walkforward_aggregates(root_output_dir)
    if failures:
        raise RuntimeError(f"{len(failures)} walk-forward year subprocess(es) failed: {failures}")
    print(f"[walk-forward] all subprocess years completed: {[r['year'] for r in completed]}")


def run_walkforward(args):
    """
    Publication-grade 2015-2023 walk-forward symbolic distillation.

    For each test year Y:
      1. Load final_WF_exp1_k1_testY and its selected checkpoint.
      2. Fit the three requested formula families using only years <= Y-2.
      3. Save each PySR model, formula, full Hall-of-Fame, and manifest
         immediately after its regression completes.
      4. Evaluate all formulas, the raw agent, the smoothed agent, and BS on
         validation year Y-1 and testing year Y.
      5. Refresh aggregate CSVs so completed years can be analyzed while later
         years are still running.
    """
    root_output_dir = Path(args.output_dir)
    root_output_dir.mkdir(parents=True, exist_ok=True)
    years = list(range(int(args.first_test_year), int(args.final_test_year) + 1))
    print(f"[walk-forward] years={years}, model_prefix={args.model_prefix}")
    print(f"[walk-forward] formula families={args.formula_candidates}")
    year_args_list = []
    for year in years:
        year_args = year_args_from_walkforward(args, year)
        # Keep the original root output path on every child.  year_args.output_dir
        # is the year-specific folder; root_output_dir is used only by the
        # parent/sequential runner for aggregate refreshes.
        year_args.root_output_dir = str(root_output_dir)
        year_args_list.append(year_args)

    if args.skip_completed_years:
        remaining = []
        skipped = []
        for year_args in year_args_list:
            if walkforward_year_dir_completed(year_args.output_dir):
                skipped.append(int(year_args.target_year))
            else:
                remaining.append(year_args)
        if skipped:
            print(f"[walk-forward] skipping completed years: {skipped}")
        year_args_list = remaining
        if not year_args_list:
            print("[walk-forward] no unfinished years to run")
            if args.refresh_aggregates:
                refresh_walkforward_aggregates(root_output_dir)
            return

    max_workers = max(1, int(args.parallel_years))
    if max_workers == 1 or len(year_args_list) <= 1:
        print("[walk-forward] running years sequentially")
        failures = []
        for year_args in year_args_list:
            try:
                run_one_walkforward_year(
                    year_args,
                    refresh_aggregates=bool(args.refresh_aggregates),
                )
            except Exception:
                failures.append(
                    {
                        "year": int(year_args.target_year),
                        "model_name": year_args.model_name,
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[walk-forward] year {year_args.target_year} failed")
                print(failures[-1]["traceback"])
        if args.refresh_aggregates:
            refresh_walkforward_aggregates(root_output_dir)
        if failures:
            raise RuntimeError(f"{len(failures)} walk-forward year(s) failed: {failures}")
        return

    if args.parallel_backend == "subprocess":
        run_walkforward_subprocess_scheduler(args, year_args_list, root_output_dir)
        return

    print(
        f"[walk-forward] running up to {max_workers} yearly worker processes in parallel. "
        "Children write only per-year folders; parent refreshes aggregate CSVs."
    )
    completed = []
    failures = []
    pending_years = list(year_args_list)
    running = {}

    def submit_next_year(executor):
        if not pending_years:
            return
        next_args = pending_years.pop(0)
        future = executor.submit(run_one_walkforward_year_worker, next_args)
        running[future] = next_args
        print(
            f"[walk-forward] submitted year {next_args.target_year}; "
            f"running={len(running)}, queued={len(pending_years)}"
        )

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(max_workers, len(pending_years))):
            submit_next_year(executor)

        while running:
            done, _ = wait(running.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                year_args = running.pop(future)
                try:
                    result = future.result()
                except Exception:
                    result = {
                        "year": int(year_args.target_year),
                        "model_name": year_args.model_name,
                        "output_dir": str(year_args.output_dir),
                        "status": "failed",
                        "traceback": traceback.format_exc(),
                    }

                # Keep the worker pool full before doing parent-side aggregate
                # refresh work.  This makes the "3 years at a time" behavior
                # explicit and visible in the console.
                submit_next_year(executor)

                if result.get("status") == "completed":
                    completed.append(result)
                    print(
                        f"[walk-forward] completed year {result['year']} in worker; "
                        f"outputs: {result['output_dir']}"
                    )
                    if args.refresh_aggregates:
                        refresh_walkforward_aggregates(root_output_dir)
                else:
                    failures.append(result)
                    print(f"[walk-forward] FAILED year {result.get('year')}: {result.get('error', '')}")
                    print(result.get("traceback", ""))

    if args.refresh_aggregates:
        refresh_walkforward_aggregates(root_output_dir)
    if failures:
        raise RuntimeError(f"{len(failures)} walk-forward year(s) failed: {failures}")
    print(f"[walk-forward] all parallel years completed: {[r['year'] for r in completed]}")


def run_symbolic_fit_phase_for_model(args):
    """
    Fit the selected symbolic formulas for one model, without 2023 testing.

    The multi-seed default uses this first for both seed classes so that all
    four requested regressions complete before any testing starts.
    """
    set_random_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, _ = infer_checkpoint(args.model_name, args.results_testing_dir)
    settings = load_settings_json(args.model_name)
    data_dir = ensure_walkforward_data_dir(args)

    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(data_dir)
    try:
        from empirical_deep_hedging.include.env import Env

        actor, scaler, device = load_actor_and_scaler(
            args.model_name, checkpoint, settings, args.device
        )
        env = Env(settings)
        distill_df, _ = load_or_build_distillation_pool(
            env, actor, scaler, device, data_dir, args, output_dir, checkpoint
        )
        coverage_diag, _ = coverage_diagnostics(distill_df)
        coverage_diag.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_distillation_coverage.csv",
            index=False,
        )
        print(f"[fit-phase] fitting symbolic formulas for {args.model_name}")
        _, _, fit_df, candidate_manifest = fit_symbolic_policies(
            distill_df, args, output_dir=output_dir, checkpoint=checkpoint
        )
        fit_df.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_candidate_fit_rows.csv",
            index=False,
        )
        candidate_manifest.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_candidate_manifest.csv",
            index=False,
        )
    finally:
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir


def run_single_model(args):
    set_random_seeds(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, selected_csv = infer_checkpoint(args.model_name, args.results_testing_dir)
    settings = load_settings_json(args.model_name)
    data_dir = ensure_walkforward_data_dir(args)

    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(data_dir)
    try:
        from empirical_deep_hedging.include.env import Env

        actor, scaler, device = load_actor_and_scaler(
            args.model_name, checkpoint, settings, args.device
        )
        env = Env(settings)

        distill_df, distill_path = load_or_build_distillation_pool(
            env, actor, scaler, device, data_dir, args, output_dir, checkpoint
        )

        coverage_diag, coverage_meta = coverage_diagnostics(distill_df)
        coverage_path = output_dir / f"{args.model_name}_{checkpoint}_distillation_coverage.csv"
        coverage_diag.to_csv(coverage_path, index=False)

        print("[fit] fitting bounded symbolic-regression candidates")
        models, feature_cols, fit_df, candidate_manifest = fit_symbolic_policies(
            distill_df, args, output_dir=output_dir, checkpoint=checkpoint
        )
        fit_df.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_candidate_fit_rows.csv",
            index=False,
        )

        candidate_manifest.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_candidate_manifest.csv",
            index=False,
        )

        selected_specs = [s for s in FORMULA_SPECS if s.name in args.formula_candidates]
        smooth_trade_entries = {}
        distill_eval_df = distill_df
        if selected_specs_need_smooth_target(selected_specs):
            # Build/load every smoothed target used by the selected formulas.
            # In the bandwidth experiment this creates four traded smoothed
            # policies, e.g. smooth_target_bw075, smooth_target_bw100, ...
            # Each formula is compared with its own target below; all smoothed
            # targets are also compared with the raw agent and BS.
            for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
                smoother_entry, smooth_feature_cols, _ = load_or_build_one_smoother(
                    distill_df,
                    args,
                    output_dir,
                    checkpoint,
                    target["smoother_name"],
                    bandwidth_scale=target["bandwidth_scale"],
                )
                if smooth_feature_cols != feature_cols:
                    raise ValueError("Smoother and symbolic formula feature columns diverged.")
                smooth_trade_entries[target["policy_name"]] = smoother_entry
                distill_eval_df = add_smoothed_agent_columns(
                    distill_eval_df,
                    smoother_entry,
                    feature_cols,
                    column_prefix=target["column_prefix"],
                )

        distill_fid_frames = [
            fidelity_table(
                models,
                distill_eval_df,
                feature_cols,
                "distillation_train_only_vs_raw_agent",
                target_col="agent_delta",
            )
        ]
        for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
            target_col = f"{target['column_prefix']}_delta"
            if target_col not in distill_eval_df.columns:
                continue
            distill_fid_frames.append(
                fidelity_table(
                    models,
                    distill_eval_df,
                    feature_cols,
                    f"distillation_train_only_vs_{target['policy_name']}",
                    target_col=target_col,
                )
            )
        distill_fid = pd.concat(distill_fid_frames, ignore_index=True)
        distill_fid.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_distillation_fidelity.csv",
            index=False,
        )

        print("[test] collecting 2023 states for out-of-sample fidelity")
        test_state_df = collect_test_target_states(env, actor, scaler, device, args)
        test_eval_df = test_state_df
        if smooth_trade_entries:
            for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
                test_eval_df = add_smoothed_agent_columns(
                    test_eval_df,
                    smooth_trade_entries[target["policy_name"]],
                    feature_cols,
                    column_prefix=target["column_prefix"],
                )
        test_state_df.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_test_target_states.csv",
            index=False,
        )
        if test_eval_df is not test_state_df:
            test_eval_df.to_csv(
                output_dir / f"{args.model_name}_{checkpoint}_test_target_states_with_smooth.csv",
                index=False,
            )

        test_fid_frames = [
            fidelity_table(
                models,
                test_eval_df,
                feature_cols,
                "test_2023_vs_raw_agent",
                target_col="agent_delta",
            )
        ]
        for target in unique_smoothing_targets_for_specs(selected_specs, args).values():
            target_col = f"{target['column_prefix']}_delta"
            if target_col not in test_eval_df.columns:
                continue
            test_fid_frames.append(
                fidelity_table(
                    models,
                    test_eval_df,
                    feature_cols,
                    f"test_2023_vs_{target['policy_name']}",
                    target_col=target_col,
                )
            )
        test_fid = pd.concat(test_fid_frames, ignore_index=True)
        test_fid.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_test_fidelity.csv",
            index=False,
        )

        print("[test] trading raw agent, smoothed agent, and formula candidates on 2023")
        trade_steps = trade_test_year(
            env,
            actor,
            scaler,
            device,
            models,
            feature_cols,
            args,
            smoothing_entries=smooth_trade_entries,
        )
        trade_steps_path = output_dir / f"{args.model_name}_{checkpoint}_test_trade_steps.csv"
        trade_steps.to_csv(trade_steps_path, index=False)

        rate_cache = build_rate_cache(args.cleaned_data_dir)
        episode_metrics = episode_metrics_from_steps(trade_steps, rate_cache)
        episode_metrics.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_test_episode_metrics.csv",
            index=False,
        )

        print("[bootstrap] running two-stage bootstrap comparisons")
        comparisons = [("agent_vs_bs", "agent", "bs")]
        for smooth_name in smooth_trade_entries:
            comparisons.append((f"{smooth_name}_vs_agent", smooth_name, "agent"))
            comparisons.append((f"{smooth_name}_vs_bs", smooth_name, "bs"))
        for candidate_name in models:
            spec = models[candidate_name]["spec"]
            if spec.target_source == "smooth_kernel_bs_delta_residual":
                smooth_name = smooth_policy_name_for_spec(spec, args)
                comparisons.append((f"{candidate_name}_vs_{smooth_name}", candidate_name, smooth_name))
            comparisons.append((f"{candidate_name}_vs_agent", candidate_name, "agent"))
            comparisons.append((f"{candidate_name}_vs_bs", candidate_name, "bs"))
        if args.include_pairwise_formula_bootstrap:
            names = list(models.keys())
            for i, left in enumerate(names):
                for right in names[i + 1 :]:
                    comparisons.append((f"{left}_vs_{right}", left, right))
        boot_frames = []
        for name, left, right in comparisons:
            pair_df = paired_episode_table(episode_metrics, left, right)
            pair_df.to_csv(
                output_dir / f"{args.model_name}_{checkpoint}_{name}_paired_episodes.csv",
                index=False,
            )
            boot_frames.append(summarize_bootstrap(name, pair_df, args))
        boot_summary = pd.concat(boot_frames, ignore_index=True)
        boot_summary.to_csv(
            output_dir / f"{args.model_name}_{checkpoint}_bootstrap_summary.csv",
            index=False,
        )

        write_report(
            output_dir / f"{args.model_name}_{checkpoint}_interpret_report.txt",
            args,
            checkpoint,
            selected_csv,
            coverage_meta,
            coverage_diag,
            candidate_manifest,
            distill_fid,
            test_fid,
            boot_summary,
        )

        print("[done] Wrote leakage-safe interpretability outputs to:")
        print(f"  {output_dir}")
        print(f"[done] Distillation pairs: {distill_path}")
        print(f"[done] Trade steps: {trade_steps_path}")

    finally:
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir


def args_for_model(args, model_name):
    """Copy argparse Namespace and replace the active model name."""
    out = copy.copy(args)
    out.model_name = model_name
    return out


def run(args):
    """
    Default orchestration for publication walk-forward distillation.

    By default this runs the full 2015-2023 publication workflow.
    """
    if args.repair_walkforward:
        run_repair_walkforward(args)
        return

    if args.walk_forward:
        run_walkforward(args)
        return

    if args.single_model:
        run_single_model(args)
        if args.run_smoothing:
            run_smoothing_diagnostics_for_model(args)
        return

    model_names = args.model_names
    print(f"[orchestrate] symbolic seed-test models: {model_names}")
    for model_name in model_names:
        run_symbolic_fit_phase_for_model(args_for_model(args, model_name))

    print("[orchestrate] all requested regressions are complete; starting 2023 tests")
    for model_name in model_names:
        run_single_model(args_for_model(args, model_name))

    if args.run_smoothing:
        smoothing_model_names = args.smoothing_model_names or model_names
        print(f"[orchestrate] symbolic tests complete; starting smoothing diagnostics: {smoothing_model_names}")
        for model_name in smoothing_model_names:
            run_smoothing_diagnostics_for_model(args_for_model(args, model_name))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Leakage-safe symbolic distillation for empirical deep hedging."
    )
    parser.add_argument(
        "--walk-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the publication 2015-2023 walk-forward distillation. "
            "Use --no-walk-forward with --single-model for legacy one-year debugging."
        ),
    )
    parser.add_argument(
        "--repair-walkforward",
        action="store_true",
        help=(
            "Do not fit PySR.  Reuse cached walk-forward formulas/HOF equations "
            "and regenerate train/validation/test fidelity, trading, bootstrap, "
            "HOF, reports, and aggregates to the current publication standard."
        ),
    )
    parser.add_argument(
        "--bootstrap-summary-only",
        action="store_true",
        help=(
            "With --repair-walkforward, reuse cached episode metrics and "
            "regenerate only bootstrap summary CSVs plus root aggregate CSVs. "
            "This does not refit formulas, trade policies, or rewrite fidelity tables."
        ),
    )
    parser.add_argument(
        "--model-prefix",
        default=DEFAULT_MODEL_PREFIX,
        help=(
            "Walk-forward model class prefix.  Year Y uses "
            "<model-prefix><Y>, e.g. final_WF_exp1_k1_test2023."
        ),
    )
    parser.add_argument("--first-test-year", type=int, default=DEFAULT_FIRST_TEST_YEAR)
    parser.add_argument("--final-test-year", type=int, default=DEFAULT_FINAL_TEST_YEAR)
    parser.add_argument(
        "--walkforward-data-dir-template",
        default=DEFAULT_WALKFORWARD_DATA_DIR_TEMPLATE,
        help="DATA_DIR template for year Y; must contain {year}.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--model-names",
        nargs="+",
        default=DEFAULT_MODEL_NAMES,
        help=(
            "Legacy non-walk-forward model classes.  Ignored by the default "
            "publication walk-forward runner."
        ),
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Run the legacy one-model workflow using --model-name instead of the multi-seed orchestration.",
    )
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR)
    parser.add_argument("--train-start-year", type=int, default=DEFAULT_TRAIN_START_YEAR)
    parser.add_argument("--train-end-year", type=int, default=DEFAULT_TRAIN_END_YEAR)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_VALIDATION_YEAR)
    parser.add_argument("--target-data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--cleaned-data-dir", default=DEFAULT_CLEANED_DATA_DIR)
    parser.add_argument("--results-testing-dir", default="results/testing")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--parallel-years",
        type=int,
        default=3,
        help=(
            "Number of independent walk-forward years to run at once.  Each "
            "year remains deterministic internally; parallelism is only across "
            "separate yearly processes.  Use 1 for the sequential behavior."
        ),
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["subprocess", "processpool"],
        default="subprocess",
        help=(
            "Backend for running multiple years.  subprocess is more robust "
            "for long PySR runs and writes one log file per year; processpool "
            "is kept only as a fallback."
        ),
    )
    parser.add_argument(
        "--subprocess-poll-seconds",
        type=int,
        default=30,
        help="How often the subprocess parent reports active year processes.",
    )
    parser.add_argument(
        "--refresh-aggregates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Refresh root aggregate CSVs after completed years.  In parallel "
            "mode only the parent process writes aggregates, so this is safe."
        ),
    )
    parser.add_argument(
        "--skip-completed-years",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When resuming a long walk-forward run, skip year folders that have "
            "a completed report or completion marker.  Failed/in-progress years "
            "are rerun."
        ),
    )
    parser.add_argument("--rebuild-data-dir", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--canonical-position",
        choices=["bs", "zero"],
        default="bs",
        help="Fourth actor input used when querying a three-feature target-delta surface.",
    )
    parser.add_argument(
        "--max-train-episodes",
        type=int,
        default=20000,
        help="Training episodes used to build/reuse the actor-label pool; 0 means all starts.",
    )
    parser.add_argument(
        "--rebuild-distillation-pool",
        action="store_true",
        help="Ignore a matching cached train-year state/action pool and collect it again.",
    )
    parser.add_argument(
        "--refit-symbolic-models",
        action="store_true",
        help="Ignore matching per-candidate PySR caches and refit symbolic formulas.",
    )
    parser.add_argument("--train-support-probes", type=int, default=20000)
    parser.add_argument("--general-probes", type=int, default=20000)
    parser.add_argument("--general-m-min", type=float, default=0.85)
    parser.add_argument("--general-m-max", type=float, default=1.20)
    parser.add_argument("--general-t-days-min", type=float, default=7.0)
    parser.add_argument("--general-t-days-max", type=float, default=100.0)
    parser.add_argument("--general-iv-min", type=float, default=0.05)
    parser.add_argument("--general-iv-max", type=float, default=0.80)

    parser.add_argument(
        "--formula-candidates",
        nargs="+",
        default=DEFAULT_FORMULA_CANDIDATES,
        choices=[s.name for s in FORMULA_SPECS],
        help="Bounded formula candidates to fit.",
    )
    parser.add_argument(
        "--m-bins",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 0.95, 0.975, 1.00, 1.025, 1.05, 1.075, 1.10, 1.15, 1.25],
        help="Forward-moneyness bin edges for stratified symbolic-fit sampling.",
    )
    parser.add_argument(
        "--iv-bins",
        type=float,
        nargs="+",
        default=[0.00, 0.10, 0.12, 0.16, 0.20, 0.24, 0.28, 0.35, 0.50, 0.85],
        help="IV bin edges for stratified symbolic-fit sampling.",
    )
    parser.add_argument(
        "--t-day-bins",
        type=float,
        nargs="+",
        default=[0.0, 14.0, 21.0, 35.0, 50.0, 65.0, 80.0, 100.0, 140.0],
        help="Calendar-day maturity bin edges for stratified symbolic-fit sampling.",
    )
    parser.add_argument("--gap-weight-scale", type=float, default=4.0)
    parser.add_argument(
        "--loss-gap-weight-scale",
        type=float,
        default=4.0,
        help="Extra PySR loss weight multiplier for |agent delta - BS delta| states.",
    )
    parser.add_argument("--gamma-weight-scale", type=float, default=1.5)
    parser.add_argument("--focus-weight-boost", type=float, default=4.0)
    parser.add_argument(
        "--focus-stratified-share",
        type=float,
        default=0.60,
        help="Share of focus candidate rows selected by stratified cell sampling.",
    )
    parser.add_argument("--n-sr-samples", type=int, default=50000)
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--maxsize", type=int, default=24)
    parser.add_argument("--populations", type=int, default=20)
    parser.add_argument("--bound-epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--deterministic-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask PySR to run serial/deterministically when supported; slower but more reproducible.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
        help=(
            "Torch device used to query the trained actor.  Default is CPU for "
            "publication reproducibility; use --device cuda or --device auto "
            "only if you intentionally accept possible numerical differences."
        ),
    )

    parser.add_argument("--max-test-episodes", type=int, default=0)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--include-pairwise-formula-bootstrap", action="store_true")
    parser.add_argument(
        "--evaluate-hof-formulas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate every PySR Hall-of-Fame equation on validation/test. "
            "This is needed because PySR's selected formula can be worse than "
            "other equations on the frontier."
        ),
    )
    parser.add_argument(
        "--hof-start-year",
        type=int,
        default=2017,
        help=(
            "First walk-forward test year for HOF evaluation.  Default 2017 "
            "lets an already-running 2016 job finish unchanged, then enables "
            "HOF diagnostics from the next manual restart."
        ),
    )
    parser.add_argument(
        "--hof-max-equations-per-candidate",
        type=int,
        default=0,
        help="Limit HOF equations per candidate for debugging; 0 means all equations.",
    )
    parser.add_argument(
        "--hof-selection-target",
        choices=["actual", "canonical", "smooth"],
        default="actual",
        help="Validation target used to rank HOF equations for bootstrap selection.",
    )
    parser.add_argument(
        "--hof-bootstrap-top-k",
        type=int,
        default=10,
        help="Bootstrap only the top-K validation-selected HOF equations by default.",
    )
    parser.add_argument(
        "--hof-bootstrap-all",
        action="store_true",
        help="Bootstrap every HOF equation.  This is comprehensive but can be slow.",
    )
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--run-smoothing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After symbolic tests, also run the three-smoother diagnostic. "
            "The default smoothed-symbolic experiment does not need this."
        ),
    )
    parser.add_argument(
        "--smoothing-model-names",
        nargs="+",
        default=None,
        help="Models for smoothing diagnostics; defaults to --model-names.",
    )
    parser.add_argument(
        "--refit-smoothers",
        action="store_true",
        help="Ignore matching smoothing caches and refit the three smoothers.",
    )
    parser.add_argument("--max-smoothing-rows", type=int, default=100000)
    parser.add_argument("--smoothing-neighbors", type=int, default=256)
    parser.add_argument("--smoothing-bandwidth-scale", type=float, default=1.0)
    parser.add_argument("--monotonic-max-iter", type=int, default=300)
    parser.add_argument("--monotonic-learning-rate", type=float, default=0.05)
    parser.add_argument("--monotonic-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--monotonic-l2", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
