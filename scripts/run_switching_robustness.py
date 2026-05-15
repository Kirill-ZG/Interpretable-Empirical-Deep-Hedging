"""
Small 2022 regime-switching symbolic-regression experiment.

This is intentionally separate from distill_empirical_agents.py.  It tests one narrow
research question before we consider another full walk-forward run:

    Does a piecewise symbolic policy in forward-moneyness bands reduce the
    harmful formula-agent approximation error we saw in 2022?

The script fits exactly two switching policies by default:

    1. switch2_moneyness_bs_delta_residual
       bands: m_fwd < 1.00 and m_fwd >= 1.00

    2. switch3_moneyness_bs_delta_residual
       bands: m_fwd < 0.95, 0.95 <= m_fwd < 1.05, m_fwd >= 1.05

Each band gets its own PySR model for the BS-delta residual:

    target = agent_delta - bs_delta
    traded_delta = clip(bs_delta + symbolic_residual, 0, 1)

IMPORTANT:
    This file is diagnostic code.  It reuses the already-built leakage-safe
    2022 distillation pool from results/interpret_real_walkforward and then
    evaluates on validation 2021 and test 2022.  Do not select final paper
    rules from 2022 test results.

Suggested manual run:

    python scripts/run_switching_robustness.py

Useful faster dry-ish run:

    python scripts/run_switching_robustness.py --niterations 40 --n-sr-samples 8000 --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from distill_empirical_agents import (
    FormulaSpec,
    add_walkforward_metadata,
    atomic_pickle_dump,
    atomic_to_csv,
    atomic_write_text,
    build_rate_cache,
    collect_split_target_states,
    episode_metrics_from_steps,
    fidelity_table,
    fit_pysr_model,
    infer_checkpoint,
    load_actor_and_scaler,
    load_settings_json,
    paired_episode_table,
    set_random_seeds,
    summarize_bootstrap,
    trade_split_year,
)


DEFAULT_MODEL_PREFIX = "final_WF_exp1_k1_test"
DEFAULT_TEST_YEAR = 2022
DEFAULT_OUTPUT_DIR = "results/switching_test_2022"
DEFAULT_WALKFORWARD_OUTPUT_ROOT = "results/interpret_real_walkforward"
DEFAULT_WALKFORWARD_DATA_DIR_TEMPLATE = "data_interpret_real_wf_{year}"
DEFAULT_CLEANED_DATA_DIR = "cleaned_data"
DEFAULT_RESULTS_TESTING_DIR = "results/testing"
DEFAULT_SEED = 123

FEATURE_COLS = ["forward_moneyness", "T_years", "iv"]
SWITCH_SPEC = FormulaSpec(
    name="moneyness_switch_bs_delta_residual",
    sample_kind="uniform_by_moneyness_band",
    target_kind="delta_residual",
    loss_kind="default",
    target_source="raw_agent",
    description=(
        "Piecewise moneyness-band PySR policy; each band fits "
        "agent_delta - BS_delta on training-only distillation pairs."
    ),
)


@dataclass(frozen=True)
class SwitchingConfig:
    name: str
    cuts: tuple[float, ...]


class SwitchingPySRModel:
    """
    Tiny wrapper making several per-band PySR models look like one model.

    The empirical distillation evaluator only needs a `.predict(X)` method
    returning the raw symbolic output.  Here that raw output is the BS-delta
    residual.  The bounded delta conversion remains handled by the shared
    formula evaluator because the model entry uses
    SWITCH_SPEC.target_kind == "delta_residual".
    """

    def __init__(self, cuts, band_models, band_labels):
        self.cuts = np.asarray(cuts, dtype=float)
        self.band_models = list(band_models)
        self.band_labels = list(band_labels)

    def band_index(self, moneyness):
        return np.searchsorted(self.cuts, moneyness, side="right")

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        out = np.empty(len(X), dtype=float)
        bands = self.band_index(X[:, 0])
        for band_i, model in enumerate(self.band_models):
            mask = bands == band_i
            if not np.any(mask):
                continue
            pred = np.asarray(model.predict(X[mask]), dtype=float)
            if pred.shape == ():
                pred = np.full(mask.sum(), float(pred))
            out[mask] = pred
        return out


def parse_cut_list(text):
    if text is None or str(text).strip() == "":
        return tuple()
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def band_labels(cuts):
    cuts = tuple(float(x) for x in cuts)
    edges = [-math.inf, *cuts, math.inf]
    labels = []
    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]
        if not np.isfinite(left):
            labels.append(f"m_fwd < {right:g}")
        elif not np.isfinite(right):
            labels.append(f"m_fwd >= {left:g}")
        else:
            labels.append(f"{left:g} <= m_fwd < {right:g}")
    return labels


def add_band_column(df, cuts):
    out = df.copy()
    out["moneyness_band"] = np.searchsorted(np.asarray(cuts, dtype=float), out["forward_moneyness"], side="right")
    return out


def walkforward_year_dir(args):
    model_name = f"{args.model_prefix}{args.test_year}"
    return Path(args.walkforward_output_root) / f"{args.test_year}_{model_name}"


def load_distillation_pairs(args, checkpoint):
    if args.distillation_pairs:
        path = Path(args.distillation_pairs)
    else:
        path = walkforward_year_dir(args) / f"{args.model_prefix}{args.test_year}_{checkpoint}_distillation_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Distillation pool not found: {path}. Run the 2022 empirical-distillation "
            "walk-forward outputs first, or pass --distillation-pairs."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + ["agent_delta", "bs_delta"] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df, path


def switching_paths(output_dir, model_name, checkpoint, config_name):
    stem = f"{model_name}_{checkpoint}_{config_name}"
    return {
        "model": output_dir / f"{stem}_switching_model.pkl",
        "manifest": output_dir / f"{stem}_manifest.json",
        "formula": output_dir / f"{stem}_piecewise_formula.txt",
        "band_summary": output_dir / f"{stem}_band_summary.csv",
    }


def band_paths(output_dir, model_name, checkpoint, config_name, band_i):
    stem = f"{model_name}_{checkpoint}_{config_name}_band{band_i:02d}"
    return {
        "model": output_dir / f"{stem}_pysr_model.pkl",
        "equations": output_dir / f"{stem}_pysr_equations.csv",
        "formula": output_dir / f"{stem}_symbolic_formula.txt",
        "manifest": output_dir / f"{stem}_manifest.json",
    }


def config_payload(args, model_name, checkpoint, config):
    return {
        "model_name": model_name,
        "checkpoint": int(checkpoint),
        "test_year": int(args.test_year),
        "validation_year": int(args.validation_year),
        "train_start_year": int(args.train_start_year),
        "train_end_year": int(args.train_end_year),
        "config_name": config.name,
        "cuts": list(config.cuts),
        "n_sr_samples": int(args.n_sr_samples),
        "niterations": int(args.niterations),
        "maxsize": int(args.maxsize),
        "populations": int(args.populations),
        "deterministic_search": bool(args.deterministic_search),
        "sample_allocation": args.sample_allocation,
        "seed": int(args.seed),
        "target": "agent_delta - bs_delta",
        "features": FEATURE_COLS,
    }


def band_config_payload(args, model_name, checkpoint, config, band_i, label, fit_rows, available_rows):
    payload = config_payload(args, model_name, checkpoint, config)
    payload.update(
        {
            "band": int(band_i),
            "band_label": str(label),
            "fit_rows": int(fit_rows),
            "available_rows": int(available_rows),
        }
    )
    return payload


def load_cached_switching_model(output_dir, args, model_name, checkpoint, config):
    paths = switching_paths(output_dir, model_name, checkpoint, config.name)
    if args.refit or not paths["model"].exists() or not paths["manifest"].exists():
        return None
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if manifest.get("config") != config_payload(args, model_name, checkpoint, config):
        return None
    try:
        with open(paths["model"], "rb") as f:
            model = pickle.load(f)
    except Exception:
        return None
    required = [paths["formula"], paths["band_summary"]]
    if any(not p.exists() for p in required):
        return None
    return model


def load_cached_band_model(output_dir, args, model_name, checkpoint, config, band_i, label, fit_rows, available_rows):
    paths = band_paths(output_dir, model_name, checkpoint, config.name, band_i)
    required = [paths["model"], paths["equations"], paths["formula"], paths["manifest"]]
    if args.refit or any(not p.exists() for p in required):
        return None
    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    expected = band_config_payload(
        args, model_name, checkpoint, config, band_i, label, fit_rows, available_rows
    )
    if manifest.get("config") != expected:
        return None
    try:
        with open(paths["model"], "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def sample_band_rows(band_df, n_rows, seed):
    n = min(int(n_rows), len(band_df))
    if n <= 0:
        return band_df.iloc[0:0].copy()
    return band_df.sample(n=n, random_state=int(seed)).copy()


def band_sample_sizes(df, config, args):
    bands = sorted(df["moneyness_band"].unique())
    total = min(int(args.n_sr_samples), len(df))
    if args.sample_allocation == "proportional":
        counts = df["moneyness_band"].value_counts().to_dict()
        raw = {b: max(1, int(round(total * counts[b] / len(df)))) for b in bands}
    else:
        per = max(1, int(math.ceil(total / len(bands))))
        raw = {b: per for b in bands}

    # Do not request more rows than a band has.  Any unused budget stays unused:
    # for this experiment balanced band coverage is more important than forcing
    # the total exactly to 50k by duplicating rows in sparse regimes.
    return {b: min(raw[b], int((df["moneyness_band"] == b).sum())) for b in bands}


def fit_one_switching_policy(distill_df, args, output_dir, model_name, checkpoint, config):
    cached = load_cached_switching_model(output_dir, args, model_name, checkpoint, config)
    if cached is not None:
        print(f"[switch-fit] {config.name}: loaded cached switching model")
        return cached

    df = add_band_column(distill_df.dropna(subset=FEATURE_COLS + ["agent_delta", "bs_delta"]), config.cuts)
    sizes = band_sample_sizes(df, config, args)
    labels = band_labels(config.cuts)
    band_models = []
    band_rows = []

    for band_i, label in enumerate(labels):
        band_df = df[df["moneyness_band"] == band_i].copy()
        if len(band_df) < args.min_band_rows:
            raise ValueError(
                f"{config.name} band {band_i} ({label}) has only {len(band_df)} rows, "
                f"below --min-band-rows={args.min_band_rows}. Adjust cuts or lower the threshold."
            )
        fit_df = sample_band_rows(band_df, sizes.get(band_i, 0), args.seed + 7919 * (band_i + 1))
        cached_band = load_cached_band_model(
            output_dir,
            args,
            model_name,
            checkpoint,
            config,
            band_i,
            label,
            len(fit_df),
            len(band_df),
        )
        if cached_band is None:
            X = fit_df[FEATURE_COLS].to_numpy(dtype=float)
            y = fit_df["agent_delta"].to_numpy(dtype=float) - fit_df["bs_delta"].to_numpy(dtype=float)

            print(
                f"[switch-fit] {config.name} band {band_i + 1}/{len(labels)} "
                f"{label}: fit_rows={len(fit_df)}, available_rows={len(band_df)}"
            )
            band_args = copy.copy(args)
            band_args.seed = int(args.seed + 1009 * (band_i + 1) + 37 * len(config.cuts))
            model = fit_pysr_model(X, y, band_args, ["m_fwd", "tau", "iv"], weights=None)

            paths = band_paths(output_dir, model_name, checkpoint, config.name, band_i)
            atomic_pickle_dump(model, paths["model"])
            atomic_to_csv(model.equations_, paths["equations"], index=False)
            atomic_write_text(paths["formula"], str(model.sympy()) + "\n")
            atomic_write_text(
                paths["manifest"],
                json.dumps(
                    {
                        "config": band_config_payload(
                            args,
                            model_name,
                            checkpoint,
                            config,
                            band_i,
                            label,
                            len(fit_df),
                            len(band_df),
                        ),
                        "created_utc": pd.Timestamp.utcnow().isoformat(),
                    },
                    indent=2,
                ),
            )
        else:
            print(
                f"[switch-fit] {config.name} band {band_i + 1}/{len(labels)} "
                f"{label}: loaded cached band model"
            )
            model = cached_band
            X = fit_df[FEATURE_COLS].to_numpy(dtype=float)
            y = fit_df["agent_delta"].to_numpy(dtype=float) - fit_df["bs_delta"].to_numpy(dtype=float)
        band_models.append(model)

        pred = np.asarray(model.predict(X), dtype=float)
        if pred.shape == ():
            pred = np.full(len(fit_df), float(pred))
        err = pred - y
        band_rows.append(
            {
                "config": config.name,
                "band": band_i,
                "band_label": label,
                "available_rows": len(band_df),
                "fit_rows": len(fit_df),
                "m_min": band_df["forward_moneyness"].min(),
                "m_max": band_df["forward_moneyness"].max(),
                "target_mean": float(np.mean(y)),
                "target_std": float(np.std(y)),
                "fit_mae_raw_residual": float(np.mean(np.abs(err))),
                "fit_rmse_raw_residual": float(np.sqrt(np.mean(err**2))),
                "best_equation": str(model.sympy()),
            }
        )

    switching_model = SwitchingPySRModel(config.cuts, band_models, labels)
    paths = switching_paths(output_dir, model_name, checkpoint, config.name)
    atomic_pickle_dump(switching_model, paths["model"])
    atomic_to_csv(pd.DataFrame(band_rows), paths["band_summary"], index=False)
    atomic_write_text(paths["formula"], piecewise_formula_text(config, labels, band_models))
    atomic_write_text(
        paths["manifest"],
        json.dumps(
            {
                "config": config_payload(args, model_name, checkpoint, config),
                "created_utc": pd.Timestamp.utcnow().isoformat(),
                "band_summary_file": str(paths["band_summary"]),
                "piecewise_formula_file": str(paths["formula"]),
            },
            indent=2,
        ),
    )
    return switching_model


def piecewise_formula_text(config, labels, band_models):
    lines = [
        f"{config.name}",
        "=" * len(config.name),
        "",
        "Raw symbolic outputs are BS-delta residuals.",
        "Traded delta = clip(bs_delta + residual, 0, 1).",
        "",
    ]
    for i, (label, model) in enumerate(zip(labels, band_models)):
        lines.append(f"Band {i}: {label}")
        lines.append(f"    residual = {model.sympy()}")
        lines.append("")
    return "\n".join(lines)


def model_entry(model):
    return {"model": model, "spec": SWITCH_SPEC, "bound_epsilon": 1e-4}


def evaluate_fidelity(models, df, label):
    frames = [
        fidelity_table(models, df, FEATURE_COLS, f"{label}_vs_canonical_agent", "agent_delta")
    ]
    if "actual_agent_delta" in df.columns:
        frames.append(
            fidelity_table(models, df, FEATURE_COLS, f"{label}_vs_actual_traded_agent", "actual_agent_delta")
        )
    return pd.concat(frames, ignore_index=True)


def bootstrap_outputs(episode_metrics, args, output_dir, stem, split):
    comparisons = [("agent_vs_bs", "agent", "bs")]
    for policy in sorted(p for p in episode_metrics["policy"].unique() if str(p).startswith("switch")):
        comparisons.append((f"{policy}_vs_agent", policy, "agent"))
        comparisons.append((f"{policy}_vs_bs", policy, "bs"))
    frames = []
    for name, left, right in comparisons:
        pair = paired_episode_table(episode_metrics, left, right)
        pair["comparison"] = name
        pair["left_policy"] = left
        pair["right_policy"] = right
        atomic_to_csv(pair, output_dir / f"{stem}_{name}_paired_episodes.csv", index=False)
        summary = summarize_bootstrap(name, pair, args)
        summary["split"] = split
        summary["left_policy"] = left
        summary["right_policy"] = right
        frames.append(summary)
    out = pd.concat(frames, ignore_index=True)
    atomic_to_csv(out, output_dir / f"{stem}_bootstrap_summary.csv", index=False)
    return out


def evaluate_split(env, actor, scaler, device, models, args, output_dir, checkpoint, split):
    split_year = args.validation_year if split == "validation" else args.test_year
    stem = f"{args.model_name}_{checkpoint}_{split}"

    print(f"[{split}] collecting target states for {split_year}")
    states = collect_split_target_states(env, actor, scaler, device, args, split=split, split_year=split_year)
    atomic_to_csv(states, output_dir / f"{stem}_target_states.csv", index=False)
    fid = evaluate_fidelity(models, states, f"{split}_{split_year}")
    fid = add_walkforward_metadata(fid, args, checkpoint, split=split)
    atomic_to_csv(fid, output_dir / f"{stem}_fidelity.csv", index=False)

    print(f"[{split}] trading switching formulas on {split_year}")
    trade_steps = trade_split_year(
        env,
        actor,
        scaler,
        device,
        models,
        FEATURE_COLS,
        args,
        smoothing_entries=None,
        split=split,
        split_year=split_year,
    )
    trade_steps = add_walkforward_metadata(trade_steps, args, checkpoint, split=split)
    atomic_to_csv(trade_steps, output_dir / f"{stem}_trade_steps.csv", index=False)

    rate_cache = build_rate_cache(args.cleaned_data_dir)
    episode_metrics = episode_metrics_from_steps(trade_steps, rate_cache)
    episode_metrics = add_walkforward_metadata(episode_metrics, args, checkpoint, split=split)
    atomic_to_csv(episode_metrics, output_dir / f"{stem}_episode_metrics.csv", index=False)

    print(f"[{split}] bootstrapping switching formulas on {split_year}")
    boot = bootstrap_outputs(episode_metrics, args, output_dir, stem, split)
    boot = add_walkforward_metadata(boot, args, checkpoint, split=split)
    atomic_to_csv(boot, output_dir / f"{stem}_bootstrap_summary.csv", index=False)
    return {"fidelity": fid, "episode_metrics": episode_metrics, "bootstrap": boot}


def run(args):
    set_random_seeds(args.seed)
    args.model_name = f"{args.model_prefix}{args.test_year}"
    args.target_year = int(args.test_year)
    args.validation_year = int(args.validation_year)
    args.train_end_year = int(args.test_year) - 2
    args.target_data_dir = str(args.walkforward_data_dir_template).format(year=args.test_year)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, selected_csv = infer_checkpoint(args.model_name, args.results_testing_dir)
    settings = load_settings_json(args.model_name)
    args.kappa = float(settings.get("kappa", 1.0))
    args.reward_exponent = float(settings.get("reward_exponent", 1.0))
    args.transaction_cost = float(settings.get("transaction_cost", 0.0))

    distill_df, distill_path = load_distillation_pairs(args, checkpoint)
    configs = [
        SwitchingConfig("switch2_moneyness_bs_delta_residual", parse_cut_list(args.two_band_cuts)),
        SwitchingConfig("switch3_moneyness_bs_delta_residual", parse_cut_list(args.three_band_cuts)),
    ]

    print(f"[setup] model={args.model_name}, checkpoint={checkpoint}, distillation={distill_path}")
    print(f"[setup] output_dir={output_dir}")

    models = {}
    manifest_rows = []
    for config in configs:
        start = time.time()
        model = fit_one_switching_policy(distill_df, args, output_dir, args.model_name, checkpoint, config)
        models[config.name] = model_entry(model)
        manifest_rows.append(
            {
                "policy": config.name,
                "cuts": ",".join(str(x) for x in config.cuts),
                "elapsed_minutes": (time.time() - start) / 60.0,
                **config_payload(args, args.model_name, checkpoint, config),
            }
        )
        atomic_to_csv(pd.DataFrame(manifest_rows), output_dir / f"{args.model_name}_{checkpoint}_switching_manifest.csv", index=False)

    # Training-pool fidelity, useful because this is what PySR actually fit.
    train_fid = evaluate_fidelity(models, distill_df, "distillation_train_only")
    train_fid = add_walkforward_metadata(train_fid, args, checkpoint)
    atomic_to_csv(train_fid, output_dir / f"{args.model_name}_{checkpoint}_distillation_fidelity.csv", index=False)

    old_data_dir = os.environ.get("DATA_DIR")
    os.environ["DATA_DIR"] = str(args.target_data_dir)
    try:
        from empirical_deep_hedging.include.env import Env

        actor, scaler, device = load_actor_and_scaler(args.model_name, checkpoint, settings, args.device)
        env = Env(settings)
        validation = evaluate_split(env, actor, scaler, device, models, args, output_dir, checkpoint, "validation")
        test = evaluate_split(env, actor, scaler, device, models, args, output_dir, checkpoint, "test")
    finally:
        if old_data_dir is None:
            os.environ.pop("DATA_DIR", None)
        else:
            os.environ["DATA_DIR"] = old_data_dir

    complete = {
        "status": "completed",
        "model_name": args.model_name,
        "checkpoint": int(checkpoint),
        "selected_csv": selected_csv,
        "output_dir": str(output_dir),
        "policies": list(models.keys()),
        "validation_bootstrap_rows": int(len(validation["bootstrap"])),
        "test_bootstrap_rows": int(len(test["bootstrap"])),
        "completed_utc": pd.Timestamp.utcnow().isoformat(),
    }
    atomic_write_text(output_dir / f"{args.model_name}_{checkpoint}_switching_complete.json", json.dumps(complete, indent=2))
    print(f"[done] switching experiment complete: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="2022 moneyness-regime symbolic policy test")
    parser.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    parser.add_argument("--test-year", type=int, default=DEFAULT_TEST_YEAR)
    parser.add_argument("--validation-year", type=int, default=DEFAULT_TEST_YEAR - 1)
    parser.add_argument("--train-start-year", type=int, default=2010)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--walkforward-output-root", default=DEFAULT_WALKFORWARD_OUTPUT_ROOT)
    parser.add_argument("--walkforward-data-dir-template", default=DEFAULT_WALKFORWARD_DATA_DIR_TEMPLATE)
    parser.add_argument("--cleaned-data-dir", default=DEFAULT_CLEANED_DATA_DIR)
    parser.add_argument("--results-testing-dir", default=DEFAULT_RESULTS_TESTING_DIR)
    parser.add_argument("--distillation-pairs", default="")

    parser.add_argument("--two-band-cuts", default="1.0")
    parser.add_argument("--three-band-cuts", default="0.95,1.05")
    parser.add_argument("--sample-allocation", choices=["equal", "proportional"], default="equal")
    parser.add_argument("--min-band-rows", type=int, default=1000)

    # Same PySR defaults as the empirical distillation experiments.
    parser.add_argument("--niterations", type=int, default=200)
    parser.add_argument("--n-sr-samples", dest="n_sr_samples", type=int, default=50000)
    parser.add_argument("--maxsize", type=int, default=25)
    parser.add_argument("--populations", type=int, default=15)
    parser.add_argument("--deterministic-search", action="store_true", default=True)
    parser.add_argument("--non-deterministic-search", dest="deterministic_search", action="store_false")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--refit", action="store_true")

    parser.add_argument("--device", default="cpu")
    # The canonical stock-position coordinate uses "bs" to mean -BS delta.
    parser.add_argument("--canonical-position", default="bs", choices=["bs", "zero"])
    parser.add_argument("--max-test-episodes", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--n-bootstrap", dest="n_bootstrap", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--include-pairwise-formula-bootstrap", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
