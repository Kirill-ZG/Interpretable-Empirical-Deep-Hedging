"""
Generate paper figures and tables from frozen walk-forward test results.

This script is intentionally separate from exploratory plotting workflows.  The
goal is to make the figures used in the paper reproducible, small, and readable.
The paper figures should show the central argument cleanly:

1. Ordinary terminal variance and downside terminal variance can disagree.
2. Reward and CVaR provide the objective-aligned / downside-tail view.

By default this script does not rerun the expensive bootstrap.  It reads the
cached summary in paper/figures/wf_metric_summary.csv and only redraws figures
and LaTeX tables.  Pass --recompute-summary when the underlying result CSVs or
bootstrap settings change.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


DEFAULT_PREFIX = "final_WF_exp1_k1_test"
DEFAULT_INPUT_DIR = Path("reproducibility_artifacts/paper_inputs")
DEFAULT_RESULTS_FOLDER = DEFAULT_INPUT_DIR / "results/testing"
DEFAULT_OUTPUT_DIR = Path("paper/figures")
DEFAULT_TABLE_DIR = Path("paper/tables")
DEFAULT_SUMMARY_CSV = DEFAULT_INPUT_DIR / "paper/figures/wf_metric_summary.csv"
DEFAULT_SEED_SUMMARY_DIR = DEFAULT_INPUT_DIR / "paper/figures"
DEFAULT_NEGATIVE_PNL_BREAKDOWN = DEFAULT_INPUT_DIR / "forensic_final_all_years_negative_pnl_state_breakdown.csv"
DEFAULT_DISTILLATION_RESULTS = DEFAULT_INPUT_DIR / "results/interpret_real_walkforward"
DEFAULT_FORENSIC_YEAR_SUMMARY = DEFAULT_INPUT_DIR / "forensic_final_all_years_year_summary.csv"
DEFAULT_FORENSIC_OPTION_CORRELATIONS = Path(
    DEFAULT_INPUT_DIR / "forensic_final_all_years_option_driver_correlations_by_year.csv"
)
DEFAULT_FORENSIC_OPTION_DYNAMICS = DEFAULT_INPUT_DIR / "forensic_final_all_years_option_dynamics_by_year.csv"
DEFAULT_FORENSIC_CLUSTER_TABLE = DEFAULT_INPUT_DIR / "forensic_final_all_years_cluster_table.csv"
DEFAULT_FORENSIC_GREEK_DECOMPOSITION = Path(
    DEFAULT_INPUT_DIR / "forensic_final_all_years_greek_decomposition_by_year.csv"
)
DEFAULT_FORENSIC_GREEK_DECOMPOSITION_STEPS = Path(
    DEFAULT_INPUT_DIR / "forensic_final_all_years_greek_decomposition_steps.csv"
)
DEFAULT_RHO_STEP_ASYMMETRY = DEFAULT_INPUT_DIR / "rho_variance_audit_all_years_final_theory_step_asymmetry.csv"
DEFAULT_LONG_HORIZON_RESULTS = DEFAULT_INPUT_DIR / "results/long_horizon"
DEFAULT_HULL_WHITE_RESULTS = DEFAULT_INPUT_DIR / "results/hull_white_check"
DEFAULT_HAIRCUT_RESULTS = DEFAULT_INPUT_DIR / "results/haircut_check"

DISTILLATION_POLICY_LABELS = {
    "uniform_bs_delta_residual": "Raw uniform",
    "smooth_uniform_bs_delta_residual": "Smooth uniform",
    "smooth_focus_bs_delta_residual": "Smooth focus",
}

# These are the independent final-style actor classes used only for the
# paper's random-seed robustness table.  The class named
# ``seed_final_WF_exp1_k1_test`` is intentionally omitted because in the
# current result folder it duplicates ``final_WF_exp1_k1_test`` point-for-point.
SEED_ROBUSTNESS_PREFIXES = [
    "final_WF_exp1_k1_test",
    "new_seed_final_WF_exp1_k1_test",
    "1_seed_final_WF_exp1_k1_test",
]


def tex_num(value: float, digits: int = 3, percent: bool = False) -> str:
    """Format table numbers so signs do not affect visual centering."""

    if pd.isna(value):
        return ""
    number = f"{abs(float(value)):.{digits}f}"
    if percent:
        number += r"\%"
    sign = r"\mathllap{-}" if float(value) < 0 else ""
    return rf"${sign}{number}$"


def set_y_label(ax: plt.Axes, label: str) -> None:
    """Apply the paper's y-axis label style consistently across figures.

    
    Figure 6 uses the target label style for the paper: readable prose labels,
    normal weight, and enough padding to avoid crowding tick labels.  Route all
    y-axis labels through this helper so figure-level differences do not creep
    in when individual panels are redrawn.
    """
    ax.set_ylabel(
        label,
        fontfamily="DejaVu Sans",
        fontstyle="normal",
        fontweight="normal",
        fontsize=9.8,
        labelpad=6,
    )


METRIC_INFO = {
    "reward": {
        "label": r"$\Delta$ accumulated reward",
        "short": "Reward",
        "higher_is_better": True,
    },
    "cvar": {
        "label": r"$\Delta$ CVaR 5\% of terminal P\&L",
        "short": "CVaR 5%",
        "higher_is_better": True,
    },
    "mean_pnl": {
        "label": r"$\Delta$ mean terminal P\&L",
        "short": "Mean terminal P&L",
        "higher_is_better": True,
    },
    "log_downside_variance": {
        "label": r"$\log(\mathrm{DownVar}_{Agent}/\mathrm{DownVar}_{BS})$",
        "short": "Log Downside Variance",
        "higher_is_better": False,
    },
    "log_variance": {
        "label": r"$\log(\mathrm{Var}_{Agent}/\mathrm{Var}_{BS})$",
        "short": "Log Variance",
        "higher_is_better": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Root folder containing saved inputs used to regenerate paper figures and tables.",
    )
    parser.add_argument("--results-folder", type=Path, default=DEFAULT_RESULTS_FOLDER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help="Cached walk-forward metric summary used unless --recompute-summary is passed.",
    )
    parser.add_argument(
        "--seed-summary-dir",
        type=Path,
        default=DEFAULT_SEED_SUMMARY_DIR,
        help="Folder containing cached seed-robustness summaries.",
    )
    parser.add_argument(
        "--negative-pnl-breakdown",
        type=Path,
        default=DEFAULT_NEGATIVE_PNL_BREAKDOWN,
        help=(
            "Cached forensic table used for the negative daily-P&L state breakdown. "
            "This is a lightweight paper-table input, not a bootstrap input."
        ),
    )
    parser.add_argument(
        "--distillation-results",
        type=Path,
        default=DEFAULT_DISTILLATION_RESULTS,
        help="Folder containing cached walk-forward symbolic-distillation aggregate CSVs.",
    )
    parser.add_argument(
        "--forensic-year-summary",
        type=Path,
        default=DEFAULT_FORENSIC_YEAR_SUMMARY,
        help=(
            "Cached year-level forensic decomposition used by the regime-fragility "
            "section.  This is lightweight and does not rerun model testing."
        ),
    )
    parser.add_argument(
        "--forensic-option-correlations",
        type=Path,
        default=DEFAULT_FORENSIC_OPTION_CORRELATIONS,
        help=(
            "Cached option-driver correlation diagnostics used by the "
            "regime-fragility section."
        ),
    )
    parser.add_argument(
        "--forensic-option-dynamics",
        type=Path,
        default=DEFAULT_FORENSIC_OPTION_DYNAMICS,
        help="Cached option-direction diagnostics used by the 2022 mechanism figure.",
    )
    parser.add_argument(
        "--forensic-cluster-table",
        type=Path,
        default=DEFAULT_FORENSIC_CLUSTER_TABLE,
        help=(
            "Cached cluster-level forensic table used by the 2017 mechanism table. "
            "This is a lightweight diagnostic input and does not rerun testing."
        ),
    )
    parser.add_argument(
        "--forensic-greek-decomposition",
        type=Path,
        default=DEFAULT_FORENSIC_GREEK_DECOMPOSITION,
        help="Cached Greek-style option-P&L decomposition used by the 2023 mechanism figure.",
    )
    parser.add_argument(
        "--forensic-greek-decomposition-steps",
        type=Path,
        default=DEFAULT_FORENSIC_GREEK_DECOMPOSITION_STEPS,
        help=(
            "Cached step-level Greek-style option-P&L decomposition used by "
            "the 2022 vega-parachute diagnostic."
        ),
    )
    parser.add_argument(
        "--rho-step-asymmetry",
        type=Path,
        default=DEFAULT_RHO_STEP_ASYMMETRY,
        help=(
            "Cached step-level up/down diagnostic table.  The table is produced by "
            "the exploratory audit script, but the paper script only reads it."
        ),
    )
    parser.add_argument(
        "--long-horizon-results",
        type=Path,
        default=DEFAULT_LONG_HORIZON_RESULTS,
        help=(
            "Cached long-horizon retesting results.  Pass either the root folder "
            "results/long_horizon or a direct long_horizon_pair_bootstrap_summary.csv path."
        ),
    )
    parser.add_argument(
        "--hull-white-results",
        type=Path,
        default=DEFAULT_HULL_WHITE_RESULTS,
        help=(
            "Cached Hull-White benchmark outputs. Pass either results/hull_white_check, "
            "results/hull_white_check/<prefix>, or a direct hull_white_test_bootstrap_all.csv path."
        ),
    )
    parser.add_argument(
        "--haircut-results",
        type=Path,
        default=DEFAULT_HAIRCUT_RESULTS,
        help=(
            "Cached scalar-delta-haircut benchmark outputs. Pass either results/haircut_check, "
            "results/haircut_check/<prefix>, or a direct haircut_test_bootstrap_all.csv path."
        ),
    )
    parser.add_argument(
        "--recompute-summary",
        action="store_true",
        help="Rerun the expensive two-stage bootstrap instead of using the cached summary CSV.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def extract_year(path: Path, prefix: str) -> int | None:
    match = re.match(rf"{re.escape(prefix)}(\d{{4}})_\d+\.csv$", path.name)
    if not match:
        return None
    return int(match.group(1))


def load_episode_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"{path} is empty")

    # One row is one hedge interval.  The metrics in this paper section are
    # terminal-episode metrics, so first aggregate daily/interval quantities to
    # the episode level.  The bootstrap cluster is the episode start date.
    ep = (
        raw.groupby("episode")
        .agg(
            a_pnl=("A PnL", "sum"),
            b_pnl=("B PnL", "sum"),
            a_reward=("A Reward", "sum"),
            b_reward=("B Reward", "sum"),
            start_date=("Date", "min"),
        )
        .reset_index()
    )

    # The paper metrics report terminal PnL in percent units before calculating
    # variance, CVaR, and mean differences.  Keep the convention identical.
    ep["a_pnl"] *= 100.0
    ep["b_pnl"] *= 100.0
    return ep


def downside_second_moment(values: np.ndarray) -> float:
    downside = np.where(values < 0.0, values, 0.0)
    return float(np.sum(downside**2) / len(downside))


def cvar_5(values: np.ndarray) -> float:
    n_tail = max(1, int(len(values) * 0.05))
    return float(np.mean(np.partition(values, n_tail - 1)[:n_tail]))


def calculate_metrics(sample: pd.DataFrame) -> dict[str, float]:
    a = sample["a_pnl"].to_numpy()
    b = sample["b_pnl"].to_numpy()

    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    down_a = downside_second_moment(a)
    down_b = downside_second_moment(b)

    return {
        "reward": float(np.mean(sample["a_reward"] - sample["b_reward"])),
        "cvar": cvar_5(a) - cvar_5(b),
        "mean_pnl": float(np.mean(a - b)),
        "log_variance": float(np.log(var_a / var_b)) if var_a > 0 and var_b > 0 else np.nan,
        "log_downside_variance": (
            float(np.log(down_a / down_b)) if down_a > 0 and down_b > 0 else np.nan
        ),
    }


def two_stage_bootstrap(
    ep: pd.DataFrame,
    n_bootstrap: int,
    confidence_level: float,
    rng: np.random.Generator,
) -> dict[str, dict[str, float]]:
    point = calculate_metrics(ep)
    clusters = sorted(ep["start_date"].unique())
    if not clusters:
        raise ValueError("No bootstrap clusters found")

    cluster_frames = [ep[ep["start_date"] == cluster] for cluster in clusters]
    draws = {metric: np.empty(n_bootstrap, dtype=float) for metric in METRIC_INFO}

    # Two-stage bootstrap:
    #   1. resample non-overlapping 21-day calendar clusters;
    #   2. inside each selected cluster, resample option episodes.
    for i in range(n_bootstrap):
        sampled_cluster_idx = rng.integers(0, len(cluster_frames), size=len(cluster_frames))
        sampled_parts = []
        for idx in sampled_cluster_idx:
            cluster = cluster_frames[idx]
            row_idx = rng.integers(0, len(cluster), size=len(cluster))
            sampled_parts.append(cluster.iloc[row_idx])
        sample = pd.concat(sampled_parts, ignore_index=True)
        metrics = calculate_metrics(sample)
        for metric, value in metrics.items():
            draws[metric][i] = value

    out: dict[str, dict[str, float]] = {}
    for metric, values in draws.items():
        values = values[np.isfinite(values)]
        alpha = (1.0 - confidence_level) / 2.0
        lb, ub = np.quantile(values, [alpha, 1.0 - alpha])
        lb90, ub90 = np.quantile(values, [0.05, 0.95])
        lb95, ub95 = np.quantile(values, [0.025, 0.975])
        lb99, ub99 = np.quantile(values, [0.005, 0.995])
        center = point[metric]
        out[metric] = {
            "center": center,
            "lower": float(lb),
            "upper": float(ub),
            "lower90": float(lb90),
            "upper90": float(ub90),
            "lower95": float(lb95),
            "upper95": float(ub95),
            "lower99": float(lb99),
            "upper99": float(ub99),
            "significant": bool(lb > 0.0 or ub < 0.0),
            "sig90": bool(lb90 > 0.0 or ub90 < 0.0),
            "sig95": bool(lb95 > 0.0 or ub95 < 0.0),
            "sig99": bool(lb99 > 0.0 or ub99 < 0.0),
        }
    return out


def collect_summary(args: argparse.Namespace) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    files = sorted(args.results_folder.glob(f"{args.prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No result files found for prefix {args.prefix}")

    for path in files:
        year = extract_year(path, args.prefix)
        if year is None:
            continue
        print(f"[paper figures] processing {path.name}")
        ep = load_episode_data(path)
        boot = two_stage_bootstrap(ep, args.n_bootstrap, args.confidence_level, rng)
        for metric, stats in boot.items():
            info = METRIC_INFO[metric]
            center = stats["center"]
            significant = stats["significant"]
            if significant:
                outperforms = (
                    center > 0.0 if info["higher_is_better"] else center < 0.0
                )
            else:
                outperforms = False
            rows.append(
                {
                    "year": year,
                    "metric": metric,
                    "label": info["short"],
                    "center": center,
                    "lower": stats["lower"],
                    "upper": stats["upper"],
                    "significant": significant,
                    "sig90": stats["sig90"],
                    "sig95": stats["sig95"],
                    "sig99": stats["sig99"],
                    "lower90": stats["lower90"],
                    "upper90": stats["upper90"],
                    "lower95": stats["lower95"],
                    "upper95": stats["upper95"],
                    "lower99": stats["lower99"],
                    "upper99": stats["upper99"],
                    "outperforms": outperforms,
                    "n_episodes": len(ep),
                    "n_clusters": ep["start_date"].nunique(),
                }
            )
    summary = pd.DataFrame(rows).sort_values(["metric", "year"]).reset_index(drop=True)
    return summary


def plot_metric_pair(
    summary: pd.DataFrame,
    metrics: list[str],
    filename: Path,
    title: str,
) -> None:
    years = sorted(summary["year"].unique())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(7.2, 5.15), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = summary[summary["metric"] == metric].sort_values("year")
        info = METRIC_INFO[metric]

        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.75, zorder=1)
        if 2022 in years or 2023 in years:
            ax.axvspan(2021.5, 2023.5, color="#d9d9d9", alpha=0.25, zorder=0)

        for _, row in data.iterrows():
            center = row["center"]
            yerr = np.array(
                [[center - row["lower"]], [row["upper"] - center]], dtype=float
            )
            if not row["significant"]:
                color = "#6f7f8f"
                marker = "o"
                face = "white"
                zorder = 3
            elif row["outperforms"]:
                color = "#1b8a5a"
                marker = "^" if center > 0 else "v"
                face = color
                zorder = 4
            else:
                color = "#b33a3a"
                marker = "^" if center > 0 else "v"
                face = color
                zorder = 4

            ax.errorbar(
                row["year"],
                center,
                yerr=yerr,
                fmt=marker,
                color=color,
                mfc=face,
                mec=color,
                ms=7,
                capsize=3.5,
                elinewidth=1.2,
                zorder=zorder,
            )

        ax.set_title(info["short"], fontweight="bold", pad=5)
        if metric in ["log_downside_variance", "log_variance"]:
            set_y_label(ax, "Log ratio")
        else:
            set_y_label(ax, "Agent - BS")
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", ls=":", alpha=0.45)

    not_sig = mlines.Line2D(
        [], [], color="#6f7f8f", marker="o", mfc="white", ls="None",
        label="95% CI includes zero",
    )
    sig_out = mlines.Line2D(
        [], [], color="#1b8a5a", marker="o", mfc="#1b8a5a", ls="None",
        label="Significant agent outperformance",
    )
    sig_under = mlines.Line2D(
        [], [], color="#b33a3a", marker="o", mfc="#b33a3a", ls="None",
        label="Significant agent underperformance",
    )
    fig.legend(
        handles=[not_sig, sig_out, sig_under],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def write_metric_table(summary: pd.DataFrame, output_path: Path) -> None:
    def stars(row: pd.Series) -> str:
        if row["sig99"]:
            return r"\sym{***}"
        if row["sig95"]:
            return r"\sym{**}"
        if row["sig90"]:
            return r"\sym{*}"
        return ""

    def fmt(row: pd.Series) -> str:
        return tex_num(row["center"]) + stars(row)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"Year & Reward & CVaR 5\% & Log Downside Variance & Log Variance \\",
        r"\midrule",
    ]
    for year in sorted(summary["year"].unique()):
        values = {}
        for metric in ["reward", "cvar", "log_downside_variance", "log_variance"]:
            data = summary[(summary["year"] == year) & (summary["metric"] == metric)].iloc[0]
            values[metric] = fmt(data)
        lines.append(
            f"{year} & {values['reward']} & {values['cvar']} & "
            f"{values['log_downside_variance']} & {values['log_variance']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_delta_summary(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    files = sorted(args.results_folder.glob(f"{args.prefix}*.csv"))
    if not files:
        raise FileNotFoundError(f"No result files found for prefix {args.prefix}")

    for path in files:
        year = extract_year(path, args.prefix)
        if year is None:
            continue
        raw = pd.read_csv(path)
        agent_delta = -raw["A Pos"].astype(float)
        bs_delta = -raw["B Pos"].astype(float)
        delta_gap = agent_delta - bs_delta
        rows.append(
            {
                "year": year,
                "steps": len(raw),
                "episodes": raw["episode"].nunique(),
                "mean_agent_delta": agent_delta.mean(),
                "mean_bs_delta": bs_delta.mean(),
                "mean_delta_gap": delta_gap.mean(),
                "median_delta_gap": delta_gap.median(),
                "underhedged_share": (delta_gap < 0.0).mean(),
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def write_delta_summary_table(delta_summary: pd.DataFrame, output_path: Path) -> None:
    def fmt(value: float) -> str:
        return tex_num(value)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"Year & Agent Delta & BS Delta & Agent--BS & Underhedged Share \\",
        r"\midrule",
    ]
    for _, row in delta_summary.iterrows():
        lines.append(
            f"{int(row['year'])} & {fmt(row['mean_agent_delta'])} & "
            f"{fmt(row['mean_bs_delta'])} & {fmt(row['mean_delta_gap'])} & "
            rf"{tex_num(100.0 * row['underhedged_share'], digits=1)}\% \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_delta_gap_surface(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for path in sorted(args.results_folder.glob(f"{args.prefix}*.csv")):
        year = extract_year(path, args.prefix)
        if year is None:
            continue
        raw = pd.read_csv(path)
        frames.append(
            pd.DataFrame(
                {
                    "year": year,
                    "m_fwd": raw["forward S/K"].astype(float),
                    "iv": raw["v"].astype(float),
                    "delta_gap": -raw["A Pos"].astype(float) + raw["B Pos"].astype(float),
                }
            )
        )
    if not frames:
        raise FileNotFoundError(f"No result files found for prefix {args.prefix}")
    data = pd.concat(frames, ignore_index=True)

    # Fixed economically interpretable bins make the plot reproducible and easy
    # to read.  They also avoid changing the surface when the sample changes.
    m_bins = [0.0, 0.90, 0.95, 1.05, 1.10, np.inf]
    m_labels = ["<0.90", "0.90-0.95", "0.95-1.05", "1.05-1.10", ">1.10"]
    iv_bins = [0.0, 0.12, 0.18, 0.25, np.inf]
    iv_labels = ["<12%", "12-18%", "18-25%", ">25%"]
    data["m_bin"] = pd.cut(data["m_fwd"], m_bins, labels=m_labels, include_lowest=True)
    data["iv_bin"] = pd.cut(data["iv"], iv_bins, labels=iv_labels, include_lowest=True)

    surface = (
        data.groupby(["iv_bin", "m_bin"], observed=False)
        .agg(mean_delta_gap=("delta_gap", "mean"), steps=("delta_gap", "size"))
        .reset_index()
    )
    return surface


def plot_delta_gap_surface(surface: pd.DataFrame, output_path: Path) -> None:
    m_labels = ["<0.90", "0.90-0.95", "0.95-1.05", "1.05-1.10", ">1.10"]
    iv_labels = ["<12%", "12-18%", "18-25%", ">25%"]
    matrix = (
        surface.pivot(index="iv_bin", columns="m_bin", values="mean_delta_gap")
        .reindex(index=iv_labels, columns=m_labels)
    )
    counts = (
        surface.pivot(index="iv_bin", columns="m_bin", values="steps")
        .reindex(index=iv_labels, columns=m_labels)
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    # Keep empty and single-observation bins grey, but include other sparse
    # non-empty bins in the color scale so their estimates remain visible while
    # the counts warn readers.
    matrix_for_plot = matrix.mask(counts <= 1)
    values = matrix_for_plot.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    bound = max(0.01, float(np.nanmax(np.abs(finite))) if finite.size else 0.01)
    norm = mcolors.TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0f0f0")
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    for i, iv_label in enumerate(iv_labels):
        for j, m_label in enumerate(m_labels):
            value = matrix.loc[iv_label, m_label]
            count = counts.loc[iv_label, m_label]
            if count <= 0 or pd.isna(value):
                text = "n=0"
            elif count == 1:
                text = "n=1"
            elif count < 100:
                text = f"{value:.3f}\n(n={int(count)})"
            else:
                text = f"{value:.3f}\n({int(count)})"
            ax.text(j, i, text, ha="center", va="center", fontsize=9)

    ax.set_xticks(range(len(m_labels)))
    ax.set_xticklabels(m_labels)
    ax.set_yticks(range(len(iv_labels)))
    ax.set_yticklabels(iv_labels)
    ax.set_xlabel(r"Forward moneyness $F/K$")
    set_y_label(ax, "Implied volatility")
    ax.set_title("Mean Agent - BS Delta Gap", fontweight="bold", pad=8)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Agent - BS delta")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def collect_negative_pnl_state_summary(args: argparse.Namespace) -> pd.DataFrame:
    if not args.negative_pnl_breakdown.exists():
        raise FileNotFoundError(
            f"Missing cached negative-P&L breakdown: {args.negative_pnl_breakdown}"
        )

    raw = pd.read_csv(args.negative_pnl_breakdown)
    required = {"portfolio", "state3", "steps", "sum_pnl"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(
            f"{args.negative_pnl_breakdown} is missing required columns: {sorted(missing)}"
        )

    raw["steps"] = raw["steps"].astype(float)
    raw["sum_pnl"] = raw["sum_pnl"].astype(float)
    agg = (
        raw.groupby(["portfolio", "state3"], as_index=False)
        .agg(steps=("steps", "sum"), sum_pnl=("sum_pnl", "sum"))
    )
    agg["mean_negative_step_pnl"] = agg["sum_pnl"] / agg["steps"]

    # The forensic input contains only negative daily-P&L observations.  A loss
    # share is therefore the absolute loss in a state divided by all negative
    # daily losses for that portfolio.
    totals = -agg.groupby("portfolio")["sum_pnl"].transform("sum")
    agg["loss_share"] = -agg["sum_pnl"] / totals

    wide = agg.pivot(index="state3", columns="portfolio")
    wide.columns = [f"{field}_{portfolio}" for field, portfolio in wide.columns]
    wide = wide.reset_index()
    for col in [
        "loss_share_agent",
        "loss_share_bs",
        "mean_negative_step_pnl_agent",
        "mean_negative_step_pnl_bs",
        "steps_agent",
        "steps_bs",
    ]:
        if col not in wide:
            wide[col] = np.nan

    wide["combined_loss_share"] = (
        wide["loss_share_agent"].fillna(0.0) + wide["loss_share_bs"].fillna(0.0)
    )
    return (
        wide.sort_values("combined_loss_share", ascending=False)
        .reset_index(drop=True)
    )


def pretty_state_label(state: str) -> str:
    parts = state.split("/")
    labels = []
    for part in parts:
        if part.startswith("iv_"):
            labels.append("IV " + part.removeprefix("iv_").replace("_", " ").title())
        elif part.startswith("spot_"):
            labels.append("S " + part.removeprefix("spot_").replace("_", " ").title())
        elif part.startswith("option_"):
            labels.append("Opt. " + part.removeprefix("option_").replace("_", " ").title())
        else:
            labels.append(part.replace("_", " ").title())
    return " / ".join(labels)


def write_negative_pnl_state_table(
    state_summary: pd.DataFrame, output_path: Path, n_rows: int = 6
) -> None:
    def pct(value: float) -> str:
        return rf"{tex_num(100.0 * value, digits=1)}\%"

    def pnl(value: float) -> str:
        return tex_num(value)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\tabcolsep}{5.0pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"State & Agent Share & BS Share & Agent Mean Loss & BS Mean Loss \\",
        r"\midrule",
    ]
    for _, row in state_summary.head(n_rows).iterrows():
        lines.append(
            f"{pretty_state_label(row['state3'])} & "
            f"{pct(row['loss_share_agent'])} & {pct(row['loss_share_bs'])} & "
            f"{pnl(row['mean_negative_step_pnl_agent'])} & "
            f"{pnl(row['mean_negative_step_pnl_bs'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_negative_pnl_ranked_table(
    state_summary: pd.DataFrame, output_path: Path, n_rows: int = 5
) -> None:
    def pct(value: float) -> str:
        return rf"{tex_num(100.0 * value, digits=1)}\%"

    def top_states(portfolio: str) -> pd.DataFrame:
        share_col = f"loss_share_{portfolio}"
        return (
            state_summary[["state3", share_col]]
            .dropna()
            .sort_values(share_col, ascending=False)
            .head(2)
            .reset_index(drop=True)
        )

    agent = top_states("agent")
    bs = top_states("bs")

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"Rank & Agent Loss State & Share & BS Loss State & Share \\",
        r"\midrule",
    ]
    for i in range(2):
        lines.append(
            f"{i + 1} & {pretty_state_label(agent.loc[i, 'state3'])} & "
            f"{pct(agent.loc[i, 'loss_share_agent'])} & "
            f"{pretty_state_label(bs.loc[i, 'state3'])} & "
            f"{pct(bs.loc[i, 'loss_share_bs'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\endgroup",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_regime_fragility_summary(args: argparse.Namespace) -> pd.DataFrame:
    """Build the compact all-years diagnostic table for the fragility section.

    The point of this table is not to introduce another bootstrap exercise.
    It collects already-cached forensic diagnostics that explain why the same
    asymmetric policy can look good under reward/downside variance and fragile
    under ordinary variance.  The table deliberately spans every test year, so
    the 2022 and 2023 mechanisms are compared against the full walk-forward
    panel rather than against a hand-picked neighbor.
    """
    for path in [
        args.forensic_year_summary,
        args.forensic_option_correlations,
        args.rho_step_asymmetry,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing cached regime diagnostic input: {path}")

    year_summary = pd.read_csv(args.forensic_year_summary)
    option_corr = pd.read_csv(args.forensic_option_correlations)
    step = pd.read_csv(args.rho_step_asymmetry)
    step = step[step["prefix"].eq(args.prefix)].copy()

    # Keep separate up-day and down-day rows.  The reward is step-wise and
    # asymmetric, so the sign of the spot move is central to the mechanism.
    step_wide = (
        step[step["condition"].isin(["up_day", "down_day"])]
        .pivot(index="year", columns="condition", values="mean_step_diff_pnl_100")
        .rename(
            columns={
                "down_day": "agent_minus_bs_down_day",
                "up_day": "agent_minus_bs_up_day",
            }
        )
        .reset_index()
    )

    out = (
        year_summary[
            [
                "year",
                "var_a",
                "var_b",
                "log_var_ratio",
                "log_downside_ratio",
                "mean_diff_reward",
                "rho_up_mean",
                "rho_down_mean",
                "mean_delta_gap",
                "underhedged_share",
            ]
        ]
        .merge(
            option_corr[
                [
                    "year",
                    "corr_option_pnl_spot_return",
                    "corr_option_pnl_iv_change",
                    "corr_spot_return_iv_change",
                ]
            ],
            on="year",
            how="left",
        )
        .merge(step_wide, on="year", how="left")
        .sort_values("year")
        .reset_index(drop=True)
    )
    return out


def write_regime_fragility_table(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a narrow LaTeX table with the diagnostics used in the text."""

    def num(value: float) -> str:
        return tex_num(value)

    def pct(value: float) -> str:
        return rf"{tex_num(100.0 * value, digits=1)}\%"

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{4.8pt}",
        r"\begin{tabular}{@{}cccccccc@{}}",
        r"\toprule",
        (
            r"Year & Log Var. & BS Var. & $\rho(S,\sigma)$ & "
            r"$\rho(C,\sigma)$ & Down-Day $\Delta$PnL & Up-Day $\Delta$PnL & Underhedged \\"
        ),
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{int(row['year'])} & "
            f"{num(row['log_var_ratio'])} & "
            f"{num(row['var_b'])} & "
            f"{num(row['corr_spot_return_iv_change'])} & "
            f"{num(row['corr_option_pnl_iv_change'])} & "
            f"{num(row['agent_minus_bs_down_day'])} & "
            f"{num(row['agent_minus_bs_up_day'])} & "
            f"{pct(row['underhedged_share'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_regime_fragility_diagnostics(summary: pd.DataFrame, output_path: Path) -> None:
    """Create the compact multi-panel diagnostic figure for regime fragility.

    The figure is intentionally descriptive.  IV is an implied quantity inferred
    from option prices, so the paper should avoid causal language such as
    "IV caused option prices."  The plotted correlations and decompositions are
    evidence about realized co-movement and hedge-accounting mechanisms.
    """
    years = summary["year"].to_numpy()
    highlight = "#d9d9d9"

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), sharex=True)
    ax = axes[0, 0]
    ax.plot(years, summary["var_b"], color="#2f5d8c", marker="o", lw=1.4, label="BS")
    ax.plot(years, summary["var_a"], color="#b36b2c", marker="s", lw=1.4, label="Agent")
    ax.set_title("Terminal P&L Variance", fontweight="bold", pad=5)
    set_y_label(ax, "Variance")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[0, 1]
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.plot(
        years,
        summary["corr_spot_return_iv_change"],
        color="#1b8a5a",
        marker="o",
        lw=1.4,
        label=r"$\rho(S,\sigma)$",
    )
    ax.plot(
        years,
        summary["corr_option_pnl_iv_change"],
        color="#8c4f9f",
        marker="s",
        lw=1.4,
        label=r"$\rho(C,\sigma)$",
    )
    ax.set_title("Volatility-Channel Co-Movement", fontweight="bold", pad=5)
    set_y_label(ax, "Correlation")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1, 0]
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.plot(
        years,
        summary["agent_minus_bs_down_day"],
        color="#b33a3a",
        marker="v",
        lw=1.4,
        label="Spot-down days",
    )
    ax.plot(
        years,
        summary["agent_minus_bs_up_day"],
        color="#1b8a5a",
        marker="^",
        lw=1.4,
        label="Spot-up days",
    )
    ax.set_title("Step-Level Agent-BS P&L", fontweight="bold", pad=5)
    set_y_label(ax, "Mean daily difference")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1, 1]
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.plot(
        years,
        summary["log_var_ratio"],
        color="#b36b2c",
        marker="s",
        lw=1.4,
        label="Ordinary variance",
    )
    ax.plot(
        years,
        summary["log_downside_ratio"],
        color="#2f5d8c",
        marker="o",
        lw=1.4,
        label="Downside variance",
    )
    ax.set_title("Symmetric vs Downside Dispersion", fontweight="bold", pad=5)
    set_y_label(ax, "Log ratio")
    ax.legend(frameon=False, loc="lower left")

    for ax in axes.ravel():
        # Lightly mark the years discussed in the fragility text.  This is a
        # visual guide only; the full time series remains visible in every panel.
        ax.axvspan(2016.5, 2017.5, color=highlight, alpha=0.18, zorder=0)
        ax.axvspan(2021.5, 2023.5, color=highlight, alpha=0.22, zorder=0)
        ax.set_xticks(years)
        ax.grid(True, axis="y", ls=":", alpha=0.42)
    for ax in axes[-1, :]:
        ax.set_xlabel("Test year")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_step_direction_summary(args: argparse.Namespace) -> pd.DataFrame:
    """Load cached step-level up/down diagnostics for the traded main agent."""
    if not args.rho_step_asymmetry.exists():
        raise FileNotFoundError(f"Missing step diagnostic file: {args.rho_step_asymmetry}")
    step = pd.read_csv(args.rho_step_asymmetry)
    step = step[step["prefix"].eq(args.prefix)].copy()
    if step.empty:
        raise ValueError(f"No step diagnostics found for prefix {args.prefix}")

    rows = []
    for year, group in step.groupby("year"):
        all_row = group[group["condition"].eq("all_steps")].iloc[0]
        down_row = group[group["condition"].eq("down_day")].iloc[0]
        up_row = group[group["condition"].eq("up_day")].iloc[0]
        rows.append(
            {
                "year": int(year),
                "all_steps": float(all_row["steps"]),
                "down_steps": float(down_row["steps"]),
                "up_steps": float(up_row["steps"]),
                "down_share": float(down_row["steps"] / all_row["steps"]),
                "down_agent_minus_bs": float(down_row["mean_step_diff_pnl_100"]),
                "up_agent_minus_bs": float(up_row["mean_step_diff_pnl_100"]),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def plot_2022_reward_failure(args: argparse.Namespace, output_path: Path) -> None:
    """Draw the two diagnostics used in the 2022 reward-failure subsection."""
    if not args.forensic_option_dynamics.exists():
        raise FileNotFoundError(
            f"Missing option-dynamics file: {args.forensic_option_dynamics}"
        )
    step = load_step_direction_summary(args)
    dynamics = pd.read_csv(args.forensic_option_dynamics)
    down_iv_up = (
        dynamics[dynamics["pattern"].eq("spot_down_iv_up")]
        .sort_values("year")
        .reset_index(drop=True)
    )

    years = step["year"].to_numpy()
    colors = ["#b36b2c" if year == 2022 else "#8c9aa8" for year in years]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    ax.bar(years, 100.0 * step["down_share"], color=colors, width=0.72)
    target_down = step[step["year"].eq(2022)].iloc[0]
    ax.text(
        2022,
        100.0 * target_down["down_share"] + 1.2,
        f"{100.0 * target_down['down_share']:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_title("Spot-Down Hedge Intervals", fontweight="bold", pad=5)
    set_y_label(ax, "Share of intervals (%)")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", ls=":", alpha=0.42)

    ax = axes[1]
    colors = ["#b36b2c" if year == 2022 else "#8c9aa8" for year in down_iv_up["year"]]
    ax.bar(
        down_iv_up["year"],
        down_iv_up["mean_option_pnl"],
        color=colors,
        width=0.72,
    )
    target_parachute = down_iv_up[down_iv_up["year"].eq(2022)].iloc[0]
    ax.text(
        2022,
        target_parachute["mean_option_pnl"] - 0.035,
        f"{target_parachute['mean_option_pnl']:.3f}",
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.7)
    ax.set_title("Spot Down / IV Up Call Revaluation", fontweight="bold", pad=5)
    set_y_label(ax, "Mean option P&L")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", ls=":", alpha=0.42)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def collect_2022_reward_mechanism(args: argparse.Namespace) -> pd.DataFrame:
    """Build the step-level diagnostics used in the 2022 subsection.

    
    This is intentionally rebuilt from the frozen traded testing CSVs instead
    of from aggregate forensic summary tables.  The subsection argues about the
    reward failure under the exact traded P&L convention, so the safest object
    is the raw step record that contains A/B rewards, A/B P&Ls, option prices,
    hedge positions, and the observed implied volatility path.  This is a
    lightweight pass over cached CSVs only; it does not rerun training, testing,
    or bootstrap inference.
    """
    rows = []
    for path in sorted(args.results_folder.glob(f"{args.prefix}*.csv")):
        year = extract_year(path, args.prefix)
        if year is None:
            continue

        raw = pd.read_csv(path)
        if raw.empty:
            continue

    # Keep the accounting in the same units as the paper metrics: daily P&L is
        # multiplied by 100 before aggregation.  The reward columns are already
        # in the environment's reward units and therefore are not rescaled.
        sort_cols = ["episode", "Date"]
        if "DateEnd" in raw.columns:
            sort_cols.append("DateEnd")
        raw = raw.sort_values(sort_cols).copy()
        raw["a_pnl_100"] = raw["A PnL"] * 100.0
        raw["b_pnl_100"] = raw["B PnL"] * 100.0
        raw["diff_reward"] = raw["A Reward"] - raw["B Reward"]
        raw["option_pnl_100"] = (raw["P0"] - raw["P-1"]) * 100.0

        # A PnL = option leg + underlying hedge leg + transaction cost.  For
        # the paper mechanism we want to ask whether the stock/index hedge
        # earns enough on down moves to offset the call-price loss.  The
        # reported hedge leg below is net of transaction costs, matching the
        # realized portfolio P&L seen by the reward.
        raw["agent_hedge_pnl_100"] = raw["a_pnl_100"] - raw["option_pnl_100"]
        raw["bs_hedge_pnl_100"] = raw["b_pnl_100"] - raw["option_pnl_100"]
        raw["spot_return_pct"] = (raw["S0"] / raw["S-1"] - 1.0) * 100.0

        # The first interval of an episode has no previous IV observation for a
        # within-episode IV change.  That is fine: spot/option-down diagnostics
        # do not require dIV, and the IV-up panel automatically excludes NaNs.
        raw["iv_change"] = raw.groupby("episode")["v"].diff()

        spot_down = raw["spot_return_pct"] < 0.0
        option_down = raw["option_pnl_100"] < 0.0
        bad_down = spot_down & option_down
        down_iv_up = spot_down & (raw["iv_change"] > 0.0)

        n_episodes = raw["episode"].nunique()
        if n_episodes <= 0:
            continue

        def reward_contribution(mask: pd.Series) -> float:
            # Performance tables report mean episode reward differences.  A
            # state contribution is therefore the sum of step reward
            # differences in that state divided by the number of episodes.
            return float(raw.loc[mask, "diff_reward"].sum() / n_episodes)

        option_loss = -float(raw.loc[bad_down, "option_pnl_100"].mean())
        agent_hedge_gain = float(raw.loc[bad_down, "agent_hedge_pnl_100"].mean())
        bs_hedge_gain = float(raw.loc[bad_down, "bs_hedge_pnl_100"].mean())
        recovery_denom = option_loss if option_loss > 0.0 else np.nan

        rows.append(
            {
                "year": int(year),
                "episodes": int(n_episodes),
                "steps": int(len(raw)),
                "spot_down_share": float(spot_down.mean()),
                "bad_down_share": float(bad_down.mean()),
                "bad_down_reward_contribution": reward_contribution(bad_down),
                "other_state_reward_contribution": reward_contribution(~bad_down),
                "total_reward_difference": reward_contribution(
                    pd.Series(True, index=raw.index)
                ),
                "down_iv_up_steps": int(down_iv_up.sum()),
                "down_iv_up_option_up_share": float(
                    (raw.loc[down_iv_up, "option_pnl_100"] > 0.0).mean()
                ),
                "down_iv_up_mean_option_pnl": float(
                    raw.loc[down_iv_up, "option_pnl_100"].mean()
                ),
                "bad_down_agent_mean_pnl": float(raw.loc[bad_down, "a_pnl_100"].mean()),
                "bad_down_bs_mean_pnl": float(raw.loc[bad_down, "b_pnl_100"].mean()),
                "bad_down_option_loss": option_loss,
                "bad_down_agent_hedge_gain": agent_hedge_gain,
                "bad_down_bs_hedge_gain": bs_hedge_gain,
                "bad_down_agent_recovery_ratio": agent_hedge_gain / recovery_denom,
                "bad_down_bs_recovery_ratio": bs_hedge_gain / recovery_denom,
            }
        )

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No traded testing CSVs found for prefix {args.prefix}")
    return out


def plot_2022_reward_mechanism(args: argparse.Namespace, output_path: Path) -> None:
    """Draw the rewritten 2022 reward-failure mechanism figure.

    
    This figure tests the complete narrative: bad down states are frequent,
    IV-up days usually do not turn the call revaluation positive, the
    underhedged agent recovers less of the option loss than Black-Scholes, and
    the asymmetric reward shortfall is concentrated in those step-level states.
    """
    mechanism = collect_2022_reward_mechanism(args)
    mechanism.to_csv(
        args.output_dir / "regime_2022_reward_mechanism.csv", index=False
    )

    years = mechanism["year"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.8), sharex=True)
    agent_color = "#b36b2c"
    bs_color = "#2f5d8c"
    loss_color = "#b33a3a"
    neutral_color = "#6f7f8f"
    total_color = "#222222"

    ax = axes[0, 0]
    ax.plot(
        years,
        100.0 * mechanism["spot_down_share"],
        color=neutral_color,
        marker="o",
        lw=1.4,
        label="Index down",
    )
    ax.plot(
        years,
        100.0 * mechanism["bad_down_share"],
        color=loss_color,
        marker="s",
        lw=1.5,
        label="Index down, call down",
    )
    target = mechanism[mechanism["year"].eq(2022)].iloc[0]
    ax.annotate(
        f"{100.0 * target['bad_down_share']:.1f}%",
        xy=(2022, 100.0 * target["bad_down_share"]),
        xytext=(-7, 12),
        textcoords="offset points",
        ha="right",
        fontsize=8.8,
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": loss_color},
    )
    ax.set_title("Bad Down-State Frequency", fontweight="bold", pad=5)
    set_y_label(ax, "Share of hedge intervals (%)")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[0, 1]
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.plot(
        years,
        mechanism["down_iv_up_mean_option_pnl"],
        color=loss_color,
        marker="o",
        lw=1.5,
        label="Mean call P&L",
    )
    ax.annotate(
        f"{target['down_iv_up_mean_option_pnl']:.3f}",
        xy=(2022, target["down_iv_up_mean_option_pnl"]),
        xytext=(5, -14),
        textcoords="offset points",
        ha="left",
        fontsize=8.8,
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": loss_color},
    )
    ax.set_title("Call Revaluation on IV-Up Down Days", fontweight="bold", pad=5)
    set_y_label(ax, "Mean option P&L")

    ax = axes[1, 0]
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.plot(
        years,
        mechanism["bad_down_agent_recovery_ratio"],
        color=agent_color,
        marker="s",
        lw=1.5,
        label="Agent",
    )
    ax.plot(
        years,
        mechanism["bad_down_bs_recovery_ratio"],
        color=bs_color,
        marker="o",
        lw=1.5,
        label="Black-Scholes",
    )
    ax.annotate(
        f"{target['bad_down_agent_recovery_ratio']:.2f}",
        xy=(2022, target["bad_down_agent_recovery_ratio"]),
        xytext=(-10, 16),
        textcoords="offset points",
        ha="right",
        fontsize=8.8,
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": agent_color},
    )
    ax.set_title("Hedge Recovery in Bad Down States", fontweight="bold", pad=5)
    set_y_label(ax, "Hedge gain / option loss")
    ax.legend(frameon=False, loc="lower left")

    ax = axes[1, 1]
    ax.axhline(0.0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.plot(
        years,
        mechanism["bad_down_reward_contribution"],
        color=loss_color,
        marker="s",
        lw=1.5,
        label="Index down, call down",
    )
    ax.plot(
        years,
        mechanism["other_state_reward_contribution"],
        color=neutral_color,
        marker="o",
        lw=1.3,
        label="All other states",
    )
    ax.plot(
        years,
        mechanism["total_reward_difference"],
        color=total_color,
        marker="D",
        lw=1.3,
        label="Total",
    )
    ax.annotate(
        f"{target['total_reward_difference']:.2f}",
        xy=(2022, target["total_reward_difference"]),
        xytext=(-10, -17),
        textcoords="offset points",
        ha="right",
        fontsize=8.8,
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": total_color},
    )
    ax.set_title("Mean Episode Reward Contribution", fontweight="bold", pad=5)
    set_y_label(ax, "Agent minus Black-Scholes")
    ax.legend(frameon=False, loc="lower left")

    for ax in axes.ravel():
        # A light vertical band is more academic-looking than recoloring every
        # marker; it lets the reader compare 2022 with the full test panel.
        ax.axvspan(2021.5, 2022.5, color="#d9d9d9", alpha=0.25, zorder=0)
        ax.set_xticks(years)
        ax.grid(True, axis="y", ls=":", alpha=0.42)
    for ax in axes[-1, :]:
        ax.set_xlabel("Test year")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_2022_state_and_hedge(args: argparse.Namespace, output_path: Path) -> None:
    """Draw the compact 2022 state-frequency and hedge-recovery figure.

    
    The figure keeps the two quantities that directly identify the 2022 reward
    mechanism: the frequency of index-down/call-down states and the amount of
    the call loss recovered by the underlying hedge in those states.
    """
    mechanism = collect_2022_reward_mechanism(args)
    mechanism.to_csv(
        args.output_dir / "regime_2022_reward_mechanism.csv", index=False
    )

    years = mechanism["year"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.25))

    # Use the same restrained palette as the other regime-fragility figures:
    # blue-gray for the all-year diagnostic, copper for the focal year, and a
    # darker blue only for Black-Scholes when both hedge policies appear.
    line_color = "#4f6578"
    highlight_color = "#b36b2c"
    agent_color = "#b36b2c"
    bs_color = "#2f5d8c"

    ax = axes[0]
    ax.plot(
        years,
        100.0 * mechanism["bad_down_share"],
        color=line_color,
        marker="o",
        lw=1.6,
    )
    target = mechanism[mechanism["year"].eq(2022)].iloc[0]
    ax.scatter(
        [2022],
        [100.0 * target["bad_down_share"]],
        s=72,
        color=highlight_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.annotate(
        f"{100.0 * target['bad_down_share']:.1f}%",
        xy=(2022, 100.0 * target["bad_down_share"]),
        xytext=(-18, -19),
        textcoords="offset points",
        ha="right",
        fontsize=8.8,
        color=highlight_color,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": highlight_color},
    )
    ax.set_title("Index Down and Call Down", fontweight="bold", pad=5)
    set_y_label(ax, "Share of hedge intervals (%)")
    ax.set_xlabel("Test year")

    ax = axes[1]
    ax.axhline(1.0, color="black", lw=0.9, ls="--", alpha=0.65)
    ax.plot(
        years,
        mechanism["bad_down_agent_recovery_ratio"],
        color=agent_color,
        marker="s",
        lw=1.6,
        label="Agent",
    )
    ax.plot(
        years,
        mechanism["bad_down_bs_recovery_ratio"],
        color=bs_color,
        marker="o",
        lw=1.6,
        label="Black-Scholes",
    )
    ax.annotate(
        f"{target['bad_down_agent_recovery_ratio']:.2f}",
        xy=(2022, target["bad_down_agent_recovery_ratio"]),
        xytext=(-18, 22),
        textcoords="offset points",
        ha="right",
        fontsize=8.8,
        color=agent_color,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": agent_color},
    )
    ax.set_title("Hedge Recovery in These States", fontweight="bold", pad=5)
    set_y_label(ax, "Hedge gain / call loss")
    ax.set_xlabel("Test year")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.31), ncol=2)

    for ax in axes:
        ax.axvspan(2021.5, 2022.5, color="#d9d9d9", alpha=0.24, zorder=0)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", ls=":", alpha=0.42)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def collect_2022_vega_parachute(args: argparse.Namespace) -> pd.DataFrame:
    """Summarize whether IV revaluation offsets spot losses on down days.

    
    The word "parachute" in the paper is now tied to an explicit diagnostic:
    on index-down hedge intervals, how much of the negative spot revaluation of
    the option is offset by the IV revaluation component?  The step-level Greek
    decomposition is a reduced-form accounting identity for observed option
    price changes, not a structural causal estimate.
    """
    path = args.forensic_greek_decomposition_steps
    if not path.exists():
        raise FileNotFoundError(f"Missing step-level Greek decomposition: {path}")
    steps = pd.read_csv(path)

    rows = []
    for year, group in steps.groupby("year"):
        down = group[group["dS_pct"] < 0.0].copy()
        if down.empty:
            continue
        down_iv_up = down[down["dIV"] > 0.0].copy()

        neg_spot_loss = -float(down["spot"].sum())
        positive_iv_offset = float(down["iv"].clip(lower=0.0).sum())
        net_iv_offset = float(down["iv"].sum())

        rows.append(
            {
                "year": int(year),
                "steps": int(len(group)),
                "down_steps": int(len(down)),
                "down_share": float(len(down) / len(group)),
                "mean_option_pnl_down": float(down["actual"].mean()),
                "mean_spot_component_down": float(down["spot"].mean()),
                "mean_iv_component_down": float(down["iv"].mean()),
                "positive_iv_offset_ratio": positive_iv_offset / neg_spot_loss,
                "net_iv_offset_ratio": net_iv_offset / neg_spot_loss,
                "down_iv_up_steps": int(len(down_iv_up)),
                "down_iv_up_share": float(len(down_iv_up) / len(down)),
                "down_iv_up_option_up_share": float(
                    (down_iv_up["actual"] > 0.0).mean()
                ),
                "mean_option_pnl_down_iv_up": float(down_iv_up["actual"].mean()),
                "mean_spot_component_down_iv_up": float(down_iv_up["spot"].mean()),
                "mean_iv_component_down_iv_up": float(down_iv_up["iv"].mean()),
            }
        )

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if out.empty:
        raise ValueError("No down-day rows found in Greek decomposition steps")
    return out


def plot_2022_vega_parachute(args: argparse.Namespace, output_path: Path) -> None:
    """Draw direct evidence for the weak 2022 IV offset on index-down days."""
    parachute = collect_2022_vega_parachute(args)
    parachute.to_csv(args.output_dir / "regime_2022_vega_parachute.csv", index=False)

    years = parachute["year"].to_numpy()
    target = parachute[parachute["year"].eq(2022)].iloc[0]
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 3.15))
    line_color = "#4f6578"
    highlight_color = "#b36b2c"

    # Use the direct ratio: positive IV revaluation divided by the negative
    # spot revaluation on index-down hedge intervals.
    ax.plot(
        years,
        100.0 * parachute["positive_iv_offset_ratio"],
        color=line_color,
        marker="o",
        lw=1.6,
    )
    ax.scatter(
        [2022],
        [100.0 * target["positive_iv_offset_ratio"]],
        s=72,
        color=highlight_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.annotate(
        f"{100.0 * target['positive_iv_offset_ratio']:.1f}%",
        xy=(2022, 100.0 * target["positive_iv_offset_ratio"]),
        xytext=(-9, 16),
        textcoords="offset points",
        ha="right",
        fontsize=8.8,
        color=highlight_color,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": highlight_color},
    )
    ax.set_title("Positive IV Offset of Spot Loss", fontweight="bold", pad=5)
    set_y_label(ax, "Offset ratio on index-down days (%)")
    ax.set_xlabel("Test year")
    ax.axvspan(2021.5, 2022.5, color="#d9d9d9", alpha=0.24, zorder=0)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", ls=":", alpha=0.42)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_2022_loss_source_table(args: argparse.Namespace, output_path: Path) -> None:
    """Write the compact 2022 negative-loss state table."""
    if not args.negative_pnl_breakdown.exists():
        raise FileNotFoundError(
            f"Missing cached negative-P&L breakdown: {args.negative_pnl_breakdown}"
        )
    data = pd.read_csv(args.negative_pnl_breakdown)
    data = data[(data["year"].eq(2022)) & (data["portfolio"].eq("agent"))].copy()
    data = data.sort_values("share_negative_loss", ascending=False).head(4)

    def pct(value: float) -> str:
        return rf"{tex_num(100.0 * value, digits=1)}\%"

    def num(value: float) -> str:
        return tex_num(value)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{5.5pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"State & Loss Share & Mean PnL & Option Leg & Hedge Leg \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        lines.append(
            f"{pretty_state_label(row['state3'])} & "
            f"{pct(row['share_negative_loss'])} & "
            f"{num(row['mean_pnl'])} & "
            f"{num(row['mean_option_pnl'])} & "
            f"{num(row['mean_hedge_pnl'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_2023_variance_mechanism(args: argparse.Namespace) -> pd.DataFrame:
    """Compute the compact diagnostics used in the 2023 subsection.

    
    The diagnostics read cached step-level Greek/accounting outputs and do not
    rerun model testing or bootstrap inference.

    The key regression is descriptive, not causal:

        option_step_pnl = a + b * spot_return + c * implied_vol_change + error.

    We report the spot-only R2 and the incremental R2 from adding IV after spot.
    Because IV is implied from option prices, this is an accounting coordinate
    system for realized option-surface moves, not an exogenous-shock design.
    """
    for path in [
        args.forensic_greek_decomposition_steps,
        args.forensic_year_summary,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing 2023 mechanism input: {path}")

    steps = pd.read_csv(args.forensic_greek_decomposition_steps)
    year_summary = pd.read_csv(args.forensic_year_summary)

    required_step_cols = {"year", "actual", "dS_pct", "dIV"}
    missing_step = required_step_cols.difference(steps.columns)
    if missing_step:
        raise ValueError(
            f"{args.forensic_greek_decomposition_steps} is missing columns: "
            f"{sorted(missing_step)}"
        )

    required_year_cols = {
        "year",
        "var_b",
        "var_option_leg",
        "var_bs_hedge_leg",
        "var_a",
        "log_var_ratio",
    }
    missing_year = required_year_cols.difference(year_summary.columns)
    if missing_year:
        raise ValueError(
            f"{args.forensic_year_summary} is missing columns: "
            f"{sorted(missing_year)}"
        )

    def ols_r2(frame: pd.DataFrame, x_cols: list[str]) -> float:
        """Small local OLS helper to avoid adding a statsmodels dependency."""
        y = frame["actual"].to_numpy(dtype=float)
        x_parts = [np.ones(len(frame))]
        x_parts.extend(frame[col].to_numpy(dtype=float) for col in x_cols)
        x = np.column_stack(x_parts)
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        fitted = x @ beta
        sse = float(np.sum((y - fitted) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        return np.nan if sst <= 0.0 else 1.0 - sse / sst

    rows: list[dict[str, float]] = []
    usable = steps.dropna(subset=["actual", "dS_pct", "dIV"]).copy()
    for year, frame in usable.groupby("year"):
        r2_spot = ols_r2(frame, ["dS_pct"])
        r2_spot_iv = ols_r2(frame, ["dS_pct", "dIV"])
        rows.append(
            {
                "year": int(year),
                "n_steps": int(len(frame)),
                "corr_spot_return_iv_change": float(
                    frame["dS_pct"].corr(frame["dIV"])
                ),
                "corr_option_pnl_iv_change": float(
                    frame["actual"].corr(frame["dIV"])
                ),
                "corr_option_pnl_spot_return": float(
                    frame["actual"].corr(frame["dS_pct"])
                ),
                "r2_spot_only": r2_spot,
                "r2_spot_plus_iv": r2_spot_iv,
                "incremental_iv_r2": r2_spot_iv - r2_spot,
            }
        )

    diagnostics = pd.DataFrame(rows)
    year_summary = year_summary.copy()
    # The denominator problem is easier to see after normalizing the BS residual
    # variance by the gross hedge-account variance.  This asks: after adding the
    # option-leg variance and the BS hedge-leg variance, how much variance is
    # left once their covariance is allowed to cancel?  In 2023 this residual
    # share is extremely small, which is the cleanest way to show why the log
    # agent/BS variance ratio becomes large even though the agent's absolute
    # variance is not large by historical standards.
    gross_bs_variance = (
        year_summary["var_option_leg"] + year_summary["var_bs_hedge_leg"]
    )
    year_summary["bs_residual_variance_share"] = (
        year_summary["var_b"] / gross_bs_variance
    )
    diagnostics = diagnostics.merge(
        year_summary[
            [
                "year",
                "var_a",
                "var_b",
                "log_var_ratio",
                "bs_residual_variance_share",
            ]
        ],
        on="year",
        how="left",
    ).sort_values("year")
    return diagnostics


def plot_2023_variance_mechanism(args: argparse.Namespace, output_path: Path) -> None:
    """Draw the reduced-form mechanism for the 2023 variance failure."""
    diagnostics = build_2023_variance_mechanism(args)
    channel = build_2017_variance_mechanism(args)
    diagnostics.to_csv(
        output_path.with_suffix(".csv"),
        index=False,
    )

    years = diagnostics["year"].to_numpy()
    highlight = "#b36b2c"
    line_color = "#4f6578"

    fig = plt.figure(figsize=(7.6, 5.55))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.74, wspace=0.30)
    axes = [
        fig.add_subplot(grid[0, :]),
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]),
    ]

    ax = axes[0]
    ax.plot(
        years,
        channel["corr_spot_iv"],
        color=line_color,
        marker="o",
        lw=1.55,
        ms=4.4,
        label=r"$\rho(\Delta S/S,\Delta IV)$",
    )
    ax.plot(
        years,
        channel["corr_actual_iv"],
        color="#4f7b62",
        marker="s",
        lw=1.55,
        ms=4.0,
        label=r"$\rho(\Delta C,\Delta IV\;\mathrm{part})$",
    )
    target_channel = channel[channel["year"].eq(2023)].iloc[0]
    ax.scatter(
        [2023],
        [target_channel["corr_actual_iv"]],
        s=72,
        color=highlight,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.scatter(
        [2023],
        [target_channel["corr_spot_iv"]],
        s=62,
        color=highlight,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    row_2017 = channel[channel["year"].eq(2017)].iloc[0]
    ax.scatter(
        [2017],
        [row_2017["corr_actual_iv"]],
        s=62,
        color=highlight,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.text(
        2023,
        target_channel["corr_actual_iv"] + 0.026,
        f"{target_channel['corr_actual_iv']:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=highlight,
        fontweight="bold",
    )
    ax.text(
        2023,
        target_channel["corr_spot_iv"] - 0.055,
        f"{target_channel['corr_spot_iv']:.3f}",
        ha="center",
        va="top",
        fontsize=8.8,
        color=highlight,
        fontweight="bold",
    )
    ax.text(
        2017,
        row_2017["corr_actual_iv"] + 0.026,
        f"{row_2017['corr_actual_iv']:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color=highlight,
        fontweight="bold",
    )
    ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.42)
    ax.set_ylim(-0.64, 0.155)
    ax.set_title("Volatility-Channel Correlations", fontweight="bold", pad=5)
    set_y_label(ax, "Correlation")
    # The legend originally sat on top of the correlation paths in the lower
    # left of the panel.  Place it in the deliberate gap between the top panel
    # and the lower panels, centered under the top axes.
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        fontsize=9.0,
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.7,
    )

    panels = [
        (
            axes[1],
            "r2_spot_only",
            "Spot-Only Explanatory Power",
            r"Spot-only $R^2$ (%)",
            100.0,
            "{:.1f}%",
        ),
        (
            axes[2],
            "bs_residual_variance_share",
            "Black-Scholes Residual Variance",
            "Residual variance share (%)",
            100.0,
            "{:.2f}%",
        ),
    ]

    for ax, col, title, ylabel, scale, fmt in panels:
        values = diagnostics[col].to_numpy(dtype=float) * scale
        target = diagnostics.loc[diagnostics["year"].eq(2023)].iloc[0]
        target_value = float(target[col]) * scale

        ax.plot(
            years,
            values,
            color=line_color,
            marker="o",
            lw=1.55,
            ms=4.6,
        )
        ax.scatter(
            [2023],
            [target_value],
            s=76,
            color=highlight,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        if col == "r2_spot_only":
            row_2017_spot = diagnostics.loc[diagnostics["year"].eq(2017)].iloc[0]
            value_2017 = float(row_2017_spot[col]) * scale
            ax.scatter(
                [2017],
                [value_2017],
                s=64,
                color=highlight,
                edgecolor="white",
                linewidth=0.8,
                zorder=4,
            )

        # Keep the numerical label inside the axes so it cannot float outside a
        # panel after LaTeX rescales the figure.
        y_min, y_max = np.nanmin(values), np.nanmax(values)
        pad = 0.08 * (y_max - y_min if y_max > y_min else 1.0)
        ax.set_ylim(y_min - pad, y_max + 2.3 * pad)
        va = "bottom"
        label_y = target_value + 0.55 * pad
        if label_y > ax.get_ylim()[1] - 0.2 * pad:
            label_y = target_value - 0.55 * pad
            va = "top"
        ax.text(
            2023,
            label_y,
            fmt.format(target_value),
            ha="center",
            va=va,
            fontsize=8.8,
            color=highlight,
            fontweight="bold",
        )
        if col == "r2_spot_only":
            value_2017 = float(
                diagnostics.loc[diagnostics["year"].eq(2017), col].iloc[0]
            ) * scale
            ax.text(
                2017,
                value_2017 + 0.55 * pad,
                fmt.format(value_2017),
                ha="center",
                va="bottom",
                fontsize=8.8,
                color=highlight,
                fontweight="bold",
            )

        ax.set_title(title, fontweight="bold", pad=5)
        set_y_label(ax, ylabel)

    for ax in axes:
        ax.axvspan(2022.5, 2023.5, color=highlight, alpha=0.09, zorder=0)
        ax.grid(True, axis="y", ls=":", alpha=0.38)
        ax.set_xticks(years)
        ax.tick_params(axis="x", rotation=45)
    for ax in axes[1:]:
        ax.set_xlabel("Test year")

    # The custom top-spanning GridSpec layout is spaced explicitly by the
    # GridSpec arguments above.  Avoid tight_layout here because Matplotlib
    # warns on this layout even though the saved figure is valid.
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_2023_volatility_channel(args: argparse.Namespace, output_path: Path) -> None:
    """Write the variance-mechanism figure under a compatibility filename.

    
    The current paper text uses regime_2023_variance_mechanism.pdf.  This
    wrapper also writes the same figure to regime_2023_volatility_channel.pdf
    for reproducibility of archived manuscript builds.
    """
    plot_2023_variance_mechanism(args, output_path)


def write_bs_cancellation_table(args: argparse.Namespace, output_path: Path) -> None:
    """Write the hedge-account variance decomposition for the BS benchmark."""
    if not args.forensic_year_summary.exists():
        raise FileNotFoundError(f"Missing year-summary file: {args.forensic_year_summary}")
    data = pd.read_csv(args.forensic_year_summary).sort_values("year")

    def num(value: float) -> str:
        return tex_num(value)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\setlength{\tabcolsep}{6.0pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"Year & Option Var. & Hedge Var. & $2\,\mathrm{Cov}$ & BS Var. \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        lines.append(
            f"{int(row['year'])} & "
            f"{num(row['var_option_leg'])} & "
            f"{num(row['var_bs_hedge_leg'])} & "
            f"{num(row['two_cov_option_bs_hedge'])} & "
            f"{num(row['var_b'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_2017_low_volatility(args: argparse.Namespace, output_path: Path) -> None:
    """Draw the low-volatility denominator diagnostics for 2017."""
    if not args.forensic_year_summary.exists():
        raise FileNotFoundError(f"Missing year-summary file: {args.forensic_year_summary}")
    data = pd.read_csv(args.forensic_year_summary).sort_values("year")
    years = data["year"].to_numpy()
    colors = ["#b36b2c" if year == 2017 else "#8c9aa8" for year in years]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    ax = axes[0]
    ax.bar(years, data["realized_vol_mean"], color=colors, width=0.72)
    target_2017 = data[data["year"].eq(2017)].iloc[0]
    ax.text(
        2017,
        target_2017["realized_vol_mean"] + 0.006,
        f"{target_2017['realized_vol_mean']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_title("Realized Volatility", fontweight="bold", pad=5)
    set_y_label(ax, "Mean episode realized vol.")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", ls=":", alpha=0.42)

    ax = axes[1]
    ax.bar(years, data["var_b"], color=colors, width=0.72)
    ax.text(
        2017,
        target_2017["var_b"] + 0.045,
        f"{target_2017['var_b']:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    ax.set_title("Black-Scholes Terminal Variance", fontweight="bold", pad=5)
    set_y_label(ax, "BS terminal P&L variance")
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", ls=":", alpha=0.42)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_2017_variance_mechanism(args: argparse.Namespace) -> pd.DataFrame:
    """Build the compact diagnostic table for the rewritten 2017 subsection.

    
    Section 9.3 establishes four facts in one place:

    1. the option-IV translation is weak in 2017, so IV changes do not show up
       as a strong independent driver of call-price changes;
    2. the option itself moves unusually little, which makes the benchmark
       variance denominator small in absolute P&L units;
    3. the traded policy is still, on average, below Black-Scholes delta, so the
       variance increase is not primarily a native overhedging story;
    4. the remaining policy residual is large relative to a very tight
       Black-Scholes terminal-variance denominator.

    These are reduced-form accounting diagnostics.  IV is implied from option
    prices, so the correlations below should not be read as causal estimates of
    exogenous volatility shocks.
    """
    for path in [args.forensic_year_summary, args.forensic_greek_decomposition]:
        if not path.exists():
            raise FileNotFoundError(f"Missing 2017 mechanism input: {path}")

    year_summary = pd.read_csv(args.forensic_year_summary).copy()
    greek_summary = pd.read_csv(args.forensic_greek_decomposition).copy()

    required_year_cols = {
        "year",
        "var_a",
        "var_b",
        "log_var_ratio",
        "log_downside_ratio",
        "mean_delta_gap",
        "underhedged_share",
        "var_option_leg",
        "var_bs_hedge_leg",
    }
    missing_year = required_year_cols.difference(year_summary.columns)
    if missing_year:
        raise ValueError(
            f"{args.forensic_year_summary} is missing columns: "
            f"{sorted(missing_year)}"
        )

    required_greek_cols = {
        "year",
        "actual_var",
        "corr_actual_iv",
        "corr_spot_iv",
        "corr_actual_spot",
        "mean_abs_iv_share",
    }
    missing_greek = required_greek_cols.difference(greek_summary.columns)
    if missing_greek:
        raise ValueError(
            f"{args.forensic_greek_decomposition} is missing columns: "
            f"{sorted(missing_greek)}"
        )

    year_summary["bs_gross_variance"] = (
        year_summary["var_option_leg"] + year_summary["var_bs_hedge_leg"]
    )
    year_summary["bs_residual_variance_share"] = (
        year_summary["var_b"] / year_summary["bs_gross_variance"]
    )

    diagnostics = year_summary[
        [
            "year",
            "var_a",
            "var_b",
            "log_var_ratio",
            "log_downside_ratio",
            "mean_delta_gap",
            "underhedged_share",
            "bs_residual_variance_share",
        ]
    ].merge(
        greek_summary[
            [
                "year",
                "actual_var",
                "corr_actual_iv",
                "corr_spot_iv",
                "corr_actual_spot",
                "mean_abs_iv_share",
            ]
        ],
        on="year",
        how="left",
    )
    return diagnostics.sort_values("year")


def plot_2017_variance_mechanism(args: argparse.Namespace, output_path: Path) -> None:
    """Draw the single-panel 2017 scale diagnostic used in Section 9.3."""
    diagnostics = build_2017_variance_mechanism(args)
    diagnostics.to_csv(output_path.with_suffix(".csv"), index=False)

    years = diagnostics["year"].to_numpy()
    highlight_2017 = "#b36b2c"
    line_blue = "#4f6578"

    fig, ax = plt.subplots(1, 1, figsize=(5.1, 3.15))

    def annotate_target(
        ax: plt.Axes,
        year: int,
        value: float,
        label: str,
        color: str = highlight_2017,
    ) -> None:
        """Place a small in-panel label without covering neighboring years."""
        y_min, y_max = ax.get_ylim()
        pad = 0.055 * (y_max - y_min if y_max > y_min else 1.0)
        label_y = value + pad
        va = "bottom"
        if label_y > y_max - 0.35 * pad:
            label_y = value - pad
            va = "top"
        ax.text(
            year,
            label_y,
            label,
            ha="center",
            va=va,
            fontsize=8.6,
            color=color,
            fontweight="bold",
        )

    row_2017 = diagnostics[diagnostics["year"].eq(2017)].iloc[0]

    # The volatility-correlation panel has been moved into the 2023 mechanism
    # figure, where it is used for both 2023 and 2017 comparisons.  The 2017
    # subsection now needs only this scale diagnostic: the traded call
    # revaluation variance is the smallest in the sample.
    values = diagnostics["actual_var"].to_numpy(dtype=float)
    ax.plot(years, values, color=line_blue, marker="o", lw=1.55, ms=4.4)
    ax.scatter(
        [2017],
        [row_2017["actual_var"]],
        s=72,
        color=highlight_2017,
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
    )
    ax.set_title("Call Revaluation Scale", fontweight="bold", pad=5)
    set_y_label(ax, "Call-price variance by interval")
    ax.grid(True, axis="y", ls=":", alpha=0.38)
    ax.annotate(
        f"{row_2017['actual_var']:.3f}",
        xy=(2017, row_2017["actual_var"]),
        xytext=(0, 20),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=8.6,
        color=highlight_2017,
        fontweight="bold",
        arrowprops={"arrowstyle": "-", "lw": 0.6, "color": highlight_2017},
    )

    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45)
    ax.axvspan(2016.5, 2017.5, color=highlight_2017, alpha=0.09, zorder=0)
    ax.set_xlabel("Test year")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_2017_cluster_mechanism_table(
    args: argparse.Namespace, output_path: Path
) -> None:
    """Write the compact cluster table supporting the 2017 interpretation.

    
    The table is intentionally small.  It checks whether the largest positive
    contributions to the 2017 ordinary-variance gap come from a systematic
    overhedge.  In the traded test paths they do not: the largest positive
    contributors mostly have negative mean Agent-BS delta gaps.
    """
    if not args.forensic_cluster_table.exists():
        raise FileNotFoundError(f"Missing cluster table: {args.forensic_cluster_table}")

    data = pd.read_csv(args.forensic_cluster_table)
    required_cols = {
        "year",
        "start_date",
        "episodes",
        "delta_gap_mean",
        "rho_mean",
        "return_mean",
        "iv_change_mean",
        "diff_pnl_mean",
        "diff_reward_mean",
        "sum_excess_var",
    }
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(
            f"{args.forensic_cluster_table} is missing columns: {sorted(missing)}"
        )

    year_data = data[data["year"].eq(2017)].copy()
    if year_data.empty:
        raise ValueError(f"{args.forensic_cluster_table} contains no 2017 rows")

    positive_total = year_data.loc[
        year_data["sum_excess_var"].gt(0.0), "sum_excess_var"
    ].sum()
    year_data["positive_excess_share"] = np.where(
        positive_total > 0.0,
        year_data["sum_excess_var"] / positive_total,
        np.nan,
    )
    top = (
        year_data[year_data["sum_excess_var"].gt(0.0)]
        .sort_values("sum_excess_var", ascending=False)
        .head(5)
    )

    def num(value: float) -> str:
        return tex_num(value)

    def pct(value: float) -> str:
        return tex_num(100.0 * value, digits=1, percent=True)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.12}",
        # The 2017 mechanism table has several signed diagnostics.  Keep the
        # typography compact so it does not protrude in the compiled paper.
        # Keep the table readable in the paper.  The surrounding LaTeX table
        # already requests \small, so do not override it with footnotesize.
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        (
            r"Cluster & Ep. & Agent--BS $\Delta$ & "
            r"$\rho(\Delta S,\Delta IV)$ & $\Delta$PnL & $\Delta$Reward & "
            r"Excess Var. \\"
        ),
        r"\midrule",
    ]
    for _, row in top.iterrows():
        lines.append(
            f"{row['start_date']} & "
            f"{int(row['episodes'])} & "
            f"{num(row['delta_gap_mean'])} & "
            f"{num(row['rho_mean'])} & "
            f"{num(row['diff_pnl_mean'])} & "
            f"{num(row['diff_reward_mean'])} & "
            f"{pct(row['positive_excess_share'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_distillation_fidelity(args: argparse.Namespace) -> pd.DataFrame:
    path = args.distillation_results / "walkforward_test_fidelity_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing distillation fidelity file: {path}")
    data = pd.read_csv(path)
    return data[data["target_delta_col"] == "actual_agent_delta"].copy()


def load_distillation_bootstrap(args: argparse.Namespace) -> pd.DataFrame:
    path = args.distillation_results / "walkforward_test_bootstrap_summary_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing distillation bootstrap file: {path}")
    return pd.read_csv(path)


def load_distillation_hof_fidelity(args: argparse.Namespace, split: str) -> pd.DataFrame:
    path = args.distillation_results / f"walkforward_{split}_hof_fidelity_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Hall-of-Fame fidelity file: {path}")
    return pd.read_csv(path)


def load_distillation_hof_bootstrap(args: argparse.Namespace, split: str) -> pd.DataFrame:
    path = args.distillation_results / f"walkforward_{split}_hof_all_bootstrap_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Hall-of-Fame bootstrap file: {path}")
    return pd.read_csv(path)


def intended_hof_target_name(candidate: str) -> str:
    # The raw family is fit directly to the traded actor.  The two smoothed
    # families are fit to the kernel-smoothed actor, so their parsimony rule
    # should be applied to that intended target rather than to the raw actor.
    if candidate == "uniform_bs_delta_residual":
        return "actual_traded_agent"
    if candidate in {"smooth_uniform_bs_delta_residual", "smooth_focus_bs_delta_residual"}:
        return "smooth_agent_bw100"
    raise ValueError(f"Unknown symbolic candidate family: {candidate}")


def select_parsimonious_hof_candidates(
    validation_hof_fidelity: pd.DataFrame, tolerance: float = 0.10
) -> pd.DataFrame:
    # Publication rule used in the paper: for each test year, pool every
    # Hall-of-Fame formula from the three candidate families and use one common
    # validation target, the actually traded raw agent action.  Select the
    # lowest-complexity formula whose validation MAE is within 10% of the best
    # pooled validation MAE.
    # This is the one-formula-per-year rule used for the paper's main
    # distillation results.
    selected_rows = []
    raw_target = validation_hof_fidelity[
        validation_hof_fidelity["target_name"] == "actual_traded_agent"
    ].copy()
    for test_year, group in raw_target.groupby("test_year"):
        if group.empty:
            continue
        best_mae = group["mae"].min()
        eligible = group[group["mae"] <= (1.0 + tolerance) * best_mae]
        selected = eligible.sort_values(
            ["complexity", "mae", "hof_index", "candidate"]
        ).iloc[0]
        best = group.sort_values(["mae", "complexity", "hof_index", "candidate"]).iloc[0]
        candidate = selected["candidate"]
        selected_rows.append(
            {
                "test_year": int(test_year),
                "candidate": candidate,
                "family": DISTILLATION_POLICY_LABELS[candidate],
                "policy": selected["policy"],
                "hof_index": int(selected["hof_index"]),
                "complexity": int(selected["complexity"]),
                "validation_target_name": "actual_traded_agent",
                "validation_target_mae": float(selected["mae"]),
                "best_validation_target_mae": float(best["mae"]),
                "best_fit_complexity": int(best["complexity"]),
                "best_fit_family": DISTILLATION_POLICY_LABELS[best["candidate"]],
                "equation": selected["equation"],
            }
        )
    return pd.DataFrame(selected_rows).sort_values("test_year")


def collect_parsimonious_fit_summary(
    selected: pd.DataFrame, test_hof_fidelity: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for _, choice in selected.iterrows():
        test_rows = test_hof_fidelity[
            (test_hof_fidelity["test_year"] == choice["test_year"])
            & (test_hof_fidelity["policy"] == choice["policy"])
            & (test_hof_fidelity["target_name"] == "actual_traded_agent")
        ]
        if test_rows.empty:
            continue
        test_row = test_rows.iloc[0]
        rows.append(
            {
                "test_year": choice["test_year"],
                "candidate": choice["candidate"],
                "family": choice["family"],
                "policy": choice["policy"],
                "hof_index": choice["hof_index"],
                "complexity": choice["complexity"],
                "validation_target_mae": choice["validation_target_mae"],
                "test_raw_mae": test_row["mae"],
                "test_raw_rmse": test_row["rmse"],
                "test_raw_p95": test_row["p95_abs_error"],
                "test_raw_corr": test_row["corr"],
            }
        )
    detail = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "n_years": len(detail),
                "avg_complexity": detail["complexity"].mean(),
                "validation_target_mae": detail["validation_target_mae"].mean(),
                "test_raw_mae": detail["test_raw_mae"].mean(),
                "test_raw_p95": detail["test_raw_p95"].mean(),
                "test_raw_corr": detail["test_raw_corr"].mean(),
            }
        ]
    )
    return detail, summary


def plot_parsimonious_fidelity(fit_detail: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.5))
    styles = {
        "uniform_bs_delta_residual": ("#2f5d8c", "o"),
        "smooth_uniform_bs_delta_residual": ("#b36b2c", "s"),
        "smooth_focus_bs_delta_residual": ("#1b8a5a", "^"),
    }
    fit_detail = fit_detail.sort_values("test_year")
    ax.plot(
        fit_detail["test_year"],
        fit_detail["test_raw_mae"],
        color="#555555",
        lw=1.1,
        alpha=0.75,
        zorder=1,
    )
    for candidate, label in DISTILLATION_POLICY_LABELS.items():
        data = fit_detail[fit_detail["candidate"] == candidate]
        if data.empty:
            continue
        color, marker = styles[candidate]
        ax.scatter(
            data["test_year"],
            data["test_raw_mae"],
            marker=marker,
            s=44,
            color=color,
            label=label,
            zorder=2,
        )
    ax.set_xlabel("Test year")
    set_y_label(ax, "MAE to raw traded agent delta")
    ax.set_xticks(sorted(fit_detail["test_year"].unique()))
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def symbolic_formula_label(row: pd.Series) -> str:
    prefixes = {
        "uniform_bs_delta_residual": "raw",
        "smooth_uniform_bs_delta_residual": "sUni",
        "smooth_focus_bs_delta_residual": "sFocus",
    }
    return f"{prefixes[row['candidate']]} idx{int(row['hof_index'])} c{int(row['complexity'])}"


def write_selected_formula_table(detail: pd.DataFrame, output_path: Path) -> None:
    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\setlength{\tabcolsep}{4.8pt}",
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        r"Year & Family & Complexity & Val. MAE & Test MAE & 95\% Error & Test Corr. \\",
        r"\midrule",
    ]
    for _, row in detail.sort_values("test_year").iterrows():
        lines.append(
            f"{int(row['test_year'])} & {row['family']} & {int(row['complexity'])} & "
            f"{tex_num(row['validation_target_mae'])} & {tex_num(row['test_raw_mae'])} & "
            f"{tex_num(row['test_raw_p95'])} & {tex_num(row['test_raw_corr'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def significance_star(row: pd.Series) -> str:
    if int(row.get("significant_99", 0)) == 1:
        return r"\sym{***}"
    if int(row.get("significant_95", row.get("significant", 0))) == 1:
        return r"\sym{**}"
    if int(row.get("significant_90", 0)) == 1:
        return r"\sym{*}"
    return ""


DISTILLATION_METRIC_ORDER = ["rew", "cvar", "log_down_var_ratio", "log_var_ratio"]
DISTILLATION_TO_MAIN_METRIC = {
    "rew": "reward",
    "cvar": "cvar",
    "log_down_var_ratio": "log_downside_variance",
    "log_var_ratio": "log_variance",
}


def summarize_metric_block(data: pd.DataFrame, metric_col: str = "metric") -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in DISTILLATION_METRIC_ORDER:
        block = data[data[metric_col].eq(metric)].copy()
        if block.empty:
            out[f"{metric}_mean"] = np.nan
            out[f"{metric}_fav"] = np.nan
            out[f"{metric}_bad"] = np.nan
            continue
        if metric in {"rew", "cvar"}:
            favorable = (block["point_estimate"] > 0.0) & (block["significant"] == 1)
            unfavorable = (block["point_estimate"] < 0.0) & (block["significant"] == 1)
        else:
            favorable = (block["point_estimate"] < 0.0) & (block["significant"] == 1)
            unfavorable = (block["point_estimate"] > 0.0) & (block["significant"] == 1)
        out[f"{metric}_mean"] = float(block["point_estimate"].mean())
        out[f"{metric}_fav"] = int(favorable.sum())
        out[f"{metric}_bad"] = int(unfavorable.sum())
    return out


def summarize_agent_vs_bs(summary: pd.DataFrame) -> dict[str, float]:
    rows = []
    for metric, main_metric in DISTILLATION_TO_MAIN_METRIC.items():
        block = summary[summary["metric"].eq(main_metric)].copy()
        block["metric"] = metric
        block["point_estimate"] = block["center"]
        rows.append(block[["metric", "point_estimate", "significant"]])
    return summarize_metric_block(pd.concat(rows, ignore_index=True))


def write_smoother_tables(
    bootstrap: pd.DataFrame,
    agent_bs_summary: pd.DataFrame,
    yearly_output_path: Path,
    bs_output_path: Path,
) -> None:
    data = bootstrap[
        bootstrap["comparison"].isin(
            ["smooth_target_bw100_vs_agent", "smooth_target_bw100_vs_bs"]
        )
        & bootstrap["metric"].isin(DISTILLATION_METRIC_ORDER)
    ].copy()

    def fmt_summary(row: dict[str, float], metric: str) -> str:
        return (
            f"{tex_num(row[f'{metric}_mean'])} "
            rf"({int(row[f'{metric}_fav'])}/{int(row[f'{metric}_bad'])})"
        )

    agent_row = summarize_agent_vs_bs(agent_bs_summary)
    smooth_row = summarize_metric_block(data[data["comparison"].eq("smooth_target_bw100_vs_bs")])

    yearly_lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"Year & Reward & CVaR 5\% & Log Downside Variance & Log Variance \\",
        r"\midrule",
    ]
    panel = data[data["comparison"].eq("smooth_target_bw100_vs_agent")]
    for year, group in panel.groupby("test_year"):
        pieces = [str(int(year))]
        for metric in DISTILLATION_METRIC_ORDER:
            row = group[group["metric"].eq(metric)].iloc[0]
            pieces.append(tex_num(row["point_estimate"]) + significance_star(row))
        yearly_lines.append(" & ".join(pieces) + r" \\")
    yearly_lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    yearly_output_path.write_text("\n".join(yearly_lines) + "\n", encoding="utf-8")

    bs_lines = [
        r"\begingroup",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"Comparison & Reward & CVaR 5\% & Log Downside Variance & Log Variance \\",
        r"\midrule",
        (
            "Agent--BS & "
            f"{fmt_summary(agent_row, 'rew')} & "
            f"{fmt_summary(agent_row, 'cvar')} & "
            f"{fmt_summary(agent_row, 'log_down_var_ratio')} & "
            f"{fmt_summary(agent_row, 'log_var_ratio')} \\\\"
        ),
        (
            "Smoothed agent--BS & "
            f"{fmt_summary(smooth_row, 'rew')} & "
            f"{fmt_summary(smooth_row, 'cvar')} & "
            f"{fmt_summary(smooth_row, 'log_down_var_ratio')} & "
            f"{fmt_summary(smooth_row, 'log_var_ratio')} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\endgroup",
    ]
    bs_output_path.write_text("\n".join(bs_lines) + "\n", encoding="utf-8")


def collect_parsimonious_trading_summary(
    selected: pd.DataFrame, hof_bootstrap: pd.DataFrame, agent_bs_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def sig_counts(data: pd.DataFrame, metric: str) -> tuple[int, int, int]:
        if metric in {"rew", "cvar"}:
            favorable = (data["point_estimate"] > 0.0) & (data["significant"] == 1)
            unfavorable = (data["point_estimate"] < 0.0) & (data["significant"] == 1)
        else:
            favorable = (data["point_estimate"] < 0.0) & (data["significant"] == 1)
            unfavorable = (data["point_estimate"] > 0.0) & (data["significant"] == 1)
        indistinguishable = data["significant"] == 0
        return int(favorable.sum()), int(unfavorable.sum()), int(indistinguishable.sum())

    detail_rows = []
    for _, choice in selected.iterrows():
        for rhs in ["bs", "agent"]:
            for metric in ["rew", "cvar", "log_down_var_ratio", "log_var_ratio"]:
                data = hof_bootstrap[
                    (hof_bootstrap["test_year"] == choice["test_year"])
                    & (hof_bootstrap["left_policy"] == choice["policy"])
                    & (hof_bootstrap["right_policy"] == rhs)
                    & (hof_bootstrap["metric"] == metric)
                ]
                if data.empty:
                    continue
                row = data.iloc[0].to_dict()
                row.update(
                    {
                        "candidate": choice["candidate"],
                        "family": choice["family"],
                        "selected_policy": choice["policy"],
                        "complexity": choice["complexity"],
                    }
                )
                detail_rows.append(row)
    detail = pd.DataFrame(detail_rows)

    summary_rows = []
    for rhs, label in [("bs", "Formula--BS"), ("agent", "Formula--Agent")]:
        row = {"comparison": label}
        for metric in DISTILLATION_METRIC_ORDER:
            data = detail[(detail["right_policy"] == rhs) & (detail["metric"] == metric)]
            if data.empty:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_fav"] = np.nan
                row[f"{metric}_bad"] = np.nan
                row[f"{metric}_indist"] = np.nan
                continue
            fav, bad, indist = sig_counts(data, metric)
            row[f"{metric}_mean"] = data["point_estimate"].mean()
            row[f"{metric}_fav"] = fav
            row[f"{metric}_bad"] = bad
            row[f"{metric}_indist"] = indist
        summary_rows.append(row)

    agent_row = {"comparison": "Agent--BS"}
    agent_summary = summarize_agent_vs_bs(agent_bs_summary)
    for metric in DISTILLATION_METRIC_ORDER:
        agent_row[f"{metric}_mean"] = agent_summary[f"{metric}_mean"]
        agent_row[f"{metric}_fav"] = agent_summary[f"{metric}_fav"]
        agent_row[f"{metric}_bad"] = agent_summary[f"{metric}_bad"]
        agent_row[f"{metric}_indist"] = (
            len(agent_bs_summary["year"].unique())
            - agent_summary[f"{metric}_fav"]
            - agent_summary[f"{metric}_bad"]
        )

    # Cosmetic paper-table order: show the raw agent benchmark first, then the
    # selected symbolic formula against Black-Scholes, then formula-vs-agent.
    ordered = [agent_row, summary_rows[0], summary_rows[1]]
    return detail, pd.DataFrame(ordered)


def write_parsimonious_trading_table(summary: pd.DataFrame, output_path: Path) -> None:
    def fmt_cell(row: pd.Series, metric: str) -> str:
        value = row[f"{metric}_mean"]
        fav = int(row[f"{metric}_fav"])
        bad = int(row[f"{metric}_bad"])
        return f"{tex_num(value)} ({fav}/{bad})"

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        r"Comparison & Reward & CVaR 5\% & Log Downside Variance & Log Variance \\",
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['comparison']} & "
            f"{fmt_cell(row, 'rew')} & "
            f"{fmt_cell(row, 'cvar')} & "
            f"{fmt_cell(row, 'log_down_var_ratio')} & "
            f"{fmt_cell(row, 'log_var_ratio')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# The next three helpers create the paper's robustness tables from frozen
# outputs.  They deliberately avoid any model retesting or expensive bootstrap:
# seed robustness uses point estimates from already-traded test CSVs, while the
# selection-rule and switching tables read cached bootstrap summaries generated
# by distill_empirical_agents.py and run_switching_robustness.py.

LONG_HORIZON_METRIC_LABELS = {
    "mean": "Mean terminal P&L",
    "rew": "Accumulated reward",
    "log_down_var_ratio": "Log downside variance",
    "log_var_ratio": "Log variance",
}


def load_long_horizon_pair_summary(args: argparse.Namespace) -> pd.DataFrame:
    """Read cached long-horizon pair bootstrap results.

    
    This is intentionally a pure reader.  The long-horizon script performs the
    expensive retesting/bootstrap.  The paper script only turns the frozen
    triangular panel into compact figures for the robustness section.
    """

    path = args.long_horizon_results
    if path.is_dir():
        path = path / args.prefix / "long_horizon_pair_bootstrap_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing long-horizon pair bootstrap summary: {path}"
        )
    summary = pd.read_csv(path)
    required = {
        "model_year",
        "target_year",
        "metric",
        "point_estimate",
        "significant_90",
        "significant_95",
        "significant_99",
    }
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return summary.copy()


def plot_long_horizon_heatmaps(
    long_summary: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
) -> None:
    """Draw triangular long-horizon policy-stress heatmaps.

    Green always means favorable for the agent and red means unfavorable.  The
    annotated number is the original point estimate, so log variance/downside
    variance cells remain in log-ratio units where negative values are good.
    """

    years = list(
        range(
            int(long_summary["model_year"].min()),
            int(long_summary["target_year"].max()) + 1,
        )
    )
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(7.2, 3.85 * len(metrics)),
        sharex=True,
        sharey=True,
    )
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = long_summary[long_summary["metric"] == metric].copy()
        if data.empty:
            raise ValueError(f"Long-horizon metric not found: {metric}")

        values = data.pivot(
            index="model_year", columns="target_year", values="point_estimate"
        ).reindex(index=years, columns=years)
        sig = data.pivot(
            index="model_year", columns="target_year", values="significant_95"
        ).reindex(index=years, columns=years)
        sig90 = data.pivot(
            index="model_year", columns="target_year", values="significant_90"
        ).reindex(index=years, columns=years)
        sig99 = data.pivot(
            index="model_year", columns="target_year", values="significant_99"
        ).reindex(index=years, columns=years)

        # For reward and mean P&L positive is favorable.  For variance ratios
        # negative is favorable, so multiply by -1 before coloring.  The cell
        # text still reports the economically standard point estimate.
        higher_is_better = metric in {"mean", "rew"}
        color_values = values if higher_is_better else -values
        finite = color_values.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        vmax = float(np.nanmax(np.abs(finite))) if len(finite) else 1.0
        vmax = max(vmax, 1e-6)

        cmap = plt.get_cmap("RdYlGn").copy()
        cmap.set_bad("#f3f3f3")
        image = ax.imshow(
            color_values.to_numpy(dtype=float),
            cmap=cmap,
            norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            aspect="auto",
        )

        for i, model_year in enumerate(years):
            for j, target_year in enumerate(years):
                value = values.loc[model_year, target_year]
                if pd.isna(value):
                    continue
                # Use the cached bootstrap significance levels directly:
                # one star for 90%, two for 95%, three for 99%.
                if bool(sig99.loc[model_year, target_year]):
                    star = "***"
                elif bool(sig.loc[model_year, target_year]):
                    star = "**"
                elif bool(sig90.loc[model_year, target_year]):
                    star = "*"
                else:
                    star = ""
                ax.text(
                    j,
                    i,
                    f"{value:.2f}{star}",
                    ha="center",
                    va="center",
                    fontsize=10.8,
                    color="#111111",
                )

        ax.set_title(
            LONG_HORIZON_METRIC_LABELS.get(metric, metric), fontweight="bold", pad=8
        )
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, rotation=45, ha="right")
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years)
        ax.set_xlabel("Evaluation year")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Thin white cell separators make the triangular panel readable in print.
        ax.set_xticks(np.arange(-0.5, len(years), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(years), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Favorable direction", rotation=90)

    set_y_label(axes[0], "Frozen policy year")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_or_build_seed_summary(args: argparse.Namespace, prefix: str) -> pd.DataFrame:
    """Return cached bootstrap summary for one seed-robustness prefix.

    
    The main walk-forward seed is already reported in the main performance
    table.  For robustness we build full table-style summaries only for the two
    additional independent seed classes and cache them under paper/figures.
    """

    cache_path = args.seed_summary_dir / f"robust_seed_summary_{prefix}.csv"
    required_metrics = set(METRIC_INFO)
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if required_metrics.issubset(set(cached["metric"].unique())):
            return cached

    seed_args = argparse.Namespace(**vars(args))
    seed_args.prefix = prefix
    print(f"[paper figures] building seed robustness bootstrap summary for {prefix}")
    summary = collect_summary(seed_args)
    output_cache_path = args.output_dir / f"robust_seed_summary_{prefix}.csv"
    summary.to_csv(output_cache_path, index=False)
    return summary


def write_seed_metric_table(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a Table-4-style seed robustness table with mean P&L included."""

    def stars(row: pd.Series) -> str:
        if row["sig99"]:
            return r"\sym{***}"
        if row["sig95"]:
            return r"\sym{**}"
        if row["sig90"]:
            return r"\sym{*}"
        return ""

    def fmt(row: pd.Series) -> str:
        return tex_num(row["center"]) + stars(row)

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{@{}cccccc@{}}",
        r"\toprule",
        (
            r"Year & Reward & CVaR 5\% & Mean P\&L & "
            r"Log Downside Variance & Log Variance \\"
        ),
        r"\midrule",
    ]
    for year in sorted(summary["year"].unique()):
        values = {}
        for metric in [
            "reward",
            "cvar",
            "mean_pnl",
            "log_downside_variance",
            "log_variance",
        ]:
            row = summary[
                (summary["year"] == year) & (summary["metric"] == metric)
            ].iloc[0]
            values[metric] = fmt(row)
        lines.append(
            f"{year} & {values['reward']} & {values['cvar']} & "
            f"{values['mean_pnl']} & {values['log_downside_variance']} & "
            f"{values['log_variance']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_antonov_defect_panel(
    long_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot the Antonov defect comparison from cached one-year evaluations."""

    data = long_summary[
        (long_summary["metric"] == "antonov")
        & (long_summary["model_year"] == long_summary["target_year"])
    ].copy()
    if data.empty:
        raise ValueError("No diagonal Antonov rows found in long-horizon summary")
    data = data.sort_values("model_year")

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.75, zorder=1)
    ax.axvspan(2021.5, 2023.5, color="#d9d9d9", alpha=0.25, zorder=0)

    for _, row in data.iterrows():
        center = row["point_estimate"]
        yerr = np.array(
            [[center - row["ci_low_95"]], [row["ci_high_95"] - center]],
            dtype=float,
        )
        significant = bool(row["significant_95"])
        # Lower Antonov defect is favorable, so negative differences are good.
        if not significant:
            color = "#6f7f8f"
            face = "white"
        elif center < 0.0:
            color = "#1b8a5a"
            face = color
        else:
            color = "#b33a3a"
            face = color
        ax.errorbar(
            row["model_year"],
            center,
            yerr=yerr,
            fmt="o",
            color=color,
            mfc=face,
            mec=color,
            ms=7,
            capsize=3.5,
            elinewidth=1.2,
            zorder=3,
        )

    ax.set_xticks(sorted(data["model_year"].unique()))
    ax.set_xlabel("Test year")
    set_y_label(ax, "Agent - BS defect")
    ax.set_title("Antonov Hedging-Defect Difference", fontweight="bold")
    ax.grid(True, axis="y", ls=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def collect_seed_robustness_ranges(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for prefix in SEED_ROBUSTNESS_PREFIXES:
        files = sorted(args.results_folder.glob(f"{prefix}*.csv"))
        if not files:
            raise FileNotFoundError(f"No seed-robustness result files found for {prefix}")
        for path in files:
            year = extract_year(path, prefix)
            if year is None:
                continue
            metrics = calculate_metrics(load_episode_data(path))
            rows.append(
                {
                    "prefix": prefix,
                    "year": year,
                    "reward": metrics["reward"],
                    "log_downside_variance": metrics["log_downside_variance"],
                    "log_variance": metrics["log_variance"],
                }
            )
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("No seed-robustness rows were collected")

    out_rows = []
    for year, group in data.groupby("year"):
        out_rows.append(
            {
                "year": int(year),
                "reward_min": group["reward"].min(),
                "reward_max": group["reward"].max(),
                "down_min": group["log_downside_variance"].min(),
                "down_max": group["log_downside_variance"].max(),
                "var_min": group["log_variance"].min(),
                "var_max": group["log_variance"].max(),
                "n_seeds": int(group["prefix"].nunique()),
            }
        )
    return pd.DataFrame(out_rows).sort_values("year").reset_index(drop=True)


def write_seed_robustness_table(seed_ranges: pd.DataFrame, output_path: Path) -> None:
    def interval(row: pd.Series, left: str, right: str) -> str:
        return rf"$[{row[left]:.3f},{row[right]:.3f}]$"

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{@{}cccc@{}}",
        r"\toprule",
        r"Year & Reward & Log Downside Variance & Log Variance \\",
        r"\midrule",
    ]
    for _, row in seed_ranges.iterrows():
        lines.append(
            f"{int(row['year'])} & "
            f"{interval(row, 'reward_min', 'reward_max')} & "
            f"{interval(row, 'down_min', 'down_max')} & "
            f"{interval(row, 'var_min', 'var_max')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_best_validation_fit(validation_hof_fidelity: pd.DataFrame) -> pd.DataFrame:
    raw_target = validation_hof_fidelity[
        validation_hof_fidelity["target_name"] == "actual_traded_agent"
    ].copy()
    selected = []
    for year, group in raw_target.groupby("test_year"):
        selected.append(group.sort_values(["mae", "complexity"]).iloc[0])
    return pd.DataFrame(selected).reset_index(drop=True)


def select_best_validation_metric(
    validation_hof_fidelity: pd.DataFrame,
    validation_hof_bootstrap: pd.DataFrame,
    metric: str,
    higher_is_better: bool,
) -> pd.DataFrame:
    selected = []
    data = validation_hof_bootstrap[
        (validation_hof_bootstrap["right_policy"] == "bs")
        & (validation_hof_bootstrap["metric"] == metric)
    ].copy()
    raw_target = validation_hof_fidelity[
        validation_hof_fidelity["target_name"] == "actual_traded_agent"
    ]
    for year, group in data.groupby("test_year"):
        group = group.sort_values(
            ["point_estimate", "left_policy"],
            ascending=[not higher_is_better, True],
        )
        policy = group.iloc[0]["left_policy"]
        selected.append(
            raw_target[
                (raw_target["test_year"] == year) & (raw_target["policy"] == policy)
            ].iloc[0]
        )
    return pd.DataFrame(selected).reset_index(drop=True)


def summarize_selection_rule_trading(
    selected_by_rule: dict[str, pd.DataFrame],
    test_hof_fidelity: pd.DataFrame,
    test_hof_bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    def sig_counts(data: pd.DataFrame, metric: str) -> tuple[int, int, int]:
        higher_is_better = metric in {"rew", "cvar", "mean"}
        significant = data["significant"].astype(bool)
        if higher_is_better:
            favorable = significant & (data["point_estimate"] > 0.0)
            unfavorable = significant & (data["point_estimate"] < 0.0)
        else:
            favorable = significant & (data["point_estimate"] < 0.0)
            unfavorable = significant & (data["point_estimate"] > 0.0)
        return int(favorable.sum()), int(unfavorable.sum()), int(len(data) - favorable.sum() - unfavorable.sum())

    rows = []
    raw_test_fidelity = test_hof_fidelity[
        test_hof_fidelity["target_name"] == "actual_traded_agent"
    ]
    for rule_name, selected in selected_by_rule.items():
        selected_pairs = list(zip(selected["test_year"], selected["policy"]))
        test_mae = []
        for year, policy in selected_pairs:
            match = raw_test_fidelity[
                (raw_test_fidelity["test_year"] == year)
                & (raw_test_fidelity["policy"] == policy)
            ]
            if match.empty:
                raise ValueError(f"Missing test fidelity for {rule_name}, {year}, {policy}")
            test_mae.append(float(match.iloc[0]["mae"]))

        row = {
            "rule": rule_name,
            "avg_complexity": float(selected["complexity"].mean()),
            "test_mae": float(np.mean(test_mae)),
        }
        for rhs in ["bs", "agent"]:
            for metric in ["rew", "cvar", "log_down_var_ratio", "log_var_ratio"]:
                pieces = []
                for year, policy in selected_pairs:
                    match = test_hof_bootstrap[
                        (test_hof_bootstrap["test_year"] == year)
                        & (test_hof_bootstrap["left_policy"] == policy)
                        & (test_hof_bootstrap["right_policy"] == rhs)
                        & (test_hof_bootstrap["metric"] == metric)
                    ]
                    if match.empty:
                        raise ValueError(
                            f"Missing bootstrap row for {rule_name}, {year}, {policy}, {rhs}, {metric}"
                        )
                    pieces.append(match.iloc[0])
                data = pd.DataFrame(pieces)
                fav, bad, indist = sig_counts(data, metric)
                row[f"{rhs}_{metric}_mean"] = float(data["point_estimate"].mean())
                row[f"{rhs}_{metric}_fav"] = fav
                row[f"{rhs}_{metric}_bad"] = bad
                row[f"{rhs}_{metric}_indist"] = indist
        rows.append(row)
    return pd.DataFrame(rows)


def write_selection_rule_robustness_table(summary: pd.DataFrame, output_path: Path) -> None:
    rule_labels = {
        "Parsimonious 10\\% fit": r"Pars. 10\%",
        "Best validation fit": "Val. fit",
        "Best validation reward": "Val. reward",
        "Best validation CVaR": "Val. CVaR",
        "Best validation downside": "Val. downside",
    }

    def fmt_metric(row: pd.Series, prefix: str) -> str:
        return (
            f"{tex_num(row[prefix + '_mean'])} "
            rf"({int(row[prefix + '_fav'])}/{int(row[prefix + '_bad'])})"
        )

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.16}",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        (
            r"Rule & Comp. & MAE & Rew--BS & Down--BS & "
            r"Rew--Agent & Down--Agent \\"
        ),
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{rule_labels.get(row['rule'], row['rule'])} & "
            f"{tex_num(row['avg_complexity'], digits=1)} & "
            f"{tex_num(row['test_mae'])} & "
            f"{fmt_metric(row, 'bs_rew')} & "
            f"{fmt_metric(row, 'bs_log_down_var_ratio')} & "
            f"{fmt_metric(row, 'agent_rew')} & "
            f"{fmt_metric(row, 'agent_log_down_var_ratio')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_hull_white_bootstrap(args: argparse.Namespace) -> pd.DataFrame:
    """Read the cached Hull--White benchmark bootstrap table.

    
    Hull--White is a paper robustness check generated by run_hull_white_benchmark.py.
    This script must be the only place that turns those cached benchmark rows
    into LaTeX tables, so the manuscript never depends on hand-written table
    artifacts.
    """

    path = args.hull_white_results
    if path.is_dir() and (path / "hull_white_test_bootstrap_all.csv").exists():
        path = path / "hull_white_test_bootstrap_all.csv"
    elif path.is_dir():
        path = path / args.prefix / "hull_white_test_bootstrap_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Hull-White bootstrap summary: {path}")

    data = pd.read_csv(path)
    required = {
        "comparison",
        "metric",
        "test_year",
        "point_estimate",
        "significant_90",
        "significant_95",
        "significant_99",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for col in ["significant_90", "significant_95", "significant_99"]:
        if data[col].dtype == object:
            data[col] = data[col].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            data[col] = data[col].astype(bool)
    return data.copy()


def write_hull_white_yearly_table(
    hull_bootstrap: pd.DataFrame, comparison: str, output_path: Path
) -> None:
    """Write a Table-1-style yearly Hull--White comparison table."""

    metric_order = ["rew", "cvar", "log_down_var_ratio", "log_var_ratio"]
    metric_headers = {
        "rew": "Reward",
        "cvar": r"CVaR 5\%",
        "log_down_var_ratio": "Log Downside Variance",
        "log_var_ratio": "Log Variance",
    }

    def fmt(row: pd.Series) -> str:
        if bool(row["significant_99"]):
            star = r"\sym{***}"
        elif bool(row["significant_95"]):
            star = r"\sym{**}"
        elif bool(row["significant_90"]):
            star = r"\sym{*}"
        else:
            star = ""
        return tex_num(row["point_estimate"]) + star

    data = hull_bootstrap[
        (hull_bootstrap["comparison"] == comparison)
        & (hull_bootstrap["metric"].isin(metric_order))
    ].copy()
    if data.empty:
        raise ValueError(f"No Hull-White rows found for comparison={comparison}")

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        (
            "Year & "
            + " & ".join(metric_headers[metric] for metric in metric_order)
            + r" \\"
        ),
        r"\midrule",
    ]
    for year in sorted(data["test_year"].unique()):
        cells = []
        for metric in metric_order:
            match = data[(data["test_year"] == year) & (data["metric"] == metric)]
            if match.empty:
                raise ValueError(
                    f"Missing Hull-White row for {comparison}, {year}, {metric}"
                )
            cells.append(fmt(match.iloc[0]))
        lines.append(f"{int(year)} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_hull_white_summary_table(
    hull_bootstrap: pd.DataFrame, output_path: Path
) -> None:
    """Write the compact main-text Hull--White summary table.

    Parentheses use 95% two-stage bootstrap significance to match the current
    robustness-table convention in the manuscript.
    """

    comparisons = [
        ("hw_vs_agent", "Hull-White vs. Agent"),
        ("hw_vs_selected_formula", "Hull-White vs. Formula"),
    ]
    metric_order = ["rew", "cvar", "log_down_var_ratio", "log_var_ratio"]
    metric_headers = ["Reward", r"CVaR 5\%", "Log Downside Variance", "Log Variance"]

    def counts(data: pd.DataFrame, metric: str) -> tuple[int, int]:
        higher_is_better = metric in {"rew", "cvar", "mean"}
        significant = data["significant_95"].astype(bool)
        if higher_is_better:
            favorable = significant & (data["point_estimate"] > 0.0)
            unfavorable = significant & (data["point_estimate"] < 0.0)
        else:
            favorable = significant & (data["point_estimate"] < 0.0)
            unfavorable = significant & (data["point_estimate"] > 0.0)
        return int(favorable.sum()), int(unfavorable.sum())

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.18}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}ccccc@{}}",
        r"\toprule",
        "Comparison & " + " & ".join(metric_headers) + r" \\",
        r"\midrule",
    ]
    for comparison, label in comparisons:
        cells = []
        for metric in metric_order:
            data = hull_bootstrap[
                (hull_bootstrap["comparison"] == comparison)
                & (hull_bootstrap["metric"] == metric)
            ].copy()
            if data.empty:
                raise ValueError(f"No Hull-White rows for {comparison}, {metric}")
            fav, bad = counts(data, metric)
            cells.append(f"{tex_num(data['point_estimate'].mean())} ({fav}/{bad})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_haircut_bootstrap(args: argparse.Namespace) -> pd.DataFrame:
    """Read the cached scalar-delta-haircut benchmark bootstrap table."""

    path = args.haircut_results
    if path.is_dir() and (path / "haircut_test_bootstrap_all.csv").exists():
        path = path / "haircut_test_bootstrap_all.csv"
    elif path.is_dir():
        path = path / args.prefix / "haircut_test_bootstrap_all.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing haircut bootstrap summary: {path}")

    data = pd.read_csv(path)
    required = {
        "comparison",
        "metric",
        "test_year",
        "point_estimate",
        "significant_95",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for col in ["significant_90", "significant_95", "significant_99"]:
        if col in data.columns:
            if data[col].dtype == object:
                data[col] = data[col].astype(str).str.lower().isin(["true", "1", "yes"])
            else:
                data[col] = data[col].astype(bool)
    return data.copy()


def write_haircut_summary_table(
    haircut_bootstrap: pd.DataFrame, output_path: Path
) -> None:
    """Write one compact appendix table for the scalar haircut benchmark."""

    comparisons = [
        ("haircut_vs_bs", "Haircut--BS"),
        ("haircut_vs_agent", "Haircut--Agent"),
        ("haircut_vs_selected_formula", "Haircut--Formula"),
    ]
    metric_order = ["rew", "cvar", "log_down_var_ratio", "log_var_ratio"]
    metric_headers = ["Reward", r"CVaR 5\%", "Log Downside Variance", "Log Variance"]

    def counts(data: pd.DataFrame, metric: str) -> tuple[int, int]:
        higher_is_better = metric in {"rew", "cvar", "mean"}
        significant = data["significant_95"].astype(bool)
        if higher_is_better:
            favorable = significant & (data["point_estimate"] > 0.0)
            unfavorable = significant & (data["point_estimate"] < 0.0)
        else:
            favorable = significant & (data["point_estimate"] < 0.0)
            unfavorable = significant & (data["point_estimate"] > 0.0)
        return int(favorable.sum()), int(unfavorable.sum())

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\setlength{\tabcolsep}{7pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        "Comparison & " + " & ".join(metric_headers) + r" \\",
        r"\midrule",
    ]
    for comparison, label in comparisons:
        cells = []
        for metric in metric_order:
            data = haircut_bootstrap[
                (haircut_bootstrap["comparison"] == comparison)
                & (haircut_bootstrap["metric"] == metric)
            ].copy()
            if data.empty:
                raise ValueError(f"No haircut rows for {comparison}, {metric}")
            fav, bad = counts(data, metric)
            cells.append(rf"${data['point_estimate'].mean():.3f}$ ({fav}/{bad})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_switching_negative_result(args: argparse.Namespace) -> pd.DataFrame:
    switch_dir = args.input_dir / "results/switching_test_2022"
    fidelity_path = switch_dir / "final_WF_exp1_k1_test2022_20000_test_fidelity.csv"
    bootstrap_path = switch_dir / "final_WF_exp1_k1_test2022_20000_test_bootstrap_summary.csv"
    if not fidelity_path.exists() or not bootstrap_path.exists():
        raise FileNotFoundError(f"Missing cached switching-test outputs in {switch_dir}")

    validation_hof_fidelity = load_distillation_hof_fidelity(args, "validation")
    test_hof_fidelity = load_distillation_hof_fidelity(args, "test")
    test_hof_bootstrap = load_distillation_hof_bootstrap(args, "test")
    parsimonious = select_parsimonious_hof_candidates(validation_hof_fidelity)
    policy_2022 = parsimonious[parsimonious["test_year"] == 2022].iloc[0]["policy"]

    rows = []
    raw_test_fidelity = test_hof_fidelity[
        test_hof_fidelity["target_name"] == "actual_traded_agent"
    ]
    fid = raw_test_fidelity[
        (raw_test_fidelity["test_year"] == 2022)
        & (raw_test_fidelity["policy"] == policy_2022)
    ].iloc[0]
    row = {
        "policy": "Global parsimonious",
        "mae": float(fid["mae"]),
        "p95_abs_error": float(fid["p95_abs_error"]),
    }
    for rhs in ["agent", "bs"]:
        for metric in ["rew", "log_down_var_ratio"]:
            boot = test_hof_bootstrap[
                (test_hof_bootstrap["test_year"] == 2022)
                & (test_hof_bootstrap["left_policy"] == policy_2022)
                & (test_hof_bootstrap["right_policy"] == rhs)
                & (test_hof_bootstrap["metric"] == metric)
            ].iloc[0]
            row[f"{rhs}_{metric}"] = float(boot["point_estimate"])
            row[f"{rhs}_{metric}_sig"] = bool(boot["significant"])
    rows.append(row)

    switch_fidelity = pd.read_csv(fidelity_path)
    switch_bootstrap = pd.read_csv(bootstrap_path)
    for label, policy in [
        ("2-band switching", "switch2_moneyness_bs_delta_residual"),
        ("3-band switching", "switch3_moneyness_bs_delta_residual"),
    ]:
        fid = switch_fidelity[
            (switch_fidelity["policy"] == policy)
            & (switch_fidelity["target_delta_col"] == "actual_agent_delta")
        ].iloc[0]
        row = {
            "policy": label,
            "mae": float(fid["mae"]),
            "p95_abs_error": float(fid["p95_abs_error"]),
        }
        for rhs in ["agent", "bs"]:
            for metric in ["rew", "log_down_var_ratio"]:
                boot = switch_bootstrap[
                    (switch_bootstrap["split"] == "test")
                    & (switch_bootstrap["left_policy"] == policy)
                    & (switch_bootstrap["right_policy"] == rhs)
                    & (switch_bootstrap["metric"] == metric)
                ].iloc[0]
                row[f"{rhs}_{metric}"] = float(boot["point_estimate"])
                row[f"{rhs}_{metric}_sig"] = bool(boot["significant"])
        rows.append(row)
    return pd.DataFrame(rows)


def write_switching_negative_result_table(summary: pd.DataFrame, output_path: Path) -> None:
    policy_labels = {
        "Global parsimonious": "Global",
        "2-band switching": "2-band",
        "3-band switching": "3-band",
    }

    def fmt(value: float, significant: bool, higher_is_better: bool) -> str:
        favorable = value > 0.0 if higher_is_better else value < 0.0
        star = r"\sym{*}" if significant else ""
        return tex_num(value) + star

    lines = [
        r"\begingroup",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\begin{tabular}{@{}ccccccc@{}}",
        r"\toprule",
        (
            r"Policy & MAE & 95\% err. & Rew--Agent & "
            r"Down--Agent & Rew--BS & Down--BS \\"
        ),
        r"\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{policy_labels.get(row['policy'], row['policy'])} & "
            f"{tex_num(row['mae'])} & "
            f"{tex_num(row['p95_abs_error'])} & "
            f"{fmt(row['agent_rew'], row['agent_rew_sig'], True)} & "
            f"{fmt(row['agent_log_down_var_ratio'], row['agent_log_down_var_ratio_sig'], False)} & "
            f"{fmt(row['bs_rew'], row['bs_rew_sig'], True)} & "
            f"{fmt(row['bs_log_down_var_ratio'], row['bs_log_down_var_ratio_sig'], False)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\endgroup"])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_parsimonious_smooth_focus_vs_agent(
    selected: pd.DataFrame, hof_bootstrap: pd.DataFrame, output_path: Path
) -> None:
    metrics = [
        ("rew", "Reward", True),
        ("log_down_var_ratio", "Log Downside Variance", False),
    ]
    choices = selected
    years = sorted(choices["test_year"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), sharex=True)
    for ax, (metric, title, higher_is_better) in zip(axes, metrics):
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.75)
        if 2022 in years or 2023 in years:
            ax.axvspan(2021.5, 2023.5, color="#d9d9d9", alpha=0.25, zorder=0)
        for _, choice in choices.sort_values("test_year").iterrows():
            data = hof_bootstrap[
                (hof_bootstrap["test_year"] == choice["test_year"])
                & (hof_bootstrap["left_policy"] == choice["policy"])
                & (hof_bootstrap["right_policy"] == "agent")
                & (hof_bootstrap["metric"] == metric)
            ]
            if data.empty:
                continue
            row = data.iloc[0]
            center = row["point_estimate"]
            yerr = np.array([[center - row["ci_low"]], [row["ci_high"] - center]])
            significant = bool(row["significant"])
            favorable = center > 0.0 if higher_is_better else center < 0.0
            if not significant:
                color, marker, face = "#6f7f8f", "o", "white"
            elif favorable:
                color, marker, face = "#1b8a5a", "^" if center > 0 else "v", "#1b8a5a"
            else:
                color, marker, face = "#b33a3a", "^" if center > 0 else "v", "#b33a3a"
            ax.errorbar(
                choice["test_year"],
                center,
                yerr=yerr,
                fmt=marker,
                color=color,
                mfc=face,
                mec=color,
                ms=7,
                capsize=3.5,
                elinewidth=1.2,
            )
        ax.set_title(title, fontweight="bold", pad=5)
        set_y_label(ax, "Formula - agent" if metric == "rew" else "Log ratio")
        ax.grid(True, axis="y", ls=":", alpha=0.45)
    axes[-1].set_xticks(years)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_agent_formula_cvar_vs_bs(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    hof_bootstrap: pd.DataFrame,
    output_path: Path,
) -> None:
    years = sorted(selected["test_year"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)

    formula_rows = []
    for _, choice in selected.sort_values("test_year").iterrows():
        data = hof_bootstrap[
            (hof_bootstrap["test_year"] == choice["test_year"])
            & (hof_bootstrap["left_policy"] == choice["policy"])
            & (hof_bootstrap["right_policy"] == "bs")
            & (hof_bootstrap["metric"] == "cvar")
        ]
        if data.empty:
            raise ValueError(
                f"Missing selected-formula CVaR bootstrap row for {choice['test_year']}"
            )
        row = data.iloc[0]
        formula_rows.append(
            {
                "year": int(choice["test_year"]),
                "center": float(row["point_estimate"]),
                "lower": float(row["ci_low"]),
                "upper": float(row["ci_high"]),
                "significant": bool(row["significant"]),
                "outperforms": bool(row["significant"]) and float(row["point_estimate"]) > 0.0,
            }
        )

    panels = [
        ("Agent - BS", summary[summary["metric"].eq("cvar")].copy()),
        ("Formula - BS", pd.DataFrame(formula_rows)),
    ]
    for ax, (title, data) in zip(axes, panels):
        data = data.sort_values("year")
        ax.axhline(0.0, color="black", lw=1.0, ls="--", alpha=0.75, zorder=1)
        if 2022 in years or 2023 in years:
            ax.axvspan(2021.5, 2023.5, color="#d9d9d9", alpha=0.25, zorder=0)
        for _, row in data.iterrows():
            center = float(row["center"])
            yerr = np.array([[center - row["lower"]], [row["upper"] - center]], dtype=float)
            significant = bool(row["significant"])
            favorable = center > 0.0
            if not significant:
                color, marker, face = "#6f7f8f", "o", "white"
            elif favorable:
                color, marker, face = "#1b8a5a", "^", "#1b8a5a"
            else:
                color, marker, face = "#b33a3a", "v", "#b33a3a"
            ax.errorbar(
                row["year"],
                center,
                yerr=yerr,
                fmt=marker,
                color=color,
                mfc=face,
                mec=color,
                ms=7,
                capsize=3.5,
                elinewidth=1.2,
                zorder=4 if significant else 3,
            )
        ax.set_title(title, fontweight="bold", pad=5)
        set_y_label(ax, "CVaR difference")
        ax.grid(True, axis="y", ls=":", alpha=0.45)
    axes[-1].set_xticks(years)

    not_sig = mlines.Line2D(
        [], [], color="#6f7f8f", marker="o", mfc="white", ls="None",
        label="95% CI includes zero",
    )
    sig_out = mlines.Line2D(
        [], [], color="#1b8a5a", marker="^", mfc="#1b8a5a", ls="None",
        label="Significant improvement",
    )
    sig_under = mlines.Line2D(
        [], [], color="#b33a3a", marker="v", mfc="#b33a3a", ls="None",
        label="Significant deterioration",
    )
    fig.legend(
        handles=[not_sig, sig_out, sig_under],
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def load_or_build_summary(args: argparse.Namespace) -> pd.DataFrame:
    summary_csv = args.summary_csv
    if summary_csv.exists() and not args.recompute_summary:
        print(f"[paper figures] reading cached summary {summary_csv}")
        summary = pd.read_csv(summary_csv)
        bool_cols = ["significant", "sig90", "sig95", "sig99", "outperforms"]
        for col in bool_cols:
            if col in summary.columns:
                if summary[col].dtype == object:
                    summary[col] = summary[col].astype(str).str.lower().isin(["true", "1", "yes"])
                else:
                    summary[col] = summary[col].astype(bool)
        missing_metrics = set(METRIC_INFO) - set(summary["metric"].unique())
        if not missing_metrics:
            return summary
        print(
            "[paper figures] cached summary is missing "
            f"{sorted(missing_metrics)}; recomputing bootstrap summary once"
        )

    print("[paper figures] recomputing bootstrap summary")
    summary = collect_summary(args)
    output_summary_csv = args.output_dir / "wf_metric_summary.csv"
    summary.to_csv(output_summary_csv, index=False)
    print(f"[paper figures] wrote {output_summary_csv}")
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    summary = load_or_build_summary(args)

    plot_metric_pair(
        summary,
        ["log_downside_variance", "log_variance"],
        args.output_dir / "wf_variance_downside_pair.pdf",
        "Terminal P&L Dispersion: Downside Versus Ordinary Variance",
    )
    plot_metric_pair(
        summary,
        ["reward", "mean_pnl"],
        args.output_dir / "wf_reward_pnl_pair.pdf",
        "Objective-Aligned Performance and Mean Terminal P&L",
    )
    write_metric_table(summary, args.table_dir / "wf_metric_summary.tex")
    delta_summary = collect_delta_summary(args)
    delta_summary.to_csv(args.output_dir / "learned_delta_summary.csv", index=False)
    write_delta_summary_table(delta_summary, args.table_dir / "learned_delta_summary.tex")

    delta_surface = collect_delta_gap_surface(args)
    delta_surface.to_csv(args.output_dir / "learned_delta_gap_surface.csv", index=False)
    plot_delta_gap_surface(
        delta_surface, args.output_dir / "learned_delta_gap_surface.pdf"
    )

    negative_state_summary = collect_negative_pnl_state_summary(args)
    negative_state_summary.to_csv(
        args.output_dir / "negative_pnl_state_summary.csv", index=False
    )
    write_negative_pnl_ranked_table(
        negative_state_summary, args.table_dir / "negative_pnl_ranked_summary.tex"
    )

    # Targeted regime-fragility diagnostics.  These are deliberately separate
    # from the walk-forward bootstrap figures: each figure/table supports one
    # mechanism subsection and is built from cached path-level diagnostics only.
    plot_2022_state_and_hedge(
        args, args.output_dir / "regime_2022_state_and_hedge.pdf"
    )
    plot_2022_vega_parachute(
        args, args.output_dir / "regime_2022_vega_parachute.pdf"
    )
    # This is the figure now cited in Section 9.2.  It is intentionally
    # lightweight and does not invoke the bootstrap; it only summarizes cached
    # path-level diagnostics produced by the forensic audit scripts.
    plot_2023_variance_mechanism(
        args, args.output_dir / "regime_2023_variance_mechanism.pdf"
    )
    write_bs_cancellation_table(
        args, args.table_dir / "regime_bs_cancellation.tex"
    )
    # These mechanism outputs are generated from cached forensic summaries, so
    # this remains cheap and reproducible.
    plot_2017_variance_mechanism(
        args, args.output_dir / "regime_2017_variance_mechanism.pdf"
    )
    write_2017_cluster_mechanism_table(
        args, args.table_dir / "regime_2017_cluster_mechanism.tex"
    )
    validation_hof_fidelity = load_distillation_hof_fidelity(args, "validation")
    test_hof_fidelity = load_distillation_hof_fidelity(args, "test")
    test_hof_bootstrap = load_distillation_hof_bootstrap(args, "test")
    test_bootstrap = load_distillation_bootstrap(args)
    write_smoother_tables(
        test_bootstrap,
        summary,
        args.table_dir / "distill_smoother_vs_agent_yearly.tex",
        args.table_dir / "distill_smoother_vs_bs_summary.tex",
    )
    parsimonious_selected = select_parsimonious_hof_candidates(validation_hof_fidelity)
    parsimonious_selected.to_csv(
        args.output_dir / "distill_parsimonious_selected.csv", index=False
    )
    parsimonious_fit_detail, parsimonious_fit_summary = collect_parsimonious_fit_summary(
        parsimonious_selected, test_hof_fidelity
    )
    parsimonious_fit_detail.to_csv(
        args.output_dir / "distill_parsimonious_fit_detail.csv", index=False
    )
    parsimonious_fit_summary.to_csv(
        args.output_dir / "distill_parsimonious_fit_summary.csv", index=False
    )
    write_selected_formula_table(
        parsimonious_fit_detail,
        args.table_dir / "distill_parsimonious_fit_summary.tex",
    )

    parsimonious_trading_detail, parsimonious_trading_summary = (
        collect_parsimonious_trading_summary(
            parsimonious_selected, test_hof_bootstrap, summary
        )
    )
    parsimonious_trading_detail.to_csv(
        args.output_dir / "distill_parsimonious_trading_detail.csv", index=False
    )
    parsimonious_trading_summary.to_csv(
        args.output_dir / "distill_parsimonious_trading_summary.csv", index=False
    )
    write_parsimonious_trading_table(
        parsimonious_trading_summary,
        args.table_dir / "distill_parsimonious_trading_summary.tex",
    )
    plot_parsimonious_smooth_focus_vs_agent(
        parsimonious_selected,
        test_hof_bootstrap,
        args.output_dir / "distill_parsimonious_smooth_focus_vs_agent_pair.pdf",
    )
    plot_agent_formula_cvar_vs_bs(
        summary,
        parsimonious_selected,
        test_hof_bootstrap,
        args.output_dir / "distill_agent_formula_cvar_vs_bs_pair.pdf",
    )

    # The main seed already appears in the main walk-forward performance table.
    # Robustness now reports the two additional seeds as full Table-4-style
    # metric tables, rather than as min/max intervals that mix in the main seed.
    extra_seed_prefixes = [
        prefix for prefix in SEED_ROBUSTNESS_PREFIXES if prefix != args.prefix
    ]
    for seed_prefix in extra_seed_prefixes:
        seed_summary = load_or_build_seed_summary(args, seed_prefix)
        write_seed_metric_table(
            seed_summary,
            args.table_dir / f"robust_seed_metrics_{seed_prefix}.tex",
        )

    long_horizon_summary = load_long_horizon_pair_summary(args)
    long_horizon_summary.to_csv(
        args.output_dir / "robust_long_horizon_pair_summary.csv", index=False
    )
    plot_long_horizon_heatmaps(
        long_horizon_summary,
        ["log_down_var_ratio", "log_var_ratio"],
        args.output_dir / "robust_long_horizon_variance_heatmaps.pdf",
    )
    plot_long_horizon_heatmaps(
        long_horizon_summary,
        ["rew", "mean"],
        args.output_dir / "robust_long_horizon_reward_mean_heatmaps.pdf",
    )
    validation_hof_bootstrap = load_distillation_hof_bootstrap(args, "validation")
    selection_rule_candidates = {
        "Parsimonious 10\\% fit": parsimonious_selected,
        "Best validation fit": select_best_validation_fit(validation_hof_fidelity),
        "Best validation reward": select_best_validation_metric(
            validation_hof_fidelity, validation_hof_bootstrap, "rew", True
        ),
        "Best validation CVaR": select_best_validation_metric(
            validation_hof_fidelity, validation_hof_bootstrap, "cvar", True
        ),
        "Best validation downside": select_best_validation_metric(
            validation_hof_fidelity, validation_hof_bootstrap, "log_down_var_ratio", False
        ),
    }
    selection_rule_summary = summarize_selection_rule_trading(
        selection_rule_candidates, test_hof_fidelity, test_hof_bootstrap
    )
    selection_rule_summary.to_csv(
        args.output_dir / "robust_selection_rule_summary.csv", index=False
    )
    write_selection_rule_robustness_table(
        selection_rule_summary, args.table_dir / "robust_selection_rule_summary.tex"
    )

    hull_white_bootstrap = load_hull_white_bootstrap(args)
    hull_white_bootstrap.to_csv(
        args.output_dir / "robust_hull_white_test_bootstrap.csv", index=False
    )
    write_hull_white_summary_table(
        hull_white_bootstrap, args.table_dir / "robust_hull_white_summary.tex"
    )
    write_hull_white_yearly_table(
        hull_white_bootstrap,
        "hw_vs_bs",
        args.table_dir / "robust_hull_white_vs_bs.tex",
    )
    write_hull_white_yearly_table(
        hull_white_bootstrap,
        "hw_vs_agent",
        args.table_dir / "robust_hull_white_vs_agent.tex",
    )
    write_hull_white_yearly_table(
        hull_white_bootstrap,
        "hw_vs_selected_formula",
        args.table_dir / "robust_hull_white_vs_formula.tex",
    )

    haircut_bootstrap = load_haircut_bootstrap(args)
    haircut_bootstrap.to_csv(
        args.output_dir / "robust_haircut_test_bootstrap.csv", index=False
    )
    write_haircut_summary_table(
        haircut_bootstrap, args.table_dir / "robust_haircut_summary.tex"
    )

    switching_summary = collect_switching_negative_result(args)
    switching_summary.to_csv(
        args.output_dir / "robust_switching_negative_result.csv", index=False
    )
    write_switching_negative_result_table(
        switching_summary, args.table_dir / "robust_switching_negative_result.tex"
    )
    print("[paper figures] wrote paper/figures/wf_variance_downside_pair.pdf")
    print("[paper figures] wrote paper/figures/wf_reward_pnl_pair.pdf")
    print("[paper figures] wrote paper/figures/learned_delta_gap_surface.pdf")
    print("[paper figures] wrote paper/figures/regime_2022_state_and_hedge.pdf")
    print("[paper figures] wrote paper/figures/regime_2022_vega_parachute.pdf")
    print("[paper figures] wrote paper/figures/regime_2023_variance_mechanism.pdf")
    print("[paper figures] wrote paper/figures/regime_2017_variance_mechanism.pdf")
    print(
        "[paper figures] wrote "
        "paper/figures/distill_parsimonious_smooth_focus_vs_agent_pair.pdf"
    )
    print("[paper figures] wrote paper/figures/distill_agent_formula_cvar_vs_bs_pair.pdf")
    print("[paper figures] wrote paper/figures/robust_long_horizon_variance_heatmaps.pdf")
    print("[paper figures] wrote paper/figures/robust_long_horizon_reward_mean_heatmaps.pdf")
    print("[paper figures] wrote paper/tables/wf_metric_summary.tex")
    print("[paper figures] wrote paper/tables/learned_delta_summary.tex")
    print("[paper figures] wrote paper/tables/negative_pnl_ranked_summary.tex")
    print("[paper figures] wrote paper/tables/regime_2017_cluster_mechanism.tex")
    print("[paper figures] wrote paper/tables/distill_smoother_vs_agent_yearly.tex")
    print("[paper figures] wrote paper/tables/distill_smoother_vs_bs_summary.tex")
    print("[paper figures] wrote paper/tables/distill_parsimonious_fit_summary.tex")
    print("[paper figures] wrote paper/tables/distill_parsimonious_trading_summary.tex")
    for seed_prefix in extra_seed_prefixes:
        print(f"[paper figures] wrote paper/tables/robust_seed_metrics_{seed_prefix}.tex")
    print("[paper figures] wrote paper/tables/robust_selection_rule_summary.tex")
    print("[paper figures] wrote paper/tables/robust_hull_white_summary.tex")
    print("[paper figures] wrote paper/tables/robust_hull_white_vs_bs.tex")
    print("[paper figures] wrote paper/tables/robust_hull_white_vs_agent.tex")
    print("[paper figures] wrote paper/tables/robust_hull_white_vs_formula.tex")
    print("[paper figures] wrote paper/tables/robust_haircut_summary.tex")
    print("[paper figures] wrote paper/tables/robust_switching_negative_result.tex")


if __name__ == "__main__":
    main()
