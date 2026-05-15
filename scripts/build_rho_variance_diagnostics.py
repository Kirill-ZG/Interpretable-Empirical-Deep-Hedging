"""Build rho-like variance diagnostics from walk-forward testing paths.

This script studies years in which terminal-PnL variance can deteriorate even
when reward and downside variance do not.  The main hypothesis is a breakdown
in the effective spot/variance relationship.  Because the result CSVs do not
contain calibrated Heston rho values, the script estimates empirical proxies
within each hedging episode:

1. corr(d log S, d implied vol)
2. corr(d log S, d implied variance)

These are not structural Heston parameters, but they are directly observable
from the same paths used for the reported PnLs.

Measurement convention:
    Var({sum_t daily_agent_pnl_i,t}_i) / Var({sum_t daily_bs_pnl_i,t}_i)

Variance is measured across terminal episode PnLs, not inside each episode.
The within-episode path is used only to estimate the explanatory variable for
that episode.
"""

import argparse
import glob
import math
import os
import re

import numpy as np
import pandas as pd

DEFAULT_PREFIXES = [
    "no_q_WF_exp1_k1_test",
    "2_no_q_WF_exp1_k1_test",
    "final_WF_exp1_k1_test",
]


def safe_var(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return np.var(x, ddof=1)


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def safe_beta(x, y):
    # Slope from regressing y on x with an intercept. For the rho hypothesis,
    # x is usually d log S and y is d implied variance. A more negative slope
    # means implied variance rises more strongly when spot falls.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    var_x = np.var(x, ddof=1)
    if var_x <= 1e-12:
        return np.nan
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def downside_second_moment(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    down = np.where(x < 0.0, x, 0.0)
    return float(np.mean(down ** 2))


def log_ratio(num, den):
    if num is None or den is None:
        return np.nan
    if np.isnan(num) or np.isnan(den) or num <= 1e-12 or den <= 1e-12:
        return np.nan
    return float(np.log(num / den))


def hypergeom_upper_tail(k, population_size, success_count, draw_count):
    # Exact one-sided enrichment test:
    # If high-rho episodes were randomly scattered through the year, this is
    # P[X >= k] for seeing at least k high-rho episodes in the top draw_count
    # ranked contributors. This avoids relying on scipy being installed.
    if any(
        x < 0
        for x in [k, population_size, success_count, draw_count]
    ):
        return np.nan
    if success_count > population_size or draw_count > population_size:
        return np.nan

    denom = math.comb(population_size, draw_count)
    if denom == 0:
        return np.nan

    upper = min(success_count, draw_count)
    tail = 0
    for x in range(k, upper + 1):
        tail += math.comb(success_count, x) * math.comb(
            population_size - success_count, draw_count - x
        )
    return float(tail / denom)


def safe_div(num, den):
    if den is None or np.isnan(den) or abs(den) <= 1e-12:
        return np.nan
    return float(num / den)


def fmt_float(value, digits=4):
    if value is None or pd.isna(value):
        return ""
    return ("{:,.%df}" % digits).format(float(value))


def extract_year_and_checkpoint(path, prefix):
    name = os.path.basename(path)
    pattern = r"^{}(?P<year>\d{{4}})_(?P<checkpoint>\d+)\.csv$".format(
        re.escape(prefix)
    )
    match = re.match(pattern, name)
    if not match:
        return None, None
    return int(match.group("year")), int(match.group("checkpoint"))


def discover_result_files(results_dir, prefix, years):
    files = []
    for path in sorted(glob.glob(os.path.join(results_dir, "{}*.csv".format(prefix)))):
        year, checkpoint = extract_year_and_checkpoint(path, prefix)
        if year is None:
            continue
        if years and year not in years:
            continue
        files.append(
            {
                "prefix": prefix,
                "year": year,
                "checkpoint": checkpoint,
                "path": path,
            }
        )
    return files


def load_result_file(meta):
    df = pd.read_csv(meta["path"])
    required = [
        "episode",
        "Date",
        "A PnL",
        "B PnL",
        "A Reward",
        "B Reward",
        "S0",
        "S-1",
        "v",
        "A Pos",
        "B Pos",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("{} missing columns: {}".format(meta["path"], missing))

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["episode", "Date"]).reset_index(drop=True)
    return df


def summarize_episode(group):
    group = group.sort_values("Date").reset_index(drop=True)

    # Terminal PnL convention: sum within each episode, then multiply by 100
    # to express the result in percent-like units.
    a_pnl = 100.0 * group["A PnL"].sum()
    b_pnl = 100.0 * group["B PnL"].sum()
    a_reward = group["A Reward"].sum()
    b_reward = group["B Reward"].sum()

    # Stepwise return/vol proxies. S0 and S-1 are normalized by DataKeeper to
    # the episode start, which is fine for log returns and correlations.
    s_prev = group["S-1"].to_numpy(dtype=float)
    s_now = group["S0"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlog_s = np.log(s_now / s_prev)

    iv = group["v"].to_numpy(dtype=float)
    d_iv = np.diff(iv, prepend=np.nan)
    d_var = np.diff(iv ** 2, prepend=np.nan)

    rho_iv = safe_corr(dlog_s, d_iv)
    rho_var = safe_corr(dlog_s, d_var)
    up_mask = dlog_s > 0.0
    down_mask = dlog_s < 0.0

    hedge_gap = -group["A Pos"].to_numpy(dtype=float) - (-group["B Pos"].to_numpy(dtype=float))
    # delta_gap is agent_delta - BS_delta. Negative means the agent holds a
    # lower long-delta hedge than Black-Scholes, i.e. it underhedges the short
    # call relative to BS.
    delta_gap = hedge_gap
    s_start = float(s_prev[0]) if len(s_prev) else np.nan
    s_end = float(s_now[-1]) if len(s_now) else np.nan

    return pd.Series(
        {
            "start_date": group["Date"].iloc[0],
            "end_date": group["Date"].iloc[-1],
            "expiry": group["Expiry"].iloc[0] if "Expiry" in group.columns else "",
            "a_pnl": a_pnl,
            "b_pnl": b_pnl,
            "diff_pnl": a_pnl - b_pnl,
            "a_reward": a_reward,
            "b_reward": b_reward,
            "diff_reward": a_reward - b_reward,
            "rho_iv": rho_iv,
            "rho_var": rho_var,
            "realized_vol_path": float(np.nanstd(dlog_s, ddof=1) * np.sqrt(252.0)),
            "iv_start": float(iv[0]) if len(iv) else np.nan,
            "iv_end": float(iv[-1]) if len(iv) else np.nan,
            "iv_change": float(iv[-1] - iv[0]) if len(iv) else np.nan,
            "terminal_return": float(s_end / s_start - 1.0)
            if s_start and not np.isnan(s_start) and abs(s_start) > 1e-12
            else np.nan,
            "sum_up_return": float(np.nansum(np.where(dlog_s > 0.0, dlog_s, 0.0))),
            "sum_down_return": float(np.nansum(np.where(dlog_s < 0.0, dlog_s, 0.0))),
            "up_day_share": float(np.nanmean(up_mask)) if len(up_mask) else np.nan,
            "rho_var_up_days": safe_corr(dlog_s[up_mask], d_var[up_mask]),
            "rho_var_down_days": safe_corr(dlog_s[down_mask], d_var[down_mask]),
            "beta_var_on_ret_up_days": safe_beta(dlog_s[up_mask], d_var[up_mask]),
            "beta_var_on_ret_down_days": safe_beta(dlog_s[down_mask], d_var[down_mask]),
            "mean_abs_hedge_gap": float(np.nanmean(np.abs(hedge_gap))),
            "mean_agent_delta": float(np.nanmean(-group["A Pos"].to_numpy(dtype=float))),
            "mean_bs_delta": float(np.nanmean(-group["B Pos"].to_numpy(dtype=float))),
            "mean_delta_gap": float(np.nanmean(delta_gap)),
            "underhedged_share": float(np.nanmean(delta_gap < 0.0)),
            "start_moneyness": float(group["S/K"].iloc[0]) if "S/K" in group.columns else np.nan,
            "start_t": float(group["T"].iloc[0]) if "T" in group.columns else np.nan,
        }
    )


def summarize_file(meta):
    raw = load_result_file(meta)
    # Keep this compatible with newer pandas behavior: the episode id is already
    # restored by reset_index(), so summarize_episode does not need the grouping
    # column inside the group frame.
    ep = (
        raw.groupby("episode", sort=True)
        .apply(summarize_episode, include_groups=False)
        .reset_index()
    )
    ep["prefix"] = meta["prefix"]
    ep["year"] = meta["year"]
    ep["checkpoint"] = meta["checkpoint"]
    ep["source_file"] = os.path.basename(meta["path"])

    # The paper metrics use sample variance with ddof=1. Therefore the exact
    # episode-level contribution to Var(agent) - Var(BS) is:
    #   [(A_i - mean(A))^2 - (B_i - mean(B))^2] / (n - 1)
    # The sum of excess_var_contrib over episodes equals Var(agent)-Var(BS).
    # This is the cleanest per-terminal-PnL proxy for "how much this episode
    # contributed to the agent's excess variance versus BS."
    n_episodes = len(ep)
    ep["a_centered"] = ep["a_pnl"] - ep["a_pnl"].mean()
    ep["b_centered"] = ep["b_pnl"] - ep["b_pnl"].mean()
    ep["a_sq_dev"] = ep["a_centered"] ** 2
    ep["b_sq_dev"] = ep["b_centered"] ** 2
    ep["excess_sq_dev"] = ep["a_sq_dev"] - ep["b_sq_dev"]
    if n_episodes > 1:
        ep["a_var_contrib"] = ep["a_sq_dev"] / (n_episodes - 1)
        ep["b_var_contrib"] = ep["b_sq_dev"] / (n_episodes - 1)
        ep["excess_var_contrib"] = ep["excess_sq_dev"] / (n_episodes - 1)
    else:
        ep["a_var_contrib"] = np.nan
        ep["b_var_contrib"] = np.nan
        ep["excess_var_contrib"] = np.nan

    # Downside variance is measured as the downside second moment:
    # mean(min(PnL, 0)^2). It is not centered and uses denominator n.
    # These contributions also sum exactly to downside_agent - downside_BS.
    ep["a_down_sq"] = np.where(ep["a_pnl"] < 0.0, ep["a_pnl"], 0.0) ** 2
    ep["b_down_sq"] = np.where(ep["b_pnl"] < 0.0, ep["b_pnl"], 0.0) ** 2
    if n_episodes > 0:
        ep["a_downside_contrib"] = ep["a_down_sq"] / n_episodes
        ep["b_downside_contrib"] = ep["b_down_sq"] / n_episodes
        ep["excess_downside_contrib"] = (
            ep["a_down_sq"] - ep["b_down_sq"]
        ) / n_episodes
    else:
        ep["a_downside_contrib"] = np.nan
        ep["b_downside_contrib"] = np.nan
        ep["excess_downside_contrib"] = np.nan

    return raw, ep


def summarize_model_year(ep):
    var_a = safe_var(ep["a_pnl"])
    var_b = safe_var(ep["b_pnl"])
    down_a = downside_second_moment(ep["a_pnl"])
    down_b = downside_second_moment(ep["b_pnl"])
    excess_var_total = var_a - var_b
    excess_downside_total = down_a - down_b

    # Variance decomposition:
    # A = B + D, where D is agent-minus-BS terminal PnL.
    # Var(A) - Var(B) = Var(D) + 2 Cov(B, D).
    diff = ep["diff_pnl"].to_numpy(dtype=float)
    b = ep["b_pnl"].to_numpy(dtype=float)
    var_diff = safe_var(diff)
    cov_b_diff = np.cov(b, diff, ddof=1)[0, 1] if len(ep) >= 2 else np.nan
    positive_excess_var = ep["excess_var_contrib"].clip(lower=0.0).sum()
    negative_excess_var = ep["excess_var_contrib"].clip(upper=0.0).sum()

    return {
        "prefix": ep["prefix"].iloc[0],
        "year": int(ep["year"].iloc[0]),
        "checkpoint": int(ep["checkpoint"].iloc[0]),
        "episodes": len(ep),
        "clusters": ep["start_date"].nunique(),
        "mean_a_pnl": ep["a_pnl"].mean(),
        "mean_b_pnl": ep["b_pnl"].mean(),
        "mean_diff_pnl": ep["diff_pnl"].mean(),
        "mean_diff_reward": ep["diff_reward"].mean(),
        "var_a": var_a,
        "var_b": var_b,
        "excess_var_total": excess_var_total,
        "log_var_ratio": log_ratio(var_a, var_b),
        "downside_a": down_a,
        "downside_b": down_b,
        "excess_downside_total": excess_downside_total,
        "log_downside_ratio": log_ratio(down_a, down_b),
        "rho_iv_mean": ep["rho_iv"].mean(),
        "rho_iv_median": ep["rho_iv"].median(),
        "rho_var_mean": ep["rho_var"].mean(),
        "rho_var_median": ep["rho_var"].median(),
        "rho_var_p10": ep["rho_var"].quantile(0.10),
        "rho_var_p90": ep["rho_var"].quantile(0.90),
        "realized_vol_mean": ep["realized_vol_path"].mean(),
        "iv_change_mean": ep["iv_change"].mean(),
        "mean_abs_hedge_gap": ep["mean_abs_hedge_gap"].mean(),
        "mean_delta_gap": ep["mean_delta_gap"].mean(),
        "underhedged_share": ep["underhedged_share"].mean(),
        "terminal_return_mean": ep["terminal_return"].mean(),
        "up_episode_share": (ep["terminal_return"] > 0.0).mean(),
        "rho_var_up_days_mean": ep["rho_var_up_days"].mean(),
        "rho_var_down_days_mean": ep["rho_var_down_days"].mean(),
        "beta_var_on_ret_up_days_mean": ep["beta_var_on_ret_up_days"].mean(),
        "beta_var_on_ret_down_days_mean": ep["beta_var_on_ret_down_days"].mean(),
        "var_diff": var_diff,
        "two_cov_b_diff": 2.0 * cov_b_diff if not np.isnan(cov_b_diff) else np.nan,
        "var_decomp_residual": (var_diff + 2.0 * cov_b_diff) - (var_a - var_b)
        if not np.isnan(var_diff) and not np.isnan(cov_b_diff)
        else np.nan,
        "positive_excess_var_total": positive_excess_var,
        "negative_excess_var_total": negative_excess_var,
        "positive_excess_var_to_total": safe_div(positive_excess_var, excess_var_total),
        "negative_excess_var_to_total": safe_div(negative_excess_var, excess_var_total),
        "corr_rho_var_a_sq_dev": safe_corr(ep["rho_var"], ep["a_sq_dev"]),
        "corr_rho_var_excess_sq_dev": safe_corr(ep["rho_var"], ep["excess_sq_dev"]),
        "corr_rho_var_excess_var_contrib": safe_corr(
            ep["rho_var"], ep["excess_var_contrib"]
        ),
        "corr_rho_var_positive_excess_var": safe_corr(
            ep["rho_var"], ep["excess_var_contrib"].clip(lower=0.0)
        ),
        "corr_abs_rho_var_a_sq_dev": safe_corr(np.abs(ep["rho_var"]), ep["a_sq_dev"]),
        "corr_hedge_gap_a_sq_dev": safe_corr(ep["mean_abs_hedge_gap"], ep["a_sq_dev"]),
        "corr_hedge_gap_excess_var_contrib": safe_corr(
            ep["mean_abs_hedge_gap"], ep["excess_var_contrib"]
        ),
    }


def add_rho_buckets(ep, bucket_col, n_buckets):
    ep = ep.copy()
    valid = ep[bucket_col].dropna()
    if valid.nunique() < 2:
        ep["rho_bucket"] = "all"
        return ep

    # qcut can drop duplicate bin edges in years where rho values are clumped.
    ep["rho_bucket"] = pd.qcut(
        ep[bucket_col],
        q=min(n_buckets, valid.nunique()),
        duplicates="drop",
    ).astype(str)
    ep.loc[ep[bucket_col].isna(), "rho_bucket"] = "nan"
    return ep


def summarize_bucket(group):
    var_a = safe_var(group["a_pnl"])
    var_b = safe_var(group["b_pnl"])
    down_a = downside_second_moment(group["a_pnl"])
    down_b = downside_second_moment(group["b_pnl"])
    return pd.Series(
        {
            "episodes": len(group),
            "clusters": group["start_date"].nunique(),
            "rho_var_mean": group["rho_var"].mean(),
            "rho_var_min": group["rho_var"].min(),
            "rho_var_max": group["rho_var"].max(),
            "mean_a_pnl": group["a_pnl"].mean(),
            "mean_b_pnl": group["b_pnl"].mean(),
            "log_var_ratio": log_ratio(var_a, var_b),
            "log_downside_ratio": log_ratio(down_a, down_b),
            "mean_abs_hedge_gap": group["mean_abs_hedge_gap"].mean(),
            "mean_delta_gap": group["mean_delta_gap"].mean(),
            "underhedged_share": group["underhedged_share"].mean(),
            "up_episode_share": (group["terminal_return"] > 0.0).mean(),
            "mean_realized_vol": group["realized_vol_path"].mean(),
            "mean_excess_sq_dev": group["excess_sq_dev"].mean(),
            "sum_excess_var_contrib": group["excess_var_contrib"].sum(),
            "sum_positive_excess_var_contrib": group["excess_var_contrib"]
            .clip(lower=0.0)
            .sum(),
            "sum_excess_downside_contrib": group["excess_downside_contrib"].sum(),
        }
    )


def simple_r2(ep, y_col, x_cols):
    data = ep[[y_col] + x_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) <= len(x_cols) + 2:
        return np.nan, len(data)

    y = data[y_col].to_numpy(dtype=float)
    x = data[x_cols].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    if ss_tot <= 1e-12:
        return np.nan, len(data)
    return float(1.0 - ss_res / ss_tot), len(data)


def regression_diagnostics(ep):
    ep = ep.copy()
    ep["rho_var_abs"] = np.abs(ep["rho_var"])
    ep["rho_var_sq"] = ep["rho_var"] ** 2
    ep["positive_excess_var_contrib"] = ep["excess_var_contrib"].clip(lower=0.0)
    ep["abs_excess_var_contrib"] = np.abs(ep["excess_var_contrib"])
    ep["terminal_return_pos"] = ep["terminal_return"].clip(lower=0.0)
    ep["terminal_return_neg_abs"] = (-ep["terminal_return"].clip(upper=0.0))
    ep["gap_x_up_return"] = ep["mean_delta_gap"] * ep["terminal_return_pos"]
    ep["gap_x_down_return"] = ep["mean_delta_gap"] * ep["terminal_return_neg_abs"]

    specs = [
        ("rho_only", ["rho_var"]),
        ("rho_nonlinear", ["rho_var", "rho_var_sq"]),
        ("rho_plus_vol", ["rho_var", "rho_var_sq", "realized_vol_path", "iv_change"]),
        (
            "rho_plus_state",
            [
                "rho_var",
                "rho_var_sq",
                "realized_vol_path",
                "iv_change",
                "mean_abs_hedge_gap",
                "start_moneyness",
                "start_t",
            ],
        ),
        (
            "asymmetric_hedge",
            [
                "terminal_return_pos",
                "terminal_return_neg_abs",
                "iv_change",
                "mean_delta_gap",
                "gap_x_up_return",
                "gap_x_down_return",
                "rho_var",
            ],
        ),
    ]

    rows = []
    for name, cols in specs:
        for y_col in [
            "a_var_contrib",
            "excess_var_contrib",
            "positive_excess_var_contrib",
            "excess_downside_contrib",
            "diff_pnl",
        ]:
            r2, n = simple_r2(ep, y_col, cols)
            rows.append(
                {
                    "prefix": ep["prefix"].iloc[0],
                    "year": int(ep["year"].iloc[0]),
                    "checkpoint": int(ep["checkpoint"].iloc[0]),
                    "target": y_col,
                    "spec": name,
                    "r2": r2,
                    "n": n,
                    "features": ",".join(cols),
                }
            )
    return pd.DataFrame(rows)


def print_table(title, df, max_rows=None):
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    if df.empty:
        print("(empty)")
        return
    if max_rows:
        df = df.head(max_rows)
    print(df.to_string(index=False))


def key_r2_columns(regressions):
    # The detailed regressions CSV stays long-form because that is easiest to
    # filter programmatically. For human reading, we also create a compact wide
    # table with the exact R2s that answer the present question.
    keep = regressions[
        regressions["target"].isin(
            [
                "excess_var_contrib",
                "positive_excess_var_contrib",
                "excess_downside_contrib",
            ]
        )
        & regressions["spec"].isin(["rho_only", "rho_nonlinear", "rho_plus_state"])
    ].copy()
    if keep.empty:
        return pd.DataFrame()

    keep["r2_name"] = "r2_" + keep["target"] + "_" + keep["spec"]
    wide = keep.pivot_table(
        index=["prefix", "year", "checkpoint"],
        columns="r2_name",
        values="r2",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def build_readable_summary(summary, regressions):
    readable = summary.copy()
    r2_wide = key_r2_columns(regressions)
    if not r2_wide.empty:
        readable = readable.merge(r2_wide, on=["prefix", "year", "checkpoint"], how="left")

    # Keep the report focused. The full CSVs still contain all diagnostics.
    cols = [
        "prefix",
        "year",
        "checkpoint",
        "episodes",
        "clusters",
        "var_a",
        "var_b",
        "log_var_ratio",
        "log_downside_ratio",
        "excess_var_total",
        "excess_downside_total",
        "rho_var_mean",
        "rho_var_median",
        "rho_var_p10",
        "rho_var_p90",
        "mean_abs_hedge_gap",
        "mean_delta_gap",
        "underhedged_share",
        "up_episode_share",
        "rho_var_up_days_mean",
        "rho_var_down_days_mean",
        "beta_var_on_ret_up_days_mean",
        "beta_var_on_ret_down_days_mean",
        "positive_excess_var_total",
        "negative_excess_var_total",
        "corr_rho_var_excess_var_contrib",
        "corr_rho_var_positive_excess_var",
        "r2_excess_var_contrib_rho_only",
        "r2_excess_var_contrib_rho_nonlinear",
        "r2_excess_var_contrib_rho_plus_state",
        "r2_positive_excess_var_contrib_rho_only",
        "r2_excess_downside_contrib_rho_only",
    ]
    existing = [c for c in cols if c in readable.columns]
    return readable[existing].sort_values(["prefix", "year"])


def build_hypothesis_tests(episodes, threshold, top_n):
    rows = []
    for keys, group in episodes.groupby(["prefix", "year", "checkpoint"], sort=True):
        prefix, year, checkpoint = keys
        group = group.copy()
        n = len(group)
        draw_count = min(top_n, n)
        high_rho = group["rho_var"] > threshold
        success_count = int(high_rho.sum())
        positive_excess_total = group["excess_var_contrib"].clip(lower=0.0).sum()
        excess_total = group["excess_var_contrib"].sum()

        # Diagnostic note:
        # Count high-rho episodes among "worst" episodes.
        # CURRENT TEST:
        # Keep that count, but compare it to the base rate in the same year
        # using an exact hypergeometric enrichment test. This matters because
        # in 2023 most episodes already have rho > -0.4.
        for rank_col, label in [
            ("a_var_contrib", "agent_variance"),
            ("excess_var_contrib", "excess_variance"),
        ]:
            top = group.sort_values(rank_col, ascending=False).head(draw_count)
            high_in_top = int((top["rho_var"] > threshold).sum())
            rows.append(
                {
                    "prefix": prefix,
                    "year": int(year),
                    "checkpoint": int(checkpoint),
                    "ranked_by": label,
                    "threshold": threshold,
                    "top_n": draw_count,
                    "episodes": n,
                    "high_rho_episodes": success_count,
                    "base_high_rho_share": safe_div(success_count, n),
                    "top_high_rho_count": high_in_top,
                    "top_high_rho_share": safe_div(high_in_top, draw_count),
                    "top_high_rho_enrichment_p_ge": hypergeom_upper_tail(
                        high_in_top, n, success_count, draw_count
                    ),
                    "top_sum_excess_var_contrib": top["excess_var_contrib"].sum(),
                    "top_share_total_excess_var": safe_div(
                        top["excess_var_contrib"].sum(), excess_total
                    ),
                    "top_share_positive_excess_var": safe_div(
                        top["excess_var_contrib"].clip(lower=0.0).sum(),
                        positive_excess_total,
                    ),
                    "top_mean_rho": top["rho_var"].mean(),
                    "all_mean_rho": group["rho_var"].mean(),
                }
            )

        high_group = group[high_rho]
        rows.append(
            {
                "prefix": prefix,
                "year": int(year),
                "checkpoint": int(checkpoint),
                "ranked_by": "all_high_rho_bucket",
                "threshold": threshold,
                "top_n": np.nan,
                "episodes": n,
                "high_rho_episodes": success_count,
                "base_high_rho_share": safe_div(success_count, n),
                "top_high_rho_count": np.nan,
                "top_high_rho_share": np.nan,
                "top_high_rho_enrichment_p_ge": np.nan,
                "top_sum_excess_var_contrib": high_group["excess_var_contrib"].sum(),
                "top_share_total_excess_var": safe_div(
                    high_group["excess_var_contrib"].sum(), excess_total
                ),
                "top_share_positive_excess_var": safe_div(
                    high_group["excess_var_contrib"].clip(lower=0.0).sum(),
                    positive_excess_total,
                ),
                "top_mean_rho": high_group["rho_var"].mean(),
                "all_mean_rho": group["rho_var"].mean(),
            }
        )

    return pd.DataFrame(rows)


def build_cluster_decomposition(episodes):
    rows = []
    for keys, group in episodes.groupby(["prefix", "year", "checkpoint"], sort=True):
        prefix, year, checkpoint = keys
        group = group.copy()
        base_var_a = safe_var(group["a_pnl"])
        base_var_b = safe_var(group["b_pnl"])
        base_log_ratio = log_ratio(base_var_a, base_var_b)
        positive_excess_total = group["excess_var_contrib"].clip(lower=0.0).sum()
        excess_total = group["excess_var_contrib"].sum()

        for start_date, cluster in group.groupby("start_date", sort=True):
            rest = group[group["start_date"] != start_date]
            without_log_ratio = log_ratio(safe_var(rest["a_pnl"]), safe_var(rest["b_pnl"]))
            # Positive log_ratio_reduction means this cluster is making the
            # plotted log variance ratio worse. Negative means the cluster was
            # helping the agent relative to BS.
            rows.append(
                {
                    "prefix": prefix,
                    "year": int(year),
                    "checkpoint": int(checkpoint),
                    "start_date": start_date,
                    "episodes": len(cluster),
                    "rho_var_mean": cluster["rho_var"].mean(),
                    "rho_var_median": cluster["rho_var"].median(),
                    "sum_excess_var_contrib": cluster["excess_var_contrib"].sum(),
                    "share_total_excess_var": safe_div(
                        cluster["excess_var_contrib"].sum(), excess_total
                    ),
                    "sum_positive_excess_var_contrib": cluster[
                        "excess_var_contrib"
                    ]
                    .clip(lower=0.0)
                    .sum(),
                    "share_positive_excess_var": safe_div(
                        cluster["excess_var_contrib"].clip(lower=0.0).sum(),
                        positive_excess_total,
                    ),
                    "base_log_var_ratio": base_log_ratio,
                    "log_var_ratio_without_cluster": without_log_ratio,
                    "log_ratio_reduction_if_removed": base_log_ratio
                    - without_log_ratio
                    if not pd.isna(without_log_ratio)
                    else np.nan,
                    "mean_a_pnl": cluster["a_pnl"].mean(),
                    "mean_b_pnl": cluster["b_pnl"].mean(),
                    "mean_abs_hedge_gap": cluster["mean_abs_hedge_gap"].mean(),
                    "mean_realized_vol": cluster["realized_vol_path"].mean(),
                    "mean_iv_change": cluster["iv_change"].mean(),
                    "mean_moneyness": cluster["start_moneyness"].mean(),
                    "mean_t": cluster["start_t"].mean(),
                }
            )

    return pd.DataFrame(rows)


def prepare_step_frame(raw, meta):
    step = raw.copy()
    step["prefix"] = meta["prefix"]
    step["year"] = meta["year"]
    step["checkpoint"] = meta["checkpoint"]

    # Result CSV stores hedge positions as stockOwned, which is negative for
    # a short-call hedge. Convert to positive deltas for direct comparison with
    # BS delta.
    step["agent_delta"] = -step["A Pos"].astype(float)
    step["bs_delta"] = -step["B Pos"].astype(float)
    step["delta_gap"] = step["agent_delta"] - step["bs_delta"]
    step["underhedged"] = step["delta_gap"] < 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        step["dlog_s"] = np.log(step["S0"].astype(float) / step["S-1"].astype(float))
    step["dvar"] = (
        step.groupby("episode", sort=False)["v"]
        .transform(lambda s: s.astype(float).pow(2).diff())
    )
    step["step_diff_pnl_100"] = 100.0 * (
        step["A PnL"].astype(float) - step["B PnL"].astype(float)
    )
    step["step_a_pnl_100"] = 100.0 * step["A PnL"].astype(float)
    step["step_b_pnl_100"] = 100.0 * step["B PnL"].astype(float)
    return step


def summarize_step_condition(group):
    return pd.Series(
        {
            "steps": len(group),
            "episodes": group["episode"].nunique(),
            "mean_agent_delta": group["agent_delta"].mean(),
            "mean_bs_delta": group["bs_delta"].mean(),
            "mean_delta_gap": group["delta_gap"].mean(),
            "median_delta_gap": group["delta_gap"].median(),
            "underhedged_share": group["underhedged"].mean(),
            "rho_var": safe_corr(group["dlog_s"], group["dvar"]),
            "beta_var_on_ret": safe_beta(group["dlog_s"], group["dvar"]),
            "mean_dlog_s": group["dlog_s"].mean(),
            "mean_dvar": group["dvar"].mean(),
            "mean_step_diff_pnl_100": group["step_diff_pnl_100"].mean(),
        }
    )


def build_step_asymmetry(raw_steps):
    if not raw_steps:
        return pd.DataFrame()
    steps = pd.concat(raw_steps, ignore_index=True)
    steps["spot_move"] = np.where(
        steps["dlog_s"] > 0.0,
        "up_day",
        np.where(steps["dlog_s"] < 0.0, "down_day", "flat_day"),
    )

    rows = []
    for keys, group in steps.groupby(["prefix", "year", "checkpoint"], sort=True):
        base = summarize_step_condition(group).to_dict()
        base.update(
            {
                "prefix": keys[0],
                "year": int(keys[1]),
                "checkpoint": int(keys[2]),
                "condition": "all_steps",
            }
        )
        rows.append(base)

        for condition, sub in group.groupby("spot_move", sort=True):
            row = summarize_step_condition(sub).to_dict()
            row.update(
                {
                    "prefix": keys[0],
                    "year": int(keys[1]),
                    "checkpoint": int(keys[2]),
                    "condition": condition,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def add_state_bins(step):
    step = step.copy()
    step["moneyness_bin"] = pd.cut(
        step["S/K"],
        bins=[-np.inf, 0.90, 0.95, 1.00, 1.05, 1.10, np.inf],
        labels=["<0.90", "0.90-0.95", "0.95-1.00", "1.00-1.05", "1.05-1.10", ">1.10"],
    )
    step["t_bin"] = pd.cut(
        step["T"],
        bins=[-np.inf, 35.0, 60.0, 90.0, 180.0, np.inf],
        labels=["<=35", "36-60", "61-90", "91-180", ">180"],
    )
    try:
        step["iv_bin"] = pd.qcut(
            step["v"],
            q=4,
            labels=["iv_q1", "iv_q2", "iv_q3", "iv_q4"],
            duplicates="drop",
        )
    except ValueError:
        step["iv_bin"] = "iv_all"
    return step


def build_underhedge_bins(raw_steps, min_steps):
    if not raw_steps:
        return pd.DataFrame()

    all_steps = pd.concat(raw_steps, ignore_index=True)
    rows = []
    for keys, group in all_steps.groupby(["prefix", "year", "checkpoint"], sort=True):
        group = add_state_bins(group)

        # Separate one-dimensional bins are readable. The combined 3D bin is
        # the stricter test of the previous finding: does underhedging
        # persist across IV, moneyness, and maturity combinations?
        bin_specs = [
            ("moneyness", ["moneyness_bin"]),
            ("maturity", ["t_bin"]),
            ("iv", ["iv_bin"]),
            ("moneyness_x_maturity_x_iv", ["moneyness_bin", "t_bin", "iv_bin"]),
        ]
        for bin_type, cols in bin_specs:
            for bin_values, sub in group.groupby(cols, observed=True, dropna=False):
                if len(sub) < min_steps:
                    continue
                if not isinstance(bin_values, tuple):
                    bin_values = (bin_values,)
                row = summarize_step_condition(sub).to_dict()
                row.update(
                    {
                        "prefix": keys[0],
                        "year": int(keys[1]),
                        "checkpoint": int(keys[2]),
                        "bin_type": bin_type,
                        "bin_label": " | ".join(str(v) for v in bin_values),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def summarize_episode_side(group):
    return pd.Series(
        {
            "episodes": len(group),
            "clusters": group["start_date"].nunique(),
            "mean_terminal_return": group["terminal_return"].mean(),
            "mean_iv_change": group["iv_change"].mean(),
            "mean_rho_var": group["rho_var"].mean(),
            "mean_rho_up_days": group["rho_var_up_days"].mean(),
            "mean_rho_down_days": group["rho_var_down_days"].mean(),
            "mean_delta_gap": group["mean_delta_gap"].mean(),
            "underhedged_share": group["underhedged_share"].mean(),
            "mean_a_pnl": group["a_pnl"].mean(),
            "mean_b_pnl": group["b_pnl"].mean(),
            "mean_diff_pnl": group["diff_pnl"].mean(),
            "sum_excess_var_contrib": group["excess_var_contrib"].sum(),
            "sum_positive_excess_var_contrib": group["excess_var_contrib"]
            .clip(lower=0.0)
            .sum(),
            "sum_excess_downside_contrib": group["excess_downside_contrib"].sum(),
        }
    )


def build_episode_asymmetry(episodes):
    ep = episodes.copy()
    ep["episode_side"] = np.where(
        ep["terminal_return"] > 0.0,
        "spot_up_episode",
        np.where(ep["terminal_return"] < 0.0, "spot_down_episode", "flat_episode"),
    )
    positive_excess = (
        ep.groupby(["prefix", "year", "checkpoint"])["excess_var_contrib"]
        .apply(lambda s: s.clip(lower=0.0).sum())
        .rename("positive_excess_total")
        .reset_index()
    )
    total_excess = (
        ep.groupby(["prefix", "year", "checkpoint"])["excess_var_contrib"]
        .sum()
        .rename("excess_total")
        .reset_index()
    )
    side = (
        ep.groupby(["prefix", "year", "checkpoint", "episode_side"], dropna=False)
        .apply(summarize_episode_side, include_groups=False)
        .reset_index()
    )
    side = side.merge(positive_excess, on=["prefix", "year", "checkpoint"], how="left")
    side = side.merge(total_excess, on=["prefix", "year", "checkpoint"], how="left")
    side["share_total_excess_var"] = side["sum_excess_var_contrib"] / side[
        "excess_total"
    ]
    side["share_positive_excess_var"] = side[
        "sum_positive_excess_var_contrib"
    ] / side["positive_excess_total"]
    return side


def build_theory_matrix(episodes, rho_threshold):
    ep = episodes.copy()
    ep["spot_regime"] = np.where(
        ep["terminal_return"] >= 0.0,
        "spot_up",
        "spot_down",
    )
    ep["rho_regime"] = np.where(
        ep["rho_var"] > rho_threshold,
        "rho_broken_or_weak",
        "rho_normal_or_strong",
    )

    totals = (
        ep.groupby(["prefix", "year", "checkpoint"], sort=True)
        .agg(
            total_excess_var=("excess_var_contrib", "sum"),
            total_positive_excess_var=(
                "excess_var_contrib",
                lambda s: s.clip(lower=0.0).sum(),
            ),
            total_excess_downside=("excess_downside_contrib", "sum"),
        )
        .reset_index()
    )

    rows = []
    for keys, group in ep.groupby(
        ["prefix", "year", "checkpoint", "spot_regime", "rho_regime"],
        sort=True,
    ):
        prefix, year, checkpoint, spot_regime, rho_regime = keys
        var_a = safe_var(group["a_pnl"])
        var_b = safe_var(group["b_pnl"])
        rows.append(
            {
                "prefix": prefix,
                "year": int(year),
                "checkpoint": int(checkpoint),
                "spot_regime": spot_regime,
                "rho_regime": rho_regime,
                "episodes": len(group),
                "clusters": group["start_date"].nunique(),
                "mean_terminal_return": group["terminal_return"].mean(),
                "mean_rho_var": group["rho_var"].mean(),
                "mean_rho_up_days": group["rho_var_up_days"].mean(),
                "mean_rho_down_days": group["rho_var_down_days"].mean(),
                "mean_delta_gap": group["mean_delta_gap"].mean(),
                "underhedged_share": group["underhedged_share"].mean(),
                "mean_a_pnl": group["a_pnl"].mean(),
                "mean_b_pnl": group["b_pnl"].mean(),
                "mean_diff_pnl": group["diff_pnl"].mean(),
                "mean_a_reward": group["a_reward"].mean(),
                "mean_b_reward": group["b_reward"].mean(),
                "mean_diff_reward": group["diff_reward"].mean(),
                "cell_var_a": var_a,
                "cell_var_b": var_b,
                "cell_log_var_ratio": log_ratio(var_a, var_b),
                "sum_excess_var_contrib": group["excess_var_contrib"].sum(),
                "sum_positive_excess_var_contrib": group["excess_var_contrib"]
                .clip(lower=0.0)
                .sum(),
                "sum_excess_downside_contrib": group[
                    "excess_downside_contrib"
                ].sum(),
            }
        )

    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix
    matrix["abs_sum_excess_var_contrib"] = np.abs(matrix["sum_excess_var_contrib"])
    gross = (
        matrix.groupby(["prefix", "year", "checkpoint"], sort=True)[
            "abs_sum_excess_var_contrib"
        ]
        .sum()
        .rename("gross_abs_excess_var")
        .reset_index()
    )
    matrix = matrix.merge(totals, on=["prefix", "year", "checkpoint"], how="left")
    matrix = matrix.merge(gross, on=["prefix", "year", "checkpoint"], how="left")
    matrix["share_total_excess_var"] = matrix["sum_excess_var_contrib"] / matrix[
        "total_excess_var"
    ]
    matrix["share_gross_abs_excess_var"] = matrix[
        "abs_sum_excess_var_contrib"
    ] / matrix["gross_abs_excess_var"]
    matrix["share_positive_excess_var"] = matrix[
        "sum_positive_excess_var_contrib"
    ] / matrix["total_positive_excess_var"]
    matrix["share_excess_downside"] = matrix["sum_excess_downside_contrib"] / matrix[
        "total_excess_downside"
    ]
    return matrix


def compact_final_story(summary, theory_matrix, regressions, prefix, focus_years):
    lines = []
    lines.append("Compact final-model rho story")
    lines.append("=" * 36)
    lines.append("")
    lines.append(
        "Interpretation note: low episode-level rho R2 does not refute a rho "
        "mechanism. The mechanism can operate through training-time hedge bias, "
        "conditional up/down spot-vol asymmetry, and the BS variance denominator."
    )
    lines.append("")

    s = summary[(summary["prefix"] == prefix) & (summary["year"].isin(focus_years))]
    if not s.empty:
        lines.append("Year-level facts")
        lines.append("----------------")
        cols = [
            "year",
            "episodes",
            "clusters",
            "var_a",
            "var_b",
            "log_var_ratio",
            "log_downside_ratio",
            "mean_diff_reward",
            "rho_var_mean",
            "rho_var_up_days_mean",
            "rho_var_down_days_mean",
            "mean_delta_gap",
            "underhedged_share",
        ]
        lines.append(
            s[cols].to_string(index=False, float_format=lambda x: fmt_float(x, 4))
        )
        lines.append("")

    m = theory_matrix[
        (theory_matrix["prefix"] == prefix)
        & (theory_matrix["year"].isin(focus_years))
    ].copy()
    if not m.empty:
        lines.append("2x2 matrix: spot direction x rho regime")
        lines.append("--------------------------------------")
        cols = [
            "year",
            "spot_regime",
            "rho_regime",
            "episodes",
            "mean_terminal_return",
            "mean_rho_var",
            "mean_delta_gap",
            "mean_diff_pnl",
            "mean_diff_reward",
            "share_gross_abs_excess_var",
            "share_total_excess_var",
            "share_positive_excess_var",
            "share_excess_downside",
        ]
        lines.append(
            m.sort_values(["year", "spot_regime", "rho_regime"])[cols].to_string(
                index=False, float_format=lambda x: fmt_float(x, 4)
            )
        )
        lines.append("")

    r = regressions[
        (regressions["prefix"] == prefix)
        & (regressions["year"].isin(focus_years))
        & (regressions["target"].isin(["excess_var_contrib", "diff_pnl"]))
        & (regressions["spec"].isin(["rho_only", "rho_plus_state", "asymmetric_hedge"]))
    ].copy()
    if not r.empty:
        lines.append("Regression sanity check")
        lines.append("-----------------------")
        lines.append(
            r[["year", "target", "spec", "r2", "n"]].to_string(
                index=False, float_format=lambda x: fmt_float(x, 4)
            )
        )
        lines.append("")

    lines.append("Readable takeaway")
    lines.append("-----------------")
    lines.append(
        "The most coherent interpretation is a rho/leverage-effect mechanism, "
        "but the direct cross-sectional regression of terminal excess variance "
        "on a single realized rho number is too narrow. The useful evidence is "
        "the joint pattern: persistent underhedging, much weaker spot-vol "
        "coupling on up moves, and concentration of excess variance in "
        "spot-up/weak-rho cells."
    )
    return "\n".join(lines)


def write_text_report(
    path,
    readable_summary,
    buckets,
    episodes,
    regressions,
    hypothesis_tests,
    cluster_decomposition,
    step_asymmetry,
    underhedge_bins,
    episode_asymmetry,
    theory_matrix,
    compact_story,
    focus_years,
    top,
):
    lines = []
    lines.append(compact_story)
    lines.append("")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Rho / terminal-PnL variance diagnostic report")
    lines.append("=" * 52)
    lines.append("")

    lines.append("2x2 theory matrix: spot direction x rho regime")
    lines.append("----------------------------------------------")
    matrix_cols = [
        "prefix",
        "year",
        "spot_regime",
        "rho_regime",
        "episodes",
        "mean_terminal_return",
        "mean_rho_var",
        "mean_delta_gap",
        "mean_diff_pnl",
        "mean_diff_reward",
        "share_gross_abs_excess_var",
        "share_total_excess_var",
        "share_positive_excess_var",
        "share_excess_downside",
    ]
    existing_matrix_cols = [c for c in matrix_cols if c in theory_matrix.columns]
    lines.append(
        theory_matrix[existing_matrix_cols].to_string(
            index=False, float_format=lambda x: fmt_float(x, 4)
        )
    )
    lines.append("")
    lines.append("Measurement convention")
    lines.append("----------------------")
    lines.append(
        "Variance here is sample variance across terminal episode PnLs, where "
        "each terminal PnL is the sum of daily PnLs inside one 21-day episode."
    )
    lines.append(
        "Realized rho is estimated inside each episode from the path "
        "corr(d log S, d implied variance). It is used as an explanatory "
        "variable for that episode's terminal-PnL variance contribution."
    )
    lines.append(
        "excess_var_contrib is exact: summing it over episodes gives "
        "Var(agent terminal PnL) - Var(BS terminal PnL)."
    )
    lines.append("")
    lines.append("Readable model/year summary")
    lines.append("---------------------------")
    lines.append(readable_summary.to_string(index=False, float_format=lambda x: fmt_float(x, 4)))
    lines.append("")

    lines.append("Direct high-rho hypothesis tests")
    lines.append("--------------------------------")
    lines.append(
        "The p-value is an exact hypergeometric upper-tail test for enrichment "
        "of rho above the threshold among the top ranked contributors. This "
        "guards against mistaking a high top-100 count for proof when the "
        "whole year already has a high base rate."
    )
    hypothesis_cols = [
        "prefix",
        "year",
        "ranked_by",
        "threshold",
        "base_high_rho_share",
        "top_high_rho_count",
        "top_high_rho_share",
        "top_high_rho_enrichment_p_ge",
        "top_share_total_excess_var",
        "top_share_positive_excess_var",
        "top_mean_rho",
        "all_mean_rho",
    ]
    existing_hypothesis_cols = [
        c for c in hypothesis_cols if c in hypothesis_tests.columns
    ]
    lines.append(
        hypothesis_tests[existing_hypothesis_cols].to_string(
            index=False, float_format=lambda x: fmt_float(x, 4)
        )
    )
    lines.append("")

    lines.append("Step-level asymmetry: up days vs down days")
    lines.append("------------------------------------------")
    lines.append(
        "delta_gap = agent_delta - BS_delta. Negative means underhedging "
        "relative to BS. beta_var_on_ret is the slope of d implied variance "
        "on d log S; more negative means a stronger volatility parachute when "
        "spot falls."
    )
    step_cols = [
        "prefix",
        "year",
        "condition",
        "steps",
        "episodes",
        "mean_delta_gap",
        "underhedged_share",
        "rho_var",
        "beta_var_on_ret",
        "mean_step_diff_pnl_100",
    ]
    existing_step_cols = [c for c in step_cols if c in step_asymmetry.columns]
    lines.append(
        step_asymmetry[existing_step_cols].to_string(
            index=False, float_format=lambda x: fmt_float(x, 4)
        )
    )
    lines.append("")

    lines.append("Episode-level asymmetry: spot-up vs spot-down episodes")
    lines.append("------------------------------------------------------")
    episode_side_cols = [
        "prefix",
        "year",
        "episode_side",
        "episodes",
        "clusters",
        "mean_terminal_return",
        "mean_iv_change",
        "mean_rho_var",
        "mean_delta_gap",
        "mean_diff_pnl",
        "share_total_excess_var",
        "share_positive_excess_var",
    ]
    existing_episode_side_cols = [
        c for c in episode_side_cols if c in episode_asymmetry.columns
    ]
    lines.append(
        episode_asymmetry[existing_episode_side_cols].to_string(
            index=False, float_format=lambda x: fmt_float(x, 4)
        )
    )
    lines.append("")

    if focus_years:
        focus = episodes[episodes["year"].isin(focus_years)].copy()
        if not focus.empty:
            focus_cols = [
                "prefix",
                "year",
                "episode",
                "start_date",
                "a_pnl",
                "b_pnl",
                "rho_var",
                "excess_var_contrib",
                "excess_downside_contrib",
                "mean_abs_hedge_gap",
                "realized_vol_path",
                "iv_change",
                "start_moneyness",
                "start_t",
            ]
            lines.append("Largest positive excess-variance contributors in focus years")
            lines.append("------------------------------------------------------------")
            top_positive = (
                focus.sort_values("excess_var_contrib", ascending=False)[focus_cols]
                .head(top)
            )
            lines.append(top_positive.to_string(index=False, float_format=lambda x: fmt_float(x, 4)))
            lines.append("")

            lines.append("Largest absolute excess-variance contributors in focus years")
            lines.append("------------------------------------------------------------")
            focus["abs_excess_var_contrib"] = np.abs(focus["excess_var_contrib"])
            top_abs = (
                focus.sort_values("abs_excess_var_contrib", ascending=False)[
                    focus_cols + ["abs_excess_var_contrib"]
                ]
                .head(top)
            )
            lines.append(top_abs.to_string(index=False, float_format=lambda x: fmt_float(x, 4)))
            lines.append("")

        focus_clusters = cluster_decomposition[
            cluster_decomposition["year"].isin(focus_years)
        ].copy()
        if not focus_clusters.empty:
            cluster_cols = [
                "prefix",
                "year",
                "start_date",
                "episodes",
                "rho_var_mean",
                "sum_excess_var_contrib",
                "share_total_excess_var",
                "share_positive_excess_var",
                "log_ratio_reduction_if_removed",
                "mean_abs_hedge_gap",
                "mean_realized_vol",
                "mean_iv_change",
                "mean_moneyness",
                "mean_t",
            ]
            lines.append("Focus-year cluster decomposition")
            lines.append("--------------------------------")
            lines.append(
                focus_clusters.sort_values(
                    ["prefix", "log_ratio_reduction_if_removed"],
                    ascending=[True, False],
                )[cluster_cols].to_string(
                    index=False, float_format=lambda x: fmt_float(x, 4)
                )
            )
            lines.append("")

        focus_underhedge = underhedge_bins[
            underhedge_bins["year"].isin(focus_years)
        ].copy()
        if not focus_underhedge.empty:
            underhedge_cols = [
                "prefix",
                "year",
                "bin_type",
                "bin_label",
                "steps",
                "episodes",
                "mean_delta_gap",
                "underhedged_share",
                "mean_agent_delta",
                "mean_bs_delta",
            ]
            lines.append("Focus-year underhedging by state bins")
            lines.append("-------------------------------------")
            lines.append(
                focus_underhedge[underhedge_cols].to_string(
                    index=False, float_format=lambda x: fmt_float(x, 4)
                )
            )
            lines.append("")

    lines.append("Regression R2 table")
    lines.append("-------------------")
    lines.append(
        "Interpretation: this is within-year, across-episode explanatory power. "
        "A low R2 does not contradict a strong between-year regime shift in "
        "average rho."
    )
    r2_cols = ["prefix", "year", "target", "spec", "r2", "n", "features"]
    lines.append(regressions[r2_cols].to_string(index=False, float_format=lambda x: fmt_float(x, 4)))
    lines.append("")

    lines.append("Rho bucket table")
    lines.append("----------------")
    bucket_cols = [
        "prefix",
        "year",
        "rho_bucket",
        "episodes",
        "clusters",
        "rho_var_mean",
        "log_var_ratio",
        "log_downside_ratio",
        "sum_excess_var_contrib",
        "sum_positive_excess_var_contrib",
        "sum_excess_downside_contrib",
    ]
    existing_bucket_cols = [c for c in bucket_cols if c in buckets.columns]
    lines.append(
        buckets[existing_bucket_cols].to_string(index=False, float_format=lambda x: fmt_float(x, 4))
    )
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_html_report(
    path,
    readable_summary,
    buckets,
    episodes,
    regressions,
    hypothesis_tests,
    cluster_decomposition,
    step_asymmetry,
    underhedge_bins,
    episode_asymmetry,
    theory_matrix,
    compact_story,
    focus_years,
    top,
):
    # HTML is much easier to inspect than wide CSVs and needs no optional Excel
    # dependency. The raw CSVs are still written for reproducibility.
    style = """
    <style>
      body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
      h1, h2 { margin-bottom: 0.25rem; }
      p { max-width: 980px; line-height: 1.35; }
      table { border-collapse: collapse; margin: 12px 0 28px 0; font-size: 12px; }
      th, td { border: 1px solid #ddd; padding: 5px 7px; text-align: right; }
      th { background: #f3f3f3; position: sticky; top: 0; }
      td:first-child, th:first-child { text-align: left; }
    </style>
    """
    html = ["<html><head><meta charset='utf-8'>", style, "</head><body>"]
    html.append("<h1>Rho / terminal-PnL variance diagnostic report</h1>")
    html.append("<h2>Compact story</h2>")
    html.append("<pre>{}</pre>".format(compact_story))
    html.append(
        "<p>Variance is measured across terminal episode PnLs. Realized rho is "
        "estimated within each episode and used to explain that episode's "
        "terminal-PnL contribution.</p>"
    )
    html.append(
        "<p><b>excess_var_contrib</b> is exact: its sum equals "
        "Var(agent terminal PnL) - Var(BS terminal PnL).</p>"
    )
    html.append("<h2>Readable model/year summary</h2>")
    html.append(readable_summary.to_html(index=False, float_format=lambda x: fmt_float(x, 4)))
    html.append("<h2>Direct high-rho hypothesis tests</h2>")
    html.append(
        "<p>The p-value is an exact hypergeometric upper-tail test for "
        "enrichment of rho above the threshold among the top ranked "
        "contributors.</p>"
    )
    html.append(
        hypothesis_tests.to_html(index=False, float_format=lambda x: fmt_float(x, 4))
    )
    html.append("<h2>Step-level asymmetry: up days vs down days</h2>")
    html.append(
        "<p>delta_gap = agent_delta - BS_delta. Negative means underhedging "
        "relative to BS.</p>"
    )
    html.append(step_asymmetry.to_html(index=False, float_format=lambda x: fmt_float(x, 4)))
    html.append("<h2>Episode-level asymmetry: spot-up vs spot-down episodes</h2>")
    html.append(episode_asymmetry.to_html(index=False, float_format=lambda x: fmt_float(x, 4)))
    html.append("<h2>2x2 theory matrix: spot direction x rho regime</h2>")
    html.append(theory_matrix.to_html(index=False, float_format=lambda x: fmt_float(x, 4)))

    if focus_years:
        focus = episodes[episodes["year"].isin(focus_years)].copy()
        if not focus.empty:
            focus_cols = [
                "prefix",
                "year",
                "episode",
                "start_date",
                "a_pnl",
                "b_pnl",
                "rho_var",
                "excess_var_contrib",
                "excess_downside_contrib",
                "mean_abs_hedge_gap",
                "realized_vol_path",
                "iv_change",
                "start_moneyness",
                "start_t",
            ]
            html.append("<h2>Largest positive excess-variance contributors</h2>")
            html.append(
                focus.sort_values("excess_var_contrib", ascending=False)[focus_cols]
                .head(top)
                .to_html(index=False, float_format=lambda x: fmt_float(x, 4))
            )

            focus["abs_excess_var_contrib"] = np.abs(focus["excess_var_contrib"])
            html.append("<h2>Largest absolute excess-variance contributors</h2>")
            html.append(
                focus.sort_values("abs_excess_var_contrib", ascending=False)[
                    focus_cols + ["abs_excess_var_contrib"]
                ]
                .head(top)
                .to_html(index=False, float_format=lambda x: fmt_float(x, 4))
            )

        focus_clusters = cluster_decomposition[
            cluster_decomposition["year"].isin(focus_years)
        ].copy()
        if not focus_clusters.empty:
            html.append("<h2>Focus-year cluster decomposition</h2>")
            html.append(
                focus_clusters.sort_values(
                    ["prefix", "log_ratio_reduction_if_removed"],
                    ascending=[True, False],
                ).to_html(index=False, float_format=lambda x: fmt_float(x, 4))
            )

        focus_underhedge = underhedge_bins[
            underhedge_bins["year"].isin(focus_years)
        ].copy()
        if not focus_underhedge.empty:
            html.append("<h2>Focus-year underhedging by state bins</h2>")
            html.append(
                focus_underhedge.to_html(
                    index=False, float_format=lambda x: fmt_float(x, 4)
                )
            )

    html.append("<h2>Regression R2 table</h2>")
    html.append(
        "<p>This is within-year, across-episode explanatory power. A low R2 "
        "does not contradict a strong between-year regime shift in average rho.</p>"
    )
    html.append(regressions.to_html(index=False, float_format=lambda x: fmt_float(x, 4)))
    html.append("<h2>Rho bucket table</h2>")
    html.append(buckets.to_html(index=False, float_format=lambda x: fmt_float(x, 4)))
    html.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))


def run_analysis(args):
    prefixes = args.prefix or DEFAULT_PREFIXES
    years = set(args.year) if args.year else None

    metas = []
    for prefix in prefixes:
        metas.extend(discover_result_files(args.results_dir, prefix, years))

    if not metas:
        raise FileNotFoundError(
            "No result files found in {} for prefixes {}".format(
                args.results_dir, prefixes
            )
        )

    all_episodes = []
    all_summaries = []
    all_buckets = []
    all_regressions = []
    all_raw_steps = []

    for meta in metas:
        print(
            "Processing {} year {} checkpoint {}...".format(
                meta["prefix"], meta["year"], meta["checkpoint"]
            )
        )
        raw, ep = summarize_file(meta)
        all_raw_steps.append(prepare_step_frame(raw, meta))
        ep = add_rho_buckets(ep, args.rho_column, args.buckets)
        all_episodes.append(ep)
        all_summaries.append(summarize_model_year(ep))

        bucket_summary = (
            ep.groupby(["prefix", "year", "checkpoint", "rho_bucket"], dropna=False)
            .apply(summarize_bucket, include_groups=False)
            .reset_index()
        )
        all_buckets.append(bucket_summary)
        all_regressions.append(regression_diagnostics(ep))

    episodes = pd.concat(all_episodes, ignore_index=True)
    summary = pd.DataFrame(all_summaries).sort_values(["prefix", "year"])
    buckets = pd.concat(all_buckets, ignore_index=True)
    regressions = pd.concat(all_regressions, ignore_index=True)
    readable_summary = build_readable_summary(summary, regressions)
    hypothesis_tests = build_hypothesis_tests(
        episodes, args.rho_threshold, args.top_test_n
    )
    cluster_decomposition = build_cluster_decomposition(episodes)
    step_asymmetry = build_step_asymmetry(all_raw_steps)
    underhedge_bins = build_underhedge_bins(all_raw_steps, args.min_bin_steps)
    episode_asymmetry = build_episode_asymmetry(episodes)
    theory_matrix = build_theory_matrix(episodes, args.rho_threshold)
    compact_story = compact_final_story(
        summary,
        theory_matrix,
        regressions,
        args.story_prefix,
        args.story_year,
    )

    # Column order chosen to match the research question: first performance,
    # then rho proxies, then decomposition/explanatory diagnostics.
    summary_cols = [
        "prefix",
        "year",
        "checkpoint",
        "episodes",
        "clusters",
        "log_var_ratio",
        "log_downside_ratio",
        "mean_diff_reward",
        "mean_diff_pnl",
        "rho_var_mean",
        "rho_var_median",
        "rho_var_p10",
        "rho_var_p90",
        "realized_vol_mean",
        "iv_change_mean",
        "mean_abs_hedge_gap",
        "mean_delta_gap",
        "underhedged_share",
        "terminal_return_mean",
        "up_episode_share",
        "rho_var_up_days_mean",
        "rho_var_down_days_mean",
        "beta_var_on_ret_up_days_mean",
        "beta_var_on_ret_down_days_mean",
        "excess_var_total",
        "excess_downside_total",
        "positive_excess_var_total",
        "negative_excess_var_total",
        "var_diff",
        "two_cov_b_diff",
        "corr_rho_var_a_sq_dev",
        "corr_rho_var_excess_sq_dev",
        "corr_rho_var_excess_var_contrib",
        "corr_rho_var_positive_excess_var",
        "corr_abs_rho_var_a_sq_dev",
        "corr_hedge_gap_a_sq_dev",
        "corr_hedge_gap_excess_var_contrib",
    ]
    print_table("Model/year terminal-PnL variance and rho diagnostics", summary[summary_cols])
    print_table("Compact human-readable summary", readable_summary)

    bucket_cols = [
        "prefix",
        "year",
        "checkpoint",
        "rho_bucket",
        "episodes",
        "clusters",
        "rho_var_mean",
        "log_var_ratio",
        "log_downside_ratio",
        "mean_excess_sq_dev",
        "sum_excess_var_contrib",
        "sum_positive_excess_var_contrib",
        "sum_excess_downside_contrib",
        "mean_abs_hedge_gap",
        "mean_realized_vol",
    ]
    print_table("Rho-bucket diagnostics", buckets[bucket_cols])

    reg_cols = ["prefix", "year", "checkpoint", "target", "spec", "r2", "n", "features"]
    print_table("How much variance contribution is explained by rho proxies? (OLS R2)", regressions[reg_cols])

    hypothesis_cols = [
        "prefix",
        "year",
        "ranked_by",
        "threshold",
        "base_high_rho_share",
        "top_high_rho_count",
        "top_high_rho_share",
        "top_high_rho_enrichment_p_ge",
        "top_share_total_excess_var",
        "top_share_positive_excess_var",
    ]
    print_table(
        "Direct high-rho enrichment tests",
        hypothesis_tests[hypothesis_cols],
    )

    step_cols = [
        "prefix",
        "year",
        "condition",
        "steps",
        "episodes",
        "mean_delta_gap",
        "underhedged_share",
        "rho_var",
        "beta_var_on_ret",
        "mean_step_diff_pnl_100",
    ]
    print_table("Step-level asymmetry: up days vs down days", step_asymmetry[step_cols])

    episode_side_cols = [
        "prefix",
        "year",
        "episode_side",
        "episodes",
        "clusters",
        "mean_terminal_return",
        "mean_iv_change",
        "mean_rho_var",
        "mean_delta_gap",
        "mean_diff_pnl",
        "share_total_excess_var",
        "share_positive_excess_var",
    ]
    print_table(
        "Episode-level asymmetry: spot-up vs spot-down episodes",
        episode_asymmetry[episode_side_cols],
    )

    matrix_cols = [
        "prefix",
        "year",
        "spot_regime",
        "rho_regime",
        "episodes",
        "mean_terminal_return",
        "mean_rho_var",
        "mean_delta_gap",
        "mean_diff_pnl",
        "mean_diff_reward",
        "share_gross_abs_excess_var",
        "share_total_excess_var",
        "share_positive_excess_var",
        "share_excess_downside",
    ]
    story_matrix = theory_matrix[
        (theory_matrix["prefix"] == args.story_prefix)
        & (theory_matrix["year"].isin(args.story_year))
    ]
    print_table(
        "Compact 2x2 theory matrix for story years",
        story_matrix[matrix_cols],
    )

    if args.focus_year:
        focus = episodes[episodes["year"].isin(args.focus_year)]
        if not focus.empty:
            focus_cols = [
                "prefix",
                "year",
                "checkpoint",
                "episode",
                "start_date",
                "a_pnl",
                "b_pnl",
                "diff_pnl",
                "rho_var",
                "rho_iv",
                "realized_vol_path",
                "iv_change",
                "mean_abs_hedge_gap",
                "a_sq_dev",
                "excess_sq_dev",
                "excess_var_contrib",
                "excess_downside_contrib",
                "start_moneyness",
                "start_t",
            ]
            top = focus.sort_values("a_sq_dev", ascending=False)[focus_cols].head(args.top)
            print_table("Largest agent terminal-PnL variance contributors", top)
            top_excess = (
                focus.sort_values("excess_var_contrib", ascending=False)[focus_cols]
                .head(args.top)
            )
            print_table("Largest positive excess-variance contributors", top_excess)

    if args.write_prefix:
        summary.to_csv(args.write_prefix + "_summary.csv", index=False)
        readable_summary.to_csv(args.write_prefix + "_readable_summary.csv", index=False)
        buckets.to_csv(args.write_prefix + "_rho_buckets.csv", index=False)
        regressions.to_csv(args.write_prefix + "_rho_regressions.csv", index=False)
        episodes.to_csv(args.write_prefix + "_episodes.csv", index=False)
        hypothesis_tests.to_csv(
            args.write_prefix + "_hypothesis_tests.csv", index=False
        )
        cluster_decomposition.to_csv(
            args.write_prefix + "_cluster_decomposition.csv", index=False
        )
        step_asymmetry.to_csv(
            args.write_prefix + "_step_asymmetry.csv", index=False
        )
        underhedge_bins.to_csv(
            args.write_prefix + "_underhedge_bins.csv", index=False
        )
        episode_asymmetry.to_csv(
            args.write_prefix + "_episode_asymmetry.csv", index=False
        )
        theory_matrix.to_csv(
            args.write_prefix + "_theory_matrix.csv", index=False
        )
        with open(args.write_prefix + "_compact_story.txt", "w", encoding="utf-8") as f:
            f.write(compact_story)
        write_text_report(
            args.write_prefix + "_human_report.txt",
            readable_summary,
            buckets,
            episodes,
            regressions,
            hypothesis_tests,
            cluster_decomposition,
            step_asymmetry,
            underhedge_bins,
            episode_asymmetry,
            theory_matrix,
            compact_story,
            args.focus_year,
            args.top,
        )
        write_html_report(
            args.write_prefix + "_human_report.html",
            readable_summary,
            buckets,
            episodes,
            regressions,
            hypothesis_tests,
            cluster_decomposition,
            step_asymmetry,
            underhedge_bins,
            episode_asymmetry,
            theory_matrix,
            compact_story,
            args.focus_year,
            args.top,
        )
        print("\nWrote:")
        print("  {}_summary.csv".format(args.write_prefix))
        print("  {}_readable_summary.csv".format(args.write_prefix))
        print("  {}_rho_buckets.csv".format(args.write_prefix))
        print("  {}_rho_regressions.csv".format(args.write_prefix))
        print("  {}_episodes.csv".format(args.write_prefix))
        print("  {}_hypothesis_tests.csv".format(args.write_prefix))
        print("  {}_cluster_decomposition.csv".format(args.write_prefix))
        print("  {}_step_asymmetry.csv".format(args.write_prefix))
        print("  {}_underhedge_bins.csv".format(args.write_prefix))
        print("  {}_episode_asymmetry.csv".format(args.write_prefix))
        print("  {}_theory_matrix.csv".format(args.write_prefix))
        print("  {}_compact_story.txt".format(args.write_prefix))
        print("  {}_human_report.txt".format(args.write_prefix))
        print("  {}_human_report.html".format(args.write_prefix))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose terminal-PnL variance failures using realized spot-vol "
            "correlation proxies from results/testing CSV files."
        )
    )
    parser.add_argument("--results-dir", default="results/testing")
    parser.add_argument(
        "--prefix",
        action="append",
        help=(
            "Model class prefix, e.g. no_q_WF_exp1_k1_test. "
            "Can be repeated. Defaults to the two no_q classes."
        ),
    )
    parser.add_argument("--year", action="append", type=int, help="Year to include. Can be repeated.")
    parser.add_argument("--focus-year", action="append", type=int, default=[2023])
    parser.add_argument(
        "--story-year",
        action="append",
        type=int,
        default=[2022, 2023],
        help="Years emphasized in the compact story output. Can be repeated.",
    )
    parser.add_argument(
        "--story-prefix",
        default="final_WF_exp1_k1_test",
        help="Model prefix emphasized in the compact story output.",
    )
    parser.add_argument("--rho-column", default="rho_var", choices=["rho_var", "rho_iv"])
    parser.add_argument(
        "--rho-threshold",
        type=float,
        default=-0.4,
        help="Threshold used in the direct high-rho enrichment test.",
    )
    parser.add_argument(
        "--top-test-n",
        type=int,
        default=100,
        help="Number of top contributors used in the high-rho enrichment test.",
    )
    parser.add_argument(
        "--min-bin-steps",
        type=int,
        default=50,
        help="Minimum step rows required before reporting an underhedging state bin.",
    )
    parser.add_argument("--buckets", type=int, default=5)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--write-prefix", default="rho_variance_audit")
    args = parser.parse_args()

    run_analysis(args)


if __name__ == "__main__":
    main()
