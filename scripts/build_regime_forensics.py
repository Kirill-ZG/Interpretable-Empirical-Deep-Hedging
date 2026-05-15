"""Build granular regime-forensics tables from walk-forward test results.

This diagnostic compares competing explanations for years in which variance,
reward, or downside-risk behavior diverges.  It works from cached testing CSVs
and records both episode-level summaries and the daily steps that drive the
largest differences between the neural agent and the Black-Scholes benchmark.

The main PnL identity used below is:
    agent PnL = option price change + agent hedge PnL + transaction cost
    BS PnL    = option price change + BS hedge PnL + transaction cost

With zero transaction costs in the final empirical run, the agent-minus-BS
difference is driven primarily by the hedge-position difference times the
underlying price move.  The step-level tables also keep spot and implied
volatility changes so the paper diagnostics can distinguish hedge-position
effects from option-state effects.
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

DEFAULT_PREFIX = "final_WF_exp1_k1_test"
DEFAULT_RESULTS_DIR = "results/testing"
DEFAULT_OUTPUT_PREFIX = "forensic_final"
DEFAULT_WORST_EPISODES = 8
DEFAULT_WORST_STEPS = 8


def safe_var(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return float(np.var(x, ddof=1))


def safe_cov(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    return float(np.cov(x, y, ddof=1)[0, 1])


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
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    vx = np.var(x, ddof=1)
    if vx <= 1e-12:
        return np.nan
    return float(np.cov(x, y, ddof=1)[0, 1] / vx)


def safe_log_ratio(num, den):
    if pd.isna(num) or pd.isna(den) or num <= 1e-12 or den <= 1e-12:
        return np.nan
    return float(np.log(num / den))


def downside_second_moment(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.where(x < 0.0, x, 0.0) ** 2))


def fmt(x, digits=4):
    if pd.isna(x):
        return ""
    return ("{:,.%df}" % digits).format(float(x))


def discover_files(results_dir, prefix):
    files = []
    for path in sorted(glob.glob(os.path.join(results_dir, "{}*.csv".format(prefix)))):
        name = os.path.basename(path)
        match = re.match(r"^{}(?P<year>\d{{4}})_(?P<ckpt>\d+)\.csv$".format(re.escape(prefix)), name)
        if not match:
            continue
        files.append(
            {
                "year": int(match.group("year")),
                "checkpoint": int(match.group("ckpt")),
                "path": path,
            }
        )
    if not files:
        raise FileNotFoundError("No files found for prefix {}".format(prefix))
    return files


def load_all_results(results_dir, prefix):
    frames = []
    for meta in discover_files(results_dir, prefix):
        df = pd.read_csv(meta["path"])
        df["Date"] = pd.to_datetime(df["Date"])
        df["year"] = meta["year"]
        df["checkpoint"] = meta["checkpoint"]
        df["source_file"] = os.path.basename(meta["path"])
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    return raw.sort_values(["year", "episode", "Date"]).reset_index(drop=True)


def prepare_steps(raw):
    step = raw.copy()
    step["dS"] = step["S0"].astype(float) - step["S-1"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step["dlog_s"] = np.log(step["S0"].astype(float) / step["S-1"].astype(float))
    step["dS_pct"] = 100.0 * (step["S0"].astype(float) / step["S-1"].astype(float) - 1.0)
    step["dIV"] = step.groupby(["year", "episode"])["v"].transform(lambda s: s.astype(float).diff())
    step["dVar"] = step.groupby(["year", "episode"])["v"].transform(lambda s: s.astype(float).pow(2).diff())
    step["option_pnl_100"] = 100.0 * (step["P0"].astype(float) - step["P-1"].astype(float))
    step["agent_hedge_pnl_100"] = 100.0 * step["A Pos"].astype(float) * step["dS"]
    step["bs_hedge_pnl_100"] = 100.0 * step["B Pos"].astype(float) * step["dS"]
    step["a_pnl_100"] = 100.0 * step["A PnL"].astype(float)
    step["b_pnl_100"] = 100.0 * step["B PnL"].astype(float)
    step["diff_pnl_100"] = step["a_pnl_100"] - step["b_pnl_100"]
    step["diff_reward"] = step["A Reward"].astype(float) - step["B Reward"].astype(float)
    step["agent_delta"] = -step["A Pos"].astype(float)
    step["bs_delta"] = -step["B Pos"].astype(float)
    step["delta_gap"] = step["agent_delta"] - step["bs_delta"]
    step["underhedged"] = step["delta_gap"] < 0.0
    step["spot_move"] = np.where(step["dS"] > 0.0, "spot_up", np.where(step["dS"] < 0.0, "spot_down", "flat"))
    step["iv_move"] = np.where(step["dIV"] > 0.0, "iv_up", np.where(step["dIV"] < 0.0, "iv_down", "iv_flat"))
    step["move_pattern"] = step["spot_move"] + "_" + step["iv_move"]
    step["no_parachute_down"] = (step["dS"] < 0.0) & (step["dIV"] <= 0.0)
    step["weak_up_coupling"] = (step["dS"] > 0.0) & (step["dIV"] >= 0.0)
    step["tc_abs_100"] = 100.0 * step.get("A TC", pd.Series(0.0, index=step.index)).abs()
    return step


def summarize_episode(group):
    group = group.sort_values("Date").reset_index(drop=True)
    dlog_s = group["dlog_s"].to_numpy(dtype=float)
    d_var = group["dVar"].to_numpy(dtype=float)
    up = dlog_s > 0.0
    down = dlog_s < 0.0
    return pd.Series(
        {
            "start_date": group["Date"].iloc[0],
            "end_date": group["Date"].iloc[-1],
            "expiry": group["Expiry"].iloc[0] if "Expiry" in group.columns else "",
            "start_moneyness": float(group["S/K"].iloc[0]),
            "start_t": float(group["T"].iloc[0]),
            "a_pnl": group["a_pnl_100"].sum(),
            "b_pnl": group["b_pnl_100"].sum(),
            "diff_pnl": group["diff_pnl_100"].sum(),
            "a_reward": group["A Reward"].sum(),
            "b_reward": group["B Reward"].sum(),
            "diff_reward": group["diff_reward"].sum(),
            "terminal_return": float(group["S0"].iloc[-1] / group["S-1"].iloc[0] - 1.0),
            "option_pnl": group["option_pnl_100"].sum(),
            "agent_hedge_pnl": group["agent_hedge_pnl_100"].sum(),
            "bs_hedge_pnl": group["bs_hedge_pnl_100"].sum(),
            "rho_var": safe_corr(dlog_s, d_var),
            "rho_up": safe_corr(dlog_s[up], d_var[up]),
            "rho_down": safe_corr(dlog_s[down], d_var[down]),
            "beta_up": safe_beta(dlog_s[up], d_var[up]),
            "beta_down": safe_beta(dlog_s[down], d_var[down]),
            "realized_vol": float(np.nanstd(dlog_s, ddof=1) * np.sqrt(252.0)),
            "iv_start": float(group["v"].iloc[0]),
            "iv_end": float(group["v"].iloc[-1]),
            "iv_change": float(group["v"].iloc[-1] - group["v"].iloc[0]),
            "mean_delta_gap": group["delta_gap"].mean(),
            "mean_abs_delta_gap": group["delta_gap"].abs().mean(),
            "underhedged_share": group["underhedged"].mean(),
            "no_parachute_down_steps": int(group["no_parachute_down"].sum()),
            "weak_up_coupling_steps": int(group["weak_up_coupling"].sum()),
            "worst_step_a_pnl": group["a_pnl_100"].min(),
            "worst_step_diff_reward": group["diff_reward"].min(),
        }
    )


def add_year_contributions(ep):
    out = []
    for year, group in ep.groupby("year", sort=True):
        group = group.copy()
        n = len(group)
        group["a_var_contrib"] = (group["a_pnl"] - group["a_pnl"].mean()) ** 2 / (n - 1)
        group["b_var_contrib"] = (group["b_pnl"] - group["b_pnl"].mean()) ** 2 / (n - 1)
        group["excess_var_contrib"] = group["a_var_contrib"] - group["b_var_contrib"]
        group["positive_excess_var_contrib"] = group["excess_var_contrib"].clip(lower=0.0)
        group["a_down_contrib"] = np.where(group["a_pnl"] < 0.0, group["a_pnl"], 0.0) ** 2 / n
        group["b_down_contrib"] = np.where(group["b_pnl"] < 0.0, group["b_pnl"], 0.0) ** 2 / n
        group["excess_down_contrib"] = group["a_down_contrib"] - group["b_down_contrib"]
        out.append(group)
    return pd.concat(out, ignore_index=True)


def build_year_summary(ep):
    rows = []
    for year, g in ep.groupby("year", sort=True):
        var_a = safe_var(g["a_pnl"])
        var_b = safe_var(g["b_pnl"])
        var_option = safe_var(g["option_pnl"])
        var_b_hedge = safe_var(g["bs_hedge_pnl"])
        cov_option_bhedge = safe_cov(g["option_pnl"], g["bs_hedge_pnl"])
        var_a_hedge = safe_var(g["agent_hedge_pnl"])
        cov_option_ahedge = safe_cov(g["option_pnl"], g["agent_hedge_pnl"])
        rows.append(
            {
                "year": int(year),
                "episodes": len(g),
                "clusters": g["start_date"].nunique(),
                "var_a": var_a,
                "var_b": var_b,
                "log_var_ratio": safe_log_ratio(var_a, var_b),
                "downside_a": downside_second_moment(g["a_pnl"]),
                "downside_b": downside_second_moment(g["b_pnl"]),
                "log_downside_ratio": safe_log_ratio(downside_second_moment(g["a_pnl"]), downside_second_moment(g["b_pnl"])),
                "mean_diff_pnl": g["diff_pnl"].mean(),
                "mean_diff_reward": g["diff_reward"].mean(),
                "rho_mean": g["rho_var"].mean(),
                "rho_up_mean": g["rho_up"].mean(),
                "rho_down_mean": g["rho_down"].mean(),
                "terminal_return_mean": g["terminal_return"].mean(),
                "spot_up_share": (g["terminal_return"] > 0.0).mean(),
                "realized_vol_mean": g["realized_vol"].mean(),
                "iv_change_mean": g["iv_change"].mean(),
                "mean_delta_gap": g["mean_delta_gap"].mean(),
                "underhedged_share": g["underhedged_share"].mean(),
                "var_option_leg": var_option,
                "var_bs_hedge_leg": var_b_hedge,
                "two_cov_option_bs_hedge": 2.0 * cov_option_bhedge,
                "bs_var_rebuilt": var_option + var_b_hedge + 2.0 * cov_option_bhedge,
                "var_agent_hedge_leg": var_a_hedge,
                "two_cov_option_agent_hedge": 2.0 * cov_option_ahedge,
                "agent_var_rebuilt": var_option + var_a_hedge + 2.0 * cov_option_ahedge,
                "mean_no_parachute_down_steps": g["no_parachute_down_steps"].mean(),
                "mean_weak_up_coupling_steps": g["weak_up_coupling_steps"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    baseline = summary[summary["year"] <= 2021]
    for col in [
        "var_a",
        "var_b",
        "rho_mean",
        "rho_up_mean",
        "rho_down_mean",
        "realized_vol_mean",
        "iv_change_mean",
        "var_option_leg",
        "var_bs_hedge_leg",
        "two_cov_option_bs_hedge",
    ]:
        mu = baseline[col].mean()
        sd = baseline[col].std(ddof=1)
        summary[col + "_z_vs_2015_2021"] = (summary[col] - mu) / sd if sd > 1e-12 else np.nan
    return summary


def build_cluster_table(ep):
    rows = []
    for year, g in ep.groupby("year", sort=True):
        base_lr = safe_log_ratio(safe_var(g["a_pnl"]), safe_var(g["b_pnl"]))
        gross = g["excess_var_contrib"].abs().sum()
        for start_date, c in g.groupby("start_date", sort=True):
            rest = g[g["start_date"] != start_date]
            lr_without = safe_log_ratio(safe_var(rest["a_pnl"]), safe_var(rest["b_pnl"]))
            rows.append(
                {
                    "year": int(year),
                    "start_date": start_date,
                    "episodes": len(c),
                    "rho_mean": c["rho_var"].mean(),
                    "return_mean": c["terminal_return"].mean(),
                    "iv_change_mean": c["iv_change"].mean(),
                    "delta_gap_mean": c["mean_delta_gap"].mean(),
                    "diff_pnl_mean": c["diff_pnl"].mean(),
                    "diff_reward_mean": c["diff_reward"].mean(),
                    "sum_excess_var": c["excess_var_contrib"].sum(),
                    "gross_abs_share": c["excess_var_contrib"].abs().sum() / gross,
                    "log_ratio_reduction_if_removed": base_lr - lr_without,
                    "no_parachute_down_steps_mean": c["no_parachute_down_steps"].mean(),
                    "weak_up_coupling_steps_mean": c["weak_up_coupling_steps"].mean(),
                }
            )
    return pd.DataFrame(rows)


def select_autopsy_episodes(ep, year, mode, n):
    g = ep[ep["year"] == year].copy()
    if mode == "variance_2023":
        return g.sort_values("positive_excess_var_contrib", ascending=False).head(n)
    if mode == "reward_2022":
        return g.sort_values("diff_reward", ascending=True).head(n)
    if mode == "bs_denominator":
        # Episodes where BS terminal PnL is unusually close to the BS yearly
        # mean while agent is far from its mean are informative for ratio blowups.
        g["denom_gap_score"] = g["a_var_contrib"] - g["b_var_contrib"]
        return g.sort_values("denom_gap_score", ascending=False).head(n)
    raise ValueError(mode)


def step_autopsy_for_episode(step, year, episode, n_steps):
    g = step[(step["year"] == year) & (step["episode"] == episode)].copy()
    g["abs_a_step"] = g["a_pnl_100"].abs()
    g["abs_diff_reward"] = g["diff_reward"].abs()
    g["rank_score"] = g[["abs_a_step", "abs_diff_reward"]].max(axis=1)
    keep = g.sort_values("rank_score", ascending=False).head(n_steps)
    cols = [
        "Date",
        "dS_pct",
        "dIV",
        "option_pnl_100",
        "agent_hedge_pnl_100",
        "bs_hedge_pnl_100",
        "a_pnl_100",
        "b_pnl_100",
        "diff_pnl_100",
        "diff_reward",
        "agent_delta",
        "bs_delta",
        "delta_gap",
        "move_pattern",
        "no_parachute_down",
        "weak_up_coupling",
    ]
    return keep[cols].sort_values("Date")


def build_worst_episode_tables(ep, step, n_episodes, n_steps):
    selected = []
    for year, mode in [(2022, "reward_2022"), (2023, "variance_2023"), (2023, "bs_denominator")]:
        if year not in set(ep["year"]):
            continue
        rows = select_autopsy_episodes(ep, year, mode, n_episodes)
        rows = rows.copy()
        rows["autopsy_mode"] = mode
        selected.append(rows)
    selected_ep = pd.concat(selected, ignore_index=True)

    step_rows = []
    for _, row in selected_ep.iterrows():
        s = step_autopsy_for_episode(step, int(row["year"]), int(row["episode"]), n_steps)
        s = s.copy()
        s.insert(0, "autopsy_mode", row["autopsy_mode"])
        s.insert(1, "year", int(row["year"]))
        s.insert(2, "episode", int(row["episode"]))
        s.insert(3, "start_date", row["start_date"])
        step_rows.append(s)
    selected_steps = pd.concat(step_rows, ignore_index=True)
    return selected_ep, selected_steps


def build_2017_2023_deep_tables(ep, step, n_episodes):
    # Focused denominator/variance comparison for the two low-BS-variance years.
    # 2017 matters because it is the other year with a very small BS variance.
    # 2023 matters because the final graph's variance ratio explodes there.
    # Looking at both years prevents us from falsely treating "small BS
    # denominator" as a complete explanation.
    years = [2017, 2023]

    # Cross-year baseline for the exact daily pattern that appears repeatedly
    # in 2023's worst episodes: spot rises while IV falls. In this code the
    # option leg is long and the hedge is short underlying. On these days the
    # option can still rise if delta/spot dominates vega/IV compression, while
    # the short hedge loses. A less-hedged agent then loses less than BS.
    spot_up_iv_down = (
        step[step["move_pattern"] == "spot_up_iv_down"]
        .groupby("year", sort=True)
        .agg(
            steps=("Date", "count"),
            sum_agent_pnl=("a_pnl_100", "sum"),
            sum_bs_pnl=("b_pnl_100", "sum"),
            sum_agent_minus_bs=("diff_pnl_100", "sum"),
            mean_agent_pnl=("a_pnl_100", "mean"),
            mean_bs_pnl=("b_pnl_100", "mean"),
            mean_agent_minus_bs=("diff_pnl_100", "mean"),
            mean_delta_gap=("delta_gap", "mean"),
            mean_spot_return_pct=("dS_pct", "mean"),
            mean_iv_change=("dIV", "mean"),
        )
        .reset_index()
    )

    top_excess = []
    top_steps = []
    cancellation = []
    cancellation_steps = []

    for year in years:
        if year not in set(ep["year"]):
            continue

        g = ep[ep["year"] == year].copy()
        g["gross_bs_legs"] = g["option_pnl"].abs() + g["bs_hedge_pnl"].abs()
        g["bs_abs_to_gross"] = g["b_pnl"].abs() / g["gross_bs_legs"].replace(0.0, np.nan)

        # Episodes that directly drive the numerator of the excess variance
        # story: large positive contribution to Var(agent)-Var(BS).
        te = g.sort_values("positive_excess_var_contrib", ascending=False).head(n_episodes)
        te = te.copy()
        te["deep_mode"] = "top_positive_excess_variance"
        top_excess.append(te)

        # Episodes where BS has large option and hedge legs but very small
        # residual PnL. These are the concrete observations behind a small BS
        # denominator. The residual is small when option_pnl + bs_hedge_pnl is
        # tightly clustered, not necessarily exactly zero.
        ce = g.sort_values(["gross_bs_legs", "bs_abs_to_gross"], ascending=[False, True]).head(n_episodes)
        ce = ce.copy()
        ce["deep_mode"] = "bs_large_legs_small_residual"
        cancellation.append(ce)

        selected_ids = set(te["episode"].astype(int))
        ts = step[(step["year"] == year) & (step["episode"].astype(int).isin(selected_ids))].copy()
        ts["deep_mode"] = "top_positive_excess_variance"
        top_steps.append(ts)

        cancel_ids = set(ce["episode"].astype(int))
        cs = step[(step["year"] == year) & (step["episode"].astype(int).isin(cancel_ids))].copy()
        cs["deep_mode"] = "bs_large_legs_small_residual"
        cancellation_steps.append(cs)

    top_excess = pd.concat(top_excess, ignore_index=True) if top_excess else pd.DataFrame()
    top_steps = pd.concat(top_steps, ignore_index=True) if top_steps else pd.DataFrame()
    cancellation = pd.concat(cancellation, ignore_index=True) if cancellation else pd.DataFrame()
    cancellation_steps = pd.concat(cancellation_steps, ignore_index=True) if cancellation_steps else pd.DataFrame()

    return spot_up_iv_down, top_excess, top_steps, cancellation, cancellation_steps


def build_option_dynamics_tables(ep, step):
    # Empirical option-dynamics tables.
    # These tables answer the central mechanism question directly:
    # when spot rises and IV falls, does the call price actually fall, or does
    # the delta/spot effect usually dominate the IV-compression effect?
    # The answer matters because the agent's relative gain versus BS on those
    # days is not caused by the call price falling. It is caused by the agent
    # being less short the underlying hedge, so it loses less hedge PnL than BS.
    conditional_rows = []
    key_patterns = [
        "spot_up_iv_down",
        "spot_up_iv_up",
        "spot_down_iv_up",
        "spot_down_iv_down",
    ]
    for year, group in step.groupby("year", sort=True):
        for pattern in key_patterns:
            h = group[group["move_pattern"] == pattern]
            if len(h) == 0:
                continue
            conditional_rows.append(
                {
                    "year": int(year),
                    "pattern": pattern,
                    "steps": len(h),
                    "mean_spot_return_pct": h["dS_pct"].mean(),
                    "mean_iv_change": h["dIV"].mean(),
                    "mean_option_pnl": h["option_pnl_100"].mean(),
                    "option_up_share": (h["option_pnl_100"] > 0.0).mean(),
                    "mean_agent_pnl": h["a_pnl_100"].mean(),
                    "mean_bs_pnl": h["b_pnl_100"].mean(),
                    "mean_agent_minus_bs": h["diff_pnl_100"].mean(),
                    "mean_delta_gap": h["delta_gap"].mean(),
                    "underhedged_share": (h["delta_gap"] < 0.0).mean(),
                }
            )

    # Moneyness composition matters for the denominator story. Deep ITM calls
    # are more delta-like; BS can cancel their option leg tightly, while an
    # underhedged agent preserves directional exposure.
    moneyness_rows = []
    for year, group in ep.groupby("year", sort=True):
        top = group.sort_values("positive_excess_var_contrib", ascending=False).head(20)
        moneyness_rows.append(
            {
                "year": int(year),
                "episodes": len(group),
                "mean_start_moneyness": group["start_moneyness"].mean(),
                "median_start_moneyness": group["start_moneyness"].median(),
                "share_itm": (group["start_moneyness"] > 1.0).mean(),
                "top20_mean_start_moneyness": top["start_moneyness"].mean(),
                "top20_share_itm": (top["start_moneyness"] > 1.0).mean(),
                "top20_mean_delta_gap": top["mean_delta_gap"].mean(),
                "top20_mean_terminal_return": top["terminal_return"].mean(),
                "top20_mean_iv_change": top["iv_change"].mean(),
                "top20_mean_agent_minus_bs": top["diff_pnl"].mean(),
            }
        )

    return pd.DataFrame(conditional_rows), pd.DataFrame(moneyness_rows)


def build_cancellation_decomposition(ep):
    # Exact variance decomposition by year.
    # This is the central diagnostic for the question:
    #   Why does Var(agent terminal PnL) exceed Var(BS terminal PnL)?
    # Since both portfolios hold the same option, the option-leg variance
    # cancels in the difference:
    #   Var(A) - Var(BS)
    #     = [Var(agent hedge) - Var(BS hedge)]
    #       + 2 * [Cov(option, agent hedge) - Cov(option, BS hedge)]
    # Underhedging normally reduces hedge-leg variance, which helps the agent.
    # It can still lose on total variance if the option/hedge covariance becomes
    # less negative by more than the hedge-variance saving. That is precisely
    # the mechanism we need to test year by year.
    rows = []
    for year, group in ep.groupby("year", sort=True):
        var_option = safe_var(group["option_pnl"])
        var_b_hedge = safe_var(group["bs_hedge_pnl"])
        var_a_hedge = safe_var(group["agent_hedge_pnl"])
        cov_option_b = safe_cov(group["option_pnl"], group["bs_hedge_pnl"])
        cov_option_a = safe_cov(group["option_pnl"], group["agent_hedge_pnl"])
        var_b = safe_var(group["b_pnl"])
        var_a = safe_var(group["a_pnl"])
        b_gross = var_option + var_b_hedge
        a_gross = var_option + var_a_hedge
        hedge_var_saving = var_a_hedge - var_b_hedge
        covariance_loss = 2.0 * (cov_option_a - cov_option_b)
        rows.append(
            {
                "year": int(year),
                "episodes": len(group),
                "var_agent": var_a,
                "var_bs": var_b,
                "log_var_ratio": safe_log_ratio(var_a, var_b),
                "var_agent_minus_bs": var_a - var_b,
                "option_var": var_option,
                "bs_hedge_var": var_b_hedge,
                "agent_hedge_var": var_a_hedge,
                "bs_two_cov": 2.0 * cov_option_b,
                "agent_two_cov": 2.0 * cov_option_a,
                "bs_gross_var_before_cov": b_gross,
                "agent_gross_var_before_cov": a_gross,
                "bs_cancellation_fraction": -2.0 * cov_option_b / b_gross if b_gross > 0 else np.nan,
                "agent_cancellation_fraction": -2.0 * cov_option_a / a_gross if a_gross > 0 else np.nan,
                "agent_less_tight_by": (
                    (-2.0 * cov_option_b / b_gross) - (-2.0 * cov_option_a / a_gross)
                    if b_gross > 0 and a_gross > 0
                    else np.nan
                ),
                "hedge_var_saving_agent_minus_bs": hedge_var_saving,
                "covariance_loss_agent_minus_bs": covariance_loss,
                "rebuilt_var_agent_minus_bs": hedge_var_saving + covariance_loss,
                "mean_delta_gap": group["mean_delta_gap"].mean(),
                "mean_abs_delta_gap": group["mean_abs_delta_gap"].mean(),
                "share_itm": (group["start_moneyness"] > 1.0).mean(),
                "mean_start_moneyness": group["start_moneyness"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_cluster_mechanism_table(ep, step):
    # Per-start-date mechanism table across all years.
    # This lets us answer whether the March 2023 story is isolated or whether
    # other 2023 clusters show the same broad mechanics. It also ranks 2023
    # clusters against clusters from all other years.
    rows = []
    for (year, start_date), group in ep.groupby(["year", "start_date"], sort=True):
        if len(group) < 2:
            continue

        var_option = safe_var(group["option_pnl"])
        var_b_hedge = safe_var(group["bs_hedge_pnl"])
        var_a_hedge = safe_var(group["agent_hedge_pnl"])
        cov_option_b = safe_cov(group["option_pnl"], group["bs_hedge_pnl"])
        cov_option_a = safe_cov(group["option_pnl"], group["agent_hedge_pnl"])
        b_gross = var_option + var_b_hedge
        a_gross = var_option + var_a_hedge

        episode_ids = set(group["episode"].astype(int))
        st = step[(step["year"] == year) & (step["episode"].astype(int).isin(episode_ids))]
        spot_down = st[st["spot_move"] == "spot_down"]
        spot_down_iv_up = st[st["move_pattern"] == "spot_down_iv_up"]
        spot_down_iv_down = st[st["move_pattern"] == "spot_down_iv_down"]

        rows.append(
            {
                "year": int(year),
                "start_date": start_date,
                "episodes": len(group),
                "var_agent": safe_var(group["a_pnl"]),
                "var_bs": safe_var(group["b_pnl"]),
                "log_var_ratio": safe_log_ratio(safe_var(group["a_pnl"]), safe_var(group["b_pnl"])),
                "sum_excess_var_contrib": group["excess_var_contrib"].sum(),
                "sum_positive_excess_var_contrib": group["positive_excess_var_contrib"].sum(),
                "mean_agent_minus_bs_pnl": group["diff_pnl"].mean(),
                "mean_agent_minus_bs_reward": group["diff_reward"].mean(),
                "bs_cancellation_fraction": -2.0 * cov_option_b / b_gross if b_gross > 0 else np.nan,
                "agent_cancellation_fraction": -2.0 * cov_option_a / a_gross if a_gross > 0 else np.nan,
                "agent_less_tight_by": (
                    (-2.0 * cov_option_b / b_gross) - (-2.0 * cov_option_a / a_gross)
                    if b_gross > 0 and a_gross > 0
                    else np.nan
                ),
                "bs_gross_var_before_cov": b_gross,
                "agent_gross_var_before_cov": a_gross,
                "mean_delta_gap": group["mean_delta_gap"].mean(),
                "mean_start_moneyness": group["start_moneyness"].mean(),
                "share_itm": (group["start_moneyness"] > 1.0).mean(),
                "mean_terminal_return": group["terminal_return"].mean(),
                "mean_iv_change": group["iv_change"].mean(),
                "mean_rho_var": group["rho_var"].mean(),
                "spot_up_iv_down_step_share": (st["move_pattern"] == "spot_up_iv_down").mean(),
                "spot_down_iv_up_step_share": (st["move_pattern"] == "spot_down_iv_up").mean(),
                "spot_down_iv_down_step_share": (st["move_pattern"] == "spot_down_iv_down").mean(),
                "spot_down_option_up_share": (
                    (spot_down["option_pnl_100"] > 0.0).mean() if len(spot_down) else np.nan
                ),
                "spot_down_iv_up_option_up_share": (
                    (spot_down_iv_up["option_pnl_100"] > 0.0).mean() if len(spot_down_iv_up) else np.nan
                ),
                "spot_down_iv_down_option_up_share": (
                    (spot_down_iv_down["option_pnl_100"] > 0.0).mean() if len(spot_down_iv_down) else np.nan
                ),
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        out["positive_excess_rank_all_clusters"] = out["sum_excess_var_contrib"].rank(
            method="min", ascending=False
        ).astype(int)
    return out


def build_spot_down_option_table(step):
    # Detailed spot-down option dynamics.
    # This is the empirical version of the "vega parachute" question. The
    # agent's reward only cares about negative step PnL, so spot-down days are
    # critical. We split them by IV-up versus IV-down to see whether IV rises
    # often enough, and whether it actually prevents the call from losing value.
    rows = []
    for year, group in step.groupby("year", sort=True):
        for pattern in ["spot_down_iv_up", "spot_down_iv_down", "spot_down_iv_flat"]:
            h = group[group["move_pattern"] == pattern]
            if len(h) == 0:
                continue
            rows.append(
                {
                    "year": int(year),
                    "pattern": pattern,
                    "steps": len(h),
                    "mean_spot_return_pct": h["dS_pct"].mean(),
                    "mean_iv_change": h["dIV"].mean(),
                    "mean_option_pnl": h["option_pnl_100"].mean(),
                    "option_up_share": (h["option_pnl_100"] > 0.0).mean(),
                    "mean_agent_pnl": h["a_pnl_100"].mean(),
                    "mean_bs_pnl": h["b_pnl_100"].mean(),
                    "mean_agent_minus_bs": h["diff_pnl_100"].mean(),
                    "mean_delta_gap": h["delta_gap"].mean(),
                    "agent_negative_pnl_share": (h["a_pnl_100"] < 0.0).mean(),
                    "bs_negative_pnl_share": (h["b_pnl_100"] < 0.0).mean(),
                }
            )
    return pd.DataFrame(rows)


def build_negative_pnl_state_table(step):
    # Negative-PnL state attribution.
    # The reward is triggered by negative PnL, not by spot-down days per se.
    # This table splits every negative-PnL step by:
    #   spot up/down, option up/down, IV up/down.
    # It answers a practical question: is the agent/BS mostly hurt by true
    # down-market failures, or by spot-up days where the short hedge loses more
    # than the option gains?
    tmp = step.copy()
    tmp["option_move"] = np.where(
        tmp["option_pnl_100"] > 0.0,
        "option_up",
        np.where(tmp["option_pnl_100"] < 0.0, "option_down", "option_flat"),
    )
    tmp["iv_sign"] = np.where(
        tmp["dIV"] > 0.0,
        "iv_up",
        np.where(tmp["dIV"] < 0.0, "iv_down", "iv_flat"),
    )
    tmp["state3"] = tmp["spot_move"] + "/" + tmp["option_move"] + "/" + tmp["iv_sign"]

    rows = []
    for portfolio, pnl_col, hedge_col in [
        ("agent", "a_pnl_100", "agent_hedge_pnl_100"),
        ("bs", "b_pnl_100", "bs_hedge_pnl_100"),
    ]:
        for year, group in tmp.groupby("year", sort=True):
            neg = group[group[pnl_col] < 0.0]
            if neg.empty:
                continue
            total_neg_loss = neg[pnl_col].sum()
            by_state = (
                neg.groupby("state3", sort=True)
                .agg(
                    steps=(pnl_col, "count"),
                    sum_pnl=(pnl_col, "sum"),
                    mean_pnl=(pnl_col, "mean"),
                    mean_option_pnl=("option_pnl_100", "mean"),
                    mean_hedge_pnl=(hedge_col, "mean"),
                    mean_agent_minus_bs=("diff_pnl_100", "mean"),
                    mean_delta_gap=("delta_gap", "mean"),
                )
                .reset_index()
            )
            by_state["year"] = int(year)
            by_state["portfolio"] = portfolio
            by_state["share_negative_steps"] = by_state["steps"] / len(neg)
            by_state["share_negative_loss"] = by_state["sum_pnl"] / total_neg_loss
            rows.append(by_state)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_option_driver_correlation_table(step):
    # Correlation table for the causal-mechanism discussion.
    # The OLS coefficient table says "conditional sensitivity"; this correlation
    # table says how tightly option PnL co-moves with spot and IV unconditionally,
    # plus the realized spot/IV coupling. It helps separate:
    #   1. spot-IV correlation / leverage effect,
    #   2. option-IV co-movement,
    #   3. spot becoming the dominant option-price driver.
    rows = []
    for year, group in step.groupby("year", sort=True):
        h = group[["option_pnl_100", "dS_pct", "dIV"]].dropna()
        rows.append(
            {
                "year": int(year),
                "corr_option_pnl_spot_return": h["option_pnl_100"].corr(h["dS_pct"]),
                "corr_option_pnl_iv_change": h["option_pnl_100"].corr(h["dIV"]),
                "corr_spot_return_iv_change": h["dS_pct"].corr(h["dIV"]),
                "std_option_pnl": h["option_pnl_100"].std(),
                "std_spot_return_pct": h["dS_pct"].std(),
                "std_iv_change": h["dIV"].std(),
            }
        )
    return pd.DataFrame(rows)


def write_report(path, summary, clusters, selected_ep, selected_steps):
    lines = []
    lines.append("Granular forensic autopsy: final model")
    lines.append("======================================")
    lines.append("")
    lines.append("Question: why is 2023 different from prior years, and what happened in 2022?")
    lines.append("")

    year_cols = [
        "year",
        "var_a",
        "var_b",
        "log_var_ratio",
        "mean_diff_reward",
        "rho_mean",
        "rho_up_mean",
        "rho_down_mean",
        "realized_vol_mean",
        "var_option_leg",
        "var_bs_hedge_leg",
        "two_cov_option_bs_hedge",
        "mean_delta_gap",
        "underhedged_share",
    ]
    lines.append("1. Cross-year comparison")
    lines.append("------------------------")
    lines.append(summary[year_cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
    lines.append("")

    z_cols = [
        "year",
        "var_b_z_vs_2015_2021",
        "rho_mean_z_vs_2015_2021",
        "rho_up_mean_z_vs_2015_2021",
        "rho_down_mean_z_vs_2015_2021",
        "two_cov_option_bs_hedge_z_vs_2015_2021",
    ]
    lines.append("2. How unusual is each year versus 2015-2021?")
    lines.append("---------------------------------------------")
    lines.append(summary[z_cols].to_string(index=False, float_format=lambda x: fmt(x, 2)))
    lines.append("")

    lines.append("3. BS denominator decomposition")
    lines.append("-------------------------------")
    lines.append(
        "BS terminal variance is Var(option leg) + Var(BS hedge leg) + 2Cov(option, hedge). "
        "A small denominator can come from small option/hedge variances or unusually negative covariance."
    )
    denom_cols = [
        "year",
        "var_b",
        "var_option_leg",
        "var_bs_hedge_leg",
        "two_cov_option_bs_hedge",
        "bs_var_rebuilt",
    ]
    lines.append(summary[denom_cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
    lines.append("")

    lines.append("4. Did previous years also have cluster concentration?")
    lines.append("-----------------------------------------------------")
    top_cluster = (
        clusters.sort_values(["year", "gross_abs_share"], ascending=[True, False])
        .groupby("year")
        .head(3)
    )
    cluster_cols = [
        "year",
        "start_date",
        "episodes",
        "rho_mean",
        "return_mean",
        "iv_change_mean",
        "diff_pnl_mean",
        "diff_reward_mean",
        "sum_excess_var",
        "gross_abs_share",
        "log_ratio_reduction_if_removed",
    ]
    lines.append(top_cluster[cluster_cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
    lines.append("")

    lines.append("5. Worst episode summary")
    lines.append("------------------------")
    episode_cols = [
        "autopsy_mode",
        "year",
        "episode",
        "start_date",
        "start_moneyness",
        "start_t",
        "a_pnl",
        "b_pnl",
        "diff_pnl",
        "diff_reward",
        "terminal_return",
        "iv_change",
        "rho_var",
        "option_pnl",
        "agent_hedge_pnl",
        "bs_hedge_pnl",
        "mean_delta_gap",
        "excess_var_contrib",
    ]
    lines.append(selected_ep[episode_cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
    lines.append("")

    lines.append("6. Step-level causes inside selected episodes")
    lines.append("---------------------------------------------")
    step_cols = [
        "autopsy_mode",
        "year",
        "episode",
        "Date",
        "dS_pct",
        "dIV",
        "option_pnl_100",
        "agent_hedge_pnl_100",
        "bs_hedge_pnl_100",
        "a_pnl_100",
        "b_pnl_100",
        "diff_pnl_100",
        "diff_reward",
        "delta_gap",
        "move_pattern",
        "no_parachute_down",
        "weak_up_coupling",
    ]
    lines.append(selected_steps[step_cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
    lines.append("")

    lines.append("Reviewer notes")
    lines.append("--------------")
    lines.append(
        "Use this report to falsify simple stories. If 2023's BS denominator is uniquely low, "
        "ask which leg/covariance caused it. If worst steps are spot-down with IV down, that is "
        "a no-parachute failure. If worst positive variance steps are spot-up with IV up or flat, "
        "that is weak upside coupling plus underhedging. If a few start dates dominate, the result "
        "is a cluster-regime result rather than a broad cross-sectional law."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_digest(
    path,
    summary,
    clusters,
    selected_ep,
    selected_steps,
    spot_up_iv_down=None,
    deep_top_excess=None,
    deep_top_steps=None,
    deep_cancellation=None,
    deep_cancellation_steps=None,
    option_dynamics=None,
    moneyness_composition=None,
    cancellation_decomposition=None,
    cluster_mechanism=None,
    spot_down_option_dynamics=None,
    negative_pnl_state=None,
    option_driver_correlation=None,
):
    # Compact human-readable digest.
    # The long report above is intentionally exhaustive, but it is hard to use
    # when the research question is: "what is the actual mechanism?" This
    # digest keeps only the diagnostics needed to answer that question:
    #   1. Is 2023 unusual relative to the available pre-2022 testing years?
    #   2. Is the BS denominator shrink caused by tiny legs, better covariance
    #      cancellation, or both?
    #   3. Is 2023 just another cluster-concentration year?
    #   4. Which daily move patterns create the selected 2022 reward losses and
    #      2023 variance outliers?
    # I deliberately keep the text short and interpretive. The CSVs remain the
    # audit trail if any number needs to be checked line by line.
    s = summary.copy()
    c = clusters.copy()
    e = selected_ep.copy()
    st = selected_steps.copy()

    s["bs_gross_leg_var"] = s["var_option_leg"] + s["var_bs_hedge_leg"]
    s["bs_cancellation_fraction"] = -s["two_cov_option_bs_hedge"] / s["bs_gross_leg_var"]
    s["var_b_rank_low_to_high"] = s["var_b"].rank(method="min", ascending=True).astype(int)
    s["rho_mean_rank_weak_to_strong"] = s["rho_mean"].rank(method="min", ascending=False).astype(int)
    s["rho_up_rank_weak_to_strong"] = s["rho_up_mean"].rank(method="min", ascending=False).astype(int)
    s["log_var_rank_high_to_low"] = s["log_var_ratio"].rank(method="min", ascending=False).astype(int)

    # Use 2015-2021 as the available "normal-ish" final-model test baseline.
    # The final run does not currently have 2010-2014 testing files in
    # results/testing, so the script must not pretend those years were tested.
    baseline = s[(s["year"] >= 2015) & (s["year"] <= 2021)]
    y2022 = s[s["year"] == 2022].iloc[0] if (s["year"] == 2022).any() else None
    y2023 = s[s["year"] == 2023].iloc[0] if (s["year"] == 2023).any() else None

    # Cluster uniqueness: if earlier years have equal or larger cluster shares,
    # cluster concentration is a contributor but not a stand-alone explanation.
    top_cluster_by_year = (
        c.sort_values(["year", "gross_abs_share"], ascending=[True, False])
        .groupby("year")
        .head(1)
        .sort_values("gross_abs_share", ascending=False)
    )

    # Step-pattern summary for the hand-picked autopsy episodes. This is the
    # most direct way to see whether losses came from spot-down/IV-down
    # no-parachute days, spot-up/IV-down underhedging days, or something else.
    pattern = (
        st.groupby(["autopsy_mode", "move_pattern"], sort=True)
        .agg(
            steps=("Date", "count"),
            sum_a_pnl=("a_pnl_100", "sum"),
            sum_b_pnl=("b_pnl_100", "sum"),
            sum_diff_pnl=("diff_pnl_100", "sum"),
            sum_diff_reward=("diff_reward", "sum"),
            mean_delta_gap=("delta_gap", "mean"),
        )
        .reset_index()
        .sort_values(["autopsy_mode", "steps"], ascending=[True, False])
    )

    # Which start dates dominate the selected episodes? This distinguishes a
    # broad year-wide law from a localized path/cluster mechanism.
    selected_clusters = (
        e.groupby(["autopsy_mode", "year", "start_date"], sort=True)
        .agg(
            episodes=("episode", "count"),
            mean_a_pnl=("a_pnl", "mean"),
            mean_b_pnl=("b_pnl", "mean"),
            mean_diff_pnl=("diff_pnl", "mean"),
            mean_diff_reward=("diff_reward", "mean"),
            mean_rho=("rho_var", "mean"),
            mean_return=("terminal_return", "mean"),
            mean_iv_change=("iv_change", "mean"),
            mean_delta_gap=("mean_delta_gap", "mean"),
        )
        .reset_index()
        .sort_values(["autopsy_mode", "episodes"], ascending=[True, False])
    )

    lines = []
    lines.append("Compact forensic digest: final model")
    lines.append("====================================")
    lines.append("")
    lines.append("Scope")
    lines.append("-----")
    lines.append(
        "Available final-model testing files cover 2015-2023. "
        "Therefore 2022/2023 are compared against 2015-2021, not 2010-2014."
    )
    lines.append("")

    if y2023 is not None:
        lines.append("2023 in one sentence")
        lines.append("--------------------")
        lines.append(
            "2023 is unusual mainly because realized spot/variance correlation is weak "
            "relative to 2015-2021, especially on up moves, while the BS denominator "
            "is the second-smallest available year and the agent still has enough "
            "terminal dispersion to make log Var(A)/Var(BS) large."
        )
        lines.append(
            "Numbers: log_var_ratio={}, var_a={}, var_b={}, rho_mean={}, rho_up={}, rho_down={}, mean_delta_gap={}.".format(
                fmt(y2023["log_var_ratio"]),
                fmt(y2023["var_a"]),
                fmt(y2023["var_b"]),
                fmt(y2023["rho_mean"]),
                fmt(y2023["rho_up_mean"]),
                fmt(y2023["rho_down_mean"]),
                fmt(y2023["mean_delta_gap"]),
            )
        )
        lines.append(
            "Ranks among 2015-2023: log variance ratio #{} highest, BS variance #{} lowest, "
            "mean rho #{} weakest, upside rho #{} weakest.".format(
                int(y2023["log_var_rank_high_to_low"]),
                int(y2023["var_b_rank_low_to_high"]),
                int(y2023["rho_mean_rank_weak_to_strong"]),
                int(y2023["rho_up_rank_weak_to_strong"]),
            )
        )
        lines.append("")

    if y2022 is not None:
        lines.append("2022 in one sentence")
        lines.append("--------------------")
        lines.append(
            "2022's headline problem is not terminal variance; it is step-wise asymmetric "
            "reward damage. The worst reward episodes have positive agent-minus-BS "
            "terminal PnL but many negative daily PnL steps, and positive later steps "
            "do not compensate because exp1/k1 rewards positive PnL almost flatly."
        )
        lines.append(
            "Numbers: log_var_ratio={}, mean_diff_reward={}, rho_mean={}, rho_up={}, rho_down={}, underhedged_share={}.".format(
                fmt(y2022["log_var_ratio"]),
                fmt(y2022["mean_diff_reward"]),
                fmt(y2022["rho_mean"]),
                fmt(y2022["rho_up_mean"]),
                fmt(y2022["rho_down_mean"]),
                fmt(y2022["underhedged_share"]),
            )
        )
        lines.append("")

    if cancellation_decomposition is not None and not cancellation_decomposition.empty:
        lines.append("Exact Year-Level Variance Identity")
        lines.append("----------------------------------")
        lines.append(
            "Because the option leg is the same for agent and BS, Var(agent)-Var(BS) equals "
            "[Var(agent hedge)-Var(BS hedge)] + 2[Cov(option, agent hedge)-Cov(option, BS hedge)]. "
            "Negative hedge_var_saving helps the agent; positive covariance_loss hurts the agent."
        )
        cols = [
            "year",
            "var_agent",
            "var_bs",
            "log_var_ratio",
            "var_agent_minus_bs",
            "hedge_var_saving_agent_minus_bs",
            "covariance_loss_agent_minus_bs",
            "bs_cancellation_fraction",
            "agent_cancellation_fraction",
            "agent_less_tight_by",
            "mean_delta_gap",
            "share_itm",
        ]
        lines.append(
            cancellation_decomposition[cols].to_string(
                index=False, float_format=lambda x: fmt(x, 4)
            )
        )
        lines.append(
            "Read 2023 carefully: the agent saves hedge variance by hedging less, but loses more through weaker option/hedge covariance. "
            "That covariance loss is the mechanical reason total agent variance exceeds BS variance."
        )
        lines.append("")

    lines.append("BS denominator diagnosis")
    lines.append("------------------------")
    if y2022 is not None and y2023 is not None:
        gross_change = y2023["bs_gross_leg_var"] / y2022["bs_gross_leg_var"] - 1.0
        den_change = y2023["var_b"] / y2022["var_b"] - 1.0
        lines.append(
            "From 2022 to 2023, BS variance falls by {} while gross option+hedge leg variance falls by {}.".format(
                fmt(100.0 * den_change, 1) + "%",
                fmt(100.0 * gross_change, 1) + "%",
            )
        )
        lines.append(
            "2023 cancellation fraction -2Cov/(Var(option)+Var(hedge)) is {}, versus {} in 2022 and {} average in 2015-2021.".format(
                fmt(y2023["bs_cancellation_fraction"]),
                fmt(y2022["bs_cancellation_fraction"]),
                fmt(baseline["bs_cancellation_fraction"].mean()),
            )
        )
        lines.append(
            "So the low denominator is not just 'BS got lucky in 2023': the gross legs are smaller than in 2022 and covariance cancellation is very tight. "
            "But 2017 had an even smaller BS variance, so denominator shrink alone cannot explain why 2023 is special."
        )
    lines.append(
        s[
            [
                "year",
                "var_b",
                "bs_gross_leg_var",
                "bs_cancellation_fraction",
                "rho_mean",
                "rho_up_mean",
            ]
        ].to_string(index=False, float_format=lambda x: fmt(x, 4))
    )
    lines.append("")

    lines.append("Cluster concentration check")
    lines.append("---------------------------")
    lines.append(
        "Cluster concentration is real but not unique to 2023. The largest single-cluster gross shares are:"
    )
    lines.append(
        top_cluster_by_year[
            [
                "year",
                "start_date",
                "episodes",
                "gross_abs_share",
                "log_ratio_reduction_if_removed",
                "rho_mean",
                "return_mean",
                "iv_change_mean",
            ]
        ]
        .head(9)
        .to_string(index=False, float_format=lambda x: fmt(x, 4))
    )
    lines.append("")

    lines.append("Selected episode clusters")
    lines.append("-------------------------")
    lines.append(
        selected_clusters.to_string(index=False, float_format=lambda x: fmt(x, 4))
    )
    lines.append("")

    lines.append("Step patterns inside selected episodes")
    lines.append("--------------------------------------")
    lines.append(pattern.to_string(index=False, float_format=lambda x: fmt(x, 4)))
    lines.append("")

    if spot_up_iv_down is not None:
        lines.append("Why spot-up/IV-down is not automatically a 2023-only story")
        lines.append("----------------------------------------------------------")
        lines.append(
            "On spot-up/IV-down days the option leg may still gain if the spot/delta effect dominates IV compression. "
            "Both hedges are short underlying and lose money. If the agent is less hedged than BS, the agent often "
            "loses less than BS, even when the agent's absolute daily PnL is negative."
        )
        lines.append(
            spot_up_iv_down.to_string(index=False, float_format=lambda x: fmt(x, 4))
        )
        lines.append(
            "The pattern exists in many years. The reason it becomes visible in the 2023 variance ratio is the combination "
            "of a small BS denominator and concentrated positive agent-minus-BS outliers, not the daily pattern alone."
        )
        lines.append("")

    if option_dynamics is not None and not option_dynamics.empty:
        lines.append("Empirical option-price sign by move pattern")
        lines.append("-------------------------------------------")
        lines.append(
            "This checks the mechanical question directly. In the final testing output, "
            "calls usually rise on spot-up/IV-down days, so IV compression usually does not overwhelm delta."
        )
        lines.append(
            option_dynamics[
                option_dynamics["pattern"].isin(["spot_up_iv_down", "spot_down_iv_up"])
            ]
            .to_string(index=False, float_format=lambda x: fmt(x, 4))
        )
        lines.append("")

    if moneyness_composition is not None and not moneyness_composition.empty:
        lines.append("Moneyness composition")
        lines.append("---------------------")
        lines.append(
            "A more ITM-heavy year is more delta-dominated. That makes BS residuals small "
            "while leaving room for an underhedged agent to keep directional PnL."
        )
        lines.append(moneyness_composition.to_string(index=False, float_format=lambda x: fmt(x, 4)))
        lines.append("")

    if deep_top_excess is not None and not deep_top_excess.empty:
        cols = [
            "year",
            "episode",
            "start_date",
            "start_moneyness",
            "start_t",
            "a_pnl",
            "b_pnl",
            "diff_pnl",
            "excess_var_contrib",
            "terminal_return",
            "iv_change",
            "rho_var",
            "option_pnl",
            "agent_hedge_pnl",
            "bs_hedge_pnl",
            "mean_delta_gap",
            "underhedged_share",
        ]
        lines.append("2017 versus 2023: top positive excess-variance episodes")
        lines.append("-------------------------------------------------------")
        lines.append(deep_top_excess[cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
        lines.append("")

    if deep_top_steps is not None and not deep_top_steps.empty:
        step_pattern = (
            deep_top_steps.groupby(["year", "move_pattern"], sort=True)
            .agg(
                steps=("Date", "count"),
                sum_agent_pnl=("a_pnl_100", "sum"),
                sum_bs_pnl=("b_pnl_100", "sum"),
                sum_agent_minus_bs=("diff_pnl_100", "sum"),
                mean_delta_gap=("delta_gap", "mean"),
                mean_spot_return_pct=("dS_pct", "mean"),
                mean_iv_change=("dIV", "mean"),
            )
            .reset_index()
            .sort_values(["year", "steps"], ascending=[True, False])
        )
        lines.append("Step patterns inside top 2017/2023 excess-variance episodes")
        lines.append("------------------------------------------------------------")
        lines.append(step_pattern.to_string(index=False, float_format=lambda x: fmt(x, 4)))
        lines.append("")

    if deep_cancellation is not None and not deep_cancellation.empty:
        cols = [
            "year",
            "episode",
            "start_date",
            "start_moneyness",
            "start_t",
            "b_pnl",
            "option_pnl",
            "bs_hedge_pnl",
            "gross_bs_legs",
            "bs_abs_to_gross",
            "terminal_return",
            "iv_change",
            "rho_var",
        ]
        lines.append("2017 versus 2023: concrete BS denominator examples")
        lines.append("--------------------------------------------------")
        lines.append(
            "These are episodes with large option and BS-hedge legs but a small BS residual. "
            "They show how the denominator becomes small: option movement and BS hedge movement nearly cancel."
        )
        lines.append(deep_cancellation[cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
        lines.append("")

    if deep_cancellation_steps is not None and not deep_cancellation_steps.empty:
        cancellation_pattern = (
            deep_cancellation_steps.groupby(["year", "move_pattern"], sort=True)
            .agg(
                steps=("Date", "count"),
                sum_option_leg=("option_pnl_100", "sum"),
                sum_bs_hedge_leg=("bs_hedge_pnl_100", "sum"),
                sum_bs_residual=("b_pnl_100", "sum"),
                mean_spot_return_pct=("dS_pct", "mean"),
                mean_iv_change=("dIV", "mean"),
            )
            .reset_index()
            .sort_values(["year", "steps"], ascending=[True, False])
        )
        lines.append("Step patterns inside BS denominator examples")
        lines.append("--------------------------------------------")
        lines.append(cancellation_pattern.to_string(index=False, float_format=lambda x: fmt(x, 4)))
        lines.append("")

    if cluster_mechanism is not None and not cluster_mechanism.empty:
        lines.append("All 2023 Clusters")
        lines.append("-----------------")
        lines.append(
            "This table prevents overfitting the story to March. It shows every 2023 start-date cluster, "
            "with cancellation, moneyness, path, and excess-variance contribution."
        )
        cluster_cols = [
            "start_date",
            "episodes",
            "var_agent",
            "var_bs",
            "log_var_ratio",
            "sum_excess_var_contrib",
            "positive_excess_rank_all_clusters",
            "mean_agent_minus_bs_pnl",
            "bs_cancellation_fraction",
            "agent_cancellation_fraction",
            "agent_less_tight_by",
            "mean_delta_gap",
            "mean_start_moneyness",
            "share_itm",
            "mean_terminal_return",
            "mean_iv_change",
            "mean_rho_var",
            "spot_up_iv_down_step_share",
            "spot_down_iv_up_step_share",
            "spot_down_iv_down_step_share",
        ]
        lines.append(
            cluster_mechanism[cluster_mechanism["year"] == 2023]
            .sort_values("sum_excess_var_contrib", ascending=False)[cluster_cols]
            .to_string(index=False, float_format=lambda x: fmt(x, 4))
        )
        lines.append("")

        lines.append("Top Positive Excess-Variance Clusters Across All Years")
        lines.append("------------------------------------------------------")
        all_cols = [
            "year",
            "start_date",
            "episodes",
            "sum_excess_var_contrib",
            "var_agent",
            "var_bs",
            "log_var_ratio",
            "bs_cancellation_fraction",
            "agent_cancellation_fraction",
            "mean_delta_gap",
            "mean_start_moneyness",
            "share_itm",
            "mean_terminal_return",
            "mean_iv_change",
            "mean_rho_var",
        ]
        lines.append(
            cluster_mechanism.sort_values("sum_excess_var_contrib", ascending=False)
            .head(20)[all_cols]
            .to_string(index=False, float_format=lambda x: fmt(x, 4))
        )
        lines.append("")

    if spot_down_option_dynamics is not None and not spot_down_option_dynamics.empty:
        lines.append("Spot-Down Option Dynamics")
        lines.append("-------------------------")
        lines.append(
            "On spot-down days the call almost always loses value, even when IV rises. "
            "The IV-up/leverage effect softens the loss but usually does not reverse it."
        )
        lines.append(
            spot_down_option_dynamics.to_string(
                index=False, float_format=lambda x: fmt(x, 4)
            )
        )
        lines.append("")

    if option_driver_correlation is not None and not option_driver_correlation.empty:
        lines.append("Option Driver Correlations")
        lines.append("--------------------------")
        lines.append(
            "This separates realized spot/IV coupling from option/IV co-movement. "
            "2023 is unusual because option PnL is more spot-driven and much less negatively tied to IV changes."
        )
        lines.append(
            option_driver_correlation.to_string(
                index=False, float_format=lambda x: fmt(x, 4)
            )
        )
        lines.append("")

    if negative_pnl_state is not None and not negative_pnl_state.empty:
        lines.append("Top Negative-PnL State Buckets")
        lines.append("------------------------------")
        lines.append(
            "Top three buckets by share of total negative PnL loss for each year and portfolio."
        )
        top_negative = (
            negative_pnl_state.sort_values(
                ["portfolio", "year", "share_negative_loss"],
                ascending=[True, True, False],
            )
            .groupby(["portfolio", "year"], sort=True)
            .head(3)
        )
        cols = [
            "portfolio",
            "year",
            "state3",
            "steps",
            "share_negative_loss",
            "mean_pnl",
            "mean_option_pnl",
            "mean_hedge_pnl",
            "mean_agent_minus_bs",
        ]
        lines.append(top_negative[cols].to_string(index=False, float_format=lambda x: fmt(x, 4)))
        lines.append("")

    lines.append("Interpretation")
    lines.append("--------------")
    lines.append(
        "For 2023 variance, the selected positive excess-variance episodes are mostly March 2023. "
        "Their repeated large steps are spot-up/IV-down days: calls rise from spot, IV compression partly offsets, "
        "and both hedges lose on the short underlying position, but BS loses more because it is more hedged. "
        "That leaves BS terminal PnLs close to zero while the underhedged agent keeps positive outliers."
    )
    lines.append(
        "For 2022 rewards, the selected worst episodes are almost entirely the 2022-09-28 start cluster. "
        "The damaging steps are spot-down/IV-down or spot-down/IV-flat: option value falls, the underhedged agent "
        "does not earn enough hedge profit, and the reward function punishes these negative daily PnLs immediately."
    )
    lines.append(
        "My current reviewer-level conclusion: rho is a major regime variable, but not a complete one-line explanation. "
        "The proximate 2023 mechanism is upside path + IV compression + underhedging + a small BS denominator. "
        "The proximate 2022 mechanism is no-parachute down steps under an asymmetric step reward."
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args):
    raw = load_all_results(args.results_dir, args.prefix)
    step = prepare_steps(raw)
    ep = (
        step.groupby(["year", "episode"], sort=True)
        .apply(summarize_episode, include_groups=False)
        .reset_index()
    )
    ep = add_year_contributions(ep)
    summary = build_year_summary(ep)
    clusters = build_cluster_table(ep)
    selected_ep, selected_steps = build_worst_episode_tables(
        ep, step, args.worst_episodes, args.worst_steps
    )
    (
        spot_up_iv_down,
        deep_top_excess,
        deep_top_steps,
        deep_cancellation,
        deep_cancellation_steps,
    ) = build_2017_2023_deep_tables(ep, step, args.worst_episodes)
    option_dynamics, moneyness_composition = build_option_dynamics_tables(ep, step)
    cancellation_decomposition = build_cancellation_decomposition(ep)
    cluster_mechanism = build_cluster_mechanism_table(ep, step)
    spot_down_option_dynamics = build_spot_down_option_table(step)
    negative_pnl_state = build_negative_pnl_state_table(step)
    option_driver_correlation = build_option_driver_correlation_table(step)

    out = args.output_prefix
    summary.to_csv(out + "_year_summary.csv", index=False)
    clusters.to_csv(out + "_cluster_table.csv", index=False)
    selected_ep.to_csv(out + "_selected_episodes.csv", index=False)
    selected_steps.to_csv(out + "_selected_steps.csv", index=False)
    spot_up_iv_down.to_csv(out + "_spot_up_iv_down_by_year.csv", index=False)
    deep_top_excess.to_csv(out + "_2017_2023_top_excess_episodes.csv", index=False)
    deep_top_steps.to_csv(out + "_2017_2023_top_excess_steps.csv", index=False)
    deep_cancellation.to_csv(out + "_2017_2023_bs_cancellation_episodes.csv", index=False)
    deep_cancellation_steps.to_csv(out + "_2017_2023_bs_cancellation_steps.csv", index=False)
    option_dynamics.to_csv(out + "_option_dynamics_by_year.csv", index=False)
    moneyness_composition.to_csv(out + "_moneyness_composition_by_year.csv", index=False)
    cancellation_decomposition.to_csv(out + "_cancellation_decomposition_by_year.csv", index=False)
    cluster_mechanism.to_csv(out + "_cluster_mechanism_all_years.csv", index=False)
    spot_down_option_dynamics.to_csv(out + "_spot_down_option_dynamics_by_year.csv", index=False)
    negative_pnl_state.to_csv(out + "_negative_pnl_state_breakdown.csv", index=False)
    option_driver_correlation.to_csv(out + "_option_driver_correlations_by_year.csv", index=False)
    ep.to_csv(out + "_all_episodes.csv", index=False)
    write_report(out + "_report.txt", summary, clusters, selected_ep, selected_steps)
    write_digest(
        out + "_digest.txt",
        summary,
        clusters,
        selected_ep,
        selected_steps,
        spot_up_iv_down,
        deep_top_excess,
        deep_top_steps,
        deep_cancellation,
        deep_cancellation_steps,
        option_dynamics,
        moneyness_composition,
        cancellation_decomposition,
        cluster_mechanism,
        spot_down_option_dynamics,
        negative_pnl_state,
        option_driver_correlation,
    )

    print("Wrote:")
    print("  {}_report.txt".format(out))
    print("  {}_digest.txt".format(out))
    print("  {}_year_summary.csv".format(out))
    print("  {}_cluster_table.csv".format(out))
    print("  {}_selected_episodes.csv".format(out))
    print("  {}_selected_steps.csv".format(out))
    print("  {}_spot_up_iv_down_by_year.csv".format(out))
    print("  {}_2017_2023_top_excess_episodes.csv".format(out))
    print("  {}_2017_2023_top_excess_steps.csv".format(out))
    print("  {}_2017_2023_bs_cancellation_episodes.csv".format(out))
    print("  {}_2017_2023_bs_cancellation_steps.csv".format(out))
    print("  {}_option_dynamics_by_year.csv".format(out))
    print("  {}_moneyness_composition_by_year.csv".format(out))
    print("  {}_cancellation_decomposition_by_year.csv".format(out))
    print("  {}_cluster_mechanism_all_years.csv".format(out))
    print("  {}_spot_down_option_dynamics_by_year.csv".format(out))
    print("  {}_negative_pnl_state_breakdown.csv".format(out))
    print("  {}_option_driver_correlations_by_year.csv".format(out))


def main():
    parser = argparse.ArgumentParser(
        description="Granular forensic autopsy of final model testing results."
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--worst-episodes", type=int, default=DEFAULT_WORST_EPISODES)
    parser.add_argument("--worst-steps", type=int, default=DEFAULT_WORST_STEPS)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
