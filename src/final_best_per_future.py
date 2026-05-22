"""Final best 1-day strategy per futures with stability analysis.

- IF/IC/IH: full history (2017-01-17+)
- IM: post-IM-listing only (2022-07-22+)

For each best strategy, compute:
  * yearly returns (compounded)
  * monthly returns matrix (year x month)
  * stability metrics: % positive years/months, std of yearly returns,
    Calmar (return/max_dd), worst-year, worst-month
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORT_DIR, CN_FUTURES
from src.strategy import run_grid, backtest, StrategySpec

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

IM_LISTING = "2022-07-22"
MIN_TRADES = 30


def find_best(summary: pd.DataFrame, cn_fut: str, min_trades: int = MIN_TRADES):
    d = summary[(summary["cn_fut"] == cn_fut) & (summary["n_trades"] >= min_trades)]
    d = d.dropna(subset=["sharpe_ann"])
    if len(d) == 0:
        return None
    return d.sort_values("sharpe_ann", ascending=False).iloc[0]


def trade_log_for_spec(row: pd.Series, sample_start: str | None) -> pd.DataFrame:
    spec = StrategySpec(
        name=f"final_{row['cn_fut']}",
        us_code=row["us_code"], cn_fut=row["cn_fut"],
        family=row["family"], threshold=float(row["threshold"]),
        sample_start=sample_start,
    )
    trades, _ = backtest(spec)
    return trades


def yearly_monthly_returns(trades: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Compounded yearly and (year x month) returns from trade-level net returns."""
    if len(trades) == 0:
        return pd.Series(dtype=float), pd.DataFrame()
    t = trades.copy()
    t["entry_date"] = pd.to_datetime(t["entry_date"])
    t["year"] = t["entry_date"].dt.year
    t["month"] = t["entry_date"].dt.month
    # Yearly compounded
    yearly = t.groupby("year").apply(
        lambda g: (1 + g["net_ret"]).prod() - 1, include_groups=False
    )
    # Yearly trade count + hit rate
    ystats = t.groupby("year").agg(
        n_trades=("net_ret", "size"),
        hit_rate=("net_ret", lambda x: float((x > 0).mean())),
        mean_bps=("net_ret", lambda x: float(x.mean()) * 10000),
        std_bps=("net_ret", lambda x: float(x.std(ddof=1)) * 10000 if len(x) > 1 else 0),
        worst_bps=("net_ret", lambda x: float(x.min()) * 10000),
        best_bps=("net_ret", lambda x: float(x.max()) * 10000),
    )
    ystats["return_pct"] = (yearly * 100).round(3)
    # Monthly compounded matrix
    monthly_ser = t.groupby(["year", "month"]).apply(
        lambda g: (1 + g["net_ret"]).prod() - 1, include_groups=False
    )
    monthly = monthly_ser.unstack("month")
    monthly = monthly.reindex(columns=range(1, 13))
    return ystats, monthly


def stability_metrics(yearly_returns: pd.Series, trades: pd.DataFrame) -> dict:
    if len(yearly_returns) == 0:
        return {}
    nets = trades["net_ret"].values
    wealth = np.cumprod(1 + nets)
    dd = wealth / np.maximum.accumulate(wealth) - 1
    total_ret = float((wealth[-1] - 1) * 100)
    max_dd = float(dd.min() * 100)
    return {
        "n_years": int(len(yearly_returns)),
        "pct_positive_years": float((yearly_returns > 0).mean()),
        "best_year_pct": float(yearly_returns.max() * 100),
        "worst_year_pct": float(yearly_returns.min() * 100),
        "yearly_mean_pct": float(yearly_returns.mean() * 100),
        "yearly_std_pct": float(yearly_returns.std(ddof=1) * 100) if len(yearly_returns) > 1 else 0,
        "total_return_pct": total_ret,
        "max_dd_pct": max_dd,
        "calmar": float(total_ret / abs(max_dd)) if max_dd < 0 else np.nan,
    }


def plot_yearly(yearly_data: dict, filename: str):
    """yearly_data: {cn_fut: pd.Series of yearly returns}"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    for i, (cn_fut, yseries) in enumerate(yearly_data.items()):
        ax = axes[i]
        if len(yseries) == 0:
            ax.set_title(f"{cn_fut}: no data")
            continue
        vals = yseries.values * 100
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
        ax.bar(yseries.index.astype(str), vals, color=colors, alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(j, v + (0.4 if v >= 0 else -0.4), f"{v:+.1f}%",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(f"{cn_fut}")
        ax.set_ylabel("Annual return %")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Yearly returns — 4 best strategies", y=1.0, fontsize=13)
    fig.tight_layout()
    out = REPORT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_monthly_heatmap(monthly_data: dict, filename: str):
    """monthly_data: {cn_fut: DataFrame (year x month)}"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for i, (cn_fut, mdf) in enumerate(monthly_data.items()):
        ax = axes[i]
        if mdf.empty:
            ax.set_title(f"{cn_fut}: no data")
            continue
        vals = mdf.values * 100  # in pct
        vmax = max(abs(np.nanmin(vals)), abs(np.nanmax(vals)))
        im = ax.imshow(vals, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(12))
        ax.set_xticklabels([str(m) for m in range(1, 13)])
        ax.set_yticks(range(len(mdf.index)))
        ax.set_yticklabels(mdf.index.astype(str))
        ax.set_title(f"{cn_fut} — monthly returns %")
        # Annotate
        for r in range(vals.shape[0]):
            for c in range(vals.shape[1]):
                v = vals[r, c]
                if not np.isnan(v):
                    color = "white" if abs(v) > vmax * 0.6 else "black"
                    ax.text(c, r, f"{v:+.1f}", ha="center", va="center",
                            fontsize=7, color=color)
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Monthly returns heatmap — 4 best strategies", y=1.0, fontsize=13)
    fig.tight_layout()
    out = REPORT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 220)

    # ---- 1) Full-sample grid for IF/IC/IH ----
    print("=== Running full-sample 1-day grid (for IF/IC/IH) ===")
    summary_full, _ = run_grid(sample_start=None, out_suffix="_full_history_1day")
    print(f"Total specs: {len(summary_full)}")

    # ---- 2) IM-era grid for IM ----
    print(f"\n=== Running IM-era 1-day grid (for IM, sample_start={IM_LISTING}) ===")
    summary_im, _ = run_grid(sample_start=IM_LISTING, out_suffix="_im_era_1day")
    print(f"Total specs: {len(summary_im)}")

    # ---- 3) Pick best per futures ----
    best_rows = []
    for cn_fut in ["IF", "IC", "IH"]:
        r = find_best(summary_full, cn_fut, MIN_TRADES)
        if r is not None:
            r = r.copy()
            r["sample_used"] = "full_history"
            best_rows.append(r)

    r = find_best(summary_im, "IM", MIN_TRADES)
    if r is not None:
        r = r.copy()
        r["sample_used"] = "im_era"
        best_rows.append(r)

    best = pd.DataFrame(best_rows)
    cols = ["cn_fut", "family", "us_code", "threshold", "sample_used", "n_trades",
            "hit_rate", "mean_ret_bps", "total_ret_pct", "sharpe_ann",
            "max_dd_pct", "t_stat"]
    print("\n===== BEST per future =====")
    print(best[cols].to_string(index=False))
    best.to_csv(REPORT_DIR / "final_best_per_future.csv", index=False)

    # ---- 4) Per-strategy yearly + monthly + stability ----
    yearly_data = {}
    monthly_data = {}
    stability_rows = []
    yearly_stats_combined = {}

    for _, row in best.iterrows():
        cn_fut = row["cn_fut"]
        sample_start = IM_LISTING if cn_fut == "IM" else None
        trades = trade_log_for_spec(row, sample_start=sample_start)
        ystats, monthly = yearly_monthly_returns(trades)
        yearly_data[cn_fut] = ystats["return_pct"] / 100  # back to fraction
        monthly_data[cn_fut] = monthly
        yearly_stats_combined[cn_fut] = ystats
        stab = stability_metrics(yearly_data[cn_fut], trades)
        stab["cn_fut"] = cn_fut
        stab["family"] = row["family"]
        stab["us_code"] = row["us_code"]
        stab["threshold"] = row["threshold"]
        stability_rows.append(stab)

        # Save per-strategy details
        ystats.to_csv(REPORT_DIR / f"yearly_stats_{cn_fut}.csv")
        monthly.to_csv(REPORT_DIR / f"monthly_returns_{cn_fut}.csv")

    stab_df = pd.DataFrame(stability_rows)
    print("\n===== Stability metrics =====")
    show = ["cn_fut", "family", "us_code", "threshold", "n_years",
            "pct_positive_years", "best_year_pct", "worst_year_pct",
            "yearly_mean_pct", "yearly_std_pct", "total_return_pct",
            "max_dd_pct", "calmar"]
    print(stab_df[show].to_string(index=False))
    stab_df.to_csv(REPORT_DIR / "final_stability_metrics.csv", index=False)

    print("\n===== Yearly returns table =====")
    for cn_fut, ystats in yearly_stats_combined.items():
        print(f"\n--- {cn_fut} ---")
        print(ystats[["n_trades", "hit_rate", "mean_bps", "std_bps",
                       "worst_bps", "best_bps", "return_pct"]].to_string())

    print("\n===== Plotting =====")
    print("  yearly bars:", plot_yearly(yearly_data, "final_yearly_returns.png"))
    print("  monthly heatmaps:", plot_monthly_heatmap(monthly_data, "final_monthly_heatmaps.png"))
