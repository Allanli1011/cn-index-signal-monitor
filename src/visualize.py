"""Visualizations for hypothesis testing results.

Generates:
  1. Scatter + asymmetric regression lines for each (US, CN) pair (12 panels)
  2. Bucket bar charts (CN mean | US bucket) for cc and oc returns
  3. Heatmaps of asymmetry metrics across the 3x4 panel
  4. Probability comparison (P(CN down|US down) vs P(CN up|US up))
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import US_ASSETS, CN_INDEX_ASSETS, REPORT_DIR
from src.alignment import align

# Don't try to render Chinese in default matplotlib font; stick to ASCII labels
mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

CN_PRETTY = {"HS300": "CSI 300", "ZZ500": "CSI 500", "ZZ1000": "CSI 1000", "SSE50": "SSE 50"}
US_PRETTY = {"SPX": "S&P 500", "NDX": "Nasdaq", "RUT": "Russell 2000"}


def _asymmetric_fit(x, y):
    x_pos = np.where(x > 0, x, 0.0)
    x_neg = np.where(x < 0, x, 0.0)
    X = np.column_stack([np.ones_like(x), x_pos, x_neg])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def scatter_grid(y_col: str = "cn_ret_cc", filename: str = "scatter_grid_cc.png"):
    fig, axes = plt.subplots(len(US_ASSETS), len(CN_INDEX_ASSETS),
                              figsize=(16, 11), sharex=True, sharey=True)
    for i, us in enumerate(US_ASSETS):
        for j, cn in enumerate(CN_INDEX_ASSETS):
            ax = axes[i, j]
            df = align(us, cn, "cn_index")
            d = df[["us_ret_cc", y_col]].dropna()
            x, y = d["us_ret_cc"].values, d[y_col].values
            ax.scatter(x, y, s=4, alpha=0.18, color="#1f77b4")
            beta = _asymmetric_fit(x, y)
            xs = np.linspace(x.min(), x.max(), 200)
            xs_pos = np.where(xs > 0, xs, 0.0)
            xs_neg = np.where(xs < 0, xs, 0.0)
            yhat = beta[0] + beta[1]*xs_pos + beta[2]*xs_neg
            ax.plot(xs, yhat, color="crimson", linewidth=1.6, label=f"β+={beta[1]:+.2f}, β-={beta[2]:+.2f}")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.legend(loc="upper left", fontsize=7)
            if i == 0:
                ax.set_title(CN_PRETTY[cn], fontsize=10)
            if j == 0:
                ax.set_ylabel(US_PRETTY[us], fontsize=10)
            if i == len(US_ASSETS)-1:
                ax.set_xlabel("US prior-day return", fontsize=9)

    fig.suptitle(f"A-share {y_col} vs US prior-day close-to-close return  (with asymmetric fit)",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = REPORT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def bucket_bars(y_col: str, filename: str):
    """Bar chart of mean CN return by US-return bucket, for each pair."""
    buckets_csv = REPORT_DIR / "magnitude_bucket_stats.csv"
    bk = pd.read_csv(buckets_csv)
    bk = bk[bk["y_col"] == y_col].copy()
    order = ["us_le_-2pct", "us_in_-2_-1pct", "us_in_-1_0pct",
             "us_in_0_1pct", "us_in_1_2pct", "us_ge_2pct"]
    pretty_lbl = ["US≤-2%", "-2~-1%", "-1~0%", "0~1%", "1~2%", "US≥2%"]
    bk["bucket_idx"] = bk["bucket"].map({b: i for i, b in enumerate(order)})

    fig, axes = plt.subplots(len(US_ASSETS), len(CN_INDEX_ASSETS),
                              figsize=(16, 9), sharex=True, sharey=True)
    for i, us in enumerate(US_ASSETS):
        for j, cn in enumerate(CN_INDEX_ASSETS):
            ax = axes[i, j]
            sub = bk[(bk["us_code"] == us) & (bk["cn_code"] == cn)].sort_values("bucket_idx")
            vals = sub["cn_mean"].values * 100  # bps to pct: *100 gives pct, *10000 gives bps
            colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]
            counts = sub["count"].astype(int).values
            ax.bar(np.arange(len(order)), vals, color=colors, alpha=0.85)
            for k, (v, n) in enumerate(zip(vals, counts)):
                ax.text(k, v + (0.02 if v >= 0 else -0.02),
                        f"n={n}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(pretty_lbl, rotation=35, fontsize=7)
            if i == 0:
                ax.set_title(CN_PRETTY[cn], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{US_PRETTY[us]}\n(CN mean, %)", fontsize=9)

    return_type = "close-to-close" if y_col == "cn_ret_cc" else "open-to-close"
    fig.suptitle(f"Mean A-share {return_type} return by US prior-day return bucket",
                 fontsize=13, y=0.995)
    fig.tight_layout()
    out = REPORT_DIR / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def asymmetry_heatmap():
    """3x4 heatmap of (|β-| - |β+|) and asymmetry p-values."""
    summary = pd.read_csv(REPORT_DIR / "asymmetric_regression_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, y_col, ttl in [
        (axes[0], "cn_ret_cc", "Close-to-close: |β−| − |β+| (higher = stronger downside transmission)"),
        (axes[1], "cn_ret_oc", "Open-to-close: |β−| − |β+|"),
    ]:
        d = summary[summary["y_col"] == y_col].copy()
        d["diff"] = d["beta_neg"].abs() - d["beta_pos"].abs()
        pivot = d.pivot(index="us_code", columns="cn_code", values="diff")
        pivot = pivot.reindex(index=list(US_ASSETS), columns=list(CN_INDEX_ASSETS))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.2, vmax=0.2, aspect="auto")
        ax.set_xticks(range(len(CN_INDEX_ASSETS))); ax.set_xticklabels([CN_PRETTY[c] for c in CN_INDEX_ASSETS])
        ax.set_yticks(range(len(US_ASSETS)));     ax.set_yticklabels([US_PRETTY[c] for c in US_ASSETS])
        ax.set_title(ttl, fontsize=10)
        # Annotate cells with diff value and significance stars
        p_pivot = d.pivot(index="us_code", columns="cn_code", values="asymmetry_p").reindex(
            index=list(US_ASSETS), columns=list(CN_INDEX_ASSETS))
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                p = p_pivot.values[i, j]
                stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
                ax.text(j, i, f"{v:+.3f}\n{stars}", ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 0.12 else "black")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.tight_layout()
    out = REPORT_DIR / "asymmetry_heatmap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def conditional_prob_chart():
    """Bar chart comparing P(CN down|US down) vs P(CN up|US up) for cc returns."""
    buckets_csv = REPORT_DIR / "magnitude_bucket_stats.csv"
    bk = pd.read_csv(buckets_csv)
    bk = bk[bk["y_col"] == "cn_ret_cc"].copy()

    # For each pair, look at extreme buckets
    fig, ax = plt.subplots(figsize=(12, 5))
    pairs = []
    p_down = []
    p_up = []
    for us in US_ASSETS:
        for cn in CN_INDEX_ASSETS:
            sub_down = bk[(bk.us_code==us) & (bk.cn_code==cn) & (bk.bucket=="us_le_-2pct")]
            sub_up   = bk[(bk.us_code==us) & (bk.cn_code==cn) & (bk.bucket=="us_ge_2pct")]
            if len(sub_down) and len(sub_up):
                pairs.append(f"{us}\n{CN_PRETTY[cn]}")
                p_down.append(sub_down["cn_pct_down"].values[0])
                p_up.append(1 - sub_up["cn_pct_down"].values[0])

    x = np.arange(len(pairs))
    w = 0.4
    ax.bar(x - w/2, p_down, w, label="P(CN down | US ≤ -2%)", color="#d62728", alpha=0.85)
    ax.bar(x + w/2, p_up,   w, label="P(CN up   | US ≥ +2%)", color="#2ca02c", alpha=0.85)
    ax.axhline(0.5, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(pairs, fontsize=8)
    ax.set_ylabel("Conditional probability")
    ax.set_ylim(0.3, 0.8)
    ax.legend()
    ax.set_title("Tail-conditional probabilities: A-share close-to-close direction given US prior-day move")
    fig.tight_layout()
    out = REPORT_DIR / "conditional_prob_tails.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("Generating scatter grid (close-to-close)...")
    print("  ->", scatter_grid("cn_ret_cc", "scatter_grid_cc.png"))
    print("Generating scatter grid (open-to-close)...")
    print("  ->", scatter_grid("cn_ret_oc", "scatter_grid_oc.png"))
    print("Generating bucket bars (close-to-close)...")
    print("  ->", bucket_bars("cn_ret_cc", "buckets_cc.png"))
    print("Generating bucket bars (open-to-close)...")
    print("  ->", bucket_bars("cn_ret_oc", "buckets_oc.png"))
    print("Generating asymmetry heatmap...")
    print("  ->", asymmetry_heatmap())
    print("Generating conditional prob chart...")
    print("  ->", conditional_prob_chart())
