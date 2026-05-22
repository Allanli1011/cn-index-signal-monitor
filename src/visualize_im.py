"""Visualizations focused on IM and the post-2022-07-22 sub-sample.

Generates:
  - Heatmap of asymmetry strength on futures panel (3 US x 4 futures) including IM
  - Heatmap of asymmetry strength on IM-era sub-sample for both indices and futures
  - Side-by-side comparison: full-sample vs IM-era beta_neg (close-to-close)
  - Bucket bars for futures (including IM)
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import US_ASSETS, CN_INDEX_ASSETS, CN_FUTURES, REPORT_DIR

mpl.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

US_PRETTY = {"SPX": "S&P 500", "NDX": "Nasdaq", "RUT": "Russell 2000"}
CN_IDX_PRETTY = {"HS300": "CSI 300", "ZZ500": "CSI 500", "ZZ1000": "CSI 1000", "SSE50": "SSE 50"}
CN_FUT_PRETTY = {"IF": "IF (CSI300)", "IC": "IC (CSI500)", "IM": "IM (CSI1000)", "IH": "IH (SSE50)"}


def _heatmap(ax, pivot_diff, pivot_p, title, vmin=-0.25, vmax=0.25,
             row_labels=None, col_labels=None):
    im = ax.imshow(pivot_diff.values, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot_diff.shape[1]))
    ax.set_xticklabels(col_labels or list(pivot_diff.columns), fontsize=9)
    ax.set_yticks(range(pivot_diff.shape[0]))
    ax.set_yticklabels(row_labels or list(pivot_diff.index), fontsize=9)
    ax.set_title(title, fontsize=10)
    for i in range(pivot_diff.shape[0]):
        for j in range(pivot_diff.shape[1]):
            v = pivot_diff.values[i, j]
            p = pivot_p.values[i, j]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            color = "white" if abs(v) > 0.15 else "black"
            ax.text(j, i, f"{v:+.3f}\n{stars}", ha="center", va="center",
                    fontsize=8, color=color)
    return im


def heatmap_futures_full_vs_im_era():
    """Two-panel heatmap: full-sample futures asymmetry vs IM-era asymmetry (both futures)."""
    ext = pd.read_csv(REPORT_DIR / "asymmetric_regression_extended.csv")
    im_era = pd.read_csv(REPORT_DIR / "asymmetric_regression_im_era.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, df, ttl in [
        (axes[0], ext, "Full-sample futures (native ranges)\n|β−| − |β+|, close-to-close"),
        (axes[1], im_era[im_era["cn_kind"] == "cn_future"],
                  "IM-era only (2022-07-22+) — futures\n|β−| − |β+|, close-to-close"),
    ]:
        d = df[(df["y_col"] == "cn_ret_cc") & (df["cn_kind"] == "cn_future")].copy()
        d["diff"] = d["beta_neg"].abs() - d["beta_pos"].abs()
        piv = d.pivot(index="us_code", columns="cn_code", values="diff").reindex(
            index=list(US_ASSETS), columns=list(CN_FUTURES))
        pp = d.pivot(index="us_code", columns="cn_code", values="asymmetry_p").reindex(
            index=list(US_ASSETS), columns=list(CN_FUTURES))
        im = _heatmap(ax, piv, pp, ttl,
                       row_labels=[US_PRETTY[u] for u in US_ASSETS],
                       col_labels=[CN_FUT_PRETTY[c] for c in CN_FUTURES])
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out = REPORT_DIR / "heatmap_futures_full_vs_im_era.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def beta_neg_comparison_bar():
    """Bar chart comparing beta_neg full sample vs IM-era for every (US, CN) pair on futures."""
    ext = pd.read_csv(REPORT_DIR / "asymmetric_regression_extended.csv")
    im_era = pd.read_csv(REPORT_DIR / "asymmetric_regression_im_era.csv")

    ext_f = ext[(ext["y_col"] == "cn_ret_cc") & (ext["cn_kind"] == "cn_future")].copy()
    im_f  = im_era[(im_era["y_col"] == "cn_ret_cc") & (im_era["cn_kind"] == "cn_future")].copy()

    pairs = [(u, c) for u in US_ASSETS for c in CN_FUTURES]
    labels = [f"{u}\n{CN_FUT_PRETTY[c]}" for u, c in pairs]
    full_vals = []
    im_vals = []
    for u, c in pairs:
        row_full = ext_f[(ext_f["us_code"] == u) & (ext_f["cn_code"] == c)]
        row_im   = im_f[(im_f["us_code"] == u) & (im_f["cn_code"] == c)]
        full_vals.append(row_full["beta_neg"].values[0] if len(row_full) else np.nan)
        im_vals.append(row_im["beta_neg"].values[0] if len(row_im) else np.nan)

    x = np.arange(len(pairs))
    w = 0.4
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x - w/2, full_vals, w, label="Full sample (futures native range)", color="#4c72b0")
    ax.bar(x + w/2, im_vals,   w, label="IM-era (2022-07-22+)",                color="#dd8452")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("β⁻  (sensitivity to US negative returns)")
    ax.set_title("Downside transmission β⁻ — full-sample vs IM-era, close-to-close, on futures")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = REPORT_DIR / "beta_neg_full_vs_im_era.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def bucket_bars_futures(suffix: str = "_extended"):
    """Bucket bars (close-to-close) for futures including IM, using extended bucket file."""
    bk = pd.read_csv(REPORT_DIR / f"magnitude_bucket{suffix}.csv")
    bk = bk[(bk["y_col"] == "cn_ret_cc") & (bk["cn_kind"] == "cn_future")].copy()
    order = ["us_le_-2pct", "us_in_-2_-1pct", "us_in_-1_0pct",
             "us_in_0_1pct", "us_in_1_2pct", "us_ge_2pct"]
    pretty_lbl = ["US≤-2%", "-2~-1%", "-1~0%", "0~1%", "1~2%", "US≥2%"]
    bk["bucket_idx"] = bk["bucket"].map({b: i for i, b in enumerate(order)})

    fig, axes = plt.subplots(len(US_ASSETS), len(CN_FUTURES),
                              figsize=(16, 9), sharex=True, sharey=True)
    for i, us in enumerate(US_ASSETS):
        for j, cn in enumerate(CN_FUTURES):
            ax = axes[i, j]
            sub = bk[(bk["us_code"] == us) & (bk["cn_code"] == cn)].sort_values("bucket_idx")
            if len(sub) == 0:
                ax.set_title(f"{CN_FUT_PRETTY[cn]} (no data)", fontsize=9)
                continue
            vals = sub["cn_mean"].values * 100
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
                ax.set_title(CN_FUT_PRETTY[cn], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{US_PRETTY[us]}\n(CN_fut mean, %)", fontsize=9)

    label = "full sample, futures native ranges" if suffix == "_extended" else "IM-era (2022-07-22+)"
    fig.suptitle(f"Mean A-share futures close-to-close return by US prior-day bucket — {label}",
                 fontsize=12, y=0.995)
    fig.tight_layout()
    out = REPORT_DIR / f"buckets_futures{suffix}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def im_focused_summary():
    """Print a focused IM-only summary across all three samples for quick reading."""
    ext = pd.read_csv(REPORT_DIR / "asymmetric_regression_extended.csv")
    im_era = pd.read_csv(REPORT_DIR / "asymmetric_regression_im_era.csv")

    print("\n========== IM (CSI 1000 futures) — full data range vs IM-era restriction ==========")
    print("(IM data starts 2022-07-22, so these are identical for IM. Indices ZZ1000 included for compare.)")
    rows = []
    for src_name, src in [("full_sample", ext), ("im_era_constrained", im_era)]:
        for us in US_ASSETS:
            for code, kind in [("ZZ1000", "cn_index"), ("IM", "cn_future")]:
                m = src[(src["us_code"] == us) & (src["cn_code"] == code) &
                        (src["cn_kind"] == kind) & (src["y_col"] == "cn_ret_cc")]
                if len(m):
                    r = m.iloc[0]
                    rows.append({
                        "sample": src_name, "us": us, "cn": f"{kind}:{code}",
                        "n": int(r["n"]), "beta_pos": r["beta_pos"], "beta_neg": r["beta_neg"],
                        "p_pos": r["p_pos"], "p_neg": r["p_neg"],
                        "asym_p": r["asymmetry_p"], "r2": r["r2"],
                    })
    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    print("Heatmap futures full vs IM-era ->",   heatmap_futures_full_vs_im_era())
    print("Beta_neg full vs IM-era bars     ->", beta_neg_comparison_bar())
    print("Bucket bars futures (extended)   ->", bucket_bars_futures("_extended"))
    print("Bucket bars futures (IM-era)     ->", bucket_bars_futures("_im_era"))
    im_focused_summary()
