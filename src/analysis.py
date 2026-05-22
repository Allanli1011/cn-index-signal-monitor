"""Hypothesis testing: asymmetric correlation US -> A-share.

For each (US index, CN index) pair we estimate:

    r_CN = alpha + beta_plus * max(r_US, 0) + beta_minus * min(r_US, 0) + eps

with HAC (Newey-West) standard errors. The hypothesis "US drops transmit more
strongly than US rises" implies |beta_minus| > |beta_plus|, tested via a
Wald restriction beta_minus = beta_plus.

We also compute conditional probabilities and expected returns stratified by
US-move direction and magnitude.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import US_ASSETS, CN_INDEX_ASSETS, CN_FUTURES, REPORT_DIR
from src.alignment import align

IM_LISTING_DATE = "2022-07-22"  # CSI 1000 futures (IM) listed

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def asymmetric_regression(df: pd.DataFrame, y_col: str, x_col: str = "us_ret_cc",
                          hac_lags: int = 5) -> dict:
    """Fit r_y = a + b_pos * x_pos + b_neg * x_neg + e with Newey-West errors."""
    d = df[[x_col, y_col]].dropna()
    if len(d) < 60:
        return {"n": len(d), "error": "insufficient data"}
    x = d[x_col].values
    y = d[y_col].values
    x_pos = np.where(x > 0, x, 0.0)
    x_neg = np.where(x < 0, x, 0.0)
    X = sm.add_constant(np.column_stack([x_pos, x_neg]))
    model = sm.OLS(y, X)
    res = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    # Wald test for symmetry: b_pos = b_neg
    R = np.array([[0, 1, -1]])  # contrast: b_pos - b_neg
    wald = res.wald_test(R, scalar=True)

    return {
        "n": int(len(d)),
        "alpha": float(res.params[0]),
        "beta_pos": float(res.params[1]),
        "beta_neg": float(res.params[2]),
        "se_pos": float(res.bse[1]),
        "se_neg": float(res.bse[2]),
        "t_pos": float(res.tvalues[1]),
        "t_neg": float(res.tvalues[2]),
        "p_pos": float(res.pvalues[1]),
        "p_neg": float(res.pvalues[2]),
        "asymmetry_F": float(wald.statistic),
        "asymmetry_p": float(wald.pvalue),
        "r2": float(res.rsquared),
    }


def conditional_stats(df: pd.DataFrame, y_col: str, x_col: str = "us_ret_cc") -> dict:
    """Conditional probabilities and means by US direction & magnitude bucket."""
    d = df[[x_col, y_col]].dropna()
    n = len(d)
    if n == 0:
        return {"n": 0}

    us = d[x_col].values
    cn = d[y_col].values

    # Sign-conditional
    us_down = us < 0
    us_up = us > 0
    cn_down = cn < 0
    cn_up = cn > 0

    p_cn_down_given_us_down = (us_down & cn_down).sum() / max(us_down.sum(), 1)
    p_cn_up_given_us_up     = (us_up & cn_up).sum()     / max(us_up.sum(), 1)
    p_cn_down_given_us_up   = (us_up & cn_down).sum()   / max(us_up.sum(), 1)
    p_cn_up_given_us_down   = (us_down & cn_up).sum()   / max(us_down.sum(), 1)

    e_cn_given_us_down = float(cn[us_down].mean()) if us_down.any() else np.nan
    e_cn_given_us_up   = float(cn[us_up].mean())   if us_up.any()   else np.nan

    # Magnitude buckets — focus on tails
    out_mag = {}
    for label, mask in [
        ("us_le_-2pct", us <= -0.02),
        ("us_in_-2_-1pct", (us > -0.02) & (us <= -0.01)),
        ("us_in_-1_0pct",  (us > -0.01) & (us <  0.00)),
        ("us_in_0_1pct",   (us >= 0.00) & (us <  0.01)),
        ("us_in_1_2pct",   (us >= 0.01) & (us <  0.02)),
        ("us_ge_2pct",     us >= 0.02),
    ]:
        if mask.any():
            out_mag[label] = {
                "count": int(mask.sum()),
                "cn_mean": float(cn[mask].mean()),
                "cn_median": float(np.median(cn[mask])),
                "cn_pct_down": float((cn[mask] < 0).mean()),
            }
        else:
            out_mag[label] = {"count": 0}

    return {
        "n": n,
        "p_cn_down_given_us_down": float(p_cn_down_given_us_down),
        "p_cn_up_given_us_up": float(p_cn_up_given_us_up),
        "p_cn_down_given_us_up": float(p_cn_down_given_us_up),
        "p_cn_up_given_us_down": float(p_cn_up_given_us_down),
        "e_cn_given_us_down": e_cn_given_us_down,
        "e_cn_given_us_up": e_cn_given_us_up,
        "asymmetry_prob": float(p_cn_down_given_us_down - p_cn_up_given_us_up),
        "asymmetry_mean": float((-e_cn_given_us_down) - e_cn_given_us_up),
        "by_bucket": out_mag,
    }


def run_pair(us_code: str, cn_code: str, cn_kind: str = "cn_index",
             y_col: str = "cn_ret_cc", sample_start: str | None = None) -> dict:
    """Combine regression + conditional stats for one pair, optionally restricting sample."""
    df = align(us_code, cn_code, cn_kind)
    if sample_start:
        df = df[df["cn_date"] >= pd.Timestamp(sample_start)].reset_index(drop=True)
    reg = asymmetric_regression(df, y_col=y_col)
    cond = conditional_stats(df, y_col=y_col)
    return {
        "us_code": us_code,
        "cn_code": cn_code,
        "cn_kind": cn_kind,
        "y_col": y_col,
        "sample_start": sample_start or "data_start",
        **reg,
        **{f"cond_{k}": v for k, v in cond.items() if k != "by_bucket"},
        "by_bucket": cond.get("by_bucket"),
    }


def _run_panel(plan, y_cols, sample_start, suffix):
    """Generic runner: iterate (US, CN, return_type) over a given list of (cn_code, cn_kind)."""
    rows, bucket_rows = [], []
    for us_code in US_ASSETS:
        for cn_code, cn_kind in plan:
            for y_col in y_cols:
                r = run_pair(us_code, cn_code, cn_kind, y_col=y_col, sample_start=sample_start)
                if "error" in r:
                    continue
                rows.append({k: v for k, v in r.items() if k != "by_bucket"})
                if r.get("by_bucket"):
                    for bucket, stats in r["by_bucket"].items():
                        bucket_rows.append({
                            "us_code": us_code, "cn_code": cn_code, "cn_kind": cn_kind,
                            "y_col": y_col, "sample_start": sample_start or "data_start",
                            "bucket": bucket, **stats,
                        })
    summary = pd.DataFrame(rows)
    buckets = pd.DataFrame(bucket_rows)
    summary.to_csv(REPORT_DIR / f"asymmetric_regression{suffix}.csv", index=False)
    buckets.to_csv(REPORT_DIR / f"magnitude_bucket{suffix}.csv", index=False)
    return summary, buckets


def run_all(y_cols=("cn_ret_cc", "cn_ret_oc")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Original behaviour: indices only, full sample. Kept for backward compatibility."""
    plan = [(c, "cn_index") for c in CN_INDEX_ASSETS]
    summary, buckets = _run_panel(plan, y_cols, sample_start=None, suffix="_summary")
    return summary, buckets


def run_extended(y_cols=("cn_ret_cc", "cn_ret_oc")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All 8 CN assets (indices + futures), each on its native data range (futures from 2017+,
    IM from 2022-07-22+). For looking at futures-specific dynamics including IM."""
    plan = (
        [(c, "cn_index") for c in CN_INDEX_ASSETS]
        + [(c, "cn_future") for c in CN_FUTURES]
    )
    return _run_panel(plan, y_cols, sample_start=None, suffix="_extended")


def run_im_era(y_cols=("cn_ret_cc", "cn_ret_oc")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apples-to-apples: all 8 CN assets restricted to IM-era (2022-07-22+)."""
    plan = (
        [(c, "cn_index") for c in CN_INDEX_ASSETS]
        + [(c, "cn_future") for c in CN_FUTURES]
    )
    return _run_panel(plan, y_cols, sample_start=IM_LISTING_DATE, suffix="_im_era")


def print_summary(summary: pd.DataFrame, title: str = ""):
    """Print a human-friendly view of the headline results."""
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    for y_col in summary["y_col"].unique():
        print(f"\n{'='*110}")
        print(f"  {title}  Y = {y_col}  ({'close-to-close' if y_col=='cn_ret_cc' else 'open-to-close'})")
        print(f"{'='*110}")
        d = summary[summary["y_col"] == y_col].copy()
        d["|b-|>|b+|"] = (d["beta_neg"].abs() > d["beta_pos"].abs()).map({True: "Y", False: " "})
        d["asym_sig"]  = d["asymmetry_p"].apply(lambda p: "***" if p<0.01 else ("**" if p<0.05 else ("*" if p<0.10 else "")))
        cols = ["us_code", "cn_kind", "cn_code", "n", "beta_pos", "beta_neg",
                "p_pos", "p_neg", "asymmetry_F", "asymmetry_p", "asym_sig",
                "|b-|>|b+|", "r2"]
        cols = [c for c in cols if c in d.columns]
        print(d[cols].to_string(index=False))


if __name__ == "__main__":
    print("\n### 1) Original: indices only, full sample ###")
    s_orig, _ = run_all()
    print_summary(s_orig, title="[indices, full sample]")

    print("\n\n### 2) Extended: indices + futures (each on native range) ###")
    s_ext, _ = run_extended()
    print_summary(s_ext, title="[indices+futures, native ranges]")

    print("\n\n### 3) IM-era apples-to-apples: all assets restricted to 2022-07-22+ ###")
    s_im, _ = run_im_era()
    print_summary(s_im, title="[2022-07-22+, all assets]")

    print(f"\n\nSaved CSVs:")
    for tag in ["_summary", "_extended", "_im_era"]:
        print(f"  - asymmetric_regression{tag}.csv")
        print(f"  - magnitude_bucket{tag}.csv")
