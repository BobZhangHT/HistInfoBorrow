"""
real_setup.py — Real-data parameter estimation for RADISH application.

Mirrors CAHB §5.2 setup so reviewers can compare directly:
  - Current trial : HORIZON (ZOL), age >= 80, race == 1
  - Historical   : FIT.vert + FIT.clin pooled, age >= 78, race == 1
  - Outcome      : standardized hip BMD at 24 months (dxhhp_24)
  - Subgroups    : 2x2 of (X1=falls, X2=frx)  -> subgroup id = X1 + 2*X2
  - Outcome model:
        Y_std = beta0(X1,X2) + beta1(X1,X2) * menyrs_std
                              + beta2(X1,X2) * Y0_std
                              + delta * Z + eps,    eps ~ N(0, sigma^2)
    estimated separately on HORIZON (-> beta_k, delta) and FIT (-> alpha_k, delta_h).

Outputs (pickled to real_data/real_params.pkl):
    moments     : standardization means/sds (computed on HORIZON)
    beta_curr   : list of 4 length-3 vectors (per-subgroup intercept/menyrs/Y0)
    alpha_hist  : list of 4 length-3 vectors  (FIT-fit equivalent)
    delta_curr  : float (HORIZON treatment effect)
    delta_hist  : float (FIT treatment effect)
    sigma_curr  : float (HORIZON residual SD on standardized scale)
    sigma_hist  : float (FIT residual SD on standardized scale)
    subgrp_dist_curr : (4,) probabilities of (X1,X2) under HORIZON
    subgrp_dist_hist : (4,) probabilities under FIT
    kde_curr / kde_hist : per-subgroup gaussian KDE on (menyrs, Y0)
                          stored as scipy.stats.gaussian_kde objects
    nH_total    : pooled FIT total N (used for default historical sample size)

Key empirical quantities also saved for Figure 1:
    bias_per_subgrp[k]  = empirical mean of (Y_FIT.control - Y_ZOL.control) at subgroup k
    var_h_per_subgrp[k] = empirical Var(Y_FIT.control) at subgroup k (proxy for tau^2_H precision)
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT     = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "real_data" / "dat_merge.csv"
OUT_PKL  = ROOT / "real_data" / "real_params.pkl"

CUR_AGE_MIN  = 80   # ZOL age cutoff (CAHB used 80; corresponds to ~90th pct)
HIST_AGE_MIN = 78   # FIT age cutoff (CAHB used 78)


def _load_filter():
    df = pd.read_csv(DATA_CSV)
    # Filter: SOF dropped, race == 1, age cutoffs by study
    df = df[df["STUDY"] != "SOF"]
    df = df[df["race"] == 1]
    cur_mask  = (df["STUDY"] == "ZOL")  & (df["age"] >= CUR_AGE_MIN)
    hist_mask = (df["STUDY"] != "ZOL") & (df["age"] >= HIST_AGE_MIN)
    df = df[cur_mask | hist_mask].copy()
    # Rename / select analysis columns
    df["falls"] = df["falls"].astype(float)
    df["frx"]   = df["frxvert"].astype(float)
    df["Y0"]    = df["dxhhp_0"].astype(float)
    df["Y"]     = df["dxhhp_24"].astype(float)
    df["Z"]     = df["TRTN"].astype(float)
    keep = ["falls", "frx", "menyrs", "Y0", "Y", "Z", "STUDY"]
    df = df[keep].dropna()
    return df


def _fit_outcome_model(d, mn_menyrs, sd_menyrs, mn_Y0, sd_Y0, mn_Y, sd_Y):
    """OLS: Y_std ~ Z + sg + sg:menyrs_std + sg:Y0_std (per-subgroup intercept,
    pooled treatment effect across subgroups; matches CAHB Section 5.2).
    Returns (intercepts(4,), menyrs_slopes(4,), Y0_slopes(4,), delta, sigma).
    """
    d = d.copy()
    d["menyrs1"] = (d["menyrs"] - mn_menyrs) / sd_menyrs
    d["Y01"]     = (d["Y0"]     - mn_Y0)     / sd_Y0
    d["Y1"]      = (d["Y"]      - mn_Y)      / sd_Y
    d["sg"]      = (d["frx"] * 2 + d["falls"]).astype(int)
    intercepts = np.zeros(4); slope_m = np.zeros(4); slope_y0 = np.zeros(4)
    delta_per_sg = np.zeros(4); n_per_sg = np.zeros(4)
    resid_all = []
    for k in range(4):
        s = d[d["sg"] == k]
        if len(s) < 6:
            continue
        X = np.column_stack([np.ones(len(s)), s["Z"], s["menyrs1"], s["Y01"]])
        y = s["Y1"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercepts[k] = beta[0]; delta_per_sg[k] = beta[1]
        slope_m[k]    = beta[2]; slope_y0[k]     = beta[3]
        n_per_sg[k]   = len(s)
        resid_all.append(y - X @ beta)
    resid_all = np.concatenate(resid_all)
    delta = float((delta_per_sg * n_per_sg).sum() / n_per_sg.sum())
    sigma = float(np.std(resid_all, ddof=4))
    return intercepts, slope_m, slope_y0, delta, sigma


def main():
    d = _load_filter()
    cur  = d[d["STUDY"] == "ZOL"].copy()
    hist = d[d["STUDY"] != "ZOL"].copy()
    print(f"[real_setup] HORIZON (current) n = {len(cur)}, "
          f"FIT (historical) n = {len(hist)}")

    # Standardization moments from HORIZON (current)
    mn_menyrs = float(cur["menyrs"].mean()); sd_menyrs = float(cur["menyrs"].std())
    mn_Y0     = float(cur["Y0"].mean());     sd_Y0     = float(cur["Y0"].std())
    mn_Y      = float(cur["Y"].mean());      sd_Y      = float(cur["Y"].std())
    moments = dict(mn_menyrs=mn_menyrs, sd_menyrs=sd_menyrs,
                   mn_Y0=mn_Y0,         sd_Y0=sd_Y0,
                   mn_Y=mn_Y,           sd_Y=sd_Y)

    # Per-subgroup linear fit: HORIZON
    b0c, b1c, b2c, delta_c, sig_c = _fit_outcome_model(
        cur, mn_menyrs, sd_menyrs, mn_Y0, sd_Y0, mn_Y, sd_Y)
    a0c, a1c, a2c, delta_h, sig_h = _fit_outcome_model(
        hist, mn_menyrs, sd_menyrs, mn_Y0, sd_Y0, mn_Y, sd_Y)

    # Per-subgroup empirical bias and variance on STANDARDIZED control Y
    cur["sg"]  = (cur["frx"] * 2 + cur["falls"]).astype(int)
    hist["sg"] = (hist["frx"] * 2 + hist["falls"]).astype(int)
    cur["Y1"]  = (cur["Y"]  - mn_Y) / sd_Y
    hist["Y1"] = (hist["Y"] - mn_Y) / sd_Y
    bias_per_sg = np.zeros(4)
    var_h_per_sg = np.zeros(4)
    n_h_per_sg   = np.zeros(4, int)
    for k in range(4):
        cy = cur [(cur ["sg"] == k) & (cur ["Z"] == 0)]["Y1"].to_numpy()
        hy = hist[(hist["sg"] == k) & (hist["Z"] == 0)]["Y1"].to_numpy()
        if len(cy) > 0 and len(hy) > 0:
            bias_per_sg[k]  = float(hy.mean() - cy.mean())
            var_h_per_sg[k] = float(hy.var(ddof=1))
            n_h_per_sg[k]   = len(hy)

    # KDE on (menyrs, Y0) per subgroup
    kde_curr  = []
    kde_hist  = []
    for k in range(4):
        s_c = cur [cur ["sg"] == k][["menyrs", "Y0"]].to_numpy().T
        s_h = hist[hist["sg"] == k][["menyrs", "Y0"]].to_numpy().T
        kde_curr.append(gaussian_kde(s_c) if s_c.shape[1] > 4 else None)
        kde_hist.append(gaussian_kde(s_h) if s_h.shape[1] > 4 else None)

    # Subgroup probabilities
    sg_dist_curr = np.array([(cur ["sg"] == k).mean() for k in range(4)])
    sg_dist_hist = np.array([(hist["sg"] == k).mean() for k in range(4)])

    out = dict(
        moments      = moments,
        beta_curr    = dict(b0=b0c, b_menyrs=b1c, b_Y0=b2c),
        alpha_hist   = dict(b0=a0c, b_menyrs=a1c, b_Y0=a2c),
        delta_curr   = delta_c,
        delta_hist   = delta_h,
        sigma_curr   = sig_c,
        sigma_hist   = sig_h,
        sg_dist_curr = sg_dist_curr,
        sg_dist_hist = sg_dist_hist,
        kde_curr     = kde_curr,
        kde_hist     = kde_hist,
        bias_per_sg  = bias_per_sg,
        var_h_per_sg = var_h_per_sg,
        n_h_per_sg   = n_h_per_sg,
        n_curr       = len(cur),
        n_hist       = len(hist),
    )
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)

    # Console summary
    print(f"[real_setup] sigma_curr = {sig_c:.4f}, sigma_hist = {sig_h:.4f}")
    print(f"[real_setup] delta_curr = {delta_c:.4f}, delta_hist = {delta_h:.4f}")
    print(f"[real_setup] subgroup distributions:")
    print(f"             current : {np.round(sg_dist_curr, 3)}")
    print(f"             history : {np.round(sg_dist_hist, 3)}")
    print(f"[real_setup] per-subgroup empirical control-Y bias (hist - curr):")
    for k in range(4):
        x1, x2 = k & 1, (k >> 1) & 1
        print(f"             sg{k+1} (X1={x1}, X2={x2}):  "
              f"bias={bias_per_sg[k]:+.4f}  var_H={var_h_per_sg[k]:.4f}  "
              f"n_H={n_h_per_sg[k]}")
    print(f"[real_setup] saved -> {OUT_PKL}")


if __name__ == "__main__":
    main()
