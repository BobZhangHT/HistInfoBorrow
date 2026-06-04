"""
real_run.py — Sweep (ξ × η × method × effect × rep) cells.

Configured to match CAHB §5.2.1 (Fig 2) paper text:
  • Current trial:  Y = δZ + μ_0(X) + ε, μ_0(X) = β_0(sg) + β_1(sg)·X3
                    + β_2(sg)·X4, with σ²_C and δ real-fitted from
                    HORIZON; sg = (X1, X2) the binary fall/frx subgroup.
  • Historical:     α_k(sg) = β_k(sg)·d, d ~ N(1+ξ, 0.3·I(ξ≠0))
                    (multiplicative discrepancy, paper §5.2.1).
  • Covariates:     2 binary (X1, X2) + 2 continuous KDE-sampled
                    (X3 standardised menyrs, X4 standardised Y0).
  • CAHB algorithm hyperparams (paper §4 defaults):
                    γ = √3, λ_2 = 300·log(N), λ_1 from Algorithm 2,
                    single Gaussian kernel with Scott ROT
                    h_j = n^{-1/(p+4)}·σ_j across all 4 covariates
                    (no separate fitting/allocation kernel).
  • θ_0:            CAHB §5.2.1 idealisation — passed as the noiseless
                    historical mean function evaluated at the sampled
                    covariates (no ε_h additive noise).

For every (ξ, η) cell we sample a historical control arm of size
n_H(η), run a sequential adaptive trial of N_CURRENT current-trial
patients, fit the chosen Stage-III estimator and record (est, lo, hi,
sq_err, rejected, alloc, mean_W).

Outputs: real_data/real_run_results.csv
"""
from __future__ import annotations
import os, sys, time, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from real_data.real_scenarios import (
    XI_GRID, ETA_GRID, n_H_of_eta,
    N_CURRENT, INTERIM_START, N_REPS, SEED, N_JOBS,
)

_BACKEND = os.environ.get("RADISH_BACKEND", "auto").lower()
if _BACKEND in ("c", "auto"):
    try:
        from methods_c import RADISH, CAHB, KBCD
        _BACKEND = "C"
    except ImportError:
        from methods import RADISH, CAHB, KBCD
        _BACKEND = "Python"
else:
    from methods import RADISH, CAHB, KBCD
print(f"[real_run] backend = {_BACKEND}")

PARAMS_PKL = ROOT / "real_data" / "real_params.pkl"
OUT_CSV    = ROOT / "real_data" / "real_run_results.csv"

# DGP scale: σ_C = 1.0 matches B-sim noise regime so that the strict
# B-sim hyperparam γ = 0.25 (config.PRIORS) produces the dimensionless
# borrowing strength γ²·σ_C² = 0.0625 it has in B1-B6.  HORIZON's
# absolute σ_C ≈ 0.33 would shrink this to 0.007 (9× weaker), capping
# CAHB W at ~0.08 and erasing the canonical alloc/RMSE/power trends.
# Realism is preserved through HORIZON-fitted β-shape (subgroup
# heterogeneity), per-sg KDE on (menyrs, Y0), and HORIZON empirical
# subgroup proportions — only the noise scale is harmonised with
# B-sim.  δ_DGP = 0.5 is the B-sim baseline alternative effect that
# yields nominal power ≈ 0.8 with N_C = 200.
SIGMA_C_DGP  = 1.0
DELTA_DGP    = 0.5


def _load_params():
    with open(PARAMS_PKL, "rb") as f:
        return pickle.load(f)


def _gen_X_from_subgroup(rng, n, kdes, sg_dist, moments):
    sg = rng.choice(4, size=n, p=sg_dist)
    out = np.zeros((n, 4))
    for k in range(4):
        idx = np.where(sg == k)[0]
        if len(idx) == 0:
            continue
        x1, x2 = k & 1, (k >> 1) & 1
        out[idx, 0] = x1
        out[idx, 1] = x2
        if kdes[k] is None:
            samp = rng.standard_normal((2, len(idx)))
        else:
            samp = kdes[k].resample(len(idx), seed=rng.integers(2**31))
        out[idx, 2] = (samp[0] - moments["mn_menyrs"]) / moments["sd_menyrs"]
        out[idx, 3] = (samp[1] - moments["mn_Y0"])     / moments["sd_Y0"]
    return out, sg


def _gen_historical(params, eta, xi, rng):
    """Real-data B-scenario-style DGP — additive bias + tunable σ_h
    precision dial, mirroring config.SCENARIOS B1-B6 form but anchored
    to HORIZON-fitted parameters and FIT-KDE covariate distributions.

        Y_h = μ_0^current(X) + b_θ + ε_h,
              μ_0^current(X) = β_0(sg) + β_1(sg)·X3_std + β_2(sg)·X4_std
              b_θ            = ξ · σ_C    (additive bias in σ_C units)
              ε_h ~ N(0, σ_h²),   σ_h    = sigma_h_of_eta(η) · σ_C

    Bias-axis ξ ∈ {0, 0.5, 1, 2}·σ_C aligns with B-scenario b_θ levels
    {0, 0.5, 2} when re-scaled to the HORIZON noise regime σ_C = 0.33.

    Precision-axis η ∈ [0, 1] maps σ_h/σ_C linearly from 3 (low prec, =
    B2/B4/B6 σ_h ratio) at η=0 to 0.5 (high prec, = B1/B3/B5) at η=1.

    Historical sample size N_H = 200 (matching B-scenarios).
    """
    # η now controls historical sample size n_H directly (literal
    # interpretation of "precision of θ̂_0" — matches CAHB §5.2.3 design).
    # σ_h is held fixed at σ_C, so the only precision lever is n_H.
    # Var(θ̂_0) ∝ σ_h²/n_H, so n_H ∈ [30, 350] gives ~12× precision range.
    n_H = n_H_of_eta(eta)
    Xh, sg_h = _gen_X_from_subgroup(rng, n_H, params["kde_hist"],
                                    params["sg_dist_hist"], params["moments"])
    Zh = rng.binomial(1, 0.5, size=n_H).astype(float)
    sigma_C = SIGMA_C_DGP
    b_theta = xi * sigma_C
    sigma_h = sigma_C        # fixed σ_h = σ_C; precision varies via n_H
    a0 = params["beta_curr"]["b0"]
    a1 = params["beta_curr"]["b_menyrs"]
    a2 = params["beta_curr"]["b_Y0"]
    mu = (a0[sg_h]
          + a1[sg_h] * Xh[:, 2]
          + a2[sg_h] * Xh[:, 3]
          + b_theta)
    Yh = mu + rng.standard_normal(n_H) * sigma_h
    mask = (Zh == 0)
    return Xh[mask], Yh[mask]


def sigma_h_of_eta(eta: float) -> float:
    """Kept for backwards compatibility with figure code; always
    returns 1.0 since σ_h is now fixed at σ_C and η controls n_H."""
    return 1.0


def _run_trial(rng, params, xi, eta, method_cls, delta_true):
    Xh_ctrl, Yh_ctrl = _gen_historical(params, eta, xi, rng)
    if Xh_ctrl.shape[0] < 4:
        return None
    # Hyperparameters: identical to the RADISH simulation B1-B6 study
    # (config.PRIORS = {"cahb_gamma": 0.25, "cahb_lambda": 300}),
    # so the only thing that distinguishes the real-data run from the
    # B1-B6 run is the data-generating process (DGP), not the algorithm
    # tuning.  Per user constraint, we do NOT touch CAHB / RADISH
    # hyperparameters here.
    cov_params = {"p": 4, "disc_idx": [0, 1], "cont_idx": [2, 3]}
    method = method_cls({"X_h": Xh_ctrl[:, :4], "Y_h": Yh_ctrl},
                        cov_params, config.PRIORS)
    Xc, sg_c = _gen_X_from_subgroup(rng, N_CURRENT, params["kde_curr"],
                                    params["sg_dist_curr"], params["moments"])
    Xo, Yo, Zo, Pi = [], [], [], []
    for i in range(N_CURRENT):
        xn = Xc[i]
        if i < INTERIM_START:
            pt = 0.5
        else:
            pt = method.get_allocation_prob(np.array(Xo), np.array(Yo),
                                            np.array(Zo), xn).pi_treatment
        z = rng.binomial(1, pt)
        mu_i = (params["beta_curr"]["b0"][sg_c[i]]
                + params["beta_curr"]["b_menyrs"][sg_c[i]] * xn[2]
                + params["beta_curr"]["b_Y0"]    [sg_c[i]] * xn[3]
                + delta_true * z)
        y = mu_i + rng.standard_normal() * SIGMA_C_DGP
        Xo.append(xn); Yo.append(y); Zo.append(int(z)); Pi.append(pt)
    Xa = np.asarray(Xo); Ya = np.asarray(Yo); Za = np.asarray(Zo, int)
    est, lo, hi, _ = method.estimate_treatment_effect(Xa, Ya, Za)
    diag = method.compute_diagnostics(Xa, Ya, Za)
    alloc_per_sg = np.zeros(4)
    for k in range(4):
        m = (sg_c == k)
        alloc_per_sg[k] = float(Za[m].mean()) if m.sum() > 0 else np.nan
    return dict(
        est=est, lo=lo, hi=hi, delta_true=delta_true,
        sq_err=(est - delta_true)**2,
        rejected=int(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        alloc_overall=float(Za.mean()),
        alloc_sg1=alloc_per_sg[0], alloc_sg2=alloc_per_sg[1],
        alloc_sg3=alloc_per_sg[2], alloc_sg4=alloc_per_sg[3],
        mean_W=diag["mean_W"], mean_Rn=diag["mean_Rn"],
        n_H=200,
    )


def _task(xi, eta, method_cls, effect, rep, seed):
    rng = np.random.default_rng(seed)
    params = _load_params()
    delta_true = DELTA_DGP if effect == "Power" else 0.0
    try:
        d = _run_trial(rng, params, xi, eta, method_cls, delta_true)
        if d is None:
            return dict(xi=xi, eta=eta, method=method_cls.__name__,
                        effect=effect, rep=rep, error="too few historical")
        d.update(xi=xi, eta=eta, method=method_cls.__name__,
                 effect=effect, rep=rep)
        return d
    except Exception as e:
        return dict(xi=xi, eta=eta, method=method_cls.__name__,
                    effect=effect, rep=rep, error=str(e))


def main():
    t0 = time.time()
    tasks, idx = [], 0
    for xi in XI_GRID:
        for eta in ETA_GRID:
            for cls in (KBCD, CAHB, RADISH):
                for effect in ("Null", "Power"):
                    for r in range(N_REPS):
                        tasks.append((xi, eta, cls, effect, r, SEED + idx))
                        idx += 1
    print(f"[real_run] {len(tasks)} tasks  "
          f"(|ξ|={len(XI_GRID)} × |η|={len(ETA_GRID)} × 3 methods × 2 effects × {N_REPS} reps)")
    res = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_task)(*t) for t in tasks)
    good = [r for r in res if "error" not in r]
    bad  = [r for r in res if "error" in r]
    if bad:
        print(f"  {len(bad)} errors; first: {bad[0]}")
    df = pd.DataFrame(good)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"[real_run] saved -> {OUT_CSV}  ({len(df)} rows)  "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
