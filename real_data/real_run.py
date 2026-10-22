"""Run the HORIZON-FIT semi-synthetic RADISH experiment.

For each conflict level xi, historical-size level eta, method, hypothesis,
and replication, the script generates a historical control sample and a
sequential current trial. The historical shift is mean-zero under the target
distribution but differs between the X1 strata for every xi > 0; eta maps to
n_H in [30, 350]. HORIZON-FIT-calibrated covariate
distributions, subgroup response surfaces, and sigma_C are used.
The moderate alternative delta=0.14 avoids saturated power at N=200.  KBCD,
CAHB, RADISH, and the conflict levels share random-number streams within each
(eta, effect, replication) block.

Output: real_data/real_run_results.csv
"""
from __future__ import annotations
import hashlib, json, os, sys, time, pickle

# Avoid nested BLAS threads inside the process-level simulation workers.
for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from real_data.real_scenarios import (
    XI_GRID, ETA_GRID, LOCAL_PATH_HIGH, global_local_regimes, n_H_of_eta,
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
MANIFEST_JSON = ROOT / "real_data" / "real_run_manifest.json"
GLOBAL_LOCAL_CSV = ROOT / "real_data" / "real_global_local_results.csv"
GLOBAL_LOCAL_MANIFEST = ROOT / "real_data" / "real_global_local_manifest.json"

# The outcome scale uses the HORIZON residual variance. The controlled DGP
# combines the HORIZON response surface with FIT-calibrated historical
# covariate distributions and subgroup proportions. Historical conflict has
# target-distribution average zero and differs by X1 stratum. The compact grid
# spans compatibility through severe local conflict; it is
# not intended to duplicate the multiplicative perturbation in Jin et al. (2023).
SIGMA_C_DGP  = float(np.sqrt(0.11))   # ≈ 0.33, from HORIZON residuals
DELTA_DGP    = 0.140                  # moderate alternative; avoids saturated power

# CAHB-specific settings from Jin et al. (2023, Section 4). RADISH and
# KBCD ignore these entries, so the proposed method is not tuned to the
# evaluation grid.
REAL_PRIORS = dict(config.PRIORS)
REAL_PRIORS.update(cahb_gamma=float(np.sqrt(3.0)), cahb_lambda=300.0)


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


def _gen_historical(params, eta, xi, rng, regime=None):
    """Generate n_H historical controls on the additive conflict path.

    The conditional mean uses the HORIZON response surface plus an X1-specific
    shift. The two X1 regions move in opposite directions while their
    target-population average remains zero. Thus xi=0 is exactly compatible,
    every positive xi is covariate dependent, and a marginal mean comparison
    alone cannot detect the constructed conflict. Historical precision varies
    only through n_H(eta), with sigma_H fixed at sigma_C.
    """
    n_H = n_H_of_eta(eta)
    Xh, sg_h = _gen_X_from_subgroup(rng, n_H, params["kde_hist"],
                                    params["sg_dist_hist"], params["moments"])

    sigma_C = SIGMA_C_DGP
    sigma_h = sigma_C

    # Transparent location-conflict path on the HORIZON outcome scale.
    # xi=0 is exactly compatible; the seven-point log-spaced grid ends at xi=1, where
    # both X1 regions have severe local conflict.
    a0 = params["beta_curr"]["b0"]
    a1 = params["beta_curr"]["b_menyrs"]
    a2 = params["beta_curr"]["b_Y0"]
    p_fall = float(params["sg_dist_curr"][1] + params["sg_dist_curr"][3])
    if regime is None:
        centered_x1 = (Xh[:, 0] - p_fall) / np.sqrt(p_fall * (1.0 - p_fall))
        ell_high = np.sqrt((1.0 - p_fall) / p_fall)
        shift_sd = LOCAL_PATH_HIGH * float(xi) * ell_high * centered_x1
    else:
        g_sd = float(regime["g_sd"])
        ell_sd = float(regime["ell_sd"])
        centered_x1 = (Xh[:, 0] - p_fall) / np.sqrt(p_fall * (1.0 - p_fall))
        shift_sd = g_sd + ell_sd * centered_x1
    shift = sigma_C * shift_sd
    mu = (a0[sg_h]
          + a1[sg_h] * Xh[:, 2]
          + a2[sg_h] * Xh[:, 3]
          + shift)
    Yh = mu + rng.standard_normal(n_H) * sigma_h
    # n_H denotes available historical controls, not a randomized cohort.
    return Xh, Yh


def sigma_h_of_eta(eta: float) -> float:
    """Backwards-compatible helper; eta changes n_H, not historical noise."""
    del eta
    return SIGMA_C_DGP


def _run_trial(rng, params, xi, eta, method_cls, delta_true, regime=None):
    Xh_ctrl, Yh_ctrl = _gen_historical(params, eta, xi, rng, regime=regime)
    if Xh_ctrl.shape[0] < 4:
        return None
    # RADISH uses the common defaults. The CAHB comparator uses the
    # real-data-study settings in REAL_PRIORS (gamma=sqrt(3), lambda base
    # coefficient 300, yielding the implemented radius 300*log(N)), declared
    # once above and reported by the generated design table.
    cov_params = {"p": 4, "disc_idx": [0, 1], "cont_idx": [2, 3]}
    method = method_cls({"X_h": Xh_ctrl[:, :4], "Y_h": Yh_ctrl},
                        cov_params, REAL_PRIORS)
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
    x1_0 = Xa[:, 0] == 0
    x1_1 = Xa[:, 0] == 1
    return dict(
        est=est, lo=lo, hi=hi, delta_true=delta_true,
        sq_err=(est - delta_true)**2,
        rejected=int(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
        alloc_overall=float(Za.mean()),
        alloc_sg1=alloc_per_sg[0], alloc_sg2=alloc_per_sg[1],
        alloc_sg3=alloc_per_sg[2], alloc_sg4=alloc_per_sg[3],
        alloc_x1_0=float(Za[x1_0].mean()) if x1_0.any() else np.nan,
        alloc_x1_1=float(Za[x1_1].mean()) if x1_1.any() else np.nan,
        mean_W=diag["mean_W"], mean_Rn=diag["mean_Rn"],
        mean_W_x1_0=diag.get("mean_W_x1_0", np.nan),
        mean_W_x1_1=diag.get("mean_W_x1_1", np.nan),
        mean_Rn_x1_0=diag.get("mean_Rn_x1_0", np.nan),
        mean_Rn_x1_1=diag.get("mean_Rn_x1_1", np.nan),
        n_H=int(Xh_ctrl.shape[0]),
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


def _task_global_local(regime, method_cls, effect, rep, seed):
    rng = np.random.default_rng(seed)
    params = _load_params()
    delta_true = DELTA_DGP if effect == "Power" else 0.0
    eta = max(ETA_GRID)
    try:
        d = _run_trial(
            rng, params, xi=None, eta=eta, method_cls=method_cls,
            delta_true=delta_true, regime=regime,
        )
        if d is None:
            return dict(regime=regime["regime"], method=method_cls.__name__,
                        effect=effect, rep=rep, error="too few historical")
        p1 = float(params["sg_dist_curr"][1] + params["sg_dist_curr"][3])
        scale = np.sqrt(p1 * (1.0 - p1))
        b0_sd = regime["g_sd"] - regime["ell_sd"] * p1 / scale
        b1_sd = regime["g_sd"] + regime["ell_sd"] * (1.0 - p1) / scale
        d.update(
            regime=regime["regime"], regime_label=regime["label"],
            g_sd=float(regime["g_sd"]), ell_sd=float(regime["ell_sd"]),
            b0_sd=float(b0_sd), b1_sd=float(b1_sd), eta=eta,
            method=method_cls.__name__, effect=effect, rep=rep,
        )
        return d
    except Exception as e:
        return dict(regime=regime["regime"], method=method_cls.__name__,
                    effect=effect, rep=rep, error=str(e))


def main(n_reps=None, n_jobs=None, out_csv=None):
    n_reps = N_REPS if n_reps is None else int(n_reps)
    n_jobs = N_JOBS if n_jobs is None else int(n_jobs)
    t0 = time.time()
    tasks = []
    # Common random numbers pair methods and conflict levels within each
    # (eta, effect, replication) block, isolating the xi sensitivity path.
    for i_xi, xi in enumerate(XI_GRID):
        for i_eta, eta in enumerate(ETA_GRID):
            for i_eff, effect in enumerate(("Null", "Power")):
                for r in range(n_reps):
                    seed = SEED + r + n_reps * (i_eff + 2 * i_eta)
                    for cls in (KBCD, CAHB, RADISH):
                        tasks.append((xi, eta, cls, effect, r, seed))
    print(f"[real_run] {len(tasks)} tasks  "
          f"(|ξ|={len(XI_GRID)} × |η|={len(ETA_GRID)} × 3 methods × 2 effects × {n_reps} reps)")
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_task)(*t) for t in tasks)
    good = [r for r in res if "error" not in r]
    bad  = [r for r in res if "error" in r]
    if bad:
        print(f"  {len(bad)} errors; first: {bad[0]}")
    df = pd.DataFrame(good)
    out_csv = OUT_CSV if out_csv is None else Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[real_run] saved -> {out_csv}  ({len(df)} rows)  "
          f"in {(time.time()-t0)/60:.1f} min")


def main_global_local(n_reps=None, n_jobs=None, out_csv=None):
    """Run the compact mean-shift by local-heterogeneity diagnostic."""
    n_reps = N_REPS if n_reps is None else int(n_reps)
    n_jobs = N_JOBS if n_jobs is None else int(n_jobs)
    params = _load_params()
    p1 = float(params["sg_dist_curr"][1] + params["sg_dist_curr"][3])
    regimes = global_local_regimes(p1)
    if out_csv is None:
        out_csv = (
            GLOBAL_LOCAL_CSV if n_reps == N_REPS else
            ROOT / "real_data" / f"real_global_local_pilot_{n_reps}.csv"
        )
    out_csv = Path(out_csv)
    t0 = time.time()
    tasks = []
    for regime in regimes:
        for i_eff, effect in enumerate(("Null", "Power")):
            for rep in range(n_reps):
                # Pair all methods and regimes within each effect/replication.
                seed = SEED + rep + n_reps * i_eff
                for cls in (KBCD, CAHB, RADISH):
                    tasks.append((regime, cls, effect, rep, seed))
    print(
        f"[real_run:global-local] {len(tasks)} tasks "
        f"(4 regimes x 3 methods x 2 effects x {n_reps} reps)"
    )
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_task_global_local)(*task) for task in tasks
    )
    good = [row for row in res if "error" not in row]
    bad = [row for row in res if "error" in row]
    if bad:
        print(f"  {len(bad)} errors; first: {bad[0]}")
    df = pd.DataFrame(good)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(
        f"[real_run:global-local] saved -> {out_csv} ({len(df)} rows) "
        f"in {(time.time()-t0)/60:.1f} min"
    )
    return out_csv


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_results(path=OUT_CSV):
    """Validate the calibrated-study grid and write its SHA-256 manifest."""
    path = Path(path)
    df = pd.read_csv(path)
    methods, effects = ("KBCD", "CAHB", "RADISH"), ("Null", "Power")
    expected_rows = (
        len(XI_GRID) * len(ETA_GRID) * len(methods) * len(effects) * N_REPS
    )
    assert len(df) == expected_rows, (len(df), expected_rows)
    assert set(df["method"]) == set(methods)
    assert set(df["effect"]) == set(effects)
    assert np.allclose(sorted(df["xi"].unique()), XI_GRID)
    assert np.allclose(sorted(df["eta"].unique()), ETA_GRID)
    keys = ["xi", "eta", "method", "effect", "rep"]
    assert not df.duplicated(keys).any(), "duplicate result identifiers"
    counts = df.groupby(keys[:-1], observed=True).size()
    assert counts.eq(N_REPS).all()
    assert df["rep"].min() == 0 and df["rep"].max() == N_REPS - 1
    for eta in ETA_GRID:
        observed = df.loc[np.isclose(df["eta"], eta), "n_H"].unique()
        assert len(observed) == 1 and int(observed[0]) == n_H_of_eta(eta)
    null_delta = df.loc[df["effect"] == "Null", "delta_true"].unique()
    power_delta = df.loc[df["effect"] == "Power", "delta_true"].unique()
    assert len(null_delta) == 1 and np.isclose(null_delta[0], 0.0)
    assert len(power_delta) == 1 and np.isclose(power_delta[0], DELTA_DGP)
    required = [
        "est", "lo", "hi", "sq_err", "alloc_overall", "mean_W", "mean_Rn",
        "mean_W_x1_0", "mean_W_x1_1", "mean_Rn_x1_0", "mean_Rn_x1_1",
    ]
    assert np.isfinite(df[required].to_numpy(float)).all()
    report = {
        "result_file": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "rows": int(len(df)),
        "replications_per_cell": int(N_REPS),
        "xi_grid": [float(x) for x in XI_GRID],
        "eta_grid": [float(x) for x in ETA_GRID],
        "historical_sizes": [int(n_H_of_eta(e)) for e in ETA_GRID],
        "seed": int(SEED),
        "local_path_high": float(LOCAL_PATH_HIGH),
        "shift_path": "(b0,b1)/sigma_C=L*xi*(-1,(1-p1)/p1)",
        "methods": list(methods),
        "effects": list(effects),
        "delta_null": 0.0,
        "delta_power": float(DELTA_DGP),
        "validation": "passed",
    }
    MANIFEST_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate_global_local_results(path=GLOBAL_LOCAL_CSV):
    """Validate the formal four-regime diagnostic and write its manifest."""
    path = Path(path)
    df = pd.read_csv(path)
    params = _load_params()
    p1 = float(params["sg_dist_curr"][1] + params["sg_dist_curr"][3])
    regimes = global_local_regimes(p1)
    regime_ids = [item["regime"] for item in regimes]
    methods, effects = ("KBCD", "CAHB", "RADISH"), ("Null", "Power")
    expected_rows = len(regime_ids) * len(methods) * len(effects) * N_REPS
    assert len(df) == expected_rows, (len(df), expected_rows)
    assert set(df["regime"]) == set(regime_ids)
    assert set(df["method"]) == set(methods)
    assert set(df["effect"]) == set(effects)
    keys = ["regime", "method", "effect", "rep"]
    assert not df.duplicated(keys).any(), "duplicate result identifiers"
    assert df.groupby(keys[:-1], observed=True).size().eq(N_REPS).all()
    required = [
        "est", "lo", "hi", "sq_err", "alloc_overall", "alloc_x1_0",
        "alloc_x1_1", "mean_W",
        "mean_W_x1_0", "mean_W_x1_1", "g_sd", "ell_sd", "b0_sd", "b1_sd",
    ]
    assert np.isfinite(df[required].to_numpy(float)).all()
    report = {
        "result_file": str(path.relative_to(ROOT)),
        "sha256": _sha256(path),
        "rows": int(len(df)),
        "replications_per_cell": int(N_REPS),
        "seed": int(SEED),
        "p_x1_1_current": p1,
        "regimes": regimes,
        "methods": list(methods),
        "effects": list(effects),
        "validation": "passed",
    }
    GLOBAL_LOCAL_MANIFEST.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=None,
                    help=f"replications per cell (default {N_REPS}); "
                         f"use a small value for a smoke test")
    ap.add_argument("--jobs", type=int, default=None,
                    help=f"parallel workers (default {N_JOBS})")
    ap.add_argument(
        "--validate-only", action="store_true",
        help="validate the existing result file and write its manifest",
    )
    ap.add_argument(
        "--design", choices=("xi", "global-local"), default="xi",
        help="experiment to run or validate (default: xi)",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="optional output CSV; useful for a named pilot run",
    )
    args = ap.parse_args()
    if args.validate_only:
        report = (
            validate_results() if args.design == "xi" else
            validate_global_local_results()
        )
        print(json.dumps(report, indent=2))
    elif args.design == "global-local":
        main_global_local(args.reps, args.jobs, args.output)
    else:
        main(args.reps, args.jobs, args.output)
