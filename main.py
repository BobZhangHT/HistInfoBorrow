"""
main.py — Monte Carlo Simulation Runner (with subgroup tracking)
=================================================================
Records per-patient X1 subgroup for allocation ratio analysis,
plus borrowing diagnostics (mean W, R_n, D_PDC) showing how each
method responds to historical estimator precision τ²_H.

Usage:
  python main.py                              # demo (10 reps)
  python main.py --mode full --jobs 4         # 3×2 factorial (500 reps)
  python main.py --mode precision --jobs 4    # σ_H gradient sweep
"""
import argparse, os, time
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import config

# Prefer C-backed methods (~5–8× faster). Fall back to Python if DLL absent.
_BACKEND = os.environ.get("RADISH_BACKEND", "auto").lower()
if _BACKEND in ("c", "auto"):
    try:
        from methods_c import RADISH, CAHB, KBCD, _as_2d
        _BACKEND = "C"
    except ImportError as _e:
        if _BACKEND == "c":
            raise
        from methods import RADISH, CAHB, KBCD, _as_2d
        _BACKEND = "Python (C lib not built; run `python build_c.py`)"
else:
    from methods import RADISH, CAHB, KBCD, _as_2d
    _BACKEND = "Python"
print(f"[main] backend = {_BACKEND}")

# ── Data-generating mechanism ────────────────────────────────────
def gen_cov(n, rng):
    X = np.zeros((n, 2)); X[:,0] = rng.binomial(1,.5,n); X[:,1] = rng.standard_normal(n)
    return X

def mu0c(X):
    X = np.atleast_2d(X); b = config.BETA
    return b[0] + b[1]*X[:,0] + b[2]*X[:,1] + b[3]*X[:,0]*X[:,1]

def gen_hist(sc_key, rng, scenarios=None):
    scenarios = scenarios or config.SCENARIOS
    sc = scenarios[sc_key]; X = gen_cov(config.N_HISTORICAL, rng)
    return X, mu0c(X) + sc["b_theta"] + rng.standard_normal(len(X))*sc["sigma_h"]

def gen_outcome(X, Z, delta, rng):
    X = np.atleast_2d(X); Z = np.asarray(Z, float)
    sig = np.where(Z == 0, config.SIGMA_0, config.SIGMA_1)
    return mu0c(X) + delta*Z + rng.standard_normal(len(Z))*sig

# ── Single trial ─────────────────────────────────────────────────
def run_trial(sc_key, eff_key, mcls, rep, rng, scenarios=None):
    scenarios = scenarios or config.SCENARIOS
    td = config.EFFECT_SIZES[eff_key]
    Xh, Yh = gen_hist(sc_key, rng, scenarios)
    m = mcls({"X_h": Xh, "Y_h": Yh}, config.COVARIATE_PARAMS, config.PRIORS)
    Xs = gen_cov(config.N_CURRENT, rng)
    Xo, Yo, Zo, Pi = [], [], [], []
    for i in range(config.N_CURRENT):
        xn = Xs[i]
        pt = 0.5 if i < config.INTERIM_START else \
             m.get_allocation_prob(np.array(Xo), np.array(Yo),
                                   np.array(Zo), xn).pi_treatment
        z = rng.binomial(1, pt)
        y = gen_outcome(xn.reshape(1,-1), np.array([z]), td, rng)[0]
        Xo.append(xn); Yo.append(y); Zo.append(z); Pi.append(pt)
    Xa, Ya, Za = np.array(Xo), np.array(Yo), np.array(Zo)
    est, cl, ch, pp = m.estimate_treatment_effect(Xa, Ya, Za)
    rej = int(np.isfinite(cl) and np.isfinite(ch) and (cl > 0 or ch < 0))

    # ── Borrowing diagnostics ──
    try:
        diag = m.compute_diagnostics(Xa, Ya, Za)
    except Exception:
        diag = {"mean_W": np.nan, "mean_Rn": np.nan,
                "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}

    # ── Subgroup allocation ratios (X1=0 vs X1=1) ──
    x1 = Xa[:, 0]
    mask0 = x1 == 0; mask1 = x1 == 1
    alloc_x1_0 = float(np.mean(Za[mask0])) if mask0.sum() > 0 else np.nan
    alloc_x1_1 = float(np.mean(Za[mask1])) if mask1.sum() > 0 else np.nan

    return dict(
        scenario=sc_key, effect_type=eff_key, method=m.name, rep=rep,
        true_delta=td, estimated_delta=est, ci_low=cl, ci_high=ch,
        rejected=rej, estimation_bias=est-td,
        allocation_ratio=float(np.mean(Za)),
        alloc_x1_0=alloc_x1_0,
        alloc_x1_1=alloc_x1_1,
        n_x1_0=int(mask0.sum()),
        n_x1_1=int(mask1.sum()),
        mean_W=diag["mean_W"],
        mean_Rn=diag["mean_Rn"],
        mean_Dpdc=diag["mean_Dpdc"],
        mean_tau2_H=diag["mean_tau2_H"],
        # Scenario factors (for precision-grid plots)
        b_theta=float(scenarios[sc_key]["b_theta"]),
        sigma_h=float(scenarios[sc_key]["sigma_h"]),
    )

def _task(sc, ef, mc, r, seed, scenarios):
    try: return run_trial(sc, ef, mc, r, np.random.default_rng(seed), scenarios)
    except Exception as e:
        return dict(error=str(e), scenario=sc, effect_type=ef,
                    method=mc.__name__, rep=r)

# ── Parallel runner ──────────────────────────────────────────────
def run_sim(mode, seed=2026, n_jobs=4):
    nreps = config.get_mode_replications(mode)
    scenarios = config.get_mode_scenarios(mode)
    tasks, idx = [], 0
    for sc in scenarios:
        for ef in config.EFFECT_SIZES:
            for mc in [RADISH, CAHB, KBCD]:
                for r in range(nreps):
                    tasks.append((sc, ef, mc, r, seed+idx, scenarios)); idx += 1
    print(f"[{mode}] {len(tasks)} tasks  (reps={nreps}, jobs={n_jobs}, "
          f"scenarios={len(scenarios)})")
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_task)(*t) for t in tasks)
    good = [r for r in res if "error" not in r]
    bad  = [r for r in res if "error" in r]
    if bad: print(f"  {len(bad)} errors; first: {bad[0]}")
    return pd.DataFrame(good)

# ── Metrics ──────────────────────────────────────────────────────
def metrics(df, scenarios=None):
    df = df.copy(); td = df["true_delta"]
    df["coverage"] = ((df.ci_low <= td) & (td <= df.ci_high)).astype(int)
    df["width"] = df.ci_high - df.ci_low
    df["sq_err"] = (df.estimated_delta - td)**2
    sc_order = list(scenarios.keys()) if scenarios is not None else config.SCENARIO_ORDER
    rows = []
    for sc in sc_order:
        for ef in config.EFFECT_ORDER:
            for mt in config.METHOD_ORDER:
                s = df[(df.scenario==sc)&(df.effect_type==ef)&(df.method==mt)]
                if s.empty: continue
                rr = s.rejected.mean()
                rows.append(dict(Scenario=sc, Effect=ef, Method=mt, N=len(s),
                    Bias=round(s.estimation_bias.mean(),4),
                    RMSE=round(np.sqrt(s.sq_err.mean()),4),
                    Rejection=round(rr,4), Coverage=round(s.coverage.mean(),4),
                    Width=round(s.width.mean(),4),
                    Alloc_Overall=round(s.allocation_ratio.mean(),4),
                    Alloc_X1_0=round(s.alloc_x1_0.mean(),4),
                    Alloc_X1_1=round(s.alloc_x1_1.mean(),4),
                    Mean_W=round(s.mean_W.mean(skipna=True),4) if "mean_W" in s else "",
                    Mean_Rn=round(s.mean_Rn.mean(skipna=True),4) if "mean_Rn" in s else "",
                    Mean_Dpdc=round(s.mean_Dpdc.mean(skipna=True),4) if "mean_Dpdc" in s else "",
                    Type_I_Error=round(rr,4) if ef=="Null" else "",
                    Power=round(rr,4) if ef=="Power" else ""))
    return pd.DataFrame(rows)

# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo","full","precision"], default="demo")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    args = ap.parse_args()
    t0 = time.time()
    df = run_sim(args.mode, args.seed, args.jobs)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min  ({len(df)} results)")
    out = config.get_mode_output_dir(args.mode); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"raw_results.csv", index=False)
    scenarios = config.get_mode_scenarios(args.mode)
    tbl = metrics(df, scenarios); tbl.to_csv(out/"metrics.csv", index=False)
    print("\n" + tbl.to_string(index=False))
    # Generate plots
    try:
        import analysis
        analysis.run_analysis(df, out, mode=args.mode)
        print(f"\nPlots saved to {out}/plots/")
    except Exception as e:
        print(f"Plot generation skipped: {e}")
