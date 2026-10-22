"""Monte Carlo runner for the RADISH simulation studies.

The runner records overall and X1-stratum-specific allocation and borrowing
diagnostics. Paper modes are ``six``, ``local_path``, and ``global_local``;
``full``, ``precision``, ``bias``, and ``hist_size`` provide supplementary
sensitivity analyses. Use ``--validate-only`` to verify saved results without
rerunning a simulation.
"""
import argparse, hashlib, json, os, time

# Prevent nested BLAS threading inside joblib workers. This must be set before
# importing NumPy so parallel simulations remain deterministic and efficient.
for _thread_var in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_var] = "1"

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
def gen_cov(n, rng, p_x1=0.5):
    X = np.zeros((n, 2)); X[:,0] = rng.binomial(1, p_x1, n); X[:,1] = rng.standard_normal(n)
    return X

def mu0c(X):
    X = np.atleast_2d(X); b = config.BETA
    return b[0] + b[1]*X[:,0] + b[2]*X[:,1] + b[3]*X[:,0]*X[:,1]

def b_hist(X, sc):
    """Historical location shift at each covariate point.

    Scenarios carrying a b_local pair shift the historical outcome separately
    inside the two levels of the binary covariate X1. The shifts may differ in
    both strata; b_theta remains the study-level marginal summary.
    """
    if sc.get("b_local") is not None:
        b0, b1 = sc["b_local"]
        x1 = np.atleast_2d(X)[:, 0]
        return np.where(x1 == 1, float(b1), float(b0))
    return float(sc["b_theta"])

def _b_pair(sc):
    """Stratum-wise historical shift (b(x1=0), b(x1=1)) of one scenario."""
    pair = sc.get("b_local")
    return pair if pair is not None else (sc["b_theta"], sc["b_theta"])

def gen_hist(sc_key, rng, scenarios=None, n_historical=None,
             max_n_historical=None):
    scenarios = scenarios or config.SCENARIOS
    sc = scenarios[sc_key]
    n_historical = config.N_HISTORICAL if n_historical is None else int(n_historical)
    draw_size = n_historical if max_n_historical is None else int(max_n_historical)
    if n_historical <= 0 or draw_size < n_historical:
        raise ValueError("historical sample sizes must satisfy 0 < n_historical <= max_n_historical")
    X_full = gen_cov(draw_size, rng, sc.get("p_x1", 0.5))
    Y_full = (mu0c(X_full) + b_hist(X_full, sc)
              + rng.standard_normal(draw_size) * sc["sigma_h"])
    # Drawing the largest requested cohort and taking prefixes makes the
    # historical-size sensitivity comparison nested within each replication.
    return X_full[:n_historical], Y_full[:n_historical]

def gen_outcome(X, Z, delta, rng):
    X = np.atleast_2d(X); Z = np.asarray(Z, float)
    sig = np.where(Z == 0, config.SIGMA_0, config.SIGMA_1)
    return mu0c(X) + delta*Z + rng.standard_normal(len(Z))*sig

# ── Single trial ─────────────────────────────────────────────────
def run_trial(sc_key, eff_key, mcls, rep, rng=None, scenarios=None,
              n_historical=None, rng_hist=None, max_n_historical=None,
              hist_data=None, rng_cov=None, rng_alloc=None,
              rng_outcome=None):
    scenarios = scenarios or config.SCENARIOS
    td = config.EFFECT_SIZES[eff_key]
    n_historical = config.N_HISTORICAL if n_historical is None else int(n_historical)
    if rng is None and any(x is None for x in (rng_cov, rng_alloc, rng_outcome)):
        raise ValueError("supply either rng or all concurrent RNG streams")
    hist_rng = rng if rng_hist is None else rng_hist
    cov_rng = rng if rng_cov is None else rng_cov
    alloc_rng = rng if rng_alloc is None else rng_alloc
    outcome_rng = rng if rng_outcome is None else rng_outcome
    if hist_data is None:
        Xh, Yh = gen_hist(sc_key, hist_rng, scenarios, n_historical,
                          max_n_historical)
    else:
        Xh, Yh = hist_data
        Xh = np.asarray(Xh, dtype=float)
        Yh = np.asarray(Yh, dtype=float)
        if len(Xh) != n_historical or len(Yh) != n_historical:
            raise ValueError("precomputed historical data has the wrong size")
    m = mcls({"X_h": Xh, "Y_h": Yh}, config.COVARIATE_PARAMS, config.PRIORS)
    Xs = gen_cov(config.N_CURRENT, cov_rng, scenarios[sc_key].get("p_x1", 0.5))
    alloc_uniforms = alloc_rng.random(config.N_CURRENT)
    outcome_errors = outcome_rng.standard_normal(config.N_CURRENT)
    Xo, Yo, Zo, Pi = [], [], [], []
    for i in range(config.N_CURRENT):
        xn = Xs[i]
        pt = 0.5 if i < config.INTERIM_START else \
             m.get_allocation_prob(np.array(Xo), np.array(Yo),
                                   np.array(Zo), xn).pi_treatment
        z = int(alloc_uniforms[i] < pt)
        sigma = config.SIGMA_1 if z == 1 else config.SIGMA_0
        y = float(mu0c(xn.reshape(1, -1))[0] + td * z
                  + outcome_errors[i] * sigma)
        Xo.append(xn); Yo.append(y); Zo.append(z); Pi.append(pt)
    Xa, Ya, Za = np.array(Xo), np.array(Yo), np.array(Zo)
    est, cl, ch, pp = m.estimate_treatment_effect(Xa, Ya, Za)
    rej = int(np.isfinite(cl) and np.isfinite(ch) and (cl > 0 or ch < 0))
    # Wald standard error recovered from the reported interval; analysis.py
    # uses the same z quantile for the calibrated-power calculation.
    se_est = ((ch - cl) / (2.0 * 1.959964)
              if np.isfinite(cl) and np.isfinite(ch) else np.nan)
    z_abs = (abs(est) / se_est
             if np.isfinite(se_est) and se_est > 0 else np.nan)

    # ── Borrowing diagnostics ──
    try:
        diag = m.compute_diagnostics(Xa, Ya, Za)
    except Exception:
        diag = {"mean_W": np.nan, "mean_Rn": np.nan,
                "mean_W_x1_0": np.nan, "mean_W_x1_1": np.nan,
                "mean_Rn_x1_0": np.nan, "mean_Rn_x1_1": np.nan,
                "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}

    # ── Subgroup allocation ratios (X1=0 vs X1=1) ──
    x1 = Xa[:, 0]
    mask0 = x1 == 0; mask1 = x1 == 1
    alloc_x1_0 = float(np.mean(Za[mask0])) if mask0.sum() > 0 else np.nan
    alloc_x1_1 = float(np.mean(Za[mask1])) if mask1.sum() > 0 else np.nan

    return dict(
        scenario=sc_key, effect_type=eff_key, method=m.name, rep=rep,
        n_historical=n_historical,
        true_delta=td, estimated_delta=est, ci_low=cl, ci_high=ch,
        rejected=rej, estimation_bias=est-td, se_est=se_est, z_abs=z_abs,
        allocation_ratio=float(np.mean(Za)),
        alloc_x1_0=alloc_x1_0,
        alloc_x1_1=alloc_x1_1,
        n_x1_0=int(mask0.sum()),
        n_x1_1=int(mask1.sum()),
        mean_W=diag.get("mean_W", np.nan),
        mean_Rn=diag.get("mean_Rn", np.nan),
        mean_W_x1_0=diag.get("mean_W_x1_0", np.nan),
        mean_W_x1_1=diag.get("mean_W_x1_1", np.nan),
        mean_Rn_x1_0=diag.get("mean_Rn_x1_0", np.nan),
        mean_Rn_x1_1=diag.get("mean_Rn_x1_1", np.nan),
        mean_Dpdc=diag.get("mean_Dpdc", np.nan),
        mean_tau2_H=diag.get("mean_tau2_H", np.nan),
        # Scenario factors (for precision-grid plots)
        b_theta=float(scenarios[sc_key]["b_theta"]),
        sigma_h=float(scenarios[sc_key]["sigma_h"]),
        b_local_x1_0=float(_b_pair(scenarios[sc_key])[0]),
        b_local_x1_1=float(_b_pair(scenarios[sc_key])[1]),
    )

def _task(sc, ef, mc, r, seed, scenarios):
    try:
        hist_rng, cov_rng, alloc_rng, outcome_rng = _paired_rngs(seed)
        return run_trial(
            sc, ef, mc, r, None, scenarios,
            rng_hist=hist_rng, rng_cov=cov_rng, rng_alloc=alloc_rng,
            rng_outcome=outcome_rng,
        )
    except Exception as e:
        return dict(error=str(e), scenario=sc, effect_type=ef,
                    method=mc.__name__, rep=r)


def _paired_rngs(seed):
    """Independent latent streams shared by paired methods and scenarios."""
    children = np.random.SeedSequence(int(seed)).spawn(4)
    return tuple(np.random.default_rng(child) for child in children)


def _hist_size_rngs(seed):
    """Alias documenting the same four-stream scheme for size comparisons."""
    return _paired_rngs(seed)


def _hist_size_task(sc, ef, mc, r, seed, scenarios, n_historical,
                    max_n_historical):
    """Run one paired historical-size task with independent RNG streams."""
    try:
        hist_rng, cov_rng, alloc_rng, outcome_rng = _hist_size_rngs(seed)
        return run_trial(
            sc, ef, mc, r, None, scenarios,
            n_historical=n_historical,
            rng_hist=hist_rng,
            max_n_historical=max_n_historical,
            rng_cov=cov_rng,
            rng_alloc=alloc_rng,
            rng_outcome=outcome_rng,
        )
    except Exception as e:
        return dict(error=str(e), scenario=sc, effect_type=ef,
                    method=mc.__name__, rep=r,
                    n_historical=n_historical)


# ── Parallel runner ──────────────────────────────────────────────
def run_sim(mode, seed=2026, n_jobs=4, nreps=None):
    nreps = config.get_mode_replications(mode) if nreps is None else int(nreps)
    scenarios = config.get_mode_scenarios(mode)
    # Common random numbers: the task seed depends only on the effect and
    # the replication, never on the scenario or the method. Covariates,
    # innovations, and the concurrent trial stream are therefore identical
    # across scenarios and methods, while the historical location shift
    # changes deterministically with the scenario. A method that ignores
    # historical outcomes is then exactly invariant across scenarios, which
    # is what the KBCD column of the operating-characteristics table must
    # show, and all methods remain paired within a replication.
    effect_order = list(config.EFFECT_SIZES)
    tasks = []
    for sc in scenarios:
        for i_eff, ef in enumerate(effect_order):
            for mc in [RADISH, CAHB, KBCD]:
                for r in range(nreps):
                    paired_seed = int(seed) + r + nreps * i_eff
                    tasks.append((sc, ef, mc, r, paired_seed, scenarios))
    print(f"[{mode}] {len(tasks)} tasks  (reps={nreps}, jobs={n_jobs}, "
          f"scenarios={len(scenarios)})")
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_task)(*t) for t in tasks)
    good = [r for r in res if "error" not in r]
    bad  = [r for r in res if "error" in r]
    if bad: print(f"  {len(bad)} errors; first: {bad[0]}")
    return pd.DataFrame(good)


def run_hist_size_sim(seed=2026, n_jobs=4, nreps=None):
    """Paired sensitivity study over config.HISTORICAL_SIZE_GRID.

    Within each scenario/effect/replication, every method and historical
    sample size receives the same historical and concurrent random-number
    streams. Historical samples are nested prefixes of the largest cohort.
    """
    nreps = config.N_REPS_HIST_SIZE if nreps is None else int(nreps)
    scenarios = config.get_mode_scenarios("hist_size")
    hist_sizes = [int(n) for n in config.HISTORICAL_SIZE_GRID]
    if not hist_sizes or min(hist_sizes) <= 0:
        raise ValueError("HISTORICAL_SIZE_GRID must contain positive sizes")
    max_n_historical = max(hist_sizes)
    tasks = []
    for sc in scenarios:
        for i_eff, ef in enumerate(config.EFFECT_SIZES):
            for mc in [RADISH, CAHB, KBCD]:
                for r in range(nreps):
                    paired_seed = int(seed) + r + nreps * i_eff
                    for n_historical in hist_sizes:
                        tasks.append((sc, ef, mc, r, paired_seed, scenarios,
                                      n_historical, max_n_historical))
    print(f"[hist_size] {len(tasks)} tasks (reps={nreps}, jobs={n_jobs}, "
          f"scenarios={len(scenarios)}, N_H={hist_sizes})")
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_hist_size_task)(*t) for t in tasks)
    good = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    if bad:
        print(f"  {len(bad)} errors; first: {bad[0]}")
    out = pd.DataFrame(good)
    if len(out) != len(tasks):
        raise RuntimeError(
            f"historical-size study returned {len(out)} successful rows; "
            f"expected {len(tasks)}"
        )
    keys = ["scenario", "effect_type", "method", "rep", "n_historical"]
    if out.duplicated(keys).any():
        raise RuntimeError("hist_size output contains duplicate task keys")
    expected_rows = (len(scenarios) * len(config.EFFECT_ORDER)
                     * len(config.METHOD_ORDER) * nreps * len(hist_sizes))
    if len(out) != expected_rows:
        raise RuntimeError(
            f"hist_size output has {len(out)} rows; expected {expected_rows}"
        )
    out.attrs["hist_size_manifest"] = {
        "seed": int(seed),
        "historical_sizes": hist_sizes,
        "rows_computed": int(len(out)),
        "rows_written": int(len(out)),
        "coupling": (
            "historical sizes use nested prefixes of one independent "
            "maximum-size historical cohort; concurrent covariates, allocation "
            "uniforms, and outcome innovations use three separate child RNG "
            "streams; methods and sample sizes share paired latent draws "
            "within each replication"
        ),
        "rng_scheme_version": 2,
        "child_streams": ["historical", "covariates", "allocation", "outcomes"],
        "max_n_historical": int(max_n_historical),
    }
    return out

# ── Metrics ──────────────────────────────────────────────────────
def _abs_z(part):
    """Equivalence-test statistic |estimate| / SE from the reported interval."""
    half = (part.ci_high.to_numpy(float)
            - part.ci_low.to_numpy(float)) / (2.0 * 1.959964)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs(part.estimated_delta.to_numpy(float)) / half
    z[~np.isfinite(z)] = np.nan
    return z


def _calibrated_power(df, sc_order, alpha=None):
    """Calibrated power per (scenario, method).

    Each cell uses its own finite-sample critical value: the empirical
    (1 - alpha) quantile of |z| under the null effect. Nominal size is met
    before the power of the alternative is read off.
    """
    alpha = config.ALPHA if alpha is None else float(alpha)
    out = {}
    for sc in sc_order:
        for mt in config.METHOD_ORDER:
            cell = df[(df.scenario == sc) & (df.method == mt)]
            zn = _abs_z(cell[cell.effect_type == "Null"])
            zn = zn[np.isfinite(zn)]
            zp = _abs_z(cell[cell.effect_type == "Power"])
            zp = zp[np.isfinite(zp)]
            if zn.size == 0 or zp.size == 0:
                out[(sc, mt)] = np.nan
                continue
            out[(sc, mt)] = float(np.mean(zp > np.quantile(zn, 1 - alpha)))
    return out


def metrics(df, scenarios=None):
    df = df.copy(); td = df["true_delta"]
    df["coverage"] = ((df.ci_low <= td) & (td <= df.ci_high)).astype(int)
    df["width"] = df.ci_high - df.ci_low
    df["sq_err"] = (df.estimated_delta - td)**2
    sc_order = list(scenarios.keys()) if scenarios is not None else config.SCENARIO_ORDER
    cal_power = _calibrated_power(df, sc_order)
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
                    Power_calibrated=(round(cal_power[(sc, mt)], 4)
                                      if ef == "Power"
                                      and np.isfinite(cal_power[(sc, mt)]) else ""),
                    Power=round(rr,4) if ef=="Power" else ""))
    return pd.DataFrame(rows)


def metrics_by_history_size(df, scenarios=None):
    """Summarize operating characteristics separately for each N_H."""
    rows = []
    for n_historical in sorted(df.n_historical.unique()):
        part = metrics(df[df.n_historical == n_historical], scenarios)
        part.insert(0, "N_Historical", int(n_historical))
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_saved_results(mode="full", result_dir=None, reps=None):
    """Validate a saved synthetic experiment against the code contract."""
    out = Path(result_dir) if result_dir else config.get_mode_output_dir(mode)
    raw_path, metrics_path = out / "raw_results.csv", out / "metrics.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw results: {raw_path}")
    df = pd.read_csv(raw_path)
    scenarios = config.get_mode_scenarios(mode)
    reps = config.get_mode_replications(mode) if reps is None else int(reps)
    if reps < 1:
        raise ValueError("replications must be positive")
    methods, effects = tuple(config.METHOD_ORDER), tuple(config.EFFECT_ORDER)
    required = {
        "scenario", "effect_type", "method", "rep", "true_delta",
        "estimated_delta", "ci_low", "ci_high", "rejected",
        "estimation_bias", "allocation_ratio", "b_theta", "sigma_h",
        "se_est", "z_abs", "b_local_x1_0", "b_local_x1_1",
    }
    if mode == "hist_size":
        required.add("n_historical")
    missing = required.difference(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    multipliers = len(scenarios) * len(methods) * len(effects) * reps
    if mode == "hist_size":
        multipliers *= len(config.HISTORICAL_SIZE_GRID)
    assert len(df) == multipliers, (len(df), multipliers)
    assert set(df["scenario"]) == set(scenarios)
    assert set(df["method"]) == set(methods)
    assert set(df["effect_type"]) == set(effects)
    keys = ["scenario", "effect_type", "method", "rep"]
    if mode == "hist_size":
        keys.insert(0, "n_historical")
        assert tuple(sorted(df["n_historical"].unique())) == tuple(
            sorted(config.HISTORICAL_SIZE_GRID)
        )
    assert not df.duplicated(keys).any(), "duplicate simulation identifiers"
    essential = [
        "true_delta", "estimated_delta", "ci_low", "ci_high",
        "rejected", "allocation_ratio", "b_theta", "sigma_h",
    ]
    assert np.isfinite(df[essential].to_numpy(float)).all()
    assert df["allocation_ratio"].between(0, 1).all()
    assert df["rejected"].isin([0, 1]).all()
    assert (df["ci_low"] <= df["ci_high"]).all()
    assert np.isfinite(df["se_est"].to_numpy(float)).all()
    assert (df["se_est"] >= 0).all()
    for name, spec in scenarios.items():
        part = df[df["scenario"] == name]
        assert np.allclose(part["b_theta"], spec["b_theta"])
        assert np.allclose(part["sigma_h"], spec["sigma_h"])
        b0, b1 = _b_pair(spec)
        assert np.allclose(part["b_local_x1_0"], b0)
        assert np.allclose(part["b_local_x1_1"], b1)
    expected = (
        metrics_by_history_size(df, scenarios)
        if mode == "hist_size" else metrics(df, scenarios)
    ).reset_index(drop=True)
    summary_matches = metrics_path.exists()
    if summary_matches:
        saved = pd.read_csv(metrics_path).reset_index(drop=True)
        for column in expected.select_dtypes(include="object"):
            expected[column] = expected[column].mask(expected[column].eq(""))
        pd.testing.assert_frame_equal(
            saved, expected, check_dtype=False, check_exact=False,
            atol=5e-5, rtol=0,
        )
    report = {
        "mode": mode,
        "raw_results": str(raw_path),
        "sha256": _sha256(raw_path),
        "rows": int(len(df)),
        "cells": int(df.groupby(keys[:-1], observed=True).ngroups),
        "replications_per_cell": int(reps),
        "scenarios": list(scenarios),
        "methods": list(methods),
        "effects": list(effects),
        "summary_matches_metrics_csv": summary_matches,
        "validation": "passed",
    }
    if mode == "hist_size":
        report["historical_sizes"] = [
            int(x) for x in sorted(config.HISTORICAL_SIZE_GRID)
        ]
    return report

# ── CLI ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "full", "precision", "bias",
                                       "hist_size", "global_local", "six",
                                       "local_path"], default="demo")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=config.N_JOBS)
    ap.add_argument("--reps", type=int, default=None,
                    help="override replications for smoke tests or sensitivity runs")
    ap.add_argument(
        "--validate-only", action="store_true",
        help="validate existing outputs for --mode without rerunning simulation",
    )
    args = ap.parse_args()
    if args.validate_only:
        print(json.dumps(validate_saved_results(args.mode, reps=args.reps), indent=2))
        raise SystemExit(0)
    t0 = time.time()
    if args.mode == "hist_size":
        df = run_hist_size_sim(args.seed, args.jobs, args.reps)
    else:
        df = run_sim(args.mode, args.seed, args.jobs, args.reps)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min  ({len(df)} results)")
    out = config.get_mode_output_dir(args.mode); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out/"raw_results.csv", index=False)
    if args.mode == "hist_size" and df.attrs.get("hist_size_manifest"):
        (out / "hist_size_pairing_manifest.json").write_text(
            json.dumps(df.attrs["hist_size_manifest"], indent=2),
            encoding="utf-8",
        )
        print(f"[hist_size] pairing manifest written to "
              f"{out / 'hist_size_pairing_manifest.json'}")
    scenarios = config.get_mode_scenarios(args.mode)
    if args.mode == "hist_size":
        tbl = metrics_by_history_size(df, scenarios)
    else:
        tbl = metrics(df, scenarios)
    tbl.to_csv(out/"metrics.csv", index=False)
    print("\n" + tbl.to_string(index=False))
    # Generate plots
    try:
        if args.mode == "hist_size":
            import historical_size_analysis
            historical_size_analysis.run_analysis(df, out)
        else:
            import analysis
            analysis.run_analysis(df, out, mode=args.mode)
        print(f"\nPlots saved to {out}/plots/")
    except Exception as e:
        print(f"Plot generation skipped: {e}")
