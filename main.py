"""
main.py — Monte Carlo Simulation Runner (with subgroup tracking)
=================================================================
Records per-patient X1 subgroup for allocation ratio analysis,
plus borrowing diagnostics (mean W, R_n, D_PDC) showing how each
method responds to historical estimator precision τ²_H.

Usage:
  python main.py                              # four-regime demo (10 reps)
  python main.py --mode full --jobs 4         # four-regime 4×2 grid (1000 reps)
  python main.py --mode precision --jobs 4    # sigma_H gradient sweep
  python main.py --mode bias --jobs 4         # focused low-conflict path
  python main.py --mode hist_size --jobs 4    # paired N_H sensitivity
"""
import argparse, copy, hashlib, json, os, time

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
def gen_cov(n, rng):
    X = np.zeros((n, 2)); X[:,0] = rng.binomial(1,.5,n); X[:,1] = rng.standard_normal(n)
    return X

def mu0c(X):
    X = np.atleast_2d(X); b = config.BETA
    return b[0] + b[1]*X[:,0] + b[2]*X[:,1] + b[3]*X[:,0]*X[:,1]

def gen_hist(sc_key, rng, scenarios=None, n_historical=None,
             max_n_historical=None):
    scenarios = scenarios or config.SCENARIOS
    sc = scenarios[sc_key]
    n_historical = config.N_HISTORICAL if n_historical is None else int(n_historical)
    draw_size = n_historical if max_n_historical is None else int(max_n_historical)
    if n_historical <= 0 or draw_size < n_historical:
        raise ValueError("historical sample sizes must satisfy 0 < n_historical <= max_n_historical")
    X_full = gen_cov(draw_size, rng)
    Y_full = (mu0c(X_full) + sc["b_theta"]
              + rng.standard_normal(draw_size) * sc["sigma_h"])
    # Drawing the largest requested cohort and taking prefixes makes the
    # historical-size sensitivity comparison nested within each replication.
    return X_full[:n_historical], Y_full[:n_historical]

def gen_outcome(X, Z, delta, rng):
    X = np.atleast_2d(X); Z = np.asarray(Z, float)
    sig = np.where(Z == 0, config.SIGMA_0, config.SIGMA_1)
    return mu0c(X) + delta*Z + rng.standard_normal(len(Z))*sig

# ── Single trial ─────────────────────────────────────────────────
def run_trial(sc_key, eff_key, mcls, rep, rng, scenarios=None,
              n_historical=None, rng_hist=None, max_n_historical=None,
              hist_data=None):
    scenarios = scenarios or config.SCENARIOS
    td = config.EFFECT_SIZES[eff_key]
    n_historical = config.N_HISTORICAL if n_historical is None else int(n_historical)
    hist_rng = rng if rng_hist is None else rng_hist
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
        n_historical=n_historical,
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


def _hist_size_task(sc, ef, mc, r, seed, scenarios, n_historical,
                    max_n_historical):
    """Run one paired historical-size task with independent RNG streams."""
    try:
        hist_seed, trial_seed = np.random.SeedSequence(seed).spawn(2)
        return run_trial(
            sc, ef, mc, r, np.random.default_rng(trial_seed), scenarios,
            n_historical=n_historical,
            rng_hist=np.random.default_rng(hist_seed),
            max_n_historical=max_n_historical,
        )
    except Exception as e:
        return dict(error=str(e), scenario=sc, effect_type=ef,
                    method=mc.__name__, rep=r,
                    n_historical=n_historical)


def _primary_full_task_seed(base_seed, sc, ef, mcls, rep, scenarios):
    """Return the seed used by run_sim('full') for one task.

    run_sim enumerates scenario, effect, method, and replication in this
    order and increments a single task index. Keeping this mapping explicit
    makes the historical-size extension a genuine common-random-number
    comparison with the primary full experiment.
    """
    scenario_order = list(scenarios.keys())
    effect_order = list(config.EFFECT_SIZES.keys())
    method_order = [RADISH, CAHB, KBCD]
    try:
        si = scenario_order.index(sc)
        ei = effect_order.index(ef)
        mi = method_order.index(mcls)
    except ValueError as exc:
        raise ValueError("task is not in the primary full grid") from exc
    offset = (((si * len(effect_order) + ei) * len(method_order) + mi)
              * config.N_REPS_FULL + int(rep))
    return int(base_seed) + offset


def _load_primary_full_rows(seed, nreps, scenarios):
    """Load and validate primary N_H=200 rows for reuse.

    The full result file does not carry its seed as a column, so reuse is
    intentionally restricted to the documented default seed. Silently
    mixing a custom-seed historical run with the primary file would destroy
    the claimed pairing.
    """
    if int(seed) != 2026:
        raise ValueError(
            "hist_size reuse requires --seed 2026 because the existing "
            "full/raw_results.csv has no seed metadata; rerun the full grid "
            "with the requested seed before reusing it."
        )
    path = config.get_mode_output_dir("full") / "raw_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"primary full result file not found: {path}. "
            "Run `python main.py --mode full --jobs N` first."
        )
    frame = pd.read_csv(path)
    required = {"scenario", "effect_type", "method", "rep",
                "n_historical", "estimated_delta", "ci_low", "ci_high"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"primary full result is missing columns: {missing}")
    if "error" in frame.columns:
        raise ValueError("primary full result contains an error column")
    expected_scenarios = list(scenarios.keys())
    expected_methods = list(config.METHOD_ORDER)
    if set(frame.scenario) != set(expected_scenarios):
        raise ValueError("primary full result scenarios do not match hist_size")
    if set(frame.effect_type) != set(config.EFFECT_ORDER):
        raise ValueError("primary full result effects do not match hist_size")
    if set(frame.method) != set(expected_methods):
        raise ValueError("primary full result methods do not match hist_size")
    if set(frame.n_historical.astype(int)) != {config.N_HISTORICAL}:
        raise ValueError("primary full result is not an N_H=200 file")
    expected_total = (len(expected_scenarios) * len(config.EFFECT_ORDER)
                      * len(expected_methods) * config.N_REPS_FULL)
    if len(frame) != expected_total:
        raise ValueError(
            f"primary full result has {len(frame)} rows; expected "
            f"{expected_total} ({len(expected_scenarios)} scenarios x "
            f"{len(config.EFFECT_ORDER)} effects x {len(expected_methods)} "
            f"methods x {config.N_REPS_FULL} reps)"
        )
    keys = ["scenario", "effect_type", "method", "rep"]
    if frame.duplicated(keys).any():
        raise ValueError("primary full result contains duplicate task keys")
    counts = frame.groupby(keys, dropna=False).size()
    if not np.all(counts.to_numpy() == 1):
        raise ValueError("primary full result does not have one row per task")
    if not np.isfinite(frame[["estimated_delta", "ci_low", "ci_high"]]
                       .to_numpy(dtype=float)).all():
        raise ValueError("primary full result contains non-finite estimates")
    if nreps < 1 or nreps > config.N_REPS_FULL:
        raise ValueError("hist_size replications must be within the primary full grid")
    out = frame[frame.rep.astype(int) < int(nreps)].copy()
    expected_short = (len(expected_scenarios) * len(config.EFFECT_ORDER)
                      * len(expected_methods) * int(nreps))
    if len(out) != expected_short:
        raise ValueError("filtered primary full rows do not cover the requested reps")
    return out


def _nested_history_from_primary(sc_key, primary_seed, scenarios):
    """Build the N_H=400 extension while preserving the full N_H=200 stream.

    The primary stream first generates the 200 historical observations. Its
    post-history state is copied: one copy generates the additional 200
    historical observations, while the original state is used for the
    concurrent trial. Thus the first 200 historical observations and every
    concurrent random draw are identical to the primary full run.
    """
    primary_rng = np.random.default_rng(int(primary_seed))
    x_first, y_first = gen_hist(
        sc_key, primary_rng, scenarios,
        n_historical=config.N_HISTORICAL,
        max_n_historical=config.N_HISTORICAL,
    )
    post_history_state = copy.deepcopy(primary_rng.bit_generator.state)
    extension_rng = np.random.default_rng()
    extension_rng.bit_generator.state = copy.deepcopy(post_history_state)
    x_extra, y_extra = gen_hist(
        sc_key, extension_rng, scenarios,
        n_historical=config.N_HISTORICAL,
    )
    x_hist = np.vstack([x_first, x_extra])
    y_hist = np.concatenate([y_first, y_extra])
    return primary_rng, (x_hist, y_hist)


def _hist_size_reuse_task(sc, ef, mc, rep, primary_seed, scenarios):
    """Compute only the new N_H=400 row coupled to a primary full row."""
    try:
        trial_rng, hist_data = _nested_history_from_primary(
            sc, primary_seed, scenarios)
        return run_trial(
            sc, ef, mc, rep, trial_rng, scenarios,
            n_historical=2 * config.N_HISTORICAL,
            hist_data=hist_data,
        )
    except Exception as e:
        return dict(error=str(e), scenario=sc, effect_type=ef,
                    method=mc.__name__, rep=rep,
                    n_historical=2 * config.N_HISTORICAL)


def _validate_primary_stream_alignment(primary_rows, seed, scenarios):
    """Replay deterministic probes against stored primary full rows.

    This catches changes in task ordering, RNG stream construction, or method
    labels before the expensive N_H=400 extension is launched.
    """
    scenario_order = list(scenarios.keys())
    probes = [
        (scenario_order[0], "Null", KBCD, 0),
        (scenario_order[2], "Power", RADISH, 0),
        (scenario_order[-1], "Null", CAHB, 0),
    ]
    method_names = {RADISH: "RADISH", CAHB: "CAHB", KBCD: "KBCD"}
    cols = ["estimated_delta", "ci_low", "ci_high", "allocation_ratio",
            "mean_W", "mean_Rn", "mean_Dpdc"]
    for sc, ef, mc, rep in probes:
        task_seed = _primary_full_task_seed(seed, sc, ef, mc, rep, scenarios)
        generated = run_trial(
            sc, ef, mc, rep, np.random.default_rng(task_seed), scenarios,
            n_historical=config.N_HISTORICAL,
        )
        stored = primary_rows[
            (primary_rows.scenario == sc)
            & (primary_rows.effect_type == ef)
            & (primary_rows.method == method_names[mc])
            & (primary_rows.rep.astype(int) == rep)
        ]
        if len(stored) != 1:
            raise ValueError(
                f"primary full probe row not found for "
                f"{sc}/{ef}/{method_names[mc]}/{rep}"
            )
        stored = stored.iloc[0]
        for col in cols:
            a = generated.get(col, np.nan)
            b = stored.get(col, np.nan)
            if np.isfinite(a) and np.isfinite(b):
                if not np.isclose(float(a), float(b), rtol=1e-10, atol=1e-10):
                    raise ValueError(
                        f"primary full RNG replay mismatch for {sc}/{ef}/"
                        f"{method_names[mc]}/{rep}/{col}: {a} != {b}"
                    )
            elif not (pd.isna(a) and pd.isna(b)):
                raise ValueError(
                    f"primary full RNG replay mismatch for {sc}/{ef}/"
                    f"{method_names[mc]}/{rep}/{col}"
                )
    return True

# ── Parallel runner ──────────────────────────────────────────────
def run_sim(mode, seed=2026, n_jobs=4):
    nreps = config.get_mode_replications(mode)
    scenarios = config.get_mode_scenarios(mode)
    tasks, idx = [], 0
    if mode == "bias":
        # Common random numbers pair all methods and xi values within each
        # (effect, replication) block.  The historical location shift then
        # changes deterministically along xi while covariates, innovations,
        # and the current-trial stream are held fixed.  In particular, the
        # KBCD path is constant across xi, as it should be for a method that
        # does not use historical outcomes.
        effect_order = list(config.EFFECT_SIZES)
        for sc in scenarios:
            for i_eff, ef in enumerate(effect_order):
                for mc in [RADISH, CAHB, KBCD]:
                    for r in range(nreps):
                        paired_seed = int(seed) + r + nreps * i_eff
                        tasks.append((sc, ef, mc, r, paired_seed, scenarios))
    else:
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


def run_hist_size_sim(seed=2026, n_jobs=4, nreps=None):
    """Paired sensitivity study over config.HISTORICAL_SIZE_GRID.

    Within each scenario/effect/replication, every method and historical
    sample size receives the same historical and concurrent random-number
    streams. Historical samples are nested prefixes of the largest cohort.
    """
    nreps = config.N_REPS_HIST_SIZE if nreps is None else int(nreps)
    scenarios = config.get_mode_scenarios("hist_size")
    hist_sizes = [int(n) for n in config.HISTORICAL_SIZE_GRID]
    expected_sizes = [config.N_HISTORICAL, 2 * config.N_HISTORICAL]
    if hist_sizes != expected_sizes:
        raise ValueError(
            f"hist_size reuse expects HISTORICAL_SIZE_GRID={expected_sizes}, "
            f"got {hist_sizes}"
        )
    primary = _load_primary_full_rows(seed, nreps, scenarios)
    _validate_primary_stream_alignment(primary, seed, scenarios)
    tasks = []
    for sc in scenarios:
        for ef in config.EFFECT_SIZES:
            for mc in [RADISH, CAHB, KBCD]:
                for r in range(nreps):
                    primary_seed = _primary_full_task_seed(
                        seed, sc, ef, mc, r, scenarios)
                    tasks.append((sc, ef, mc, r, primary_seed, scenarios))
    print(f"[hist_size] {len(primary) + len(tasks)} output rows from "
          f"{len(tasks)} new N_H=400 tasks (reps={nreps}, jobs={n_jobs}, "
          f"scenarios={len(scenarios)}, N_H={hist_sizes})")
    print(f"[hist_size] reused primary N_H=200 rows: {len(primary)}; "
          f"avoided recomputation: {len(primary)} tasks "
          f"(50.0% of the two-size workload)")
    res = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_hist_size_reuse_task)(*t) for t in tasks)
    good = [r for r in res if "error" not in r]
    bad = [r for r in res if "error" in r]
    if bad:
        print(f"  {len(bad)} errors; first: {bad[0]}")
    new_rows = pd.DataFrame(good)
    if len(new_rows) != len(tasks):
        raise RuntimeError(
            f"N_H=400 extension returned {len(new_rows)} successful rows; "
            f"expected {len(tasks)}"
        )
    out = pd.concat([primary, new_rows], ignore_index=True)
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
        "primary_full_file": str(
            config.get_mode_output_dir("full") / "raw_results.csv"),
        "primary_n_historical": int(config.N_HISTORICAL),
        "new_n_historical": int(2 * config.N_HISTORICAL),
        "primary_rows_reused": int(len(primary)),
        "new_rows_computed": int(len(new_rows)),
        "rows_written": int(len(out)),
        "stream_alignment_probes": 3,
        "coupling": (
            "exact primary N_H=200 historical prefix and concurrent RNG "
            "stream; N_H=400 uses a cloned post-prefix stream for its "
            "additional historical cohort"
        ),
        "limitation": (
            "The N_H=400 historical extension is a deterministic coupled "
            "counterfactual, not an independent historical cohort draw."
        ),
    }
    return out

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


def validate_saved_results(mode="full", result_dir=None):
    """Validate a saved synthetic experiment against the code contract."""
    out = Path(result_dir) if result_dir else config.get_mode_output_dir(mode)
    raw_path, metrics_path = out / "raw_results.csv", out / "metrics.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing raw results: {raw_path}")
    df = pd.read_csv(raw_path)
    scenarios = config.get_mode_scenarios(mode)
    reps = config.get_mode_replications(mode)
    methods, effects = tuple(config.METHOD_ORDER), tuple(config.EFFECT_ORDER)
    required = {
        "scenario", "effect_type", "method", "rep", "true_delta",
        "estimated_delta", "ci_low", "ci_high", "rejected",
        "estimation_bias", "allocation_ratio", "b_theta", "sigma_h",
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
    for name, spec in scenarios.items():
        part = df[df["scenario"] == name]
        assert np.allclose(part["b_theta"], spec["b_theta"])
        assert np.allclose(part["sigma_h"], spec["sigma_h"])
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
                                       "hist_size"], default="demo")
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
        print(json.dumps(validate_saved_results(args.mode), indent=2))
        raise SystemExit(0)
    t0 = time.time()
    if args.mode == "hist_size":
        df = run_hist_size_sim(args.seed, args.jobs, args.reps)
    else:
        if args.reps is not None:
            raise ValueError("--reps is currently supported only for --mode hist_size")
        df = run_sim(args.mode, args.seed, args.jobs)
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
