"""Simulation scenarios and runtime settings for RADISH.

Historical conflict is covariate dependent. For the binary covariate X1,
``b_local=(b0,b1)`` defines the two stratum-specific shifts and ``b_theta``
records their marginal summary. The manuscript uses three complementary grids:
the eight-cell conflict-by-precision experiment, a nine-point local-conflict path,
and a four-cell diagnostic crossing marginal conflict with local heterogeneity.
The broader ``full`` grid is retained for supplementary sensitivity analyses.
"""
import numpy as np
from pathlib import Path

# ── Sample sizes & trial structure ────────────────────────────────
N_HISTORICAL  = 200
N_CURRENT     = 200
INTERIM_START = 20
ALPHA         = 0.05

# ── Covariate & outcome model ────────────────────────────────────
COVARIATE_DIM = 2
COVARIATE_PARAMS = {"p": 2, "disc_idx": [0], "cont_idx": [1]}
BETA = np.array([0.0, 0.5, 0.5, 0.3])   # μ₀^C(X) = β₀ + β₁X₁ + β₂X₂ + β₃X₁X₂
SIGMA_0 = 1.0    # control arm noise SD
SIGMA_1 = 1.2    # treatment arm noise SD (heteroscedastic)

EFFECT_SIZES = {"Null": 0.0, "Power": 0.5}

# ── Manuscript six-scenario grid (stable B1--B6 identifiers) ─────────
# Y^H = μ₀^C(X) + b_θ + ε_H,   ε_H ~ N(0, σ_H²)
#
# Bias levels:   b = 0.0 (none), 0.5 (moderate), 2.0 (large)
# Precision:     σ_H = 0.5 (high precision), 3.0 (low precision)

SCENARIOS = {
    "B1_noBias_highPrec":  {"b_local": (0.0, 0.0), "b_theta": 0.0, "sigma_h": 0.5,
        "description": "Ideal: unbiased, precise — maximum borrowing benefit"},
    "B2_noBias_lowPrec":   {"b_local": (0.0, 0.0), "b_theta": 0.0, "sigma_h": 3.0,
        "description": "Unbiased but noisy — limited borrowing value"},
    "L1_lowBias_highPrec": {"b_local": (0.04, 0.12), "b_theta": 0.08, "sigma_h": 0.5,
        "description": "Small local conflict, precise — useful borrowing should remain"},
    "L2_lowBias_lowPrec":  {"b_local": (0.04, 0.12), "b_theta": 0.08, "sigma_h": 3.0,
        "description": "Small local conflict, noisy — limited information gain"},
    "B3_mdBias_highPrec":  {"b_local": (0.25, 0.75), "b_theta": 0.5, "sigma_h": 0.5,
        "description": "Moderate bias, precise — CAHB over-borrows via circular trap"},
    "B4_mdBias_lowPrec":   {"b_local": (0.25, 0.75), "b_theta": 0.5, "sigma_h": 3.0,
        "description": "Moderate bias, noisy — bias hidden; less borrowing but still contaminated"},
    "B5_lgBias_highPrec":  {"b_local": (1.5, 2.5), "b_theta": 2.0, "sigma_h": 0.5,
        "description": "Large bias, precise — all methods detect conflict"},
    "B6_lgBias_lowPrec":   {"b_local": (1.5, 2.5), "b_theta": 2.0, "sigma_h": 3.0,
        "description": "Large bias, noisy — all methods detect conflict"},
}

def _local(b0, b1, sigma_h, description):
    """Primary-grid scenario defined by stratum-wise historical shifts."""
    return {"b_local": (float(b0), float(b1)),
            "b_theta": 0.5 * (float(b0) + float(b1)),
            "sigma_h": float(sigma_h),
            "description": description}


def _path(b1, description):
    """One-to-three local-shift path used by the mechanism figure."""
    return _local(float(b1) / 3.0, float(b1), 0.5, description)


LOCAL_PATH_SCENARIOS = {
    "P0_compatible": _local(0.0, 0.0, 0.5, "Exact compatibility"),
    "P1_0p001": _path(0.001, "Very small local conflict"),
    "P2_0p003": _path(0.003, "Very small local conflict"),
    "P3_0p01": _path(0.01, "Small local conflict"),
    "P4_0p03": _path(0.03, "Small local conflict"),
    "P5_0p10": _path(0.10, "Low local conflict"),
    "P6_0p30": _path(0.30, "Moderate local conflict"),
    "P7_1p0": _path(1.00, "High local conflict"),
    "P8_3p0": _path(3.00, "Severe local conflict"),
}


# ── Extended four-regime grid used by demo, full, and hist_size modes ───
# The compatible stratum is b0 = 0 throughout; the conflict lives in the
# x1 = 1 stratum only, at twice the marginal level of the legacy grid.
PRIMARY_SCENARIOS = SCENARIOS

# ── Kernel ───────────────────────────────────────────────────────
# A single Gaussian product kernel is used for every stage (Stage I/II/III
# and the allocation rule).  Its per-dimension bandwidth is set by Scott's
# rule of thumb h_j = n^{-1/(p+4)} * sigma_j (see methods.kernel_bandwidths);
# there is no separate fitting/allocation kernel and no hand-tuned bandwidth.

# Backwards-compatible name used by older downstream scripts.
DEMO_SCENARIOS = PRIMARY_SCENARIOS

# Orthogonal marginal-shift-by-local-heterogeneity diagnostic. This compact
# four-cell experiment is
# separate from the retained eight-scenario bias-by-precision study.
# The diagnostic uses P(X1=1)=1/3 and the centered contrast
# s(0)=-1/2, s(1)=1. Hence E{s(X1)}=0, while unequal absolute contrasts let
# the borrowing diagnostic visibly distinguish the two local regions.
GLOBAL_LOCAL_X1_PROB = 1.0 / 3.0
GLOBAL_SIGNAL_STRONG = 0.6
LOCAL_SIGNAL_WEAK = 0.1
LOCAL_SIGNAL_STRONG = 1.2


def _global_local(g, ell, description):
    """Scenario with b(x)=g+ell*s(x), E{s(X1)}=0, at high precision."""
    scenario = _local(g - 0.5 * ell, g + ell, 0.5, description)
    scenario["p_x1"] = GLOBAL_LOCAL_X1_PROB
    scenario["b_theta"] = float(g)
    return scenario


GLOBAL_LOCAL_SCENARIOS = {
    "GL00_weak_global_weak_local": _global_local(
        0.0, LOCAL_SIGNAL_WEAK, "Weak global and weak local conflict"),
    "GL10_strong_global_weak_local": _global_local(
        GLOBAL_SIGNAL_STRONG, LOCAL_SIGNAL_WEAK,
        "Strong global and weak local conflict"),
    "GL01_weak_global_strong_local": _global_local(
        0.0, LOCAL_SIGNAL_STRONG, "Weak global and strong local conflict"),
    "GL11_strong_global_strong_local": _global_local(
        GLOBAL_SIGNAL_STRONG, LOCAL_SIGNAL_STRONG,
        "Strong marginal shift and strong local heterogeneity; X1=0 remains compatible"),
}

# ── Method hyperparameters ───────────────────────────────────────
PRIORS = {
    # RADISH (proposed)
    "radish_nc_stabilizer":    5.0,   # N_c floor for Stage II
    "radish_n0_final":         5.0,   # N_c floor for Stage III
    # CAHB (Jin et al., 2023)
    "cahb_lambda":            300.0, # L1 projection radius
    "cahb_gamma":             0.25,  # τ prior scale
}

# ── Precision-gradient scenarios (showcases RADISH τ²_H awareness) ──
# Hold b_θ fixed and sweep σ_H over a fine grid. Tests whether each
# method correctly down-weights low-precision historical data even
# when no bias is present — RADISH's R_n explicitly contains τ²_H.
PRECISION_GRID = [0.25, 0.5, 1.0, 1.5, 2.5, 4.0]
PRECISION_BIAS_LEVELS = {"unbiased": 0.0, "modBias": 0.5}

def _build_precision_scenarios():
    out = {}
    for blab, bv in PRECISION_BIAS_LEVELS.items():
        for sh in PRECISION_GRID:
            key = f"P_{blab}_sH{sh:g}".replace(".", "p")
            ell = 0.5 * float(bv)
            out[key] = {"b_local": (float(bv) - ell, float(bv) + ell),
                        "b_theta": float(bv), "sigma_h": float(sh),
                        "description": f"Precision sweep b={bv}, σ_H={sh}",
                        "_grid_bias": blab, "_grid_sigma": float(sh)}
    return out

PRECISION_SCENARIOS = _build_precision_scenarios()

# ── Bias-gradient scenarios — direct test of Theorem 1 ────────────
# Theorem 1 predicts D_PDC ≥ c1·N_H·h̄·b² + c2·log N. Sweep b_θ on a
# fine grid at fixed σ_H = 0.5 to verify the quadratic-in-b growth
# of D_PDC and the exponential collapse of W in the unsaturated
# regime (i.e. before D_PDC hits the −log(ε_p) ≈ 27.6 cap).
# Focused low-conflict grid.  We use the same dimensionless parameter as the
# HORIZON-FIT semi-synthetic study: b_theta = 2 * xi * sigma_C. Here
# sigma_C=SIGMA_0=1, so the requested doubling grid maps to
# b_theta={0,.02,.04,.08,.16,.32}.  Moderate and severe conflict remain in
# the primary 4x2 experiment at b_theta=.5 and 2.
BIAS_XI_GRID = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16]
BIAS_GRID = [2.0 * xi * SIGMA_0 for xi in BIAS_XI_GRID]
BIAS_GRID_SIGMA_H = 0.5

def _build_bias_scenarios():
    out = {}
    for xi, b in zip(BIAS_XI_GRID, BIAS_GRID):
        key = f"G_xi{xi:g}".replace(".", "p")
        ell = 0.5 * float(b)
        out[key] = {"b_local": (float(b) - ell, float(b) + ell),
                    "b_theta": float(b), "sigma_h": float(BIAS_GRID_SIGMA_H),
                    "xi": float(xi),
                    "description": (f"Low-conflict gradient xi={xi}, "
                                    f"b_theta={b}, sigma_H={BIAS_GRID_SIGMA_H}"),
                    "_grid_bias_value": float(b),
                    "_grid_xi_value": float(xi)}
    return out

BIAS_SCENARIOS = _build_bias_scenarios()

# ── Replications ─────────────────────────────────────────────────
N_REPS_DEMO       = 10
N_REPS_FULL       = 1000
N_REPS_PRECISION  = 500   # 12 cells × 500 reps = 6000 trials/method
N_REPS_BIAS       = 500   # 6 xi cells x 2 effects x 3 methods x 500 reps
N_REPS_HIST_SIZE  = 1000
N_REPS_GLOBAL_LOCAL = 1000
N_REPS_SIX = 1000
N_REPS_LOCAL_PATH = 1000

# Historical-sample-size sensitivity analysis. The larger case doubles the
# historical cohort while holding the concurrent trial size fixed at N=200.
HISTORICAL_SIZE_GRID = [N_HISTORICAL, 2 * N_HISTORICAL]

def get_mode_replications(mode):
    return {"demo": N_REPS_DEMO,
            "full": N_REPS_FULL,
            "precision": N_REPS_PRECISION,
            "bias":  N_REPS_BIAS,
            "hist_size": N_REPS_HIST_SIZE,
            "global_local": N_REPS_GLOBAL_LOCAL,
            "six": N_REPS_SIX,
            "local_path": N_REPS_LOCAL_PATH}[mode]

def get_mode_scenarios(mode):
    """Return the scenario dict for a given mode."""
    if mode in {"demo", "full", "hist_size"}:
        return PRIMARY_SCENARIOS
    if mode == "precision":
        return PRECISION_SCENARIOS
    if mode == "bias":
        return BIAS_SCENARIOS
    if mode == "global_local":
        return GLOBAL_LOCAL_SCENARIOS
    if mode == "six":  # backwards-compatible mode name; now eight cells
        return SCENARIOS
    if mode == "local_path":
        return LOCAL_PATH_SCENARIOS
    return SCENARIOS

# ── Output & Runtime ─────────────────────────────────────────────
RESULTS_ROOT   = "results"
N_JOBS         = 4
SCENARIO_ORDER = list(PRIMARY_SCENARIOS.keys())
METHOD_ORDER   = ["KBCD", "CAHB", "RADISH"]
EFFECT_ORDER   = ["Null", "Power"]

def get_mode_output_dir(mode):
    return Path(RESULTS_ROOT) / mode
