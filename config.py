"""
config.py — Simulation configuration (3×2 factorial design).

Clean 3×2 factorial grid:
                    High precision (σ_H=0.5)        Low precision (σ_H=3.0)
  No bias (b=0)     B1: ideal borrowing              B2: noisy but unbiased
  Mod bias (b=0.5)  B3: precise bias → CAHB traps    B4: hidden bias in noise
  Large bias (b=2)  B5: all methods detect            B6: all methods detect

Key B3 vs B4 insight:
  B3: θ₀(x) precise → CAHB's variance-ratio sees high value + circular trap
      → OVER-borrows biased data → severe ATE contamination
  B4: θ₀(x) noisy → less variance reduction value → CAHB borrows less
      but still cannot detect hidden bias
  RADISH: limits borrowing in both via PDC (B3) and precision cap (B4)
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

# ── Simulation Scenarios (3×2 factorial) ──────────────────────────
# Y^H = μ₀^C(X) + b_θ + ε_H,   ε_H ~ N(0, σ_H²)
#
# Bias levels:   b = 0.0 (none), 0.5 (moderate), 2.0 (large)
# Precision:     σ_H = 0.5 (high precision), 3.0 (low precision)

SCENARIOS = {
    "B1_noBias_highPrec":  {"b_theta": 0.0, "sigma_h": 0.5,
        "description": "Ideal: unbiased, precise — maximum borrowing benefit"},
    "B2_noBias_lowPrec":   {"b_theta": 0.0, "sigma_h": 3.0,
        "description": "Unbiased but noisy — limited borrowing value"},
    "B3_mdBias_highPrec":  {"b_theta": 0.5, "sigma_h": 0.5,
        "description": "Moderate bias, precise — CAHB over-borrows via circular trap"},
    "B4_mdBias_lowPrec":   {"b_theta": 0.5, "sigma_h": 3.0,
        "description": "Moderate bias, noisy — bias hidden; less borrowing but still contaminated"},
    "B5_lgBias_highPrec":  {"b_theta": 2.0, "sigma_h": 0.5,
        "description": "Large bias, precise — all methods detect conflict"},
    "B6_lgBias_lowPrec":   {"b_theta": 2.0, "sigma_h": 3.0,
        "description": "Large bias, noisy — all methods detect conflict"},
}

# ── Kernel bandwidths ────────────────────────────────────────────
KERNEL_BANDWIDTH_BINARY     = 1.1    # Epanechnikov allocation kernel
KERNEL_BANDWIDTH_CONTINUOUS = 1.3

# ── Method hyperparameters ───────────────────────────────────────
PRIORS = {
    # RADISH (proposed)
    "radish_nc_stabilizer":    5.0,   # N_c floor for Stage II
    "radish_n_min_discrepancy": 3.0,
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
            out[key] = {"b_theta": float(bv), "sigma_h": float(sh),
                        "description": f"Precision sweep b={bv}, σ_H={sh}",
                        "_grid_bias": blab, "_grid_sigma": float(sh)}
    return out

PRECISION_SCENARIOS = _build_precision_scenarios()

# ── Replications ─────────────────────────────────────────────────
N_REPS_DEMO       = 10
N_REPS_FULL       = 500
N_REPS_PRECISION  = 200   # finer grid (12 cells) → moderate reps per cell
def get_mode_replications(mode):
    return {"demo": N_REPS_DEMO,
            "full": N_REPS_FULL,
            "precision": N_REPS_PRECISION}[mode]

def get_mode_scenarios(mode):
    """Return the scenario dict for a given mode."""
    if mode == "precision":
        return PRECISION_SCENARIOS
    return SCENARIOS

# ── Output & Runtime ─────────────────────────────────────────────
RESULTS_ROOT   = "results"
N_JOBS         = 4
SCENARIO_ORDER = list(SCENARIOS.keys())
METHOD_ORDER   = ["KBCD", "CAHB", "RADISH"]
EFFECT_ORDER   = ["Null", "Power"]

def get_mode_output_dir(mode):
    return Path(RESULTS_ROOT) / mode
