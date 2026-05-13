"""
real_scenarios.py — Two-axis design for the real-data-anchored
simulation, mirroring the B1-B6 simulation DGP form (additive bias +
σ_h precision) but anchored to HORIZON-fitted parameters and FIT-KDE
covariate distributions.

  ξ ∈ {0, 0.25, 0.5, 1, 2}   bias-axis (in σ_C units).
                             b_θ_absolute = ξ · σ_C, so ξ ∈ {0, 0.5, 2}
                             reproduce B1/B3-B4/B5-B6 b_θ levels under
                             our HORIZON σ_C ≈ 0.33 noise scale.
  η ∈ {0, 0.25, 0.5, 0.75, 1} precision-axis.  Maps σ_h/σ_C linearly
                             from 3.0 (η=0, low prec, B2/B4/B6 ratio) to
                             0.5 (η=1, high prec, B1/B3/B5 ratio).
                             N_H is fixed at 200 (matching B-scenarios).

Differentiation from B1-B6: 4 covariates (2 binary + 2 KDE-sampled
continuous from real menyrs/Y0 distributions, with subgroup-specific
β_0(sg), β_1(sg), β_2(sg) coefficients per HORIZON Table 3) vs B-sim's
2 covariates with pooled β.  σ_C = 0.33 (real) vs 1.0 (B-sim).
δ_DGP    = 0.17 (calibrated for nominal power ≈ 0.8) vs 0.5 (B-sim).
"""
from __future__ import annotations

# Bias levels mirror B1-B6 (b_θ ∈ {0, 0.5, 2}) plus interpolations.
XI_GRID  = [0.0, 0.25, 0.5, 1.0, 2.0]
# Precision dial — η → σ_h/σ_C linearly maps [0, 1] → [3.0, 0.5].
ETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

N_CURRENT     = 200     # Match B-scenarios N (config.N_CURRENT) and CAHB §5.2.1.
N_HISTORICAL  = 200     # Match B-scenarios N_HISTORICAL.
INTERIM_START = 20      # 1:1 baseline allocation for first 20 patients
N_REPS        = 1000    # Final rep count (tighten 95% CI on RMSE/power)
SEED          = 6052
N_JOBS        = 4

# η now controls historical sample size n_H directly (CAHB §5.2.3
# design): η ∈ [0, 1] linearly maps to n_H ∈ [30, 350].
NH_MIN = 30
NH_MAX = 350
def n_H_of_eta(eta: float) -> int:
    return int(round(NH_MIN + (NH_MAX - NH_MIN) * eta))
