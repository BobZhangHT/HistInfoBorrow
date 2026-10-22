"""Compact two-axis HORIZON-FIT semi-synthetic design.

Historical controls have a mean-zero, X1-dependent shift along the conflict
path. Opposing stratum-specific discrepancies grow with ``xi`` while their
target-population average remains zero. Seven log-spaced display levels trace
compatibility through high localized conflict. The precision
axis changes the number of available historical controls from 30 to 350 while
holding the historical noise scale fixed.
Current-trial covariates and response-surface coefficients are calibrated to
HORIZON/FIT.
"""
from __future__ import annotations

# Zero is exact compatibility. The six positive levels are spaced by roughly
# half a decade, so transitions near compatibility and at high conflict receive
# comparable horizontal resolution on a log scale.
XI_GRID = [0.0, 0.003162, 0.01, 0.031623, 0.10, 0.316228, 1.00]
# At xi=1, the majority and minority X1 strata differ from the concurrent
# response surface by -4 and 4(1-p1)/p1 residual SDs, respectively. This is a
# severe stress-test endpoint at which both borrowing procedures should return
# close to the concurrent-only design while the mean shift remains zero.
LOCAL_PATH_HIGH = 4.0
# Boundaries divide the displayed range [1e-3, 1] into four equal log-width
# regions. The zero-conflict cell is displayed at the 1e-3 anchor.
XI_REGIMES = [
    ("Compatible/minimal", 1e-3, 10 ** -2.25),
    ("Low", 10 ** -2.25, 10 ** -1.5),
    ("Medium", 10 ** -1.5, 10 ** -0.75),
    ("High", 10 ** -0.75, 1.05),
]
# Historical-precision dial; eta maps to the historical-control count below.
ETA_GRID = [0.0, 1.0]

# Compact factorial diagnostic that separates the target-population mean
# historical shift from its X1-dependent heterogeneity.  Values are expressed
# in units of the HORIZON residual standard deviation.  The high-heterogeneity
# value is computed from the empirical HORIZON prevalence. The separate xi
# path extends to twice this magnitude so its endpoint checks full
# no-borrowing reversion.
GLOBAL_LOCAL_LEVELS = {
    "mean_shift_weak": 0.0,
    "mean_shift_strong": 2.0,
    "heterogeneity_weak": 0.1,
}


def global_local_regimes(p1: float):
    """Return the four mean-shift by local-heterogeneity regimes.

    Under b(X1)/sigma_C = g + ell * (X1-p1)/sqrt(p1*(1-p1)), g is
    the target-population mean shift and ell is its across-stratum SD.
    """
    if not 0.0 < p1 < 1.0:
        raise ValueError("p1 must lie strictly between zero and one")
    ell_low = GLOBAL_LOCAL_LEVELS["heterogeneity_weak"]
    ell_high = ((1.0 - p1) / p1) ** 0.5
    g_low = GLOBAL_LOCAL_LEVELS["mean_shift_weak"]
    g_high = GLOBAL_LOCAL_LEVELS["mean_shift_strong"]
    return [
        dict(regime="GW_LW", g_sd=g_low, ell_sd=ell_low,
             label="Weak mean shift, weak local heterogeneity"),
        dict(regime="GS_LW", g_sd=g_high, ell_sd=ell_low,
             label="Strong mean shift, weak local heterogeneity"),
        dict(regime="GW_LS", g_sd=g_low, ell_sd=ell_high,
             label="Weak mean shift, strong local heterogeneity"),
        dict(regime="GS_LS", g_sd=g_high, ell_sd=ell_high,
             label="Strong mean shift, strong local heterogeneity"),
    ]

N_CURRENT     = 200     # Match B-scenarios N (config.N_CURRENT) and CAHB Section 5.2.1.
N_HISTORICAL  = 200     # Match B-scenarios N_HISTORICAL.
INTERIM_START = 20      # 1:1 baseline allocation for first 20 patients
N_REPS        = 1000    # Final rep count (tighten 95% CI on RMSE/power)
SEED          = 260920  # Formal-run seed.
N_JOBS        = 6

# eta controls historical sample size n_H directly (CAHB Section 5.2.3
# design): eta in [0, 1] maps linearly to n_H in [30, 350].
NH_MIN = 30
NH_MAX = 350
def n_H_of_eta(eta: float) -> int:
    return int(round(NH_MIN + (NH_MAX - NH_MIN) * eta))
