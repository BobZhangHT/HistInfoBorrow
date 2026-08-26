"""Two-axis real-data-calibrated sensitivity design.

The conflict axis combines a dense low-conflict zoom with the original broad
stress-test range. Historical controls have mean shift
``b_theta = 2 * xi * sigma_C``; ``xi=0`` is exact compatibility, the positive
levels double from 0.01 through 0.64, and ``xi=1`` is the severe-conflict
endpoint. The precision axis changes the number of available historical controls
from 30 to 350 while holding the historical noise scale fixed. Current-trial
covariates and response-surface coefficients are calibrated to HORIZON/FIT.
"""
from __future__ import annotations

# Dense compatible/low-conflict coverage plus moderate and severe endpoints.
# Plotting uses a zero-aware base-2 symlog axis.
XI_GRID  = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.00]
XI_REGIMES = [
    ("Compatible", -0.0025, 0.0283),
    ("Low conflict", 0.0283, 0.2263),
    ("Moderate conflict", 0.2263, 0.80),
    ("Severe conflict", 0.80, 1.08),
]
# Historical-precision dial; eta maps to the historical-control count below.
ETA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

N_CURRENT     = 200     # Match B-scenarios N (config.N_CURRENT) and CAHB Section 5.2.1.
N_HISTORICAL  = 200     # Match B-scenarios N_HISTORICAL.
INTERIM_START = 20      # 1:1 baseline allocation for first 20 patients
N_REPS        = 1000    # Final rep count (tighten 95% CI on RMSE/power)
SEED          = 6052
N_JOBS        = 6

# eta controls historical sample size n_H directly (CAHB Section 5.2.3
# design): eta in [0, 1] maps linearly to n_H in [30, 350].
NH_MIN = 30
NH_MAX = 350
def n_H_of_eta(eta: float) -> int:
    return int(round(NH_MIN + (NH_MAX - NH_MIN) * eta))
