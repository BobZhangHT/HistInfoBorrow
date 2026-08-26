"""
methods_c.py — C-backed implementations of KBCD, CAHB, RADISH.

Drop-in replacement for methods.py. Same class names, constructors,
and return contracts; numerical results match methods.py up to the
last bit (verified by tests/test_methods_c_consistency.py).

Usage
-----
>>> from methods_c import KBCD, CAHB, RADISH        # same as methods
... or, to keep importing methods.* but get C speed:
>>> import methods_c as methods                     # alias

Build the shared library first:
    python build_c.py
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import numpy as np

def _kernel_effective_n(weights, floor=1e-6):
    """Reference for the C RADISH kernel's Kish effective sample size.

    The native code computes this internally for each local concurrent
    neighbourhood. Keeping the scalar reference here makes the Python/C
    numerical contract explicit for parity checks.
    """
    w = np.asarray(weights, dtype=float).ravel()
    sw = float(np.sum(w))
    sw2 = float(np.dot(w, w))
    if (not np.isfinite(sw)) or (not np.isfinite(sw2)) or sw <= 1e-12 or sw2 <= 1e-12:
        return float(floor)
    return float(max((sw * sw) / sw2, floor))

# ── Locate and load the shared library ───────────────────────────
_HERE = Path(__file__).resolve().parent
_LIB_NAME = "radish_core.dll" if os.name == "nt" else "radish_core.so"
_LIB_PATH = _HERE / _LIB_NAME
if not _LIB_PATH.exists():
    raise ImportError(
        f"{_LIB_NAME} not found in {_HERE}. Build it first:\n"
        f"    python build_c.py")
_lib = ct.CDLL(str(_LIB_PATH))

# ── ctypes helpers ───────────────────────────────────────────────
_DBL  = ct.c_double
_INT  = ct.c_int
_PD   = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
_PI   = np.ctypeslib.ndpointer(dtype=np.int32,   flags="C_CONTIGUOUS")

def _arr(a, dtype=np.float64):
    """Ensure C-contiguous, correct dtype."""
    return np.ascontiguousarray(np.asarray(a, dtype=dtype))


# ── Function signatures ──────────────────────────────────────────
_lib.kernel_gauss_matrix.argtypes = [_PD, _INT, _PD, _INT, _INT, _PD, _PD]
_lib.kernel_gauss_matrix.restype  = None

_lib.normalize_rows.argtypes      = [_PD, _INT, _INT]
_lib.normalize_rows.restype       = None

_lib.kernel_gauss_vec.argtypes    = [_PD, _INT, _INT, _PD, _PD, _PD]
_lib.kernel_gauss_vec.restype     = None

_lib.kernel_epan_vec.argtypes     = [_PD, _INT, _INT, _PD, _PD, _PD]
_lib.kernel_epan_vec.restype      = None

_lib.estimate_ate_c.argtypes      = [
    _PD, _INT, _INT,           # X_eval, n_eval, p
    _PD, _PD, _INT,            # X0, Y0, n0
    _PD, _PD, _INT,            # X1, Y1, n1
    _PD, _PD, _INT,            # Xh, Yh, nh
    _PD, _PD, _DBL,            # h, W_vec, alpha
    _PD, _PD, _PD, _PD, _PD,   # outputs
]
_lib.estimate_ate_c.restype       = None

_lib.radish_set_borrow_params.argtypes = [_DBL, _DBL, _INT, _INT]
_lib.radish_set_borrow_params.restype  = None

_lib.radish_get_borrow_params.argtypes = [ct.POINTER(_DBL), ct.POINTER(_DBL),
                                           ct.POINTER(_INT), ct.POINTER(_INT)]
_lib.radish_get_borrow_params.restype = None

_lib.radish_set_adj_global.argtypes    = [_INT]
_lib.radish_set_adj_global.restype     = None


def set_adj_global(on=False):
    """Use the covariate-standardised trial-level conflict check instead of the
    raw Welch comparison of marginal means.  The raw statistic standardises by
    marginal variances, which carry Var{mu_0(X)}; that term is common to both
    cohorts, carries no conflict information, and costs power.  Constant-free.
    Process-global: set it inside each worker."""
    _lib.radish_set_adj_global(int(bool(on)))


_lib.radish_set_alloc_params.argtypes  = [_INT, _DBL, _DBL, _INT, _INT]
_lib.radish_set_alloc_params.restype   = None


def set_alloc_params(seq_global=False, phi_lo=0.0, phi_hi=1.0,
                     alloc_legacy=False, alloc_min_n0=0):
    """Stage-II options.  ``seq_global`` recomputes the trial-level conflict
    check from the controls accrued so far at every interim (published rule:
    off, leaf-only); ``phi_lo``/``phi_hi`` clip the randomization probability;
    ``alloc_legacy`` is an explicit opt-in that keeps Stage II on the published
    map even when Stage III uses the repaired one; ``alloc_min_n0`` suppresses borrowing at allocation
    until that many concurrent controls have accrued.
    Process-global: set it inside each worker."""
    _lib.radish_set_alloc_params(int(bool(seq_global)),
                                 float(phi_lo), float(phi_hi),
                                 int(bool(alloc_legacy)), int(alloc_min_n0))


MAPS = {"legacy": 0, "chisq": 1, "excess": 2, "median": 3, "eb": 4}


def set_borrow_params(lam_loc=1.0, lam_glb=1.0, use_tau=False, map="excess"):
    """Configure the commensurability map in the C kernel.

    Defaults use the centered excess-surprisal rule and retain the
    reference-predictive SE^2 precision scale. ``map`` selects how the conflict p-value
    kappa becomes the discount M:

      "legacy"  M = kappa                         (earlier-draft uncentered rule)
      "excess"  M = min(1, e * kappa)             centered one-nat calibration
      "median"  M = min(1, 2 * kappa)             constant-free
      "chisq"   M = exp{-lam (Z^2-1)_+/2}         needs lam_loc / lam_glb
      "eb"      variance-additive moment estimator, constant-free

    Only "chisq" reads lam_loc / lam_glb.  Process-global: set inside each
    worker.
    """
    m = MAPS[map] if isinstance(map, str) else int(map)
    _lib.radish_set_borrow_params(float(lam_loc), float(lam_glb),
                                  int(bool(use_tau)), m)


def get_borrow_params():
    """Return the process-local C borrowing configuration for parity tests."""
    lam_loc, lam_glb, use_tau, map_code = _DBL(), _DBL(), _INT(), _INT()
    _lib.radish_get_borrow_params(ct.byref(lam_loc), ct.byref(lam_glb),
                                  ct.byref(use_tau), ct.byref(map_code))
    return lam_loc.value, lam_glb.value, bool(use_tau.value), map_code.value


_lib.radish_diagnostics_batch.argtypes = [
    _PD, _INT,
    _PD, _PD, _INT,
    _PD, _PD, _INT,
    _INT, _PD, _DBL,
    _PD, _PD, _PD, _PD,
]
_lib.radish_diagnostics_batch.restype = None

_lib.radish_allocation_prob.argtypes  = [
    _PD, _PD, _PI, _INT,
    _PD, _PD, _INT,
    _INT, _PD, _PD, _PD, _DBL,
]
_lib.radish_allocation_prob.restype   = _DBL

_lib.kbcd_allocation_prob.argtypes    = [
    _PD, _PI, _INT, _INT, _PD, _PD,
]
_lib.kbcd_allocation_prob.restype     = _DBL

_lib.cahb_fit.argtypes = [
    _PD, _PD, _PD, _INT, _INT,
    _PD, _PD, _INT,
    _PD, _PD,
    _DBL, _DBL, _INT,
    _PD, _PD, _PD, _PD,
]
_lib.cahb_fit.restype  = _INT

_lib.cahb_compute_Rn.argtypes = [
    _PD, _PD, _INT, _INT, _PD, _DBL, _PD, _PD,
]
_lib.cahb_compute_Rn.restype  = _DBL

_lib.cahb_Rn_batch.argtypes = [
    _PD, _INT,
    _PD, _PD, _INT, _INT,
    _PD, _DBL, _PD, _PD,
]
_lib.cahb_Rn_batch.restype  = None

_lib.cahb_allocation_prob.argtypes = [
    _PD, _PD, _PD, _INT,
    _PD, _PD, _INT,
    _INT, _PD, _PD, _PD,
    _DBL, _DBL, _INT, _PD,
]
_lib.cahb_allocation_prob.restype  = _DBL


# ═══════════════════════════════════════════════════════════════════
# Shared helpers (mirrors methods.py)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AllocationResult:
    pi_treatment: float
    diagnostics: dict


def _as_2d(a):
    a = np.asarray(a, dtype=float)
    return a.reshape(1, -1) if a.ndim == 1 else a


def kernel_bandwidths(X_pool):
    """Identical to methods.kernel_bandwidths (single Gaussian kernel,
    Scott's rule of thumb h_j = n^{-1/(p+4)} * sigma_j on every dimension)."""
    X = _as_2d(X_pool); n, p = X.shape
    sig = np.std(X, axis=0, ddof=1); sig[sig < 1e-12] = 1.0
    f = n ** (-1.0 / (p + 4.0))
    return np.maximum(f * sig, 1e-12)


def _estimate_ate_c(Xeval, X0, Y0, X1, Y1, Xh, Yh, h, W_vec, alpha):
    """C-backed kernel g-formula Stage III; returns (ate, lo, hi, prob_pos).

    Robins (1986) g-computation with kernel-regression plug-ins; variance is
    the standard M-estimator sandwich for kernel projections (Härdle 1990
    §4.2; Wand & Jones 1995 §3.4.4).  Borrowing enters via the augmented
    control regression m̂₀(x) = (1−W) m̂₀^C(x) + W·m̂₀^H(x).
    """
    Xeval = _arr(_as_2d(Xeval))
    X0    = _arr(_as_2d(X0));   Y0 = _arr(Y0)
    X1    = _arr(_as_2d(X1));   Y1 = _arr(Y1)
    Xh    = _arr(_as_2d(Xh));   Yh = _arr(Yh)
    h     = _arr(np.atleast_1d(h)); W = _arr(W_vec)
    n_eval, p = Xeval.shape
    n0 = X0.shape[0]; n1 = X1.shape[0]; nh = Xh.shape[0]

    if n0 < 2 or n1 < 2:
        ate = float(np.nanmean(Y1) - np.nanmean(Y0))
        return ate, np.nan, np.nan, 0.5

    out_ate = np.zeros(1); out_lo = np.zeros(1); out_hi = np.zeros(1)
    out_pp  = np.zeros(1); out_V  = np.zeros(1)
    _lib.estimate_ate_c(
        Xeval, n_eval, p,
        X0, Y0, n0, X1, Y1, n1, Xh, Yh, nh,
        h, W, _DBL(float(alpha)),
        out_ate, out_lo, out_hi, out_pp, out_V,
    )
    return float(out_ate[0]), float(out_lo[0]), float(out_hi[0]), float(out_pp[0])


# ═══════════════════════════════════════════════════════════════════
# KBCD
# ═══════════════════════════════════════════════════════════════════

class KBCD:
    """Kernel-Based Covariate-Adaptive Design (no borrowing). C-backed."""
    name = "KBCD"

    def __init__(self, historical_data, scenario_params, priors):
        self.X_h = _arr(_as_2d(historical_data["X_h"]))
        self.Y_h = _arr(historical_data["Y_h"])
        self.h = _arr(kernel_bandwidths(self.X_h))
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new):
        if len(X_curr) < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "KBCD"})
        Xc = _arr(_as_2d(X_curr))
        Z  = _arr(np.asarray(Z_curr).ravel(), dtype=np.int32)
        x  = _arr(np.asarray(X_new, float).ravel())
        pi = _lib.kbcd_allocation_prob(Xc, Z, Xc.shape[0], Xc.shape[1],
                                       self.h, x)
        return AllocationResult(float(pi), {"R_n": 1.0, "method": "KBCD"})

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0, i1 = np.where(Z == 0)[0], np.where(Z == 1)[0]
        W = np.zeros(X.shape[0])
        return _estimate_ate_c(X, X[i0], Y[i0], X[i1], Y[i1],
                               self.X_h, self.Y_h, self.h, W, self.alpha)

    def compute_diagnostics(self, X, Y, Z):
        return {"mean_W": 0.0, "mean_Rn": 1.0,
                "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}


# ═══════════════════════════════════════════════════════════════════
# CAHB — C-backed coordinate ascent
# ═══════════════════════════════════════════════════════════════════

class CAHB:
    """Covariate-Adaptive Historical Borrowing (Jin et al., 2023). C-backed."""
    name = "CAHB"

    def __init__(self, historical_data, scenario_params, priors):
        self.X_h = _arr(_as_2d(historical_data["X_h"]))
        self.Y_h = _arr(historical_data["Y_h"])
        self.h = _arr(kernel_bandwidths(self.X_h))
        self.gamma   = float(priors.get("cahb_gamma", np.sqrt(3.0)))
        self.lam     = float(priors.get("cahb_lambda", 300.0))
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05
        self.max_iter = 50

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new):
        if len(X_curr) < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "CAHB"})
        Xc = _arr(_as_2d(X_curr)); n, p = Xc.shape
        Y  = _arr(Y_curr); Z = _arr(np.asarray(Z_curr, float))
        if (Z == 0).sum() < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "CAHB"})
        x  = _arr(np.asarray(X_new, float).ravel())
        pi = _lib.cahb_allocation_prob(
            Xc, Y, Z, n, self.X_h, self.Y_h, self.X_h.shape[0], p,
            self.h, self.h, self.h,
            self.gamma, self.lam, self.max_iter, x,
        )
        return AllocationResult(float(pi), {"R_n": np.nan, "method": "CAHB"})

    def _fit_arrays(self, X, Y, Z):
        Xc = _arr(_as_2d(X)); n, p = Xc.shape
        Y  = _arr(Y); Z = _arr(np.asarray(Z, float))
        if (Z == 0).sum() < 2:
            return None
        mu0    = np.zeros(n); tau = np.zeros(n)
        theta0 = np.zeros(n); phi = np.zeros(1)
        rc = _lib.cahb_fit(
            Xc, Y, Z, n, p, self.X_h, self.Y_h, self.X_h.shape[0],
            self.h, self.h, self.gamma, self.lam, self.max_iter,
            mu0, tau, phi, theta0,
        )
        if rc != 0: return None
        return dict(X=Xc, Y=Y, Z=Z, mu0=mu0, tau=tau, phi=float(phi[0]),
                    theta0=theta0, n=n, p=p)

    def estimate_treatment_effect(self, X, Y, Z):
        Xc = _arr(_as_2d(X)); Y = _arr(Y); Z = np.asarray(Z, int)
        i0 = np.where(Z == 0)[0]; i1 = np.where(Z == 1)[0]
        if len(i0) < 2 or len(i1) < 2:
            return float(Y[i1].mean() - Y[i0].mean()), np.nan, np.nan, 0.5
        m = self._fit_arrays(Xc, Y, Z)
        if m is None:
            return float(Y[i1].mean() - Y[i0].mean()), np.nan, np.nan, 0.5
        # R_n at every patient → W
        Rn = np.zeros(m["n"])
        _lib.cahb_Rn_batch(Xc, m["n"], Xc, m["Z"], m["n"], m["p"],
                           m["tau"], m["phi"], self.h, Rn)
        W = (Rn - 1.0) / np.maximum(Rn, 1e-12)
        return _estimate_ate_c(Xc, Xc[i0], Y[i0], Xc[i1], Y[i1],
                               self.X_h, self.Y_h, self.h, W, self.alpha)

    def compute_diagnostics(self, X, Y, Z):
        Xc = _arr(_as_2d(X)); Y = _arr(Y); Z = np.asarray(Z, int)
        if (Z == 0).sum() < 2 or (Z == 1).sum() < 2:
            return {"mean_W": np.nan, "mean_Rn": np.nan,
                    "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}
        m = self._fit_arrays(Xc, Y, Z)
        if m is None:
            return {"mean_W": np.nan, "mean_Rn": np.nan,
                    "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}
        Rn = np.zeros(m["n"])
        _lib.cahb_Rn_batch(Xc, m["n"], Xc, m["Z"], m["n"], m["p"],
                           m["tau"], m["phi"], self.h, Rn)
        W = (Rn - 1.0) / np.maximum(Rn, 1e-12)
        return {"mean_W": float(W.mean()), "mean_Rn": float(Rn.mean()),
                "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}


# ═══════════════════════════════════════════════════════════════════
# RADISH — C-backed PDC borrowing
# ═══════════════════════════════════════════════════════════════════

class RADISH:
    """Robust Adaptive Discrepancy-Informed Shrinkage (proposed). C-backed.

    The minimal finite-sample calibration that distinguishes RADISH from the
    earlier draft is applied here by construction:

      * the excess-surprisal commensurability map D = (-log kappa - 1)_+
        (``map="excess"``), whose one-nat offset is motivated by the ideal
        standard-normal probability-integral-transform calibration and is not
        asserted to equal the finite-sample mean of the kernel statistic.

    This introduces no tuning constant. Stage II uses the centered leaf-only
    allocation-time check and both stages retain the predictive-variance
    denominator. ``alloc_legacy=1``
    remains an explicit opt-in for reproducing the earlier uncentered rule.

    These are process-global C statics, so they are set in __init__: each
    joblib worker constructs its own RADISH and thereby configures its own
    process.  KBCD and CAHB read none of these globals, so a mixed run is
    unaffected.  To reproduce the earlier draft's rule for comparison, call
    ``set_borrow_params(1., 1., 0, "legacy")`` and ``set_adj_global(0)``
    after construction.
    """
    name = "RADISH"

    def __init__(self, historical_data, scenario_params, priors):
        set_borrow_params(1.0, 1.0, use_tau=False, map="excess")
        set_alloc_params(seq_global=False, phi_lo=0.0, phi_hi=1.0,
                         alloc_legacy=False, alloc_min_n0=0)
        set_adj_global(False)
        self.X_h = _arr(_as_2d(historical_data["X_h"]))
        self.Y_h = _arr(historical_data["Y_h"])
        self.h = _arr(kernel_bandwidths(self.X_h))
        self.nc_stab = float(priors.get("radish_nc_stabilizer", 5.0))
        self.n0_fin  = float(priors.get("radish_n0_final", 5.0))
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new):
        if len(X_curr) < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "RADISH"})
        Xc = _arr(_as_2d(X_curr))
        Y  = _arr(Y_curr)
        Z  = _arr(np.asarray(Z_curr).ravel(), dtype=np.int32)
        x  = _arr(np.asarray(X_new, float).ravel())
        pi = _lib.radish_allocation_prob(
            Xc, Y, Z, Xc.shape[0],
            self.X_h, self.Y_h, self.X_h.shape[0],
            Xc.shape[1], self.h, self.h, x, self.nc_stab,
        )
        return AllocationResult(float(pi),
                                {"R_n": np.nan, "method": "RADISH"})

    def estimate_treatment_effect(self, X, Y, Z):
        Xc = _arr(_as_2d(X)); Y = _arr(Y); Z = np.asarray(Z, int)
        i0 = np.where(Z == 0)[0]; i1 = np.where(Z == 1)[0]
        if len(i0) < 2 or len(i1) < 2:
            return float(np.nanmean(Y[i1]) - np.nanmean(Y[i0])), np.nan, np.nan, 0.5
        n = Xc.shape[0]
        Rn = np.zeros(n); W = np.zeros(n); D = np.zeros(n); T2 = np.zeros(n)
        _lib.radish_diagnostics_batch(
            Xc, n,
            _arr(Xc[i0]), _arr(Y[i0]), len(i0),
            self.X_h, self.Y_h, self.X_h.shape[0],
            Xc.shape[1], self.h, self.n0_fin,
            Rn, W, D, T2,
        )
        return _estimate_ate_c(Xc, Xc[i0], Y[i0], Xc[i1], Y[i1],
                               self.X_h, self.Y_h, self.h, W, self.alpha)

    def compute_diagnostics(self, X, Y, Z):
        Xc = _arr(_as_2d(X)); Y = _arr(Y); Z = np.asarray(Z, int)
        i0 = np.where(Z == 0)[0]
        if len(i0) < 2:
            return {"mean_W": np.nan, "mean_Rn": np.nan,
                    "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}
        n = Xc.shape[0]
        Rn = np.zeros(n); W = np.zeros(n); D = np.zeros(n); T2 = np.zeros(n)
        _lib.radish_diagnostics_batch(
            Xc, n,
            _arr(Xc[i0]), _arr(Y[i0]), len(i0),
            self.X_h, self.Y_h, self.X_h.shape[0],
            Xc.shape[1], self.h, self.n0_fin,
            Rn, W, D, T2,
        )
        return {"mean_W": float(W.mean()), "mean_Rn": float(Rn.mean()),
                "mean_Dpdc": float(D.mean()), "mean_tau2_H": float(T2.mean())}


__all__ = ["AllocationResult", "KBCD", "CAHB", "RADISH"]
