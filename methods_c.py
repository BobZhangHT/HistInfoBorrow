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


def fitting_bandwidths(X_pool, disc_idx=(0,), cont_idx=(1,)):
    """Identical to methods.fitting_bandwidths."""
    X = _as_2d(X_pool); n, p = X.shape
    h = np.ones(p)
    for k in disc_idx:
        h[k] = 0.1
    if cont_idx:
        Xc = X[:, list(cont_idx)]
        sig = np.std(Xc, axis=0, ddof=1); sig[sig < 1e-12] = 1.0
        f = n ** (-1.0 / (Xc.shape[1] + 4.0))
        for j, k in enumerate(cont_idx):
            h[k] = max(f * sig[j], 1e-12)
    return np.maximum(h, 1e-12)


def allocation_bandwidths(p, disc_idx=(0,), cont_idx=(1,)):
    try:
        import config as _c
        hb = float(getattr(_c, "KERNEL_BANDWIDTH_BINARY", 1.1))
        hc = float(getattr(_c, "KERNEL_BANDWIDTH_CONTINUOUS", 1.3))
    except ImportError:
        hb, hc = 1.1, 1.3
    h = np.full(p, hc)
    for k in disc_idx:
        h[k] = hb
    return h


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
        d = scenario_params.get("disc_idx", [0])
        c = scenario_params.get("cont_idx", [1])
        self.h_fit   = _arr(fitting_bandwidths(self.X_h, d, c))
        self.h_alloc = _arr(allocation_bandwidths(self.X_h.shape[1], d, c))
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
                                       self.h_alloc, x)
        return AllocationResult(float(pi), {"R_n": 1.0, "method": "KBCD"})

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0, i1 = np.where(Z == 0)[0], np.where(Z == 1)[0]
        W = np.zeros(X.shape[0])
        return _estimate_ate_c(X, X[i0], Y[i0], X[i1], Y[i1],
                               self.X_h, self.Y_h, self.h_fit, W, self.alpha)

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
        d = scenario_params.get("disc_idx", [0])
        c = scenario_params.get("cont_idx", [1])
        self.h_fit   = _arr(fitting_bandwidths(self.X_h, d, c))
        self.h_alloc = _arr(allocation_bandwidths(self.X_h.shape[1], d, c))
        self.gamma   = float(priors.get("cahb_gamma", np.sqrt(3.0)))
        self.lam     = float(priors.get("cahb_lambda", 300.0))
        # Silverman bandwidth for hist mean
        n, p = self.X_h.shape
        s = np.std(self.X_h, axis=0, ddof=1); s[s < 1e-12] = 1.0
        f = (4.0/(p+2.0))**(1.0/(p+4.0)) * n**(-1.0/(p+4.0))
        self.h_hist = _arr(np.maximum(f * s, 1e-12))
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
            self.h_fit, self.h_hist, self.h_alloc,
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
            self.h_fit, self.h_hist, self.gamma, self.lam, self.max_iter,
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
                           m["tau"], m["phi"], self.h_fit, Rn)
        W = (Rn - 1.0) / np.maximum(Rn, 1e-12)
        return _estimate_ate_c(Xc, Xc[i0], Y[i0], Xc[i1], Y[i1],
                               self.X_h, self.Y_h, self.h_fit, W, self.alpha)

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
                           m["tau"], m["phi"], self.h_fit, Rn)
        W = (Rn - 1.0) / np.maximum(Rn, 1e-12)
        return {"mean_W": float(W.mean()), "mean_Rn": float(Rn.mean()),
                "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}


# ═══════════════════════════════════════════════════════════════════
# RADISH — C-backed PDC borrowing
# ═══════════════════════════════════════════════════════════════════

class RADISH:
    """Robust Adaptive Discrepancy-Informed Shrinkage (proposed). C-backed."""
    name = "RADISH"

    def __init__(self, historical_data, scenario_params, priors):
        self.X_h = _arr(_as_2d(historical_data["X_h"]))
        self.Y_h = _arr(historical_data["Y_h"])
        d = scenario_params.get("disc_idx", [0])
        c = scenario_params.get("cont_idx", [1])
        self.h_fit   = _arr(fitting_bandwidths(self.X_h, d, c))
        self.h_alloc = _arr(allocation_bandwidths(self.X_h.shape[1], d, c))
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
            Xc.shape[1], self.h_fit, self.h_alloc, x, self.nc_stab,
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
            Xc.shape[1], self.h_fit, self.n0_fin,
            Rn, W, D, T2,
        )
        return _estimate_ate_c(Xc, Xc[i0], Y[i0], Xc[i1], Y[i1],
                               self.X_h, self.Y_h, self.h_fit, W, self.alpha)

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
            Xc.shape[1], self.h_fit, self.n0_fin,
            Rn, W, D, T2,
        )
        return {"mean_W": float(W.mean()), "mean_Rn": float(Rn.mean()),
                "mean_Dpdc": float(D.mean()), "mean_tau2_H": float(T2.mean())}


__all__ = ["AllocationResult", "KBCD", "CAHB", "RADISH"]
