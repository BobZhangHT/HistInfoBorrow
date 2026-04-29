"""
methods.py — KBCD, CAHB, and RADISH implementations
=====================================================
Three covariate-adaptive randomisation methods for two-arm trials
with optional historical control borrowing.

Stage II  : adaptive allocation (method-specific R_n)
Stage III : unified linearized ATE estimator for all methods

    Delta_hat = a1'Y1 - a0'Y0 - aH'Y_H
    V_hat     = sigma1^2 ||a1||^2 + sigma0^2 ||a0||^2 + sigmaH^2 ||aH||^2
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple
import numpy as np
from scipy.stats import norm

_EPS = 1e-12

# =====================================================================
# Data container
# =====================================================================

@dataclass
class AllocationResult:
    """Returned by get_allocation_prob()."""
    pi_treatment: float
    diagnostics: dict

# =====================================================================
# Kernel utilities
# =====================================================================

def _as_2d(arr):
    a = np.asarray(arr, dtype=float)
    return a.reshape(1, -1) if a.ndim == 1 else a

def gaussian_weights(X_ref, x_star, h):
    """1-D weights: K_h(X_ref[i] - x*) for Gaussian product kernel."""
    X_ref = _as_2d(X_ref)
    x = np.asarray(x_star, float).ravel()
    diff = (X_ref - x) / np.asarray(h, float)
    return np.exp(-0.5 * np.sum(diff**2, axis=1))

def gaussian_weight_matrix(X_ref, X_eval, h):
    """Batch kernel: returns (n_eval, n_ref) matrix."""
    Xr = _as_2d(X_ref); Xe = _as_2d(X_eval)
    h = np.asarray(h, float)
    d = Xr[None, :, :] - Xe[:, None, :]
    return np.exp(-0.5 * np.sum((d / h)**2, axis=2))

def normalise_rows(W):
    return W / np.maximum(W.sum(axis=1, keepdims=True), _EPS)

def epanechnikov_weights(X_ref, x_star, h):
    """Product Epanechnikov kernel (used for allocation)."""
    X_ref = _as_2d(X_ref)
    x = np.asarray(x_star, float).ravel()
    u = (X_ref - x) / np.maximum(np.asarray(h, float), _EPS)
    k = np.maximum(0.0, 1.0 - u**2) * 0.75
    return np.prod(k, axis=1)

def fitting_bandwidths(X_pool, disc_idx=(0,), cont_idx=(1,)):
    """Stage I/III fitting: binary h=0.1, continuous via Scott's ROT."""
    X = _as_2d(X_pool); n, p = X.shape
    h = np.ones(p)
    for k in disc_idx:
        h[k] = 0.1
    if cont_idx:
        Xc = X[:, list(cont_idx)]
        sig = np.std(Xc, axis=0, ddof=1); sig[sig < _EPS] = 1.0
        f = n ** (-1.0 / (Xc.shape[1] + 4.0))
        for j, k in enumerate(cont_idx):
            h[k] = max(f * sig[j], _EPS)
    return np.maximum(h, _EPS)

def allocation_bandwidths(p, disc_idx=(0,), cont_idx=(1,)):
    """Broad allocation kernel: binary 1.1, continuous 1.3."""
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

# =====================================================================
# Unified linearized ATE (Stage III for ALL methods)
# =====================================================================

def estimate_ate(X_eval, X0, Y0, X1, Y1, X_H, Y_H, h, W_vec, alpha=0.05):
    """
    Linearized ATE estimator with borrowing weight vector W.

    Parameters
    ----------
    W_vec : (n_eval,) array
        Borrowing weight at each evaluation point.
        W=0 means no borrowing (KBCD), W>0 means borrowing (CAHB/RADISH).

    Returns
    -------
    (ate, ci_low, ci_high, prob_positive)
    """
    X_eval = _as_2d(X_eval)
    n_eval = X_eval.shape[0]
    n0, n1, nH = len(Y0), len(Y1), len(Y_H)
    if n0 < 2 or n1 < 2:
        return float(np.nanmean(Y1) - np.nanmean(Y0)), np.nan, np.nan, 0.5

    # Normalised kernel weight matrices
    S0 = normalise_rows(gaussian_weight_matrix(X0, X_eval, h))
    S1 = normalise_rows(gaussian_weight_matrix(X1, X_eval, h))
    SH = normalise_rows(gaussian_weight_matrix(X_H, X_eval, h))

    # Modulate by borrowing weight
    W = np.asarray(W_vec, float).ravel()[:, None]
    a1 = np.mean(S1, axis=0)
    a0 = np.mean((1.0 - W) * S0, axis=0)
    aH = np.mean(W * SH, axis=0)

    # Point estimate
    ate = float(a1 @ Y1 - a0 @ Y0 - aH @ Y_H)

    # ── Residual variance σ̂²(Y|X,Z) via kernel regression ──────────
    # The paper's "within-arm sample variances" σ̂₁², σ̂₀², σ̂_H² must be
    # the CONDITIONAL variance Var(Y|X,Z), NOT the marginal Var(Y|Z).
    # Using np.var(Y,ddof=1) inflates the estimate by Var(μ(X)) — the
    # covariate-driven mean variation — producing CIs that are 20%+ too
    # wide, coverage > 95%, and Type-I error << 0.05.
    #
    # Fix: fit a kernel smoother Ŷ = S·Y within each arm and estimate
    # σ̂² = ||Y - SY||² / (n - tr(S)), where tr(S) accounts for the
    # effective degrees of freedom consumed by the smoother.
    def _resid_var(X_arm, Y_arm):
        na = len(Y_arm)
        if na < 3:
            return max(float(np.var(Y_arm, ddof=1)), _EPS) if na > 1 else _EPS
        S_self = normalise_rows(gaussian_weight_matrix(X_arm, X_arm, h))
        resid = Y_arm - S_self @ Y_arm
        tr_S = np.trace(S_self)
        dof = max(na - tr_S, 1.0)
        return max(float(np.sum(resid**2) / dof), _EPS)

    v1 = _resid_var(X1, Y1)
    v0 = _resid_var(X0, Y0)
    vH = _resid_var(X_H, Y_H)
    V = max(v1 * (a1 @ a1) + v0 * (a0 @ a0) + vH * (aH @ aH), _EPS)
    se = np.sqrt(V)
    z = norm.ppf(1.0 - alpha / 2.0)
    return ate, ate - z*se, ate + z*se, float(norm.cdf(ate/se)) if se > _EPS else 0.5

# =====================================================================
# RADISH helper: PDC discrepancy
# =====================================================================

def _pdc_discrepancy(theta_H, tau2_H, ybar0):
    """Z_n, p_n, D_PDC = -log(p_n)."""
    tau = np.sqrt(max(tau2_H, _EPS))
    Z = np.clip((ybar0 - theta_H) / tau, -1e10, 1e10)
    pu = float(norm.cdf(Z))
    pn = max(2.0 * min(pu, 1.0 - pu), _EPS)
    return Z, pn, float(-np.log(pn))

def _compute_borrowing_weight(theta, tau2_H, ybar, sigma2_0c, Nc, nc_floor):
    """Compute R_n and borrowing weight W = (R_n-1)/R_n.

    Implements eq. (7) of the paper:
        R_n(x) = 1 + sigma2_{0,c}(x) / [N_c(x) * tau2_H(x) * exp(D_PDC(x))]
    """
    Nc_s = max(Nc, nc_floor)
    _, _, Dn = _pdc_discrepancy(theta, tau2_H, ybar)
    gD = np.exp(Dn)
    PiH = 1.0 / max(tau2_H * gD, _EPS)
    PiC = max(Nc_s / sigma2_0c, _EPS)
    Rn = max(1.0, min(1.0 + PiH / PiC, 1e4))
    return Rn, (Rn - 1.0) / Rn

def radish_stage1(x, X_H, Y_H, h):
    """Stage I: local historical summary theta(x), tau2_H(x)."""
    wH = gaussian_weights(_as_2d(X_H), np.asarray(x, float).ravel(), h)
    swH = wH.sum()
    if swH <= _EPS:
        return float(np.mean(Y_H)), 1e6
    th = float(wH @ Y_H / swH)
    s2 = max(float(np.sum(wH * (Y_H - th)**2) / swH), _EPS)
    neff = max(swH**2 / max(float(wH @ wH), _EPS), 1e-6)
    return th, max(s2 / neff, _EPS)

# =====================================================================
# Neyman-optimal allocation
# =====================================================================

def phi_from_counts(n0_eff, n1_eff):
    """phi = N0*^2 / (N0*^2 + N1^2), clipped to [0,1]."""
    n0 = max(n0_eff, _EPS); n1 = max(n1_eff, _EPS)
    return float(np.clip(n0**2 / (n0**2 + n1**2), 0.0, 1.0))

# =====================================================================
# KBCD — no borrowing baseline
# =====================================================================

class KBCD:
    """Kernel-Based Covariate-Adaptive Design (Jiang et al., 2018).
    R_n(x) = 1;  allocation balances local arm sizes only."""

    def __init__(self, historical_data, scenario_params, priors):
        self.X_h = _as_2d(historical_data["X_h"])
        self.Y_h = np.asarray(historical_data["Y_h"], float)
        self.name = "KBCD"
        d = scenario_params.get("disc_idx", [0])
        c = scenario_params.get("cont_idx", [1])
        self.h_fit = fitting_bandwidths(self.X_h, d, c)
        self.h_alloc = allocation_bandwidths(self.X_h.shape[1], d, c)
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new):
        if len(X_curr) < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "KBCD"})
        X_curr = _as_2d(X_curr); Z = np.asarray(Z_curr, float).ravel()
        w = epanechnikov_weights(X_curr, X_new, self.h_alloc)
        n0 = float(w @ (1.0 - Z)); n1 = float(w @ Z)
        return AllocationResult(phi_from_counts(n0, n1),
                                {"R_n": 1.0, "method": "KBCD"})

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0, i1 = np.where(Z == 0)[0], np.where(Z == 1)[0]
        W = np.zeros(X.shape[0])
        return estimate_ate(X, X[i0], Y[i0], X[i1], Y[i1],
                            self.X_h, self.Y_h, self.h_fit, W, self.alpha)

    def compute_diagnostics(self, X, Y, Z):
        """Borrowing diagnostics: KBCD never borrows."""
        return {"mean_W": 0.0, "mean_Rn": 1.0, "mean_Dpdc": np.nan,
                "mean_tau2_H": np.nan}

# =====================================================================
# CAHB — Jin et al. (2023) coordinate-ascent borrowing
# =====================================================================

class _CAHBModel:
    """Internal: fitted CAHB model context."""
    __slots__ = ("X", "Y", "Z", "sZ", "KM", "theta0",
                 "mu0", "tau", "phi0", "phi0_sq")

class CAHB:
    """Covariate-Adaptive Historical Borrowing (Jin et al., 2023).
    R_n(x) = Var_ref / Var_borrow via coordinate-ascent model."""

    def __init__(self, historical_data, scenario_params, priors):
        self.X_h = _as_2d(historical_data["X_h"])
        self.Y_h = np.asarray(historical_data["Y_h"], float)
        self.name = "CAHB"
        d = scenario_params.get("disc_idx", [0])
        c = scenario_params.get("cont_idx", [1])
        self.h_fit = fitting_bandwidths(self.X_h, d, c)
        self.h_alloc = allocation_bandwidths(self.X_h.shape[1], d, c)
        self.gamma = float(priors.get("cahb_gamma", np.sqrt(3.0)))
        self.lam   = float(priors.get("cahb_lambda", 300.0))
        self.invg2 = 1.0 / max(self.gamma**2, _EPS)
        # Silverman bandwidth for historical kernel mean
        n, p = self.X_h.shape
        s = np.std(self.X_h, axis=0, ddof=1); s[s < _EPS] = 1.0
        f = (4/(p+2))**(1/(p+4)) * n**(-1/(p+4))
        self.h_hist = np.maximum(f * s, _EPS)
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05

    # ── Gaussian kernel matrix (fitting bandwidth) ──
    def _km(self, Xa, Xb):
        h = self.h_fit
        ci = np.diag(1.0 / np.maximum(h, 1e-3)**2)
        ld = np.sum(np.log(np.maximum(h, 1e-3)**2))
        p = Xa.shape[1]
        ln = -0.5 * (p * np.log(2*np.pi) + ld)
        diff = _as_2d(Xb)[:, None, :] - _as_2d(Xa)[None, :, :]
        return np.exp(-0.5 * np.einsum("ijk,kl,ijl->ij", diff, ci, diff) + ln)

    def _hist_mean(self, Xe):
        Xe = _as_2d(Xe); out = np.zeros(Xe.shape[0])
        for i in range(Xe.shape[0]):
            w = gaussian_weights(self.X_h, Xe[i], self.h_hist)
            s = w.sum()
            out[i] = float(w @ self.Y_h / s) if s > _EPS else 0.0
        return out

    # ── Coordinate ascent ──
    def _fit(self, X, Y, Z):
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, float)
        sZ = 1.0 - Z
        if sZ.sum() < 2:
            return None
        KM = self._km(X, X)
        th0 = self._hist_mean(X)     # actual historical theta0 (may be biased)

        # ── lambda_trunc: R code Algorithm 2 (lam.sel.fn, utils.R L837-848) ──
        # Step 1: fit mu0 from CURRENT control only (no borrowing, theta0=0)
        mu0_self = self._mu0(KM, Y, Z, np.zeros_like(Y), 1.0, np.zeros_like(Y))
        # Step 2: use mu0_self as "null theta0" → compute RAW tau (pre-projection)
        tau_null_raw = self._tau_raw(KM, Z, mu0_self, mu0_self)
        tf = tau_null_raw[np.isfinite(tau_null_raw)]
        lt = float(np.quantile(tf, 0.10)) if tf.size else 0.0

        # ── Main coordinate ascent (uses actual historical th0) ──
        tau = np.zeros(len(Y))
        phi = max(np.std(Y[Z == 0], ddof=1), 1.0) if (Z == 0).sum() > 1 else 1.0
        for _ in range(50):
            mu0 = self._mu0(KM, Y, Z, tau, phi, th0)
            phi_n = self._phi(Y, Z, mu0)
            tau_n = self._tau(KM, Z, mu0, th0, lt)
            if np.mean((mu0 - self._mu0(KM, Y, Z, tau, phi, th0))**2) < 1e-5:
                phi, tau = phi_n, tau_n; break
            phi, tau = phi_n, tau_n
        m = _CAHBModel()
        m.X = X; m.Y = Y; m.Z = Z; m.sZ = sZ; m.KM = KM
        m.theta0 = th0; m.mu0 = mu0; m.tau = tau
        m.phi0 = phi; m.phi0_sq = phi**2
        return m

    def _mu0(self, KM, Y, Z, tau, phi, th0):
        sZ = 1.0 - Z
        W = 0.5 / phi**2 + 0.5 * tau
        M = Y / (2*phi**2) + 0.5 * tau * th0
        wk = KM * sZ[:, None]
        num = np.nansum(wk * M[:, None], axis=0)
        den = np.nansum(wk * W[:, None], axis=0)
        out = np.zeros_like(num); mask = den > _EPS
        out[mask] = num[mask] / den[mask]; return out

    def _phi(self, Y, Z, mu0):
        sZ = 1.0 - Z; r = Y - mu0
        return float(np.sqrt(max((0.5*sZ@(r**2)+0.01)/(1.01+0.5*sZ.sum()), _EPS)))

    def _tau_raw(self, KM, Z, mu0, th0):
        """Raw tau values BEFORE truncation and L1 projection (for lambda_trunc calibration)."""
        sZ = 1.0 - Z; dsq = (mu0 - th0)**2
        wk = KM * sZ[:, None]
        num = np.nansum(wk, axis=0)
        den = np.nansum(wk * dsq[:, None], axis=0) + self.invg2
        return np.maximum(num / np.maximum(den, _EPS), 0.0)

    def _tau(self, KM, Z, mu0, th0, lt):
        sZ = 1.0 - Z; dsq = (mu0 - th0)**2
        wk = KM * sZ[:, None]
        num = np.nansum(wk, axis=0)
        den = np.nansum(wk * dsq[:, None], axis=0) + self.invg2
        raw = np.maximum(num / np.maximum(den, _EPS), 0.0)
        raw[raw <= lt] = 0.0
        # L1 projection
        r = self.lam * max(np.log(max(len(Z), 1)), 1.0)
        if np.isfinite(r) and raw.sum() > r:
            u = np.sort(raw)[::-1]; cs = np.cumsum(u)
            idx = np.nonzero(u * np.arange(1, len(u)+1) > (cs - r))[0]
            if idx.size:
                raw = np.maximum(raw - (cs[idx[-1]] - r)/(idx[-1]+1), 0.0)
        return raw

    # ── Posterior variance for R_n ──
    def _var_mu0(self, m, Xe, use_tau=True):
        Xe = _as_2d(Xe); n = Xe.shape[0]
        tau = m.tau if use_tau else np.zeros_like(m.tau)
        KM = self._km(Xe, m.X)
        rv = np.nansum(KM * m.sZ[:, None] * (1.0/m.phi0_sq + tau[:, None]), axis=0)
        out = np.full(n, 10.0 * m.phi0_sq)
        mask = rv > _EPS; out[mask] = 1.0 / rv[mask]
        return np.minimum(out, 10.0 * m.phi0_sq)

    def _Rn(self, m, x):
        vr = self._var_mu0(m, np.atleast_2d(x), False)[0]
        vb = self._var_mu0(m, np.atleast_2d(x), True)[0]
        if not np.isfinite(vr) or not np.isfinite(vb) or vb <= _EPS:
            return 1.0
        return float(np.clip(vr / max(vb, _EPS), 1.0, 1e4))

    # ── Public API ──
    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new):
        if len(X_curr) < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "CAHB"})
        m = self._fit(X_curr, Y_curr, np.asarray(Z_curr, float))
        if m is None:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "CAHB"})
        w = epanechnikov_weights(m.X, X_new, self.h_alloc)
        n0 = float(m.sZ @ w); n1 = float(m.Z @ w)
        Rn = self._Rn(m, X_new)
        return AllocationResult(phi_from_counts(Rn * n0, n1),
                                {"R_n": Rn, "method": "CAHB"})

    def estimate_treatment_effect(self, X, Y, Z):
        """Stage III: linearized ATE with CAHB variance-ratio W(x)."""
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        if (Z == 0).sum() < 2 or (Z == 1).sum() < 2:
            return float(Y[Z==1].mean() - Y[Z==0].mean()), np.nan, np.nan, 0.5
        m = self._fit(X, Y, Z.astype(float))
        if m is None:
            return float(Y[Z==1].mean() - Y[Z==0].mean()), np.nan, np.nan, 0.5
        n = X.shape[0]; W = np.zeros(n)
        for i in range(n):
            Rn = self._Rn(m, X[i])
            W[i] = (Rn - 1.0) / max(Rn, 1.0)
        i0, i1 = np.where(Z == 0)[0], np.where(Z == 1)[0]
        return estimate_ate(X, X[i0], Y[i0], X[i1], Y[i1],
                            self.X_h, self.Y_h, self.h_fit, W, self.alpha)

    def compute_diagnostics(self, X, Y, Z):
        """Mean borrowing weight & R_n averaged over enrolled subjects."""
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        if (Z == 0).sum() < 2 or (Z == 1).sum() < 2:
            return {"mean_W": np.nan, "mean_Rn": np.nan,
                    "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}
        m = self._fit(X, Y, Z.astype(float))
        if m is None:
            return {"mean_W": np.nan, "mean_Rn": np.nan,
                    "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}
        Rns = np.array([self._Rn(m, X[i]) for i in range(X.shape[0])])
        Ws = (Rns - 1.0) / np.maximum(Rns, _EPS)
        return {"mean_W": float(np.mean(Ws)),
                "mean_Rn": float(np.mean(Rns)),
                "mean_Dpdc": np.nan,           # CAHB has no PDC metric
                "mean_tau2_H": np.nan}         # CAHB has no explicit tau^2_H

# =====================================================================
# RADISH — PDC-based robust adaptive borrowing (proposed)
# =====================================================================

class RADISH:
    """
    Proposed method.  Key innovations:
      Stage I  : local historical prior {theta(x), tau2_H(x)}
      Stage II : PDC discrepancy D(x) = -log(p_n) drives exp(-D) discount
      Stage III: linearized ATE with data-adaptive W(x) = (R_n-1)/R_n
    """

    def __init__(self, historical_data, scenario_params, priors):
        self.X_h = _as_2d(historical_data["X_h"])
        self.Y_h = np.asarray(historical_data["Y_h"], float)
        self.name = "RADISH"
        d = scenario_params.get("disc_idx", [0])
        c = scenario_params.get("cont_idx", [1])
        self.h_fit = fitting_bandwidths(self.X_h, d, c)
        self.h_alloc = allocation_bandwidths(self.X_h.shape[1], d, c)
        self.nc_stab = float(priors.get("radish_nc_stabilizer", 5.0))
        self.n0_fin  = float(priors.get("radish_n0_final", 5.0))
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new):
        if len(X_curr) < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "RADISH"})
        X_curr = _as_2d(X_curr)
        Z = np.asarray(Z_curr, int).ravel()
        Y = np.asarray(Y_curr, float).ravel()
        i0 = np.where(Z == 0)[0]
        X0 = X_curr[i0] if i0.size else np.zeros((0, X_curr.shape[1]))
        Y0 = Y[i0] if i0.size else np.zeros(0)
        # Stage I
        th, t2 = radish_stage1(X_new, self.X_h, self.Y_h, self.h_fit)
        # Stage II control summary
        w0 = gaussian_weights(_as_2d(X0), np.asarray(X_new, float).ravel(),
                              self.h_fit) if X0.shape[0] > 0 else np.zeros(0)
        Nc = float(w0.sum())
        if Nc > _EPS:
            yb = float(w0 @ Y0 / Nc)
            s2c = max(float(np.sum(w0 * (Y0 - yb)**2) / Nc), _EPS)
        else:
            yb, s2c = 0.0, 1.0
        Rn, W = _compute_borrowing_weight(th, t2, yb, s2c, Nc, self.nc_stab)
        # Allocation
        wa = epanechnikov_weights(X_curr, X_new, self.h_alloc)
        N0 = float(wa @ (1.0 - Z.astype(float)))
        N1 = float(wa @ Z.astype(float))
        pi = phi_from_counts(Rn * N0, N1)
        return AllocationResult(pi, {"R_n": Rn, "W": W, "D_pdc": -1,
                                     "method": "RADISH"})

    def estimate_treatment_effect(self, X, Y, Z):
        """Stage III: linearized ATE with PDC-based W(x)."""
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0, i1 = np.where(Z == 0)[0], np.where(Z == 1)[0]
        if len(i0) < 2 or len(i1) < 2:
            return float(np.nanmean(Y[i1]) - np.nanmean(Y[i0])), np.nan, np.nan, 0.5
        X0, Y0 = X[i0], Y[i0]; h = self.h_fit; n = X.shape[0]
        # Batch historical kernel for speed
        KH = gaussian_weight_matrix(self.X_h, X, h)
        Wv = np.zeros(n)
        for i in range(n):
            wH = KH[i]; swH = wH.sum()
            if swH > _EPS:
                th = float(wH @ self.Y_h / swH)
                s2H = max(float(np.sum(wH*(self.Y_h - th)**2)/swH), _EPS)
                ne = max(swH**2 / max(float(wH @ wH), _EPS), 1e-6)
                t2 = max(s2H / ne, _EPS)
            else:
                th = float(np.mean(self.Y_h)); t2 = 1e6
            w0 = gaussian_weights(X0, X[i], h)
            Nc = float(w0.sum())
            if Nc > _EPS:
                yb = float(w0 @ Y0 / Nc)
                s2c = max(float(np.sum(w0*(Y0-yb)**2)/Nc), _EPS)
            else:
                yb, s2c = 0.0, 1.0
            _, Wv[i] = _compute_borrowing_weight(th, t2, yb, s2c, Nc, self.n0_fin)
        return estimate_ate(X, X0, Y0, X[i1], Y[i1],
                            self.X_h, self.Y_h, h, Wv, self.alpha)

    def compute_diagnostics(self, X, Y, Z):
        """Mean borrowing weight, R_n, D_PDC and tau^2_H averaged over subjects."""
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0 = np.where(Z == 0)[0]
        if len(i0) < 2:
            return {"mean_W": np.nan, "mean_Rn": np.nan,
                    "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}
        X0, Y0 = X[i0], Y[i0]; h = self.h_fit; n = X.shape[0]
        KH = gaussian_weight_matrix(self.X_h, X, h)
        Rns = np.zeros(n); Ws = np.zeros(n); Ds = np.zeros(n); T2 = np.zeros(n)
        for i in range(n):
            wH = KH[i]; swH = wH.sum()
            if swH > _EPS:
                th = float(wH @ self.Y_h / swH)
                s2H = max(float(np.sum(wH*(self.Y_h - th)**2)/swH), _EPS)
                ne = max(swH**2 / max(float(wH @ wH), _EPS), 1e-6)
                t2 = max(s2H / ne, _EPS)
            else:
                th = float(np.mean(self.Y_h)); t2 = 1e6
            w0 = gaussian_weights(X0, X[i], h)
            Nc = float(w0.sum())
            if Nc > _EPS:
                yb = float(w0 @ Y0 / Nc)
                s2c = max(float(np.sum(w0*(Y0-yb)**2)/Nc), _EPS)
            else:
                yb, s2c = 0.0, 1.0
            _, _, Ds[i] = _pdc_discrepancy(th, t2, yb)
            Rns[i], Ws[i] = _compute_borrowing_weight(th, t2, yb, s2c, Nc, self.n0_fin)
            T2[i] = t2
        return {"mean_W": float(np.mean(Ws)),
                "mean_Rn": float(np.mean(Rns)),
                "mean_Dpdc": float(np.mean(Ds)),
                "mean_tau2_H": float(np.mean(T2))}

# =====================================================================
__all__ = ["AllocationResult", "KBCD", "CAHB", "RADISH", "estimate_ate"]
