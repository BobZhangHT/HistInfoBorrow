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

def _kernel_effective_n(weights, floor=1e-6):
    """Return Kish effective sample size for nonnegative kernel weights.

    The local weighted mean remains normalised by sum(weights). Its
    sampling variance and precision use (sum w)**2 / sum(w**2).
    Degenerate or non-finite weights return a positive floor.
    """
    w = np.asarray(weights, dtype=float).ravel()
    sw = float(np.sum(w))
    sw2 = float(np.dot(w, w))
    if (not np.isfinite(sw)) or (not np.isfinite(sw2)) or sw <= _EPS or sw2 <= _EPS:
        return float(floor)
    return float(max((sw * sw) / sw2, floor))

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

def kernel_bandwidths(X_pool):
    """Single Gaussian product-kernel bandwidth by Scott's rule of thumb.

    Scott (1992): h_j = n^{-1/(p+4)} * sigma_j, applied to every covariate
    dimension (binary and continuous alike). This is the one and only
    kernel bandwidth used across Stage I/II/III and the allocation rule;
    there is no separate fitting vs. allocation kernel.
    """
    X = _as_2d(X_pool); n, p = X.shape
    sig = np.std(X, axis=0, ddof=1); sig[sig < _EPS] = 1.0
    f = n ** (-1.0 / (p + 4.0))
    return np.maximum(f * sig, _EPS)

# =====================================================================
# Unified linearized ATE (Stage III for ALL methods)
# =====================================================================

def estimate_ate(X_eval, X0, Y0, X1, Y1, X_H, Y_H, h, W_vec, alpha=0.05):
    """Kernel g-formula Stage III estimator.

    Robins (1986) g-computation in nonparametric kernel-regression form
    (Snowden et al. 2011, AJE; Vansteelandt & Keiding 2011, AJE):

        Δ̂ = (1/n_eval) Σ_i [m̂₁(X_i) − m̂₀(X_i)]

    with borrowing-augmented control regression m̂₀(x) = (1−W(x)) m̂₀^C(x) +
    W(x) m̂₀^H(x).  Algebraically equivalent to the weighted-Y form
    Δ̂ = a₁'Y₁ − a₀'Y₀ − a_H'Y_H with influence vectors
        a_z = (1/n_eval) Σ_i [W-modulated] S_z[i, ·].

    Variance: M-estimator sandwich for a kernel-regression projection
    (Härdle 1990 §4.2; Wand & Jones 1995 §3.4.4),
        V̂ = v̂₁ ‖a₁‖² + v̂₀ ‖a₀‖² + v̂_H ‖a_H‖²,
    with σ̂²_z(Y|X,Z=z) from a kernel smoother and the second-order DoF
    correction dof = n − 2 tr(S) + tr(S'S).

    Parameters
    ----------
    W_vec : (n_eval,) per-point borrowing weight (0 ⇒ KBCD; >0 ⇒ borrow).

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
    # σ̂²_z must estimate Var(Y|X,Z=z), the CONDITIONAL variance, not
    # the marginal Var(Y|Z) (which inflates the estimate by Var{μ_z(X)},
    # producing over-wide CIs and over-coverage).  Fit Ŷ = S·Y within
    # each arm; use second-order effective degrees of freedom
    #     dof = n − 2·tr(S) + tr(S'S)            (Wand & Jones 1995 §3.4.4)
    # rather than n − tr(S), which under-estimates Var(Y|X,Z) by ignoring
    # smoother curvature — the residual contains stochastic noise AND a
    # smoothing-bias component, and the additional tr(S'S) term pays for
    # the latter.  Shared with the trial-level conflict check via
    # _kernel_resid_var so both use identical residual variances.
    v1 = _kernel_resid_var(X1, Y1, h)
    v0 = _kernel_resid_var(X0, Y0, h)
    vH = _kernel_resid_var(X_H, Y_H, h)
    V = max(v1 * (a1 @ a1) + v0 * (a0 @ a0) + vH * (aH @ aH), _EPS)
    se = np.sqrt(V)
    z = norm.ppf(1.0 - alpha / 2.0)
    return ate, ate - z*se, ate + z*se, float(norm.cdf(ate/se)) if se > _EPS else 0.5

# =====================================================================
# RADISH helper: PDC discrepancy
# =====================================================================

_EXCESS_NAT = 1.0   # one-nat ideal standard-normal PIT reference offset


def _excess_surprisal(pn):
    """Excess surprisal (-log kappa - 1)_+ on the natural-log scale.

    The one-nat offset is a reference calibration motivated by the
    probability-integral transform under an ideal standard-normal
    discrepancy. It is not treated as the finite-sample expectation of the
    estimated kernel statistic. Subtracting it prevents small compatible
    fluctuations from automatically inducing discounting, so that the
    no-penalty state D=0 is attainable with positive probability. The map is
    equivalently an exponential discount exp(-D) with
    D=max(-log(kappa)-1, 0) per node.
    """
    return max(float(-np.log(max(pn, _EPS))) - _EXCESS_NAT, 0.0)


def _pdc_discrepancy(theta_H, tau2_H, ybar0, sigma2_0c=0.0, Nc=1.0,
                     excess=True):
    """Z_n, p_n, D_PDC computed under the canonical PDC standardization
    of Evans & Moshonov (2006, Example 1) and Nott et al. (2020):

        SE^2 = tau2_H + sigma2_{0,c} / n_eff
        Z_n  = (ȳ_c − θ) / SE
        p_n  = 2 min{Φ(Z), 1 − Φ(Z)}
        D_PDC = (−log p_n − 1)_+        (excess=True, default)
              = −log p_n                (excess=False, published rule)

    Including the data sampling variance sigma2_{0,c}/n_eff in the
    denominator calibrates Z to N(0,1) under H0 by the CLT, removing
    the kernel-smoothing miscalibration that inflated the empirical
    surprisal in earlier versions. The default sigma2_0c=0, n_eff=1
    recovers the original (uncalibrated) statistic.

    Production Stages II and III both use ``excess=True``.  The
    ``excess=False`` branch is retained only to reproduce the main5 rule in
    explicit audit runs.
    """
    se2 = max(tau2_H + sigma2_0c / max(Nc, 1.0), _EPS)
    se = np.sqrt(se2)
    Z = np.clip((ybar0 - theta_H) / se, -1e10, 1e10)
    pu = float(norm.cdf(Z))
    pn = max(2.0 * min(pu, 1.0 - pu), _EPS)
    D = _excess_surprisal(pn) if excess else float(-np.log(pn))
    return Z, pn, D

def _kernel_resid_var(X_arm, Y_arm, h):
    """Conditional outcome variance Var(Y | X) from kernel residuals.

    Smooths the arm's outcomes with the same kernel used everywhere else,
    subtracts the fit, and divides by the second-order effective residual
    degrees of freedom dof = n - 2 tr(S) + tr(S'S) (Wand & Jones 1995
    §3.4.4).  This is the conditional variance, not the marginal Var(Y),
    which would additionally absorb the spread of mu(X) across covariates.
    """
    na = len(Y_arm)
    if na < 3:
        return max(float(np.var(Y_arm, ddof=1)), _EPS) if na > 1 else _EPS
    S_self = normalise_rows(gaussian_weight_matrix(X_arm, X_arm, h))
    resid = Y_arm - S_self @ Y_arm
    tr_S = np.trace(S_self)
    tr_StS = float(np.sum(S_self * S_self))   # ‖S‖_F²
    dof = max(na - 2.0 * tr_S + tr_StS, 1.0)
    return max(float(np.sum(resid**2) / dof), _EPS)


def _global_pdc_surprisal(X_eval, X0, Y0, X_H, Y_H, h):
    """Centered trial-level conflict surprisal from the two control means.

    The statistic is the main5 Welch-style check, but its two-sided
    compatibility score is passed through the same centered one-nat map as
    the local statistic. The unused covariate arguments keep the Python/C
    call contract stable for audit variants.
    """
    n0 = len(Y0); nH = len(Y_H)
    if n0 < 2 or nH < 2:
        return 0.0
    m0 = float(np.mean(Y0)); s0 = float(np.var(Y0, ddof=1))
    mH = float(np.mean(Y_H)); sH = float(np.var(Y_H, ddof=1))
    se = np.sqrt(s0 / n0 + sH / nH)
    if se < _EPS:
        return 0.0
    Zg = np.clip((mH - m0) / se, -1e10, 1e10)
    pu = float(norm.cdf(Zg))
    pg = max(2.0 * min(pu, 1.0 - pu), _EPS)
    return _excess_surprisal(pg)


def _compute_borrowing_weight(theta, tau2_H, ybar, sigma2_0c, Nc, nc_floor,
                              Dg=0.0, legacy=False, use_tau=False):
    """Compute R_n and borrowing weight W = (R_n-1)/R_n.

    Local (leaf-node) PDC standardization (Evans & Moshonov 2006 §2,
    Example 1; Nott et al. 2020):

        SE^2 = tau2_H + sigma2_{0,c} / n_eff
        Z    = (ȳ_c − θ) / SE,    p_n = 2 min{Φ(Z), 1−Φ(Z)}
        D_local = (−log p_n − 1)_+

    Hierarchical PDC: combine the local leaf-node surprisal with a trial-
    level (root-node) surprisal D_global via Fisher (1925) addition on the
    −log p scale (Marshall & Spiegelhalter 2007; Presanis et al. 2013):

        D_PDC = D_local + D_global

    SE² is the reference-predictive variance of the concurrent local mean.
    The production rule retains this observable scale both for the conflict
    standardizer and for the discounted historical contribution to R_n. The
    alternative tau2_H denominator remains available only for the diagnostic
    ablation in ``reports/audit_main5_variants.py``.

    Pass Dg=0 for allocation-time (leaf-only) computation; pass the
    precomputed trial-level surprisal for Stage III.

    ``legacy`` controls only the discrepancy map. ``use_tau`` remains an audit
    switch; the production rule uses the same reference-predictive ``SE^2``
    precision scale in Stages II and III.
    """
    # Nc is the Kish effective local concurrent sample size supplied by
    # the caller, not the raw kernel mass. Keep the configured floor for
    # empty or extremely sparse local neighbourhoods.
    Nc_s = max(float(Nc) if np.isfinite(Nc) else 0.0, float(nc_floor))
    _, _, Dn = _pdc_discrepancy(theta, tau2_H, ybar, sigma2_0c, Nc_s,
                                excess=not legacy)
    gD = np.exp(Dn + Dg)
    # Production uses the reference-predictive SE^2 scale in both stages.
    den = (tau2_H if use_tau else
           max(tau2_H + sigma2_0c / Nc_s, _EPS))
    PiH = 1.0 / max(den * gD, _EPS)
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
        self.h = kernel_bandwidths(self.X_h)
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new):
        if len(X_curr) < 2:
            return AllocationResult(0.5, {"R_n": 1.0, "method": "KBCD"})
        X_curr = _as_2d(X_curr); Z = np.asarray(Z_curr, float).ravel()
        w = gaussian_weights(X_curr, X_new, self.h)
        n0 = float(w @ (1.0 - Z)); n1 = float(w @ Z)
        return AllocationResult(phi_from_counts(n0, n1),
                                {"R_n": 1.0, "method": "KBCD"})

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0, i1 = np.where(Z == 0)[0], np.where(Z == 1)[0]
        W = np.zeros(X.shape[0])
        return estimate_ate(X, X[i0], Y[i0], X[i1], Y[i1],
                            self.X_h, self.Y_h, self.h, W, self.alpha)

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
        self.h = kernel_bandwidths(self.X_h)
        self.gamma = float(priors.get("cahb_gamma", np.sqrt(3.0)))
        self.lam   = float(priors.get("cahb_lambda", 300.0))
        self.invg2 = 1.0 / max(self.gamma**2, _EPS)
        try:
            import config; self.alpha = config.ALPHA
        except ImportError:
            self.alpha = 0.05

    # ── Gaussian kernel matrix (fitting bandwidth) ──
    def _km(self, Xa, Xb):
        h = self.h
        ci = np.diag(1.0 / np.maximum(h, 1e-3)**2)
        ld = np.sum(np.log(np.maximum(h, 1e-3)**2))
        p = Xa.shape[1]
        ln = -0.5 * (p * np.log(2*np.pi) + ld)
        diff = _as_2d(Xb)[:, None, :] - _as_2d(Xa)[None, :, :]
        return np.exp(-0.5 * np.einsum("ijk,kl,ijl->ij", diff, ci, diff) + ln)

    def _hist_mean(self, Xe):
        Xe = _as_2d(Xe); out = np.zeros(Xe.shape[0])
        for i in range(Xe.shape[0]):
            w = gaussian_weights(self.X_h, Xe[i], self.h)
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
        w = gaussian_weights(m.X, X_new, self.h)
        n0 = float(m.sZ @ w); n1 = float(m.Z @ w)
        Rn = self._Rn(m, X_new)
        return AllocationResult(phi_from_counts(Rn * n0, n1),
                                {"R_n": Rn, "method": "CAHB"})

    def estimate_treatment_effect(self, X, Y, Z):
        """Stage III: kernel g-formula with CAHB variance-ratio W(x)."""
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
                            self.X_h, self.Y_h, self.h, W, self.alpha)

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
      Stage II : centered PDC discrepancy D(x) = (-log(p_n)-1)_+ drives
                  the exp(-D) discount
      Stage III: linearized ATE with data-adaptive W(x) = (R_n-1)/R_n
    """

    def __init__(self, historical_data, scenario_params, priors):
        self.X_h = _as_2d(historical_data["X_h"])
        self.Y_h = np.asarray(historical_data["Y_h"], float)
        self.name = "RADISH"
        self.h = kernel_bandwidths(self.X_h)
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
        th, t2 = radish_stage1(X_new, self.X_h, self.Y_h, self.h)
        # Stage II control summary
        w0 = gaussian_weights(_as_2d(X0), np.asarray(X_new, float).ravel(),
                              self.h) if X0.shape[0] > 0 else np.zeros(0)
        Nc_raw = float(w0.sum())
        Nc_eff = _kernel_effective_n(w0)
        if Nc_raw > _EPS:
            yb = float(w0 @ Y0 / Nc_raw)
            s2c = max(float(np.sum(w0 * (Y0 - yb)**2) / Nc_raw), _EPS)
        else:
            yb, s2c = 0.0, 1.0
        # Stage II uses the centered allocation-time discrepancy by default:
        # D_n^A = (-log(kappa_n)-1)_+, with no root-node contribution.  The
        # optional ``legacy=True`` path remains available in the helper for
        # reproducing the earlier uncentered allocation rule.
        Rn, W = _compute_borrowing_weight(th, t2, yb, s2c, Nc_eff, self.nc_stab,
                                          legacy=False, use_tau=False)
        # Allocation
        wa = gaussian_weights(X_curr, X_new, self.h)
        N0 = float(wa @ (1.0 - Z.astype(float)))
        N1 = float(wa @ Z.astype(float))
        pi = phi_from_counts(Rn * N0, N1)
        return AllocationResult(pi, {"R_n": Rn, "W": W, "D_pdc": -1,
                                     "method": "RADISH"})

    def estimate_treatment_effect(self, X, Y, Z):
        """Stage III: kernel g-formula with hierarchical-PDC W(x)."""
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0, i1 = np.where(Z == 0)[0], np.where(Z == 1)[0]
        if len(i0) < 2 or len(i1) < 2:
            return float(np.nanmean(Y[i1]) - np.nanmean(Y[i0])), np.nan, np.nan, 0.5
        X0, Y0 = X[i0], Y[i0]; h = self.h; n = X.shape[0]
        # Trial-level (root-node) PDC surprisal, computed once and folded
        # uniformly into every local D_PDC via Fisher-style addition.
        Dg = _global_pdc_surprisal(X, X0, Y0, self.X_h, self.Y_h, h)
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
            Nc_raw = float(w0.sum())
            Nc_eff = _kernel_effective_n(w0)
            if Nc_raw > _EPS:
                yb = float(w0 @ Y0 / Nc_raw)
                s2c = max(float(np.sum(w0*(Y0-yb)**2)/Nc_raw), _EPS)
            else:
                yb, s2c = 0.0, 1.0
            _, Wv[i] = _compute_borrowing_weight(th, t2, yb, s2c, Nc_eff,
                                                 self.n0_fin, Dg=Dg)
        return estimate_ate(X, X0, Y0, X[i1], Y[i1],
                            self.X_h, self.Y_h, h, Wv, self.alpha)

    def compute_diagnostics(self, X, Y, Z):
        """Mean borrowing weight, R_n, D_PDC and tau^2_H averaged over subjects."""
        X = _as_2d(X); Y = np.asarray(Y, float); Z = np.asarray(Z, int)
        i0 = np.where(Z == 0)[0]
        if len(i0) < 2:
            return {"mean_W": np.nan, "mean_Rn": np.nan,
                    "mean_Dpdc": np.nan, "mean_tau2_H": np.nan}
        X0, Y0 = X[i0], Y[i0]; h = self.h; n = X.shape[0]
        Dg = _global_pdc_surprisal(X, X0, Y0, self.X_h, self.Y_h, h)
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
            Nc_raw = float(w0.sum())
            Nc_eff = _kernel_effective_n(w0)
            if Nc_raw > _EPS:
                yb = float(w0 @ Y0 / Nc_raw)
                s2c = max(float(np.sum(w0*(Y0-yb)**2)/Nc_raw), _EPS)
            else:
                yb, s2c = 0.0, 1.0
            _, _, Dl = _pdc_discrepancy(th, t2, yb, s2c, max(Nc_eff, self.n0_fin))
            Ds[i] = Dl + Dg
            Rns[i], Ws[i] = _compute_borrowing_weight(th, t2, yb, s2c, Nc_eff,
                                                     self.n0_fin, Dg=Dg)
            T2[i] = t2
        return {"mean_W": float(np.mean(Ws)),
                "mean_Rn": float(np.mean(Rns)),
                "mean_Dpdc": float(np.mean(Ds)),
                "mean_tau2_H": float(np.mean(T2))}

# =====================================================================
__all__ = ["AllocationResult", "KBCD", "CAHB", "RADISH", "estimate_ate"]
