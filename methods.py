"""
methods.py

Implementation of the covariate-adaptive methods evaluated in the
CAHB-PP simulation study (Sec. 3.3 of CAHB_PP(2).pdf):

- KBCD        : Jiang et al. (2018) kernel-based biased coin design.
- CAHB        : Jin et al. (2023) covariate-adjusted historical borrowing.
- CAHB_PP     : Proposed CAHB with local power prior discount a(x).
- rMAP_KBCD   : Schmidli et al. (2014) robust MAP prior combined with KBCD.

Each method implements
    • get_allocation_prob ─ adaptive randomisation for a new subject.
    • estimate_treatment_effect ─ final analysis of the completed trial.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

@dataclass
class LocalStats:
    """Weighted local sufficient statistics."""

    weight: float
    mean: float
    variance: float

    @property
    def precision(self) -> float:
        if self.weight <= 1e-8 or not np.isfinite(self.variance):
            return 0.0
        return self.weight / max(self.variance, 1e-8)


def _as_2d(array: Sequence[float]) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _silverman_bandwidth(X: np.ndarray) -> np.ndarray:
    """
    Silverman's rule-of-thumb bandwidth for product Gaussian kernels.

    h_k = (4 / (p + 2))^(1 / (p + 4)) * n^(-1 / (p + 4)) * sigma_k
    """
    X = _as_2d(X)
    n, p = X.shape
    sigma = np.std(X, axis=0, ddof=1)
    sigma[sigma < 1e-6] = 1.0  # Guard against zero variance coordinates
    factor = (4.0 / (p + 2.0)) ** (1.0 / (p + 4.0)) * n ** (-1.0 / (p + 4.0))
    bandwidth = factor * sigma
    bandwidth[bandwidth < 1e-6] = 1.0
    return bandwidth


def _gaussian_kernel_weights(x_new: np.ndarray, X_ref: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Product Gaussian kernel weights with bandwidth vector h."""
    if X_ref.shape[0] == 0:
        return np.zeros(0)

    x_new = _as_2d(x_new)
    diff = (X_ref - x_new) / h
    norm_sq = np.sum(diff ** 2, axis=1)
    weights = np.exp(-0.5 * norm_sq)
    return weights


def _compute_local_stats(
    X_ref: np.ndarray,
    values: Optional[np.ndarray],
    x_eval: np.ndarray,
    bandwidth: np.ndarray,
) -> LocalStats:
    """Return weighted sample size, mean and variance around x_eval."""
    weights = _gaussian_kernel_weights(x_eval, X_ref, bandwidth)
    weight_sum = weights.sum()

    if weight_sum <= 1e-8 or values is None:
        return LocalStats(weight=float(weight_sum), mean=np.nan, variance=np.nan)

    values = np.asarray(values, dtype=float)
    local_mean = np.dot(weights, values) / weight_sum
    variance = np.dot(weights, (values - local_mean) ** 2) / weight_sum
    return LocalStats(weight=float(weight_sum), mean=float(local_mean), variance=float(variance))


def _normal_interval(mean: float, std: float, alpha: float) -> Tuple[float, float]:
    """Symmetric normal-theory confidence interval."""
    if not np.isfinite(std) or std <= 0.0:
        return np.nan, np.nan
    z = norm.ppf(1.0 - alpha / 2.0)
    return mean - z * std, mean + z * std


def _delta_summary(delta_samples: np.ndarray, alpha: float) -> Tuple[float, float, float, float]:
    """
    Summary statistics for posterior/plug-in treatment effect.

    Returns (delta_hat, ci_low, ci_high, prob_gt_0)
    """
    delta_samples = np.asarray(delta_samples, dtype=float)
    valid = np.isfinite(delta_samples)
    if valid.sum() == 0:
        return np.nan, np.nan, np.nan, 0.5

    delta_hat = float(delta_samples[valid].mean())
    if valid.sum() < 2:
        return delta_hat, np.nan, np.nan, 0.5

    sample_var = np.var(delta_samples[valid], ddof=1)
    std = np.sqrt(max(sample_var, 1e-8) / valid.sum())
    ci_low, ci_high = _normal_interval(delta_hat, std, alpha)
    if not np.isfinite(std) or std <= 0:
        prob = 0.5
    else:
        prob = float(norm.sf((0.0 - delta_hat) / std))

    return delta_hat, ci_low, ci_high, prob


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class BaseMethod:
    """Common interface for all covariate-adaptive methods."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        self.X_h = _as_2d(historical_data["X_h"])
        self.Y_h = np.asarray(historical_data["Y_h"], dtype=float)
        self.scenario = scenario_params
        self.priors = priors
        self.h_hist = _silverman_bandwidth(self.X_h)

        try:
            import config as _sim_config  # noqa: WPS433 (import within function)

            self.alpha = float(getattr(_sim_config, "ALPHA", 0.05))
        except ImportError:
            self.alpha = 0.05

        self.name = "Base"

    # The public interface -------------------------------------------------
    def get_allocation_prob(
        self,
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
        X_new: np.ndarray,
    ) -> float:
        """Return Pr(Z_new = 1 | current data, X_new)."""
        raise NotImplementedError

    def estimate_treatment_effect(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Final analysis returning (delta_hat, ci_low, ci_high, prob_gt_0)."""
        raise NotImplementedError

    # Shared helpers -------------------------------------------------------
    @staticmethod
    def _imbalance_probability(n0_eff: float, n1_eff: float) -> float:
        """Equation (2) from Jiang et al. (2018) expressed via effective sample sizes."""
        n0_eff = max(float(n0_eff), 1e-6)
        n1_eff = max(float(n1_eff), 1e-6)
        total = n0_eff + n1_eff
        g0 = np.clip(n0_eff / total, 1e-6, 1.0 - 1e-6)
        g1 = np.clip(n1_eff / total, 1e-6, 1.0 - 1e-6)

        inv_g0 = 1.0 / g0
        inv_g1 = 1.0 / g1
        denominator = inv_g0 + inv_g1 - 2.0
        if denominator <= 1e-8:
            return 0.5
        pi_1 = (inv_g0 - 1.0) / denominator
        return float(np.clip(pi_1, 0.0, 1.0))

    @staticmethod
    def _kernel_mean(X_src: np.ndarray, Y_src: np.ndarray, bandwidth: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
        """Vector of kernel regression predictions."""
        if X_src.shape[0] == 0:
            return np.full(X_eval.shape[0], np.nan)
        preds = []
        for x in X_eval:
            weights = _gaussian_kernel_weights(x, X_src, bandwidth)
            w_sum = weights.sum()
            if w_sum <= 1e-8:
                preds.append(np.nan)
            else:
                preds.append(float(np.dot(weights, Y_src) / w_sum))
        return np.asarray(preds)


class BorrowingMethod(BaseMethod):
    """Adds helpers for methods that borrow from historical controls."""

    def _hist_stats(self, x_eval: np.ndarray) -> LocalStats:
        return _compute_local_stats(self.X_h, self.Y_h, x_eval, self.h_hist)

    @staticmethod
    def _control_stats(
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
        x_eval: np.ndarray,
    ) -> LocalStats:
        mask_ctrl = Z_curr == 0
        if mask_ctrl.sum() < 2:
            return LocalStats(weight=0.0, mean=np.nan, variance=np.nan)
        X_ctrl = _as_2d(X_curr[mask_ctrl])
        Y_ctrl = np.asarray(Y_curr[mask_ctrl], dtype=float)
        bandwidth = _silverman_bandwidth(X_ctrl)
        return _compute_local_stats(X_ctrl, Y_ctrl, x_eval, bandwidth)

    @staticmethod
    def _treated_stats(
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
        x_eval: np.ndarray,
    ) -> LocalStats:
        mask_treated = Z_curr == 1
        if mask_treated.sum() < 2:
            return LocalStats(weight=0.0, mean=np.nan, variance=np.nan)
        X_treated = _as_2d(X_curr[mask_treated])
        Y_treated = np.asarray(Y_curr[mask_treated], dtype=float)
        bandwidth = _silverman_bandwidth(X_treated)
        return _compute_local_stats(X_treated, Y_treated, x_eval, bandwidth)


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

class KBCD(BaseMethod):
    """Kernel-Based Biased Coin Design (Jiang et al., 2018)."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "KBCD"

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new) -> float:  # noqa: D401 (docstring inherited)
        if X_curr.shape[0] < 2:
            return 0.5
        bandwidth = _silverman_bandwidth(_as_2d(X_curr))
        weights = _gaussian_kernel_weights(X_new, _as_2d(X_curr), bandwidth)
        n0_local = float(np.dot(weights, 1 - Z_curr))
        n1_local = float(np.dot(weights, Z_curr))
        return self._imbalance_probability(n0_local, n1_local)

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)

        mask_ctrl = Z == 0
        mask_trt = Z == 1
        if mask_ctrl.sum() < 2 or mask_trt.sum() < 2:
            delta_hat = float(Y[mask_trt].mean() - Y[mask_ctrl].mean())
            return delta_hat, np.nan, np.nan, 0.5

        h_ctrl = _silverman_bandwidth(X[mask_ctrl])
        h_trt = _silverman_bandwidth(X[mask_trt])
        mu0_hat = self._kernel_mean(X[mask_ctrl], Y[mask_ctrl], h_ctrl, X)
        mu1_hat = self._kernel_mean(X[mask_trt], Y[mask_trt], h_trt, X)
        delta_samples = mu1_hat - mu0_hat
        return _delta_summary(delta_samples, self.alpha)


class CAHB(BorrowingMethod):
    """Covariate-Adjusted Historical Borrowing (Jin et al., 2023)."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "CAHB"
        self.gamma = np.sqrt(3.0)  # Section 4 tuning constant

    # Internal helpers -----------------------------------------------------
    def _borrowing_components(
        self,
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
        x_eval: np.ndarray,
    ) -> Tuple[float, LocalStats, LocalStats]:
        control_stats = self._control_stats(X_curr, Y_curr, Z_curr, x_eval)
        hist_stats = self._hist_stats(x_eval)

        if control_stats.weight <= 1e-6 or hist_stats.weight <= 1e-6:
            return 1.0, control_stats, hist_stats

        delta_mu = hist_stats.mean - control_stats.mean
        tau_hat = hist_stats.weight / (self.gamma ** 2 + hist_stats.weight * delta_mu ** 2)
        phi_sq = max(control_stats.variance, 1e-6)
        R_n = 1.0 + tau_hat * phi_sq
        return float(np.clip(R_n, 1.0, 1e6)), control_stats, hist_stats

    # Public API -----------------------------------------------------------
    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new) -> float:
        if X_curr.shape[0] < 2:
            return 0.5
        R_n, _, _ = self._borrowing_components(X_curr, Y_curr, Z_curr, X_new)
        bandwidth = _silverman_bandwidth(_as_2d(X_curr))
        weights = _gaussian_kernel_weights(X_new, _as_2d(X_curr), bandwidth)
        n0_local = float(np.dot(weights, 1 - Z_curr))
        n1_local = float(np.dot(weights, Z_curr))
        n0_eff = R_n * n0_local
        return self._imbalance_probability(n0_eff, n1_local)

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)
        mask_ctrl = Z == 0
        mask_trt = Z == 1
        if mask_ctrl.sum() < 2 or mask_trt.sum() < 2:
            delta_hat = float(Y[mask_trt].mean() - Y[mask_ctrl].mean())
            return delta_hat, np.nan, np.nan, 0.5

        h_ctrl = _silverman_bandwidth(X[mask_ctrl])
        h_trt = _silverman_bandwidth(X[mask_trt])
        mu0_ctrl = self._kernel_mean(X[mask_ctrl], Y[mask_ctrl], h_ctrl, X)
        mu1_hat = self._kernel_mean(X[mask_trt], Y[mask_trt], h_trt, X)

        borrowed_mu0 = []
        for x, mu0_c in zip(X, mu0_ctrl):
            R_n, control_stats, hist_stats = self._borrowing_components(X, Y, Z, x)
            if control_stats.weight <= 1e-6 or not np.isfinite(mu0_c):
                borrowed_mu0.append(mu0_c)
                continue

            tau_hat = max((R_n - 1.0) / max(control_stats.variance, 1e-6), 0.0)
            phi_sq = max(control_stats.variance, 1e-6)
            weight_c = control_stats.weight / phi_sq
            weight_h = tau_hat
            numerator = weight_c * mu0_c + weight_h * hist_stats.mean
            denominator = weight_c + weight_h
            if denominator <= 1e-8:
                borrowed_mu0.append(mu0_c)
            else:
                borrowed_mu0.append(float(numerator / denominator))

        delta_samples = mu1_hat - np.asarray(borrowed_mu0)
        return _delta_summary(delta_samples, self.alpha)


class CAHB_PP(CAHB):
    """Proposed CAHB with local power prior (Sec. 2.4–2.5 of CAHB_PP(2).pdf)."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "CAHB-PP"
        self.a_alpha = float(self.priors.get("a_beta_a", 1.0))
        self.a_beta = float(self.priors.get("a_beta_b", 1.0))

    def _local_discount(
        self,
        control_stats: LocalStats,
        hist_stats: LocalStats,
    ) -> float:
        if hist_stats.weight <= 1e-6 or control_stats.weight <= 0:
            return 0.0

        phi_sq = max(control_stats.variance, 1e-6)
        delta_mu = hist_stats.mean - control_stats.mean
        compatibility = np.exp(-0.5 * (delta_mu ** 2) / phi_sq)
        a_post = (self.a_alpha + compatibility * hist_stats.weight) / (
            self.a_alpha + self.a_beta + hist_stats.weight
        )
        return float(np.clip(a_post, 0.0, 1.0))

    def _borrowing_components(
        self,
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
        x_eval: np.ndarray,
    ) -> Tuple[float, LocalStats, LocalStats]:
        control_stats = self._control_stats(X_curr, Y_curr, Z_curr, x_eval)
        hist_stats = self._hist_stats(x_eval)

        if hist_stats.weight <= 1e-6:
            return 1.0, control_stats, hist_stats

        sigma_h_sq = max(hist_stats.variance, 1e-6)
        sigma_c_sq = max(control_stats.variance, sigma_h_sq)

        a_x = self._local_discount(control_stats, hist_stats)
        if control_stats.weight <= 1e-6:
            R_n = 1.0 + a_x * hist_stats.weight
        else:
            R_n = 1.0 + a_x * (hist_stats.weight * sigma_c_sq) / (control_stats.weight * sigma_h_sq)
        return float(np.clip(R_n, 1.0, 1e6)), control_stats, hist_stats

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)
        mask_ctrl = Z == 0
        mask_trt = Z == 1
        if mask_ctrl.sum() < 2 or mask_trt.sum() < 2:
            delta_hat = float(Y[mask_trt].mean() - Y[mask_ctrl].mean())
            return delta_hat, np.nan, np.nan, 0.5

        h_ctrl = _silverman_bandwidth(X[mask_ctrl])
        h_trt = _silverman_bandwidth(X[mask_trt])
        mu0_ctrl = self._kernel_mean(X[mask_ctrl], Y[mask_ctrl], h_ctrl, X)
        mu1_hat = self._kernel_mean(X[mask_trt], Y[mask_trt], h_trt, X)
        theta0_hist = self._kernel_mean(self.X_h, self.Y_h, self.h_hist, X)

        borrowed_mu0 = []
        for x, mu0_c, theta_h in zip(X, mu0_ctrl, theta0_hist):
            R_n, control_stats, hist_stats = self._borrowing_components(X, Y, Z, x)
            if control_stats.weight <= 1e-6 or not np.isfinite(mu0_c):
                borrowed_mu0.append(mu0_c)
                continue

            sigma_h_sq = max(hist_stats.variance, 1e-6)
            sigma_c_sq = max(control_stats.variance, sigma_h_sq)
            a_x = self._local_discount(control_stats, hist_stats)

            weight_c = control_stats.weight / sigma_c_sq
            weight_h = a_x * hist_stats.weight / sigma_h_sq

            numerator = weight_c * mu0_c + weight_h * theta_h
            denominator = weight_c + weight_h
            if denominator <= 1e-8:
                borrowed_mu0.append(mu0_c)
            else:
                borrowed_mu0.append(float(numerator / denominator))

        delta_samples = mu1_hat - np.asarray(borrowed_mu0)
        return _delta_summary(delta_samples, self.alpha)


class rMAP_KBCD(BaseMethod):
    """Robust MAP prior applied to the control arm combined with KBCD allocation."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "rMAP-KBCD"

        self.w_robust = float(self.priors.get("rmap_weight", 0.1))
        self.m0 = float(np.mean(self.Y_h))
        self.sq_sigma_h = max(np.var(self.Y_h, ddof=1), 1e-6)
        self.v0 = self.sq_sigma_h / self.X_h.shape[0]
        self.v_vague = 100.0  # diffuse component variance

        ess_fitted = self.sq_sigma_h / max(self.v0, 1e-6)
        ess_vague = self.sq_sigma_h / self.v_vague
        self.ess_prior = (1.0 - self.w_robust) * ess_fitted + self.w_robust * ess_vague

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new) -> float:
        if X_curr.shape[0] < 2:
            return 0.5
        bandwidth = _silverman_bandwidth(_as_2d(X_curr))
        weights = _gaussian_kernel_weights(X_new, _as_2d(X_curr), bandwidth)
        n0_local = float(np.dot(weights, 1 - Z_curr))
        n1_local = float(np.dot(weights, Z_curr))

        n_ctrl_global = float(np.sum(1 - Z_curr))
        if n_ctrl_global <= 1e-6:
            R_n = 1.0 + self.ess_prior
        else:
            R_n = (n_ctrl_global + self.ess_prior) / n_ctrl_global
        return self._imbalance_probability(R_n * n0_local, n1_local)

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)
        mask_ctrl = Z == 0
        mask_trt = Z == 1
        if mask_trt.sum() < 2:
            return np.nan, np.nan, np.nan, 0.5

        h_trt = _silverman_bandwidth(X[mask_trt])
        mu1_hat = self._kernel_mean(X[mask_trt], Y[mask_trt], h_trt, X)

        Y_ctrl = Y[mask_ctrl]
        n_ctrl = Y_ctrl.shape[0]
        if n_ctrl == 0:
            mu0_post_mean = self.m0
            mu0_post_var = self.v0
        else:
            y_bar = float(Y_ctrl.mean())
            var_data = max(np.var(Y_ctrl, ddof=1) / n_ctrl, 1e-6)

            v_post_fitted = 1.0 / (1.0 / self.v0 + n_ctrl / var_data)
            m_post_fitted = v_post_fitted * (self.m0 / self.v0 + y_bar * n_ctrl / var_data)

            v_post_vague = 1.0 / (1.0 / self.v_vague + n_ctrl / var_data)
            m_post_vague = v_post_vague * (0.0 / self.v_vague + y_bar * n_ctrl / var_data)

            log_w_fitted = np.log1p(-self.w_robust) + norm.logpdf(y_bar, self.m0, np.sqrt(self.v0 + var_data))
            log_w_vague = np.log(self.w_robust) + norm.logpdf(y_bar, 0.0, np.sqrt(self.v_vague + var_data))
            mix_norm = np.logaddexp(log_w_fitted, log_w_vague)
            w_fitted = np.exp(log_w_fitted - mix_norm)
            w_vague = 1.0 - w_fitted

            mu0_post_mean = w_fitted * m_post_fitted + w_vague * m_post_vague
            mu0_post_var = (
                w_fitted * (v_post_fitted + (m_post_fitted - mu0_post_mean) ** 2)
                + w_vague * (v_post_vague + (m_post_vague - mu0_post_mean) ** 2)
            )

        mu0_hat = np.full(X.shape[0], mu0_post_mean)
        delta_samples = mu1_hat - mu0_hat
        delta_hat, ci_low, ci_high, _ = _delta_summary(delta_samples, self.alpha)

        std = np.sqrt(mu0_post_var + max(np.var(Y[mask_trt], ddof=1) / max(mask_trt.sum(), 1), 1e-8))
        prob = float(norm.sf((0.0 - delta_hat) / std)) if np.isfinite(std) and std > 0 else 0.5
        return delta_hat, ci_low, ci_high, prob


__all__ = ["KBCD", "CAHB", "CAHB_PP", "rMAP_KBCD"]
