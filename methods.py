"""
methods.py

Covariate-Adaptive Randomization Methods for CAHB-PP Simulation Study

This module implements four covariate-adaptive randomization and analysis methods
evaluated in Section 3.3 of the CAHB-PP manuscript. Each method consists of two
components:

1. **Adaptive Allocation Rule**: Computes Pr(Z = 1 | X, Data) for a new subject
   with covariates X, given the accumulating trial data.

2. **Final Analysis**: Estimates the treatment effect and produces inference
   (point estimate, confidence interval, posterior probability).

Implemented Methods:
--------------------
1. KBCD (Jiang et al., 2018):
   - Kernel-based covariate-balancing randomization
   - No historical borrowing (benchmark)
   - Balances local covariate distribution across arms
   
2. CAHB (Jin et al., 2023):
   - Covariate-adjusted historical borrowing
   - Local borrowing via posterior precision inflation R_n(x)
   - Compatibility-based borrowing weight via tau_n(x)

3. CAHB-PP (Proposed):
   - Extends CAHB with local power prior discount a(x)
   - Data-driven borrowing: a(x) ∈ [0,1] learned from compatibility
   - Beta(1,1) prior on a(x) with compatibility-based likelihood

4. rMAP-KBCD (Schmidli et al., 2014):
   - Robust meta-analytic-predictive prior
   - Global borrowing (not covariate-adjusted)
   - Mixture of informative and vague priors
   - Combined with KBCD allocation

Common Interface:
-----------------
All methods inherit from BaseMethod and implement:
    - get_allocation_prob(X_curr, Y_curr, Z_curr, X_new) -> float
    - estimate_treatment_effect(X, Y, Z) -> (delta_hat, ci_low, ci_high, prob_gt_0)

References:
-----------
- CAHB-PP manuscript (Section 2-3): references/CAHB_PP.pdf
- Jin et al. (2023): references/2023_SIM_CAHB.pdf
- Jiang et al. (2018): references/KBCD.pdf  
- Schmidli et al. (2014): references/MAP.pdf
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
    """
    Kernel-Based Covariate-Balancing Biased Coin Design (Jiang et al., 2018).
    
    KBCD is a covariate-adaptive randomization method that balances the treatment
    allocation locally around each subject's covariate value. It serves as the
    NO BORROWING benchmark in our simulations.
    
    Key Principle:
    --------------
    For a new subject with covariates x, allocate to treatment vs control in a way
    that balances the LOCAL (kernel-weighted) sample sizes in the two arms.
    
    Allocation Rule (Equation 2 in Jiang et al. 2018):
    ---------------------------------------------------
    pi_1(x) = (1/g_0(x) - 1) / (1/g_0(x) + 1/g_1(x) - 2)
    
    where:
        g_0(x) = n_0(x) / n(x)    # Local proportion in control
        g_1(x) = n_1(x) / n(x)    # Local proportion in treatment
        n_0(x) = sum_i K(x, X_i) * I(Z_i = 0)  # Kernel-weighted control count
        n_1(x) = sum_i K(x, X_i) * I(Z_i = 1)  # Kernel-weighted treatment count
    
    Kernel Function:
    ----------------
    Product Gaussian kernel:
        K(x, x') = prod_j exp(-0.5 * [(x_j - x'_j) / h_j]^2)
    
    Bandwidth: Silverman's rule-of-thumb
        h_j = (4 / (p + 2))^(1/(p+4)) * n^(-1/(p+4)) * sigma_j
    
    Final Analysis:
    ---------------
    Kernel regression estimates for each arm:
        mu_0(x) = sum_i K(x, X_i) Y_i I(Z_i=0) / sum_i K(x, X_i) I(Z_i=0)
        mu_1(x) = sum_i K(x, X_i) Y_i I(Z_i=1) / sum_i K(x, X_i) I(Z_i=1)
    
    Treatment effect: delta(x) = mu_1(x) - mu_0(x)
    
    Advantages:
    -----------
    - Achieves covariate balance without modeling assumptions
    - Automatic bandwidth selection
    - Works for continuous and discrete covariates
    
    Limitations:
    ------------
    - Does not use historical data
    - May have low precision with small local sample sizes
    - Bandwidth choice affects performance
    """

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
    """
    Covariate-Adjusted Historical Borrowing (Jin et al., 2023).
    
    CAHB adaptively borrows from historical control data based on local
    covariate-specific compatibility between historical and concurrent controls.
    
    Key Features:
    -------------
    1. Local compatibility assessment via kernel-weighted statistics
    2. Borrowing strength controlled by tau_n(x) - posterior precision from 
       historical data
    3. Effective sample size inflation R_n(x) for allocation and analysis
    
    Allocation Rule (Equation 6 in Jin et al. 2023):
    ------------------------------------------------
    The allocation probability is computed via the biased coin design with
    effective sample size inflation:
    
        pi_1(x) = f(n_0^eff(x), n_1(x))
        
    where:
        n_0^eff(x) = R_n(x) * n_0(x)    # Inflated control sample size
        R_n(x) = 1 + tau_n(x) * phi_0^2(x)  # Sample size inflation factor
        
    Borrowing Components (Section 3):
    ----------------------------------
    Compatibility measure:
        tau_n(x) = n_h(x) / (gamma^2 + n_h(x) * [theta_h(x) - theta_0c(x)]^2)
        
    where:
        - theta_h(x): Historical control mean at x
        - theta_0c(x): Current control mean at x  
        - gamma: Tuning parameter (gamma = sqrt(3))
        - n_h(x): Effective historical sample size at x
    
    Final Analysis (Section 4):
    ---------------------------
    Posterior mean for control arm at x:
        theta_0^*(x) = (w_c(x) * theta_0c(x) + w_h(x) * theta_h(x)) / (w_c + w_h)
        
    where:
        w_c(x) = n_0c(x) / phi_0^2(x)      # Current control precision
        w_h(x) = tau_n(x)                   # Historical precision contribution
    """

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "CAHB"
        # Tuning parameter gamma (Section 4 of Jin et al. 2023)
        # Controls sensitivity to historical-current discrepancy
        self.gamma = float(priors.get('cahb_gamma', np.sqrt(3.0)))

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
    """
    Proposed: Covariate-Adjusted Historical Borrowing with Power Prior (CAHB-PP).
    
    CAHB-PP extends CAHB by introducing a local, data-driven discount parameter
    a(x) ∈ [0, 1] that controls borrowing strength based on covariate-specific
    compatibility between historical and current control data.
    
    Key Innovation:
    ---------------
    Instead of borrowing all historical information at x (as CAHB does), CAHB-PP
    discounts the historical data by a(x), where a(x) is learned from the data
    via a power prior framework.
    
    Power Prior Framework (Section 2.4-2.5):
    -----------------------------------------
    Historical data contribution:
        L(theta | D_h)^{a(x)}
        
    where a(x) ~ Beta(alpha_a, beta_a) is the discount parameter.
    
    Posterior for a(x) (Equation X in manuscript):
        a(x) | D ~ Beta(alpha_a + n_h(x) * compatibility, 
                        beta_a + n_h(x) * (1 - compatibility))
        
    Compatibility score:
        compatibility(x) = exp(-0.5 * [theta_h(x) - theta_0c(x)]^2 / phi_0^2(x))
        
    This gives higher a(x) when historical and current controls are compatible.
    
    Effective Sample Size with Power Prior:
    ----------------------------------------
    R_n(x) = 1 + a(x) * [n_h(x) * sigma_c^2(x)] / [n_c(x) * sigma_h^2(x)]
    
    where:
        - a(x): Local discount parameter (posterior mean)
        - n_h(x), n_c(x): Local effective sample sizes
        - sigma_h^2(x), sigma_c^2(x): Local variance estimates
    
    Advantages over CAHB:
    ---------------------
    1. Automatic discount when historical data are incompatible
    2. Covariate-specific borrowing (local a(x) vs global)
    3. Uncertainty quantification via Beta posterior on a(x)
    4. Bounded borrowing: a(x) ∈ [0,1] ensures conservative borrowing
    
    Prior Specification:
    --------------------
    Beta(1, 1) = Uniform[0, 1] (non-informative):
        - Allows data to fully determine borrowing strength
        - a_alpha = 1, a_beta = 1
    
    Alternative: Beta(0.5, 0.5) (Jeffreys prior) for more conservatism
    
    References:
        CAHB-PP manuscript Section 2.4-2.5 (Proposed Method)
    """

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "CAHB-PP"
        
        # Beta prior hyperparameters for discount parameter a(x)
        # a(x) ~ Beta(a_alpha, a_beta)
        self.a_alpha = float(self.priors.get("a_beta_a", 1.0))
        self.a_beta = float(self.priors.get("a_beta_b", 1.0))

    def _local_discount(
        self,
        control_stats: LocalStats,
        hist_stats: LocalStats,
    ) -> float:
        """
        Computes the local power prior discount parameter a(x).
        
        This is the core innovation of CAHB-PP: a data-driven, covariate-specific
        discount that down-weights historical data when they are incompatible with
        current controls at covariate value x.
        
        Power Prior Discount (Section 2.5):
        ------------------------------------
        Prior: a(x) ~ Beta(alpha_a, beta_a)
        
        Compatibility likelihood:
            L(a | x) proportional to a^{n_h(x) * c(x)}
            
        where c(x) = compatibility score:
            c(x) = exp(-0.5 * [theta_h(x) - theta_0c(x)]^2 / phi_0^2(x))
        
        Posterior (conjugate Beta update):
            a(x) | Data ~ Beta(alpha_a + n_h(x)*c(x), beta_a + n_h(x)*(1-c(x)))
        
        We use the posterior mean E[a(x) | Data] as the point estimate.
        
        Interpretation of a(x):
        -----------------------
        - a(x) ≈ 1: Historical and current data highly compatible at x
                    -> Borrow fully (close to CAHB behavior)
        - a(x) ≈ 0.5: Moderate compatibility
                      -> Partial borrowing  
        - a(x) ≈ 0: Strong incompatibility at x
                    -> Minimal borrowing (close to KBCD behavior)
        
        Args:
            control_stats: Local statistics for current control arm at x
            hist_stats: Local statistics for historical controls at x
        
        Returns:
            Posterior mean of a(x), clipped to [0, 1]
        
        Notes:
            - If no historical data at x (hist_stats.weight ≈ 0), return 0
            - Compatibility uses current control variance as reference scale
            - Gaussian compatibility measure: sensitive to mean differences
        """
        # No historical information at x -> no borrowing
        if hist_stats.weight <= 1e-6 or control_stats.weight <= 0:
            return 0.0

        # Local variance of current control (reference scale for discrepancy)
        phi_sq = max(control_stats.variance, 1e-6)
        
        # Mean difference between historical and current controls
        delta_mu = hist_stats.mean - control_stats.mean
        
        # Compatibility score: high when means are similar
        # c(x) ∈ [0, 1], with c(x)=1 when delta_mu=0
        compatibility = np.exp(-0.5 * (delta_mu ** 2) / phi_sq)
        
        # Posterior Beta parameters
        # Effective: historical data "votes" for borrowing proportional to compatibility
        alpha_post = self.a_alpha + compatibility * hist_stats.weight
        beta_post = self.a_beta + (1.0 - compatibility) * hist_stats.weight
        
        # Posterior mean: E[a | Data] = alpha_post / (alpha_post + beta_post)
        a_post = alpha_post / (alpha_post + beta_post)
        
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
    """
    Robust Meta-Analytic-Predictive (MAP) Prior + KBCD Allocation.
    
    Combines Schmidli et al. (2014) robust MAP prior for historical borrowing
    with Jiang et al. (2018) KBCD allocation. This provides a GLOBAL BORROWING
    benchmark (non-covariate-adjusted borrowing).
    
    Robust MAP Prior (Schmidli et al. 2014, Section 3):
    ---------------------------------------------------
    The control arm mean has a mixture prior:
        
        theta_0 ~ w * Vague(theta_0) + (1-w) * Informative(theta_0)
    
    Components:
        - Informative: N(m_0, v_0)  based on historical data
        - Vague: N(0, v_vague)       diffuse prior (v_vague >> v_0)
        - Mixture weight: w = 0.1 (default)
    
    Hyperparameters from Historical Data:
        m_0 = mean(Y_h)              # Historical control mean
        v_0 = var(Y_h) / n_h         # Historical control variance / sample size
        v_vague = 100 * v_0          # Vague component (large variance)
    
    Rationale:
    ----------
    - Vague component provides robustness to historical bias
    - If current data conflict with historical, posterior shifts to vague component
    - Automatic downweighting via Bayesian model averaging
    
    Allocation Rule:
    ----------------
    Uses KBCD allocation with inflated control effective sample size:
        n_0^eff = n_0 + ESS_prior
        
    where ESS_prior is the effective sample size from the MAP prior.
    
    Final Analysis:
    ---------------
    Posterior for control mean (mixture of two Normals):
        p(theta_0 | D) = w_post * N(m_post_vague, v_post_vague) 
                         + (1 - w_post) * N(m_post_fitted, v_post_fitted)
    
    Posterior mixture weights updated via Bayes theorem using data likelihood.
    
    Comparison to CAHB/CAHB-PP:
    ---------------------------
    - Global borrowing (same for all covariates x)
    - Robust to overall historical bias via mixture
    - NOT adaptive to covariate-specific bias (e.g., Scenario S4)
    - Simpler: no kernel smoothing for borrowing
    
    Limitations:
    ------------
    - Cannot detect or adapt to subgroup-specific historical bias
    - Uniform borrowing across all covariate values
    - Mixture weight w must be pre-specified
    
    References:
        Schmidli et al. (2014), "Robust Meta-Analytic-Predictive Priors..."
        Biometrics, Section 3-4
    """

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "rMAP-KBCD"

        # Robust MAP mixture weight (typically 0.1 = 10% on vague component)
        self.w_robust = float(self.priors.get("rmap_weight", 0.1))
        
        # Historical data summary statistics (global, not covariate-adjusted)
        self.m0 = float(np.mean(self.Y_h))                         # Historical mean
        self.sq_sigma_h = max(np.var(self.Y_h, ddof=1), 1e-6)     # Historical variance
        self.v0 = self.sq_sigma_h / self.X_h.shape[0]             # Prior variance (informative)
        self.v_vague = 100.0  # Vague component variance (large relative to v0)

        # Effective sample size from MAP prior (for allocation)
        # ESS = variance / prior_variance
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
