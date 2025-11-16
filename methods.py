"""
methods.py

Implementation of the covariate-adaptive methods used in the CAHB-PP
simulation study.  The code mirrors the notation in the CAHB-PP paper and
supplementary materials as well as the original authors' R reference
implementation (`utils.R`).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline


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


@dataclass
class AllocationResult:
    """Return object for allocation probabilities with diagnostics."""

    pi_treatment: float
    diagnostics: dict


def _as_2d(array: Sequence[float]) -> np.ndarray:
    arr = np.asarray(array, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _silverman_bandwidth(X: np.ndarray) -> np.ndarray:
    """Silverman's rule of thumb bandwidth for Gaussian product kernels."""
    X = _as_2d(X)
    n, p = X.shape
    sigma = np.std(X, axis=0, ddof=1)
    sigma[sigma < 1e-6] = 1.0
    factor = (4.0 / (p + 2.0)) ** (1.0 / (p + 4.0)) * n ** (-1.0 / (p + 4.0))
    bandwidth = factor * sigma
    bandwidth[bandwidth < 1e-6] = 1.0
    return bandwidth


def _gaussian_kernel_weights(x_new: np.ndarray, X_ref: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Product Gaussian kernel weights."""
    if X_ref.shape[0] == 0:
        return np.zeros(0)
    x_new = _as_2d(x_new)
    diff = (X_ref - x_new) / h
    norm_sq = np.sum(diff ** 2, axis=1)
    return np.exp(-0.5 * norm_sq)


def _compute_local_stats(
    X_ref: np.ndarray,
    values: Optional[np.ndarray],
    x_eval: np.ndarray,
    bandwidth: np.ndarray,
) -> LocalStats:
    weights = _gaussian_kernel_weights(x_eval, X_ref, bandwidth)
    weight_sum = weights.sum()
    if weight_sum <= 1e-8 or values is None:
        return LocalStats(weight=float(weight_sum), mean=np.nan, variance=np.nan)
    values = np.asarray(values, dtype=float)
    mean = np.dot(weights, values) / weight_sum
    variance = np.dot(weights, (values - mean) ** 2) / weight_sum
    return LocalStats(weight=float(weight_sum), mean=float(mean), variance=float(variance))


def _normal_interval(mean: float, std: float, alpha: float) -> Tuple[float, float]:
    if not np.isfinite(std) or std <= 0:
        return np.nan, np.nan
    z = norm.ppf(1.0 - alpha / 2.0)
    return mean - z * std, mean + z * std


def _delta_summary(delta_samples: np.ndarray, alpha: float) -> Tuple[float, float, float, float]:
    """
    Summarize posterior samples of the treatment effect.
    
    Uses empirical quantiles for credible intervals and empirical probability.
    """
    delta_samples = np.asarray(delta_samples, dtype=float)
    mask = np.isfinite(delta_samples)
    draws = delta_samples[mask]
    if draws.size == 0:
        return np.nan, np.nan, np.nan, 0.5
    mean = float(np.mean(draws))
    if draws.size < 2:
        return mean, np.nan, np.nan, float(np.mean(draws > 0.0))
    ci_low, ci_high = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    prob = float(np.mean(draws > 0.0))
    return mean, float(ci_low), float(ci_high), prob


class BridgeSampler:
    """
    Pre-computation + Interpolation Strategy for sampling discount parameter a(x).
    
    Implements the method from CAHB-PP manuscript Algorithm 1 (Carvalho & Ibrahim, 2021):
    1. Pre-computation: Estimate log-normalizing constants on a grid
    2. Interpolation: Fit spline/GP to approximate log m_h(a)
    3. MH Sampling: Use logit-scale random walk with interpolation
    
    This handles the doubly intractable posterior p(a | data) ∝ π(a) × L_h(a) / m_h(a)
    where m_h(a) is computationally expensive.
    
    Reference: CAHB-PP manuscript Section 2.5, Algorithm 1
    """

    def __init__(
        self,
        log_prior,
        log_likelihood,
        step_size: float = 0.1,
        grid_size: int = 20,
        use_precomputed_grid: bool = True,
    ):
        """
        Initialize sampler with pre-computation and interpolation strategy.
        
        Args:
            log_prior: Function a -> log π(a), evaluates log prior density
            log_likelihood: Function a -> log L_h(μ_0, σ_h^2; a), powered likelihood
            step_size: σ_p for logit-scale random walk: ε ~ N(0, σ_p^2)
            grid_size: Number of grid points K for pre-computation
            use_precomputed_grid: If True, pre-compute normalizing constants
        """
        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.step_size = max(1e-3, float(step_size))
        self.grid_size = max(5, int(grid_size))
        
        # Pre-computation grid: A = {a_1, ..., a_K} over [0, 1]
        self.grid = np.linspace(0.01, 0.99, self.grid_size)
        self.log_norm_constants = None
        self.interpolator = None
        
        if use_precomputed_grid:
            self._precompute_normalizing_constants()

    def _sample_powered_posterior(self, a_val: float, n_samples: int = 1000, 
                                   burn_in: int = 500, thin: int = 2) -> np.ndarray:
        """
        Sample from powered posterior p_k(θ) ∝ L_h(θ)^{a_val} π(θ) using MCMC.
        
        This generates samples from the un-normalized density needed for bridge sampling.
        We use a simple Metropolis-Hastings with Gaussian proposals.
        
        Args:
            a_val: Power parameter a_k
            n_samples: Number of samples to return (after burn-in and thinning)
            burn_in: Number of burn-in iterations
            thin: Thinning interval
            
        Returns:
            Array of samples from powered posterior, shape (n_samples,)
        """
        # Initialize at mode estimate (use simple optimization)
        # For our case, θ represents the parameter in the likelihood
        # Start at a reasonable initial value
        current = 0.5
        
        def log_powered_posterior(theta):
            """Evaluate log p_k(θ) = a_val × log L_h(θ) + log π(θ)"""
            # Prior: Beta(1,1) uniform, or use the actual prior structure
            log_prior = 0.0 if 0.0 < theta < 1.0 else -np.inf
            if not np.isfinite(log_prior):
                return -np.inf
            
            # Powered likelihood
            try:
                log_like = self.log_likelihood(theta)
                if not np.isfinite(log_like):
                    return -np.inf
                return a_val * log_like + log_prior
            except Exception:
                return -np.inf
        
        # MH sampling
        samples = []
        log_current = log_powered_posterior(current)
        proposal_sd = 0.1  # Adaptive proposal
        n_accepted = 0
        
        total_iterations = burn_in + n_samples * thin
        
        for iteration in range(total_iterations):
            # Gaussian random walk proposal
            proposal = current + np.random.normal(0, proposal_sd)
            proposal = np.clip(proposal, 0.01, 0.99)  # Keep in bounds
            
            log_proposal = log_powered_posterior(proposal)
            
            # MH accept/reject
            log_accept_ratio = log_proposal - log_current
            if np.log(np.random.rand()) < log_accept_ratio:
                current = proposal
                log_current = log_proposal
                n_accepted += 1
            
            # Collect samples after burn-in with thinning
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                samples.append(current)
        
        # Adaptive tuning feedback
        accept_rate = n_accepted / total_iterations
        if accept_rate < 0.15 or accept_rate > 0.6:
            # Could adjust proposal_sd for next run, but this is per-grid-point
            pass
        
        return np.asarray(samples)
    
    def _bridge_sampling_estimator(self, samples_p: np.ndarray, samples_q: np.ndarray,
                                   log_p_unnorm, log_q_unnorm, max_iter: int = 100) -> float:
        """
        Bridge sampling estimator for log(Z_p / Z_q) where:
        - p(θ) = p_unnorm(θ) / Z_p is target distribution
        - q(θ) = q_unnorm(θ) / Z_q is proposal/reference distribution
        
        Uses iterative bridge sampling (Meng & Wong, 1996; Gronau et al., 2017).
        
        Args:
            samples_p: Samples from p(θ), shape (n_p,)
            samples_q: Samples from q(θ), shape (n_q,)
            log_p_unnorm: Function θ -> log p_unnorm(θ)
            log_q_unnorm: Function θ -> log q_unnorm(θ)
            max_iter: Maximum iterations for bridge equation
            
        Returns:
            Estimate of log(Z_p / Z_q)
        """
        n_p = len(samples_p)
        n_q = len(samples_q)
        
        # Evaluate unnormalized densities at samples
        log_p_at_p = np.array([log_p_unnorm(s) for s in samples_p])
        log_q_at_p = np.array([log_q_unnorm(s) for s in samples_p])
        log_p_at_q = np.array([log_p_unnorm(s) for s in samples_q])
        log_q_at_q = np.array([log_q_unnorm(s) for s in samples_q])
        
        # Initialize log(Z_p / Z_q) estimate
        log_ratio = 0.0
        
        # Iterative bridge sampling equation
        for _ in range(max_iter):
            log_ratio_old = log_ratio
            
            # Bridge function: h(θ) = n_q / (n_p * exp(log_q - log_p) + n_q)
            # Numerator: E_p[1 / (n_p/n_q * r + 1)] where r = p_unnorm/q_unnorm * Z_q/Z_p
            log_denom_at_p = np.logaddexp(
                np.log(n_p) + log_p_at_p - log_q_at_p - log_ratio,
                np.log(n_q)
            )
            term1 = -np.mean(log_denom_at_p) + np.log(n_q)
            
            # Denominator: E_q[1 / (n_p/n_q * r + 1)]
            log_denom_at_q = np.logaddexp(
                np.log(n_p) + log_p_at_q - log_q_at_q - log_ratio,
                np.log(n_q)
            )
            term2 = -np.mean(log_denom_at_q) + np.log(n_p)
            
            # Update: log(Z_p/Z_q) = term1 - term2
            log_ratio = term1 - term2
            
            # Check convergence
            if np.abs(log_ratio - log_ratio_old) < 1e-6:
                break
        
        return log_ratio
    
    def _precompute_normalizing_constants(self):
        """
        Pre-compute log-normalizing constants l̂_k = log m̂_h(a_k) on grid using bridge sampling.
        
        For each a_k in grid:
        1. Sample from powered posterior p_k(θ) ∝ L_h(θ)^{a_k} π(θ) using MCMC
        2. Sample from reference posterior p_0(θ) ∝ π(θ) (prior samples)
        3. Use bridge sampling to estimate log m_h(a_k) = log Z_k
        4. Store l̂_k for interpolation
        
        Bridge sampling approach (Gronau et al., 2017):
        - Estimate ratio Z_k / Z_0 where Z_0 = 1 (prior is normalized)
        - Therefore log Z_k = log(Z_k / Z_0)
        
        Reference: Gronau, Q. F., et al. (2017). "A tutorial on bridge sampling."
        """
        log_norms = []
        
        # Generate reference samples from prior (a=0 case)
        # For Beta(α,β) prior, we can sample directly
        n_ref_samples = 1000
        # Assuming Beta(1,1) = Uniform prior
        ref_samples = np.random.uniform(0.01, 0.99, n_ref_samples)
        
        def log_prior_unnorm(theta):
            """Log unnormalized prior"""
            return 0.0 if 0.0 < theta < 1.0 else -np.inf
        
        for a_k in self.grid:
            try:
                # Sample from powered posterior p_k(θ) ∝ L_h(θ)^{a_k} π(θ)
                powered_samples = self._sample_powered_posterior(
                    a_k, n_samples=1000, burn_in=500, thin=2
                )
                
                # Define log unnormalized densities
                def log_powered_unnorm(theta):
                    """Log p_k(θ) = a_k × log L_h(θ) + log π(θ)"""
                    log_prior = log_prior_unnorm(theta)
                    if not np.isfinite(log_prior):
                        return -np.inf
                    try:
                        log_like = self.log_likelihood(theta)
                        if not np.isfinite(log_like):
                            return -np.inf
                        return a_k * log_like + log_prior
                    except Exception:
                        return -np.inf
                
                # Apply bridge sampling to estimate log(Z_k / Z_0)
                # Since Z_0 = 1 (normalized prior), log Z_k = log(Z_k / Z_0)
                log_norm = self._bridge_sampling_estimator(
                    samples_p=powered_samples,
                    samples_q=ref_samples,
                    log_p_unnorm=log_powered_unnorm,
                    log_q_unnorm=log_prior_unnorm,
                    max_iter=100
                )
                
                # For numerical stability, clip extreme values
                log_norm = np.clip(log_norm, -50, 50)
                
            except Exception as e:
                # Fallback: use simple approximation if bridge sampling fails
                import warnings
                warnings.warn(f"Bridge sampling failed for a={a_k:.3f}: {e}. Using fallback.", 
                            RuntimeWarning)
                try:
                    log_like = self.log_likelihood(a_k)
                    log_norm = -0.5 * a_k * log_like if np.isfinite(log_like) else 0.0
                except Exception:
                    log_norm = 0.0
            
            log_norms.append(log_norm)
        
        self.log_norm_constants = np.array(log_norms)
        
        # Fit interpolating function: cubic spline to (a_k, l̂_k)
        try:
            self.interpolator = CubicSpline(self.grid, self.log_norm_constants, 
                                           extrapolate=False)
        except Exception:
            # Fallback to linear interpolation if spline fails
            self.interpolator = lambda a: np.interp(a, self.grid, self.log_norm_constants)

    def _log_normalizing_constant(self, a: float) -> float:
        """
        Evaluate log-normalizing constant l̂(a) via interpolation.
        
        Returns fast evaluation of exp(l̂(a)) to replace expensive m_h(a) computation.
        
        Args:
            a: Discount parameter in (0, 1)
            
        Returns:
            Interpolated log-normalizing constant
        """
        if self.interpolator is None:
            return 0.0  # No normalization if not pre-computed
        
        try:
            return float(self.interpolator(a))
        except Exception:
            # Fallback to nearest grid point
            idx = np.argmin(np.abs(self.grid - a))
            return float(self.log_norm_constants[idx])

    def _logit(self, a: float) -> float:
        """Transform a ∈ (0,1) to logit scale: logit(a) = log(a / (1-a))."""
        a = np.clip(a, 1e-8, 1.0 - 1e-8)
        return float(np.log(a / (1.0 - a)))
    
    def _expit(self, x: float) -> float:
        """Transform x ∈ R to probability scale: expit(x) = 1 / (1 + exp(-x))."""
        return float(1.0 / (1.0 + np.exp(-x)))

    def _propose_logit_scale(self, current: float) -> Tuple[float, float]:
        """
        Random-walk proposal on logit scale (Algorithm 1 in manuscript).
        
        Proposal: a_p = expit(logit(a_t) + ε), where ε ~ N(0, σ_p^2)
        
        Args:
            current: Current state a_t ∈ (0, 1)
            
        Returns:
            Tuple of (proposed a_p, Jacobian adjustment)
        """
        # Transform to logit scale
        logit_current = self._logit(current)
        
        # Random walk on logit scale
        epsilon = np.random.normal(0, self.step_size)
        logit_proposal = logit_current + epsilon
        
        # Transform back to probability scale
        proposal = self._expit(logit_proposal)
        
        # Jacobian adjustment for logit transformation
        # |da_p/dlogit(a_p)| = a_p(1 - a_p)
        # Ratio: [a_p(1-a_p)] / [a_t(1-a_t)]
        jacobian_ratio = (proposal * (1.0 - proposal)) / (current * (1.0 - current))
        
        return proposal, jacobian_ratio

    def _log_target(self, a: float) -> float:
        """
        Evaluate log-target density with interpolated normalization.
        
        Target: π(a) × L_h(a; ·) × exp(l̂(a))
        
        where exp(l̂(a)) approximates 1/m_h(a) via interpolation.
        
        Args:
            a: Discount parameter in (0, 1)
            
        Returns:
            Log-target density
        """
        if not (0.0 < a < 1.0):
            return -np.inf
        
        log_prior = self.log_prior(a)
        if not np.isfinite(log_prior):
            return -np.inf
        
        log_like = self.log_likelihood(a)
        if not np.isfinite(log_like):
            return -np.inf
        
        # Use interpolated log-normalizing constant
        log_norm = self._log_normalizing_constant(a)
        
        return log_prior + log_like - log_norm

    def sample(self, n_draws: int, burn_in: int = 100, thin: int = 1) -> np.ndarray:
        """
        Generate samples using MH with logit-scale proposal and interpolation.
        
        Algorithm (CAHB-PP manuscript Algorithm 1):
        1. Initialize a_0 = 0.5
        2. For each iteration t:
            a. Propose a_p = expit(logit(a_t) + ε), ε ~ N(0, σ_p^2)
            b. Compute acceptance ratio R with Jacobian adjustment
            c. Accept/reject using R
        3. Return samples after burn-in and thinning
        
        Acceptance ratio (from manuscript):
        R = min(1, [π(a_p)L_h(a_p;·)exp(l̂(a_t)) / π(a_t)L_h(a_t;·)exp(l̂(a_p))] 
                   × [a_p(1-a_p) / a_t(1-a_t)])
        
        Args:
            n_draws: Number of samples to return (after burn-in and thinning)
            burn_in: Number of burn-in iterations
            thin: Thinning interval
            
        Returns:
            Array of samples from p(a | data), shape (n_draws,)
        """
        samples = []
        current = 0.5  # Initialize at prior mode for Beta(1,1)
        log_target_current = self._log_target(current)
        
        total_iterations = burn_in + n_draws * thin
        n_accepted = 0
        
        for iteration in range(total_iterations):
            # Propose on logit scale with Jacobian
            proposal, jacobian_ratio = self._propose_logit_scale(current)
            log_target_proposal = self._log_target(proposal)
            
            # Metropolis-Hastings acceptance ratio with Jacobian adjustment
            # R = (target_proposal / target_current) × jacobian_ratio
            log_accept_ratio = (log_target_proposal - log_target_current) + np.log(jacobian_ratio)
            
            # Accept/reject
            if np.log(np.random.rand()) < log_accept_ratio:
                current = proposal
                log_target_current = log_target_proposal
                n_accepted += 1
            
            # Collect samples after burn-in with thinning
            if iteration >= burn_in and ((iteration - burn_in) % thin == 0):
                samples.append(current)
                if len(samples) >= n_draws:
                    break
        
        # Diagnostic: acceptance rate (useful for tuning step_size)
        acceptance_rate = n_accepted / total_iterations
        if acceptance_rate < 0.1 or acceptance_rate > 0.9:
            import warnings
            warnings.warn(
                f"MH acceptance rate = {acceptance_rate:.3f}. "
                f"Consider adjusting step_size (current: {self.step_size:.3f}).",
                RuntimeWarning
            )
        
        return np.asarray(samples, dtype=float)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class BaseMethod:
    """Common interface for simulation methods."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        self.X_h = _as_2d(historical_data["X_h"])
        self.Y_h = np.asarray(historical_data["Y_h"], dtype=float)
        self.scenario = scenario_params
        self.priors = priors
        self.h_hist = _silverman_bandwidth(self.X_h)
        try:
            import config as _cfg

            self.alpha = float(getattr(_cfg, "ALPHA", 0.05))
        except ImportError:
            self.alpha = 0.05
        self.name = "Base"
        try:
            self.posterior_draws = int(priors.get("posterior_draws", 200))
        except Exception:
            self.posterior_draws = 200

    def get_allocation_prob(
        self,
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
        X_new: np.ndarray,
    ) -> AllocationResult:
        raise NotImplementedError

    def estimate_treatment_effect(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        raise NotImplementedError

    def get_calibration_payload(self) -> list:
        """Optional calibration diagnostics (override in subclasses)."""
        return []

    @staticmethod
    def _imbalance_probability(n0_eff: float, n1_eff: float) -> float:
        n0_eff = max(float(n0_eff), 1e-6)
        n1_eff = max(float(n1_eff), 1e-6)
        total = n0_eff + n1_eff
        g0 = np.clip(n0_eff / total, 1e-6, 1.0 - 1e-6)
        g1 = np.clip(n1_eff / total, 1e-6, 1.0 - 1e-6)
        inv_g0 = 1.0 / g0
        inv_g1 = 1.0 / g1
        denom = inv_g0 + inv_g1 - 2.0
        if denom <= 1e-8:
            return 0.5
        pi = (inv_g0 - 1.0) / denom
        return float(np.clip(pi, 0.0, 1.0))

    @staticmethod
    def _kernel_mean(X_src: np.ndarray, Y_src: np.ndarray, bandwidth: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
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
    """Adds local-statistics helpers."""

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
        mask_trt = Z_curr == 1
        if mask_trt.sum() < 2:
            return LocalStats(weight=0.0, mean=np.nan, variance=np.nan)
        X_trt = _as_2d(X_curr[mask_trt])
        Y_trt = np.asarray(Y_curr[mask_trt], dtype=float)
        bandwidth = _silverman_bandwidth(X_trt)
        return _compute_local_stats(X_trt, Y_trt, x_eval, bandwidth)


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

class KBCD(BaseMethod):
    """Kernel-based covariate balancing biased coin design (Jiang et al., 2018)."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "KBCD"

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new) -> AllocationResult:
        if len(X_curr) < 2:
            return AllocationResult(pi_treatment=0.5, diagnostics={"R_n": 1.0})
        X_curr = _as_2d(X_curr)
        bandwidth = _silverman_bandwidth(X_curr)
        weights = _gaussian_kernel_weights(X_new, X_curr, bandwidth)
        n0_local = float(np.dot(weights, 1 - Z_curr))
        n1_local = float(np.dot(weights, Z_curr))
        pi = self._imbalance_probability(n0_local, n1_local)
        return AllocationResult(pi_treatment=pi, diagnostics={"R_n": 1.0})

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)
        if (Z == 0).sum() < 2 or (Z == 1).sum() < 2:
            delta = float(Y[Z == 1].mean() - Y[Z == 0].mean())
            return delta, np.nan, np.nan, 0.5
        mu0 = self._kernel_mean(X[Z == 0], Y[Z == 0], _silverman_bandwidth(X[Z == 0]), X)
        mu1 = self._kernel_mean(X[Z == 1], Y[Z == 1], _silverman_bandwidth(X[Z == 1]), X)
        delta_vals = mu1 - mu0
        if np.all(~np.isfinite(delta_vals)):
            return np.nan, np.nan, np.nan, 0.5
        # Bootstrap to approximate posterior distribution of ATE
        draws = []
        n = len(delta_vals)
        for _ in range(self.posterior_draws):
            idx = np.random.randint(0, n, n)
            X_bs = X[idx]
            Y_bs = Y[idx]
            Z_bs = Z[idx]
            if (Z_bs == 0).sum() < 2 or (Z_bs == 1).sum() < 2:
                delta_bs = float(Y_bs[Z_bs == 1].mean() - Y_bs[Z_bs == 0].mean())
                draws.append(delta_bs)
                continue
            mu0_bs = self._kernel_mean(
                X_bs[Z_bs == 0],
                Y_bs[Z_bs == 0],
                _silverman_bandwidth(X_bs[Z_bs == 0]),
                X_bs,
            )
            mu1_bs = self._kernel_mean(
                X_bs[Z_bs == 1],
                Y_bs[Z_bs == 1],
                _silverman_bandwidth(X_bs[Z_bs == 1]),
                X_bs,
            )
            delta_bs = mu1_bs - mu0_bs
            if np.all(~np.isfinite(delta_bs)):
                draws.append(np.nan)
            else:
                draws.append(float(np.nanmean(delta_bs)))
        draws_arr = np.asarray(draws, dtype=float)
        draws_arr = draws_arr[np.isfinite(draws_arr)]
        if draws_arr.size == 0:
            return np.nan, np.nan, np.nan, 0.5
        return _delta_summary(draws_arr, self.alpha)


class CAHB(BorrowingMethod):
    """Covariate-adjusted historical borrowing (Jin et al., 2023)."""

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = "CAHB"
        self.gamma = float(priors.get("cahb_gamma", np.sqrt(3.0)))
        self.max_iter = int(priors.get("cahb_max_iter", 50))
        self.lambda_proj = float(priors.get("cahb_lambda", np.inf))
        self.lambda_trunc = float(priors.get("cahb_lambda_trunc", 0.0))
        self.invgam2 = float(priors.get("cahb_invgam2", 0.0))

    # Kernel helpers ------------------------------------------------------
    @staticmethod
    def _kernel_bandwidth(X: np.ndarray) -> np.ndarray:
        return _silverman_bandwidth(_as_2d(X))

    @staticmethod
    def _gaussian_kernel(x_eval: np.ndarray, X_ref: np.ndarray, h: np.ndarray) -> np.ndarray:
        return _gaussian_kernel_weights(x_eval, X_ref, h)

    # Kernel helpers ------------------------------------------------------
    @staticmethod
    def _kernel_covariance(X: np.ndarray) -> np.ndarray:
        bandwidth = _silverman_bandwidth(X)
        variances = np.maximum(bandwidth, 1e-3) ** 2
        return np.diag(variances)

    @staticmethod
    def _kernel_components(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        cov = CAHB._kernel_covariance(X)
        cov_inv = np.linalg.inv(cov)
        log_det = np.log(np.linalg.det(cov))
        p = X.shape[1]
        log_norm = -0.5 * (p * np.log(2.0 * np.pi) + log_det)
        return cov, cov_inv, log_norm

    @staticmethod
    def _kernel_matrix(
        X_eval: np.ndarray,
        X_ref: np.ndarray,
        cov_inv: np.ndarray,
        log_norm: float,
    ) -> np.ndarray:
        """
        Compute kernel matrix K(X_ref, X_eval) using multivariate Gaussian kernel.
        
        Args:
            X_eval: Evaluation points, shape (m, p)
            X_ref: Reference points, shape (n, p)
            cov_inv: Inverse covariance matrix, shape (p, p)
            log_norm: Log normalization constant
            
        Returns:
            Kernel matrix, shape (n, m)
        """
        X_eval = _as_2d(X_eval)
        X_ref = _as_2d(X_ref)
        # diff[i, j, :] = X_ref[i, :] - X_eval[j, :]
        diff = X_ref[:, None, :] - X_eval[None, :, :]
        # Compute Mahalanobis distance: diff @ cov_inv @ diff.T
        # einsum: for each (i,j), compute sum_k sum_l diff[i,j,k] * cov_inv[k,l] * diff[i,j,l]
        expo = -0.5 * np.einsum("ijk,kl,ijl->ij", diff, cov_inv, diff)
        return np.exp(expo + log_norm)

    @staticmethod
    def _kernel_vector(
        x_eval: np.ndarray,
        X_ref: np.ndarray,
        cov_inv: np.ndarray,
        log_norm: float,
    ) -> np.ndarray:
        x_eval = np.asarray(x_eval, dtype=float).ravel()
        diff = X_ref - x_eval
        expo = -0.5 * np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
        return np.exp(expo + log_norm)

    @staticmethod
    def _project_nonnegative_l1(v: np.ndarray, radius: float) -> np.ndarray:
        v = np.maximum(v, 0.0)
        if not np.isfinite(radius) or radius <= 0.0:
            return v
        if v.sum() <= radius:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(u) + 1) > (cssv - radius))[0][-1]
        theta = (cssv[rho] - radius) / (rho + 1)
        return np.maximum(v - theta, 0.0)

    @staticmethod
    def _opt_phi0(Y: np.ndarray, Z: np.ndarray, mu0: np.ndarray) -> float:
        sZs = 1.0 - Z
        residuals = Y - mu0
        a_hyper = 0.01
        b_hyper = 0.01
        numerator = 0.5 * np.dot(sZs, residuals ** 2) + b_hyper
        denominator = 1.0 + a_hyper + 0.5 * np.sum(sZs)
        value = numerator / max(denominator, 1e-8)
        return float(np.sqrt(max(value, 1e-8)))

    def _m_opt_mu0(
        self,
        kernel_matrix: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        tau: np.ndarray,
        phi0: float,
        theta0: np.ndarray,
    ) -> np.ndarray:
        """
        Optimize mu_0(x) at each observation location.
        
        Following Algorithm 2 in Jin et al. (2023) and utils.R lines 396-411:
        Computes weighted local regression estimate combining current control data
        and historical information via precision-weighted average.
        
        Args:
            kernel_matrix: K(X_i, X_j) matrix, shape (n, m)
            Y: Observed outcomes
            Z: Treatment indicators
            tau: Current tau^2 estimates (precision weights)
            phi0: Current phi_0 estimate (observation noise)
            theta0: Historical predictions
            
        Returns:
            Array of mu_0(X_i) values, shape (m,)
        """
        sZs = 1.0 - Z
        # Precision weights: W_i = 1/(2*phi0^2) + tau^2(X_i)/2 (R code line 406)
        Ws = 0.5 / (phi0 ** 2) + 0.5 * tau
        # Weighted observations: M_i = Y_i/(2*phi0^2) + tau^2(X_i)*theta0(X_i)/2 (R code line 407)
        Ms = Y / (2.0 * phi0 ** 2) + 0.5 * tau * theta0
        
        # Apply control group mask and kernel weights (R code line 408-409)
        weighted = kernel_matrix * sZs[:, None]
        numerator = (weighted * Ms[:, None]).sum(axis=0)
        denominator = (weighted * Ws[:, None]).sum(axis=0)
        
        # Compute mu_0 = sum(K*sZ*M) / sum(K*sZ*W) (R code line 410)
        mu0 = np.zeros_like(numerator)
        mask = denominator > 1e-8
        mu0[mask] = numerator[mask] / denominator[mask]
        mu0[~mask] = 0.0
        return mu0

    def _m_opt_tau(
        self,
        kernel_matrix: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        mu0: np.ndarray,
        theta0: np.ndarray,
    ) -> np.ndarray:
        """
        Optimize tau^2(x) at each observation location.
        
        Following Algorithm 2 in Jin et al. (2023) and utils.R lines 415-478:
        1. Compute raw tau^2 estimates
        2. Apply truncation threshold (set small values to 0)
        3. Project onto L1 ball for sparsity
        
        Args:
            kernel_matrix: K(X_i, X_j) matrix, shape (n, m)
            Y: Observed outcomes
            Z: Treatment indicators
            mu0: Current mu0 estimates
            theta0: Historical predictions
            
        Returns:
            Array of tau^2(X_i) values, shape (m,)
        """
        sZs = 1.0 - Z
        diff_sq = (mu0 - theta0) ** 2
        weighted = kernel_matrix * sZs[:, None]
        
        # Numerator: weighted sum of kernel weights (R code line 466)
        numerator = (weighted).sum(axis=0)
        
        # Denominator: weighted sum of squared differences + prior variance term (R code line 468)
        denominator = (weighted * diff_sq[:, None]).sum(axis=0) + self.invgam2
        
        # Raw tau^2 estimate (R code line 469)
        tau_raw = numerator / np.maximum(denominator, 1e-8)
        tau_raw = np.maximum(tau_raw, 0.0)
        
        # Truncation: set small values to 0 (R code line 473)
        tau_raw[tau_raw <= self.lambda_trunc] = 0.0
        
        # L1 projection for sparsity (R code line 474)
        # Radius scaled by log(m) as in R implementation
        radius = self.lambda_proj
        if np.isfinite(radius):
            radius = radius * max(np.log(kernel_matrix.shape[1]), 1.0)
        return self._project_nonnegative_l1(tau_raw, radius)

    def _fit_cahb_model(
        self,
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
    ) -> Optional[dict]:
        """
        Fit CAHB model via coordinate-wise optimization.
        
        Following Algorithm 2 in Jin et al. (2023) and utils.R lines 598-669:
        Iteratively optimize mu_0, phi_0, and tau^2 until convergence.
        
        Algorithm (mu0.info.est.fn in R):
        1. Initialize tau^2 = 0, phi_0 = 1 or from previous fit
        2. Repeat until convergence:
            a. Update mu_0(X_i) for all i (weighted local regression)
            b. Update phi_0 (global variance parameter)
            c. Update tau^2(X_i) for all i (precision weights with L1 projection)
            d. Check convergence: max(||mu0_new - mu0_old||^2, ||tau_new - tau_old||^2) < 1e-5
        3. Return fitted parameters and kernel information
        
        Args:
            X_curr: Current trial covariates, shape (n, p)
            Y_curr: Current trial outcomes, shape (n,)
            Z_curr: Current trial treatment indicators, shape (n,)
            
        Returns:
            Dictionary containing fitted parameters and kernel matrices, or None if insufficient control data
        """
        X_curr = _as_2d(X_curr)
        Y_curr = np.asarray(Y_curr, dtype=float)
        Z_curr = np.asarray(Z_curr, dtype=float)
        
        # Check if we have sufficient control group data (R code line 614-615)
        if (1.0 - Z_curr).sum() < 2:
            return None

        # Compute kernel matrix components (R code line 451-453)
        _, cov_inv, log_norm = self._kernel_components(X_curr)
        kernel_matrix = self._kernel_matrix(X_curr, X_curr, cov_inv, log_norm)
        
        # Get historical predictions theta_0(X_i) (R code line 453)
        theta0 = self._kernel_mean(self.X_h, self.Y_h, self.h_hist, X_curr)

        # Initialize parameters (R code line 614-624)
        tau = np.zeros(X_curr.shape[0])
        phi0 = max(np.std(Y_curr[Z_curr == 0], ddof=1), 1.0)
        mu0_prev = None
        phi0_prev = None

        # Coordinate-wise optimization loop (R code line 626-653)
        for iter_idx in range(self.max_iter):
            # Update mu_0 (R code line 628)
            mu0 = self._m_opt_mu0(kernel_matrix, Y_curr, Z_curr, tau, phi0, theta0)
            
            # Update phi_0 (R code line 631-632)
            phi0 = self._opt_phi0(Y_curr, Z_curr, mu0)
            
            # Update tau^2 (R code line 635-637)
            tau_new = self._m_opt_tau(kernel_matrix, Y_curr, Z_curr, mu0, theta0)
            
            # Check convergence (R code line 643-651)
            if mu0_prev is not None and phi0_prev is not None:
                err_mu = np.mean((mu0 - mu0_prev) ** 2)
                err_tau = np.mean((tau_new - tau) ** 2)
                err_phi0 = (phi0 - phi0_prev) ** 2
                
                err_all = max(err_mu, err_tau, err_phi0)
                if err_all < 1e-5:
                    tau = tau_new
                    break
            
            mu0_prev = mu0
            phi0_prev = phi0
            tau = tau_new

        # Prepare return dictionary with fitted parameters (R code line 657-661)
        diff_sq = (mu0 - theta0) ** 2
        phi0_sq = phi0 ** 2
        return {
            "X": X_curr,
            "Y": Y_curr,
            "Z": Z_curr,
            "sZs": 1.0 - Z_curr,
            "cov_inv": cov_inv,
            "log_norm": log_norm,
            "kernel_matrix": kernel_matrix,
            "theta0": theta0,
            "mu0": mu0,
            "tau": tau,
            "phi0": phi0,
            "phi0_sq": phi0_sq,
            "diff_sq": diff_sq,
        }

    def _tau_at_x(self, context: dict, x_eval: np.ndarray) -> float:
        weights = self._kernel_vector(x_eval, context["X"], context["cov_inv"], context["log_norm"])
        numerator = np.dot(context["sZs"], weights)
        if numerator <= 1e-8:
            return 0.0
        denominator = np.dot(context["sZs"] * context["diff_sq"], weights) + self.gamma ** 2
        return float(np.clip(numerator / max(denominator, 1e-8), 0.0, 1e6))

    def _R_n_at_x(self, context: dict, x_eval: np.ndarray) -> float:
        weights = self._kernel_vector(x_eval, context["X"], context["cov_inv"], context["log_norm"])
        denominator = np.dot(context["sZs"], weights)
        if denominator <= 1e-8:
            return 1.0
        contributions = 1.0 + context["phi0_sq"] * context["tau"]
        numerator = np.dot(context["sZs"] * contributions, weights)
        return float(np.clip(numerator / denominator, 1.0, 1e6))

    def _posterior_mu0(self, context: dict, x_eval: np.ndarray) -> float:
        weights = self._kernel_vector(x_eval, context["X"], context["cov_inv"], context["log_norm"])
        control_weights = context["sZs"] * weights
        denominator = np.dot(control_weights, (1.0 / context["phi0_sq"]) + context["tau"])
        if denominator <= 1e-8:
            return float(self._kernel_mean(
                context["X"][context["Z"] == 0],
                context["Y"][context["Z"] == 0],
                _silverman_bandwidth(context["X"][context["Z"] == 0]),
                x_eval.reshape(1, -1),
            )[0])
        numerator = np.dot(
            control_weights,
            (context["Y"] / context["phi0_sq"]) + context["tau"] * context["theta0"],
        )
        return float(numerator / denominator)

    def _posterior_mu0_variance(self, context: dict, x_eval: np.ndarray) -> float:
        weights = self._kernel_vector(x_eval, context["X"], context["cov_inv"], context["log_norm"])
        control_weights = context["sZs"] * weights
        precision_terms = (1.0 / context["phi0_sq"]) + context["tau"]
        denom = np.dot(control_weights, precision_terms)
        if denom <= 1e-8:
            return np.inf
        return float(1.0 / max(denom, 1e-8))

    def _posterior_mu0_stats(self, context: dict, X_eval: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xs = _as_2d(X_eval)
        means = np.zeros(xs.shape[0])
        variances = np.zeros(xs.shape[0])
        for idx, x in enumerate(xs):
            means[idx] = self._posterior_mu0(context, x)
            variances[idx] = self._posterior_mu0_variance(context, x)
        return means, variances

    def _posterior_mu1_stats(
        self, context: dict, X_eval: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        xs = _as_2d(X_eval)
        mask = Z > 0.5
        if mask.sum() < 2:
            return None, None
        X_trt = context["X"][mask]
        Y_trt = context["Y"][mask]
        bandwidth = self._kernel_bandwidth(X_trt)
        means = np.zeros(xs.shape[0])
        variances = np.zeros(xs.shape[0])
        for idx, x in enumerate(xs):
            w = self._gaussian_kernel(x, X_trt, bandwidth)
            w_sum = float(np.sum(w))
            if w_sum <= 1e-8:
                means[idx] = float(np.mean(Y_trt))
                variances[idx] = np.inf
                continue
            m = float(np.dot(w, Y_trt) / w_sum)
            resid = Y_trt - m
            local_var = float(np.dot(w, resid**2) / max(w_sum, 1e-8))
            means[idx] = m
            variances[idx] = max(local_var / max(w_sum, 1e-8), 1e-8)
        return means, variances

    @staticmethod
    def _kernel_bandwidth(X: np.ndarray) -> np.ndarray:
        """Helper for local kernel regressions (shared with CAHB-PP variants)."""
        if X.size == 0:
            return np.ones(X.shape[1] if X.ndim > 1 else 1)
        return _silverman_bandwidth(_as_2d(X))

    # Public interface ----------------------------------------------------
    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new) -> AllocationResult:
        if len(X_curr) < 2:
            return AllocationResult(pi_treatment=0.5, diagnostics={"R_n": 1.0})
        context = self._fit_cahb_model(X_curr, Y_curr, Z_curr)
        if context is None:
            return AllocationResult(pi_treatment=0.5, diagnostics={"R_n": 1.0})
        weights = self._kernel_vector(X_new, context["X"], context["cov_inv"], context["log_norm"])
        n0_local = float(np.dot(context["sZs"], weights))
        n1_local = float(np.dot(context["Z"], weights))
        R_n = self._R_n_at_x(context, X_new)
        n0_eff = R_n * n0_local
        pi = self._imbalance_probability(n0_eff, n1_local)
        return AllocationResult(pi_treatment=pi, diagnostics={"R_n": R_n})

    def estimate_treatment_effect(self, X, Y, Z):
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)
        if (Z == 0).sum() < 2 or (Z == 1).sum() < 2:
            delta = float(Y[Z == 1].mean() - Y[Z == 0].mean())
            return delta, np.nan, np.nan, 0.5
        context = self._fit_cahb_model(X, Y, Z)
        if context is None:
            delta = float(Y[Z == 1].mean() - Y[Z == 0].mean())
            return delta, np.nan, np.nan, 0.5
        mu0_mean, mu0_var = self._posterior_mu0_stats(context, X)
        mu1_stats = self._posterior_mu1_stats(context, X, Y, Z)
        if mu1_stats[0] is None:
            delta = float(Y[Z == 1].mean() - Y[Z == 0].mean())
            return delta, np.nan, np.nan, 0.5
        mu1_mean, mu1_var = mu1_stats
        draws = []
        sd0 = np.sqrt(np.maximum(mu0_var, 1e-8))
        sd1 = np.sqrt(np.maximum(mu1_var, 1e-8))
        for _ in range(self.posterior_draws):
            mu0_samples = np.random.normal(mu0_mean, sd0)
            mu1_samples = np.random.normal(mu1_mean, sd1)
            draws.append(float(np.mean(mu1_samples - mu0_samples)))
        return _delta_summary(np.asarray(draws), self.alpha)


# ==============================================================================
# CAHB-PP: Covariate-Adjusted Historical Borrowing with Power Prior
# ==============================================================================
# Completely independent implementation based on CAHB-PP manuscript
# Does NOT inherit from CAHB - follows different notation and algorithm

class CAHBPowerPrior(BaseMethod):
    """
    CAHB-PP: Covariate-Adjusted Historical Borrowing with Power Prior.
    
    Independent implementation following CAHB-PP manuscript notation:
    - Uses power prior formulation: L_h(μ_0, σ_h^2)^a(x)
    - Local discount parameter a(x) estimated via bridge sampling
    - Different from CAHB's tau^2(x) approach
    
    Notation (CAHB-PP manuscript):
    - a(x): Local discount parameter in (0,1)
    - θ_h(x): Historical control mean prediction
    - π(a): Beta(α, β) prior for discount parameter  
    - R_n(x): Effective sample size ratio
    
    Reference: CAHB-PP manuscript Algorithm 1 & 2
    """

    method_label = "CAHB-PP"

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        """Initialize CAHB-PP method with historical data and priors."""
        # Initialize base
        self.X_h = _as_2d(historical_data["X_h"])
        self.Y_h = np.asarray(historical_data["Y_h"], dtype=float)
        self.scenario = scenario_params
        self.priors = priors
        self.name = self.method_label
        
        # Historical data bandwidth
        self.h_hist = _silverman_bandwidth(self.X_h)
        
        # Significance level
        try:
            import config as _cfg
            self.alpha = float(getattr(_cfg, "ALPHA", 0.05))
        except ImportError:
            self.alpha = 0.05
        
        # Power prior discount parameter prior: a ~ Beta(α_a, β_a)
        self.a_alpha = float(priors.get("a_beta_a", 1.0))
        self.a_beta = float(priors.get("a_beta_b", 1.0))
        
        # Bridge sampling parameters for a(x) estimation
        self.bridge_samples = max(8, int(priors.get("bridge_samples", 64)))
        self.bridge_burn = max(8, int(priors.get("bridge_burn_in", 32)))
        self.bridge_thin = max(1, int(priors.get("bridge_thin", 1)))
        self.bridge_step = float(priors.get("bridge_step_size", 0.12))
        
        # Calibration settings
        try:
            import config as _cfg
            self.calibration_max_samples = int(getattr(_cfg, "CALIBRATION_MAX_SAMPLES", 32))
        except ImportError:
            self.calibration_max_samples = 32

        # Cache for discount parameters and calibration
        self._discount_cache = []
        self._calibration_payload = []
    
    # -------------------------------------------------------------------------
    # Kernel functions (multivariate Gaussian)
    # -------------------------------------------------------------------------

    @staticmethod
    def _kernel_bandwidth(X: np.ndarray) -> np.ndarray:
        """Silverman's rule bandwidth for each dimension."""
        return _silverman_bandwidth(X)

    @staticmethod
    def _gaussian_kernel(x_eval: np.ndarray, X_ref: np.ndarray, h: np.ndarray) -> np.ndarray:
        """
        Multivariate Gaussian product kernel.
        
        K(x, X_i) = ∏_j φ((x_j - X_ij) / h_j)
        
        Args:
            x_eval: Evaluation point, shape (p,)
            X_ref: Reference points, shape (n, p)
            h: Bandwidth, shape (p,)
            
        Returns:
            Kernel weights, shape (n,)
        """
        return _gaussian_kernel_weights(x_eval, X_ref, h)
    
    # -------------------------------------------------------------------------
    # Historical prediction θ_h(x)
    # -------------------------------------------------------------------------
    
    def _theta_h(self, X_eval: np.ndarray) -> np.ndarray:
        """
        Compute historical control mean predictions θ_h(x).
        
        θ_h(x) = kernel smoothing estimate from historical control data
        
        Args:
            X_eval: Evaluation points, shape (m, p)
            
        Returns:
            Historical predictions, shape (m,)
        """
        X_eval = _as_2d(X_eval)
        m = X_eval.shape[0]
        theta_h = np.zeros(m)
        
        for i in range(m):
            weights = self._gaussian_kernel(X_eval[i], self.X_h, self.h_hist)
            weight_sum = weights.sum()
            if weight_sum > 1e-8:
                theta_h[i] = np.dot(weights, self.Y_h) / weight_sum
            else:
                theta_h[i] = np.mean(self.Y_h)  # Fallback to global mean
        
        return theta_h
    
    # -------------------------------------------------------------------------
    # Algorithm 1: Gibbs Sampling Framework
    # -------------------------------------------------------------------------
    
    def _gibbs_sampler_at_x(self, x_eval: np.ndarray, X_curr: np.ndarray, 
                           Y_curr: np.ndarray, Z_curr: np.ndarray,
                           n_gibbs: int = 100, burn_in: int = 50) -> dict:
        """
        Implement Algorithm 1: Variance-aware local power-prior sampler at x.
        
        Gibbs sampling scheme:
        1. Draw σ²_{0,c} ~ IG(α₀ + W₀/2, β₀ + SS₀(μ₀)/2)
        2. Draw σ²_{0,h} ~ IG(α₀ + aW_h/2, β₀ + aSS_h(μ₀)/2)
        3. Draw a ~ π(a | μ₀, σ²_{0,h}, D_h) using MH step
        4. Draw μ₀ ~ N(m₀, V₀) with precision-weighted formula
        5. Draw σ²_1 ~ IG(α₀ + W₁/2, β₀ + SS₁(μ₁)/2)
        6. Draw μ₁ ~ N(m₁, V₁)
        
        Args:
            x_eval: Evaluation point, shape (p,)
            X_curr: Current trial covariates
            Y_curr: Current trial outcomes
            Z_curr: Current trial treatment indicators
            n_gibbs: Total Gibbs iterations
            burn_in: Burn-in iterations
            
        Returns:
            Dictionary with posterior draws and summaries
        """
        from scipy.stats import invgamma, norm
        
        x_eval = np.asarray(x_eval, dtype=float).ravel()
        X_curr = _as_2d(X_curr)
        Y_curr = np.asarray(Y_curr, dtype=float)
        Z_curr = np.asarray(Z_curr, dtype=int)
        
        # Separate control and treatment groups
        ctrl_mask = (Z_curr == 0)
        trt_mask = (Z_curr == 1)
        
        if ctrl_mask.sum() < 2:
            return self._fallback_gibbs_result()
        
        X_ctrl = X_curr[ctrl_mask]
        Y_ctrl = Y_curr[ctrl_mask]
        h_ctrl = self._kernel_bandwidth(X_ctrl)
        
        # Compute local sufficient statistics for control
        ctrl_weights = self._gaussian_kernel(x_eval, X_ctrl, h_ctrl)
        W0 = ctrl_weights.sum()
        Y0_bar = np.dot(ctrl_weights, Y_ctrl) / W0 if W0 > 1e-8 else 0.0
        
        # Historical sufficient statistics
        hist_weights = self._gaussian_kernel(x_eval, self.X_h, self.h_hist)
        hist_weights = np.asarray(hist_weights, dtype=float)
        Wh = float(hist_weights.sum())
        Yh_bar = np.dot(hist_weights, self.Y_h) / Wh if Wh > 1e-8 else 0.0
        
        # Treatment sufficient statistics (if available)
        if trt_mask.sum() >= 2:
            X_trt = X_curr[trt_mask]
            Y_trt = Y_curr[trt_mask]
            h_trt = self._kernel_bandwidth(X_trt)
            trt_weights = self._gaussian_kernel(x_eval, X_trt, h_trt)
            W1 = trt_weights.sum()
            Y1_bar = np.dot(trt_weights, Y_trt) / W1 if W1 > 1e-8 else 0.0
        else:
            W1, Y1_bar = 0.0, 0.0
        
        # Prior hyperparameters
        alpha0 = self.priors.get("ig_shape", 0.01)
        beta0 = self.priors.get("ig_scale", 0.01)
        
        # Initialize chain
        mu0 = Y0_bar
        mu1 = Y1_bar if W1 > 1e-8 else Y0_bar
        sigma2_0c = 1.0
        sigma2_0h = 1.0
        sigma2_1 = 1.0
        a = 0.5
        
        # Storage for posterior draws
        draws = {
            "mu0": [], "sigma2_0c": [], "sigma2_0h": [],
            "a": [], "mu1": [], "sigma2_1": []
        }
        
        # Cache for Laplace approximation (avoid recomputing for rejected proposals)
        laplace_cache = None
        
        # Gibbs sampling loop
        for t in range(n_gibbs):
            # 1. Draw σ²_{0,c} | μ₀, D_n
            SS0 = np.dot(ctrl_weights, (Y_ctrl - mu0) ** 2)
            alpha_post = max(alpha0 + W0 / 2.0, 0.5)  # Ensure minimum shape
            beta_post = max(beta0 + SS0 / 2.0, 0.01)   # Ensure minimum scale
            
            try:
                sigma2_0c = invgamma.rvs(alpha_post, scale=beta_post)
                sigma2_0c = np.clip(sigma2_0c, 1e-4, 1e4)  # Bounded range
            except:
                sigma2_0c = SS0 / W0 if W0 > 0 else 1.0
            
            # 2. Draw σ²_{0,h} | μ₀, a, D_h
            SS_h = np.dot(hist_weights, (self.Y_h - mu0) ** 2)
            alpha_h_post = max(alpha0 + a * Wh / 2.0, 0.5)  # Ensure minimum shape
            beta_h_post = max(beta0 + a * SS_h / 2.0, 0.01)   # Ensure minimum scale
            
            try:
                sigma2_0h = invgamma.rvs(alpha_h_post, scale=beta_h_post)
                sigma2_0h = np.clip(sigma2_0h, 1e-4, 1e4)  # Bounded range
            except:
                sigma2_0h = SS_h / Wh if Wh > 0 else 1.0
            
            # 3. Draw a | μ₀, σ²_{0,h}, D_h using MH step with Laplace approximation
            a, laplace_cache = self._mh_step_discount(a, mu0, sigma2_0h, hist_weights, self.Y_h, laplace_cache)
            
            # 4. Draw μ₀ | σ²_{0,c}, σ²_{0,h}, a, D_n, D_h
            precision_curr = W0 / sigma2_0c
            precision_hist = a * Wh / sigma2_0h
            V0_inv = precision_curr + precision_hist
            
            if V0_inv > 1e-8:
                m0 = (precision_curr * Y0_bar + precision_hist * Yh_bar) / V0_inv
                V0 = 1.0 / V0_inv
                mu0 = norm.rvs(loc=m0, scale=np.sqrt(V0))
            else:
                mu0 = Y0_bar
            
            # 5. Draw σ²_1 | μ₁, D_n (if treatment data available)
            if W1 > 1e-8:
                SS1 = np.dot(trt_weights, (Y_trt - mu1) ** 2)
                alpha1_post = max(alpha0 + W1 / 2.0, 0.5)
                beta1_post = max(beta0 + SS1 / 2.0, 0.01)
                
                try:
                    sigma2_1 = invgamma.rvs(alpha1_post, scale=beta1_post)
                    sigma2_1 = np.clip(sigma2_1, 1e-4, 1e4)
                except:
                    sigma2_1 = SS1 / W1 if W1 > 0 else 1.0
                
                # 6. Draw μ₁ | σ²_1, D_n
                V1 = sigma2_1 / W1
                mu1 = norm.rvs(loc=Y1_bar, scale=np.sqrt(V1))
            
            # Store draws after burn-in
            if t >= burn_in:
                draws["mu0"].append(mu0)
                draws["sigma2_0c"].append(sigma2_0c)
                draws["sigma2_0h"].append(sigma2_0h)
                draws["a"].append(a)
                draws["mu1"].append(mu1)
                draws["sigma2_1"].append(sigma2_1)
        
        # Convert to arrays
        for key in draws:
            draws[key] = np.array(draws[key])
        
        # Compute summaries
        result = {
            "mu0_mean": float(np.mean(draws["mu0"])),
            "mu1_mean": float(np.mean(draws["mu1"])),
            "sigma2_0c_mean": float(np.mean(draws["sigma2_0c"])),
            "sigma2_0h_mean": float(np.mean(draws["sigma2_0h"])),
            "a_mean": float(np.mean(draws["a"])),
            "a_var": float(np.var(draws["a"])),
            "a_ci_low": float(np.quantile(draws["a"], self.alpha / 2.0)),
            "a_ci_high": float(np.quantile(draws["a"], 1.0 - self.alpha / 2.0)),
            "draws": draws,
        }
        
        return result
    
    def _mh_step_discount(self, a_current: float, mu0: float, sigma2_0h: float,
                         hist_weights: np.ndarray, Y_h: np.ndarray,
                         laplace_cache: dict = None) -> tuple:
        """
        Metropolis-Hastings step for discount parameter a using Normalized Power Prior.
        
        From CAHB-PP manuscript (Laplace approximation method):
        - Find mode θ̂_a = arg max f(θ; a) where f(θ; a) = a·log L_h(θ) + log π(θ)
        - Compute Hessian H(a) at the mode
        - Approximate: m̂_h(a) ≈ (2π)^{d/2} |H(a)|^{-1/2} exp(f(θ̂_a; a))
        - Acceptance ratio includes ratio of normalizing constants
        
        Args:
            a_current: Current discount parameter
            mu0: Current mean parameter (used as initial value)
            sigma2_0h: Current variance parameter (used as initial value)
            hist_weights: Historical kernel weights
            Y_h: Historical outcomes
            laplace_cache: Cache storing previous Laplace approximation results
            
        Returns:
            Tuple of (new_a, updated_cache)
        """
        # Logit-scale random walk proposal
        logit_current = np.log(a_current / (1.0 - a_current))
        epsilon = np.random.normal(0, self.bridge_step)
        logit_proposed = logit_current + epsilon
        
        # Expit (inverse logit) transformation
        a_proposed = 1.0 / (1.0 + np.exp(-logit_proposed))
        a_proposed = np.clip(a_proposed, 0.01, 0.99)
        
        # Log prior ratio: π(a_p) / π(a_t)
        log_prior_current = (self.a_alpha - 1.0) * np.log(a_current) + \
                           (self.a_beta - 1.0) * np.log(1.0 - a_current)
        log_prior_proposed = (self.a_alpha - 1.0) * np.log(a_proposed) + \
                            (self.a_beta - 1.0) * np.log(1.0 - a_proposed)
        
        # Retrieve or compute Laplace approximation for current a
        if laplace_cache is None or 'f_at_mode' not in laplace_cache:
            laplace_current = self._laplace_approximation_at_a(
                a_current, hist_weights, Y_h, mu0, sigma2_0h
            )
        else:
            laplace_current = laplace_cache
        
        # Compute Laplace approximation for proposed a
        laplace_proposed = self._laplace_approximation_at_a(
            a_proposed, hist_weights, Y_h, mu0, sigma2_0h
        )
        
        # Log normalizing constant ratio using Laplace approximation
        # log(m_h(a_t) / m_h(a_p)) = [f(θ̂_t) - 0.5·log|H_t|] - [f(θ̂_p) - 0.5·log|H_p|]
        log_norm_current = laplace_current['f_at_mode'] - 0.5 * laplace_current['log_det_hessian']
        log_norm_proposed = laplace_proposed['f_at_mode'] - 0.5 * laplace_proposed['log_det_hessian']
        
        # Jacobian adjustment for logit transformation
        log_jacobian = np.log(a_proposed * (1.0 - a_proposed)) - \
                      np.log(a_current * (1.0 - a_current))
        
        # Log acceptance ratio
        # R = [π(a_p)·exp(f(θ̂_t))/√|H_t|] / [π(a_t)·exp(f(θ̂_p))/√|H_p|] · Jacobian
        log_alpha_mh = (log_prior_proposed - log_prior_current) + \
                       (log_norm_current - log_norm_proposed) + \
                       log_jacobian
        
        # Accept/reject
        if np.log(np.random.rand()) < log_alpha_mh:
            return a_proposed, laplace_proposed
        else:
            return a_current, laplace_current
    
    def _laplace_approximation_at_a(self, a: float, hist_weights: np.ndarray, 
                                    Y_h: np.ndarray, mu0_init: float, 
                                    sigma2_init: float) -> dict:
        """
        Compute Laplace approximation to m_h(a) by finding mode and Hessian.
        
        From CAHB-PP manuscript:
        1. Define f(θ; a) = a·log L_h(θ) + log π₀(θ)
        2. Find mode θ̂_a = arg max_θ f(θ; a) via optimization
        3. Compute Hessian H(a) at θ̂_a
        4. Approximate: m̂_h(a) ≈ (2π)^{d/2} |H(a)|^{-1/2} exp(f(θ̂_a; a))
        
        Args:
            a: Discount parameter
            hist_weights: Historical kernel weights
            Y_h: Historical outcomes
            mu0_init: Initial value for mean parameter
            sigma2_init: Initial value for variance parameter
            
        Returns:
            Dictionary with:
                - theta_mode: Mode θ̂_a
                - f_at_mode: f(θ̂_a; a)
                - log_det_hessian: log|H(a)|
        """
        from scipy.optimize import minimize
        
        Wh = hist_weights.sum()
        
        # Prior hyperparameters (vague priors)
        alpha0 = self.priors.get("ig_shape", 0.01)
        beta0 = self.priors.get("ig_scale", 0.01)
        
        # Define negative log-integrand: -f(θ; a) = -a·log L_h(θ) - log π(θ)
        # θ = [μ, log(σ²)] for unconstrained optimization
        def f_value(theta_vec: np.ndarray) -> float:
            """Compute f(θ; a) = a·log L_h(θ) + log π₀(θ)."""
            theta_vec = np.asarray(theta_vec, dtype=float)
            mu = theta_vec[0]
            log_sigma2 = theta_vec[1]
            sigma2 = float(np.exp(log_sigma2))
            if not np.isfinite(sigma2):
                return 1e10
            sigma2 = float(np.clip(sigma2, 1e-6, 1e8))

            SS = np.dot(hist_weights, (Y_h - mu) ** 2)
            log_lik = -0.5 * Wh * np.log(2.0 * np.pi * sigma2) - 0.5 * SS / sigma2
            log_prior = -(alpha0 + 1.0) * log_sigma2 - beta0 / sigma2

            return float(a * log_lik + log_prior)

        
        def neg_log_integrand(theta_vec: np.ndarray) -> float:
            """Negative of f(θ; a) for minimization (L-BFGS-B)."""
            return -f_value(theta_vec)
        
        # Initial guess (on log scale for σ²)
        theta_init = np.array([mu0_init, np.log(max(sigma2_init, 1e-4))])
        
        # Optimize to find mode using L-BFGS-B
        opt_result = None

        try:
            result = minimize(
                neg_log_integrand,
                theta_init,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success or result.fun < 1e8:
                theta_mode = result.x
                f_at_mode = -result.fun
                opt_result = result
            else:
                # Fallback to initial value
                theta_mode = theta_init
                f_at_mode = -neg_log_integrand(theta_init)
        except Exception:
            theta_mode = theta_init
            f_at_mode = -neg_log_integrand(theta_init)
        
        # Compute Hessian at mode using optimizer information (fallback to finite diff)
        log_det_hessian = self._compute_log_det_hessian(theta_mode, f_value, opt_result=opt_result)
        
        return {
            'theta_mode': theta_mode,
            'f_at_mode': f_at_mode,
            'log_det_hessian': log_det_hessian,
        }
    
    def _compute_log_det_hessian(
        self,
        theta_mode: np.ndarray,
        f_func: Callable[[np.ndarray], float],
        opt_result: Optional[object] = None,
        abs_step: float = 1e-4,
        rel_step: float = 1e-3,
    ) -> float:
        """
        Compute log determinant of Hessian matrix at the mode.
        
        Prefer using optimizer-provided curvature (L-BFGS inverse Hessian) and
        fall back to stabilized finite differences when necessary.
        """
        # Attempt to use optimizer-provided inverse Hessian
        if opt_result is not None:
            hess_inv = getattr(opt_result, "hess_inv", None)
            if hess_inv is not None:
                try:
                    if hasattr(hess_inv, "todense"):
                        hess_inv_mat = np.asarray(hess_inv.todense(), dtype=float)
                    else:
                        hess_inv_mat = np.asarray(hess_inv, dtype=float)
                    if hess_inv_mat.size > 0:
                        # Hessian of objective = H_{-f}; convert inverse to Hessian
                        H_neg = np.linalg.pinv(hess_inv_mat)
                        H = -0.5 * (H_neg + H_neg.T)
                        eigvals = np.linalg.eigvalsh(H)
                        min_eig = float(np.min(eigvals))
                        if min_eig <= 0:
                            H += np.eye(H.shape[0]) * (1e-8 - min_eig + 1e-10)
                        sign, log_det = np.linalg.slogdet(H)
                        if sign > 0:
                            return float(log_det)
                except Exception:
                    warnings.warn(
                        "Optimizer Hessian inversion failed; using finite differences.",
                        RuntimeWarning,
                    )

        theta_mode = np.asarray(theta_mode, dtype=float)
        dim = theta_mode.size
        if dim == 0:
            return 0.0

        # Adaptive step per coordinate
        steps = np.clip(np.abs(theta_mode) * rel_step, abs_step, 1.0)
        base_val = float(f_func(theta_mode))

        def safe_eval(theta_vec: np.ndarray) -> float:
            try:
                val = float(f_func(theta_vec))
                if not np.isfinite(val):
                    raise ValueError
                return val
            except Exception:
                return base_val

        hessian = np.zeros((dim, dim), dtype=float)
        eye = np.eye(dim)

        for i in range(dim):
            hi = steps[i]
            e_i = eye[i]
            f_plus = safe_eval(theta_mode + hi * e_i)
            f_minus = safe_eval(theta_mode - hi * e_i)
            hessian[i, i] = (f_plus - 2.0 * base_val + f_minus) / (hi ** 2)

            for j in range(i + 1, dim):
                hj = steps[j]
                e_j = eye[j]
                f_pp = safe_eval(theta_mode + hi * e_i + hj * e_j)
                f_pm = safe_eval(theta_mode + hi * e_i - hj * e_j)
                f_mp = safe_eval(theta_mode - hi * e_i + hj * e_j)
                f_mm = safe_eval(theta_mode - hi * e_i - hj * e_j)
                mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * hi * hj)
                hessian[i, j] = mixed
                hessian[j, i] = mixed

        # Symmetrize and enforce positive definiteness
        hessian = 0.5 * (hessian + hessian.T)
        H = -hessian
        eigvals, eigvecs = np.linalg.eigh(H)
        min_eig = float(np.min(eigvals))
        if not np.isfinite(min_eig):
            raise RuntimeError("Numerical Hessian produced non-finite eigenvalues.")
        if min_eig < 1e-8:
            shift = 1e-8 - min_eig + 1e-10
            H += np.eye(dim) * shift

        sign, log_det = np.linalg.slogdet(H)
        if sign <= 0:
            # Final safeguard: add jitter proportional to trace
            jitter = max(np.trace(H), 1.0) * 1e-8
            H += np.eye(dim) * jitter
            sign, log_det = np.linalg.slogdet(H)
            if sign <= 0:
                raise RuntimeError("Numerical Hessian not positive definite at mode.")
        return float(log_det)
    
    def _fallback_gibbs_result(self) -> dict:
        """Fallback when insufficient data."""
        return {
            "mu0_mean": 0.0, "mu1_mean": 0.0,
            "sigma2_0c_mean": 1.0, "sigma2_0h_mean": 1.0,
            "a_mean": 0.0, "a_var": 0.0,
            "a_ci_low": 0.0, "a_ci_high": 0.0,
            "draws": {"mu0": np.array([]), "mu1": np.array([]),
                     "sigma2_0c": np.array([]), "sigma2_0h": np.array([]),
                     "a": np.array([]), "sigma2_1": np.array([])},
        }
    
    # -------------------------------------------------------------------------
    # Discount parameter a(x) estimation via bridge sampling (deprecated)
    # -------------------------------------------------------------------------
    
    def _estimate_discount_at_x(self, x_eval: np.ndarray, X_curr: np.ndarray, 
                                Y_curr: np.ndarray, Z_curr: np.ndarray) -> dict:
        """
        Estimate discount parameter a(x) at specific covariate value.
        
        Uses bridge sampling to sample from posterior:
        p(a | data) ∝ π(a) × L_h(data_curr | a, θ_h(x))
        
        Args:
            x_eval: Covariate value, shape (p,)
            X_curr: Current trial covariates
            Y_curr: Current trial outcomes
            Z_curr: Current trial treatment indicators
            
        Returns:
            Dictionary with mean, variance, CI, and compatibility
        """
        # Compute local statistics for control group
        control_mask = (Z_curr == 0)
        if control_mask.sum() < 2:
            return {"mean": 0.0, "var": 0.0, "ci_low": 0.0, "ci_high": 0.0, "compatibility": 0.0}
        
        X_ctrl = X_curr[control_mask]
        Y_ctrl = Y_curr[control_mask]
        h_ctrl = self._kernel_bandwidth(X_ctrl)
        
        # Local control statistics at x
        ctrl_weights = self._gaussian_kernel(x_eval, X_ctrl, h_ctrl)
        ctrl_weight_sum = ctrl_weights.sum()
        
        if ctrl_weight_sum <= 1e-8:
            return {"mean": 0.0, "var": 0.0, "ci_low": 0.0, "ci_high": 0.0, "compatibility": 0.0}
        
        ctrl_mean = np.dot(ctrl_weights, Y_ctrl) / ctrl_weight_sum
        ctrl_var = np.dot(ctrl_weights, (Y_ctrl - ctrl_mean)**2) / ctrl_weight_sum
        ctrl_var = max(ctrl_var, 1e-6)
        
        # Historical statistics at x
        hist_weights = self._gaussian_kernel(x_eval, self.X_h, self.h_hist)
        hist_weight_sum = hist_weights.sum()
        
        if hist_weight_sum <= 1e-8:
            return {"mean": 0.0, "var": 0.0, "ci_low": 0.0, "ci_high": 0.0, "compatibility": 0.0}
        
        hist_mean = np.dot(hist_weights, self.Y_h) / hist_weight_sum
        hist_var = np.dot(hist_weights, (self.Y_h - hist_mean)**2) / hist_weight_sum
        hist_var = max(hist_var, 1e-6)
        
        # Compatibility metric
        delta_mu = hist_mean - ctrl_mean
        compatibility = float(np.exp(-0.5 * (delta_mu ** 2) / ctrl_var))
        
        # Define log prior for a: π(a) = Beta(α_a, β_a)
        def log_prior_a(a_val: float) -> float:
            if not 0.0 < a_val < 1.0:
                return -np.inf
            return (self.a_alpha - 1.0) * np.log(a_val) + (self.a_beta - 1.0) * np.log(1.0 - a_val)

        # Define log-likelihood: L_h(data | a, θ_h)
        # Simplified: Normal likelihood with effective sample size a × n_h
        def log_likelihood_a(a_val: float) -> float:
            if not 0.0 < a_val < 1.0:
                return -np.inf
            
            ess_hist = max(hist_weight_sum * a_val, 1e-6)
            ess_ctrl = max(ctrl_weight_sum, 1e-6)
            
            # Combined variance
            sigma2 = (ctrl_var / ess_ctrl) + (hist_var / ess_hist)
            if not np.isfinite(sigma2) or sigma2 <= 0:
                return -np.inf
            
            # Log-likelihood
            delta = hist_mean - ctrl_mean
            return -0.5 * (np.log(2.0 * np.pi * sigma2) + (delta ** 2) / sigma2)
        
        # Bridge sampling for a(x)
        try:
            sampler = BridgeSampler(
                log_prior=log_prior_a,
                log_likelihood=log_likelihood_a,
            step_size=self.bridge_step,
                grid_size=20,
                use_precomputed_grid=True,
            )
            draws = sampler.sample(
                n_draws=self.bridge_samples,
                burn_in=self.bridge_burn,
                thin=self.bridge_thin,
            )
            if draws.size == 0:
                mean = compatibility
                var = 0.0
                ci_low = compatibility
                ci_high = compatibility
            else:
                mean = float(np.clip(np.mean(draws), 0.0, 1.0))
                var = float(np.var(draws, ddof=1)) if draws.size > 1 else 0.0
            ci_low, ci_high = np.quantile(draws, [self.alpha / 2.0, 1.0 - self.alpha / 2.0])
            ci_low = float(np.clip(ci_low, 0.0, 1.0))
            ci_high = float(np.clip(ci_high, 0.0, 1.0))
        except Exception:
            # Fallback to compatibility metric
            mean = compatibility
            var = 0.0
            ci_low = compatibility
            ci_high = compatibility
        
        return {
            "mean": mean,
            "var": max(var, 0.0),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "compatibility": compatibility,
        }

    # -------------------------------------------------------------------------
    # Power prior posterior for control mean μ_0(x)
    # -------------------------------------------------------------------------
    
    def _posterior_mu0_power_prior(self, X_curr: np.ndarray, Y_curr: np.ndarray, 
                                   Z_curr: np.ndarray, discount_array: np.ndarray,
                                   sigma2_0c_array: np.ndarray, sigma2_0h_array: np.ndarray) -> np.ndarray:
        """
        Compute posterior mean for μ_0(x) using power prior with variance weighting.
        
        From CAHB-PP manuscript:
        V₀⁻¹ = W₀/σ²_{0,c} + a·W_h/σ²_{0,h}
        m₀ = (W₀·Ȳ₀/σ²_{0,c} + a·W_h·Ȳ_h/σ²_{0,h}) / V₀⁻¹
        
        Args:
            X_curr: Current covariates, shape (n, p)
            Y_curr: Current outcomes, shape (n,)
            Z_curr: Treatment indicators, shape (n,)
            discount_array: Discount parameters a(X_i), shape (n,)
            sigma2_0c_array: Current control variances σ²_{0,c}(X_i), shape (n,)
            sigma2_0h_array: Historical control variances σ²_{0,h}(X_i), shape (n,)
            
        Returns:
            Posterior means μ_0(X_i), shape (n,)
        """
        X_curr = _as_2d(X_curr)
        n = X_curr.shape[0]
        mu0_post = np.zeros(n)
        
        # Control group mask
        ctrl_mask = (Z_curr == 0)
        if ctrl_mask.sum() < 1:
            return self._theta_h(X_curr)  # Fallback to historical
        
        X_ctrl = X_curr[ctrl_mask]
        Y_ctrl = Y_curr[ctrl_mask]
        h_ctrl = self._kernel_bandwidth(X_ctrl)
        
        # Historical predictions
        theta_h = self._theta_h(X_curr)
        
        # For each point, compute variance-weighted posterior mean
        for i in range(n):
            x_i = X_curr[i]
            a_i = discount_array[i]
            sigma2_c = max(sigma2_0c_array[i], 1e-6)
            sigma2_h = max(sigma2_0h_array[i], 1e-6)
            
            # Current control: W₀ (weights), Ȳ₀ (weighted mean)
            ctrl_weights = self._gaussian_kernel(x_i, X_ctrl, h_ctrl)
            W0 = ctrl_weights.sum()
            Y0_bar = np.dot(ctrl_weights, Y_ctrl) / W0 if W0 > 1e-8 else 0.0
            
            # Historical: W_h (weights), Ȳ_h (weighted mean)
            hist_weights = self._gaussian_kernel(x_i, self.X_h, self.h_hist)
            Wh = hist_weights.sum()
            Yh_bar = np.dot(hist_weights, self.Y_h) / Wh if Wh > 1e-8 else theta_h[i]
            
            # Precision-weighted posterior: V₀⁻¹ and m₀
            precision_curr = W0 / sigma2_c
            precision_hist = a_i * Wh / sigma2_h
            V0_inv = precision_curr + precision_hist
            
            if V0_inv > 1e-8:
                numerator = precision_curr * Y0_bar + precision_hist * Yh_bar
                mu0_post[i] = numerator / V0_inv
        else:
                mu0_post[i] = theta_h[i]  # Fallback
        
        return mu0_post
    
    # -------------------------------------------------------------------------
    # Effective sample size R_n(x)
    # -------------------------------------------------------------------------
    
    def _R_n_at_x(self, x_eval: np.ndarray, X_curr: np.ndarray, Z_curr: np.ndarray, 
                  discount_at_x: float, sigma2_0c: float, sigma2_0h: float) -> float:
        """
        Compute effective sample size ratio R_n(x) with variance weighting.
        
        From CAHB-PP manuscript:
        R_n(x) = V_ref(x) / V_bor(x) = 1 + a(x) · W_h(x) · σ²_{0,c}(x) / (W₀(x) · σ²_{0,h}(x))
        
        Args:
            x_eval: Evaluation point
            X_curr: Current covariates
            Z_curr: Treatment indicators
            discount_at_x: Discount a(x)
            sigma2_0c: Current control variance σ²_{0,c}(x)
            sigma2_0h: Historical control variance σ²_{0,h}(x)
            
        Returns:
            R_n(x) >= 1
        """
        ctrl_mask = (Z_curr == 0)
        if ctrl_mask.sum() < 1:
            return 1.0
        
        X_ctrl = X_curr[ctrl_mask]
        h_ctrl = self._kernel_bandwidth(X_ctrl)
        
        # Current control weights W₀(x)
        ctrl_weights = self._gaussian_kernel(x_eval, X_ctrl, h_ctrl)
        W0 = ctrl_weights.sum()
        
        # Historical weights W_h(x)
        hist_weights = self._gaussian_kernel(x_eval, self.X_h, self.h_hist)
        Wh = hist_weights.sum()
        
        # Variance-based R_n formula
        sigma2_c = max(sigma2_0c, 1e-6)
        sigma2_h = max(sigma2_0h, 1e-6)
        
        if W0 > 1e-8:
            R_n = 1.0 + discount_at_x * Wh * sigma2_c / (W0 * sigma2_h)
        else:
            R_n = 1.0
        
        return float(np.clip(R_n, 1.0, 1e6))
    
    # -------------------------------------------------------------------------
    # Public interface (maintains compatibility)
    # -------------------------------------------------------------------------
    
    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new) -> AllocationResult:
        """
        Compute allocation probability for new subject.
        
        Following CAHB-PP Algorithm 2:
        1. Call Algorithm 1 (Gibbs sampler) at x = X_new
        2. Estimate R_n(X_new) from posterior draws
        3. Compute allocation probabilities π₀, π₁ via (2)
        4. Apply biased coin design
        """
        X_curr = _as_2d(X_curr)
        Y_curr = np.asarray(Y_curr, dtype=float)
        Z_curr = np.asarray(Z_curr, dtype=int)
        X_new = np.asarray(X_new, dtype=float).ravel()
        
        if len(X_curr) < 2:
            return AllocationResult(pi_treatment=0.5, diagnostics={"R_n": 1.0, "a_mean": 0.0})
        
        # Algorithm 1: Gibbs sampler at X_new
        gibbs_result = self._gibbs_sampler_at_x(X_new, X_curr, Y_curr, Z_curr,
                                                n_gibbs=100, burn_in=50)
        
        # Extract posterior means
        a_mean = gibbs_result["a_mean"]
        sigma2_0c_mean = gibbs_result["sigma2_0c_mean"]
        sigma2_0h_mean = gibbs_result["sigma2_0h_mean"]
        
        # Compute R_n(X_new) using posterior means
        R_n = self._R_n_at_x(X_new, X_curr, Z_curr, a_mean, sigma2_0c_mean, sigma2_0h_mean)
        
        # Local sample sizes
        h_curr = self._kernel_bandwidth(X_curr)
        weights = self._gaussian_kernel(X_new, X_curr, h_curr)
        n0_local = float(np.dot(weights, 1 - Z_curr))
        n1_local = float(np.dot(weights, Z_curr))
        
        # Effective control sample size
        n0_eff = R_n * n0_local
        
        # Biased coin probability (Algorithm 2 step)
        pi = self._imbalance_probability(n0_eff, n1_local)
        
        # Diagnostics
        diag = {
            "R_n": R_n,
            "a_mean": a_mean,
            "a_ci_low": gibbs_result["a_ci_low"],
            "a_ci_high": gibbs_result["a_ci_high"],
            "sigma2_0c": sigma2_0c_mean,
            "sigma2_0h": sigma2_0h_mean,
        }
        
        return AllocationResult(pi_treatment=pi, diagnostics=diag)

    def estimate_treatment_effect(self, X, Y, Z) -> Tuple[float, float, float, float]:
        """
        Estimate average treatment effect δ = E[μ_1(X) - μ_0(X)].
        
        Following CAHB-PP with Gibbs sampling:
        1. Run Gibbs sampler (Algorithm 1) at each X_i to get posterior draws
        2. Extract μ_0(X_i) and μ_1(X_i) draws
        3. Compute δ_i = μ_1(X_i) - μ_0(X_i)
        4. Aggregate to δ_ATE^(s) = average over X_i for each draw s
        5. Return posterior summary
        """
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)
        
        if (Z == 0).sum() < 2 or (Z == 1).sum() < 2:
            delta = float(Y[Z == 1].mean() - Y[Z == 0].mean())
            return delta, np.nan, np.nan, 0.5
        
        # Run Gibbs sampler at each X_i
        n = X.shape[0]
        self._discount_cache = []
        mu0_draws_all = []
        mu1_draws_all = []
        
        for i in range(n):
            gibbs_result = self._gibbs_sampler_at_x(X[i], X, Y, Z,
                                                    n_gibbs=100, burn_in=50)
            mu0_draws = np.asarray(gibbs_result["draws"]["mu0"], dtype=float)
            mu1_draws = np.asarray(gibbs_result["draws"]["mu1"], dtype=float)
            mu0_draws_all.append(mu0_draws)
            mu1_draws_all.append(mu1_draws if mu1_draws.size > 0 else np.full_like(mu0_draws, gibbs_result["mu1_mean"]))
            
            summary = {
                "mean": gibbs_result["a_mean"],
                "var": gibbs_result["a_var"],
                "ci_low": gibbs_result["a_ci_low"],
                "ci_high": gibbs_result["a_ci_high"],
                "compatibility": 1.0,
            }
            self._discount_cache.append(summary)
        
        # Align draws across subjects by truncating to min length
        min_len = min((len(draws) for draws in mu0_draws_all), default=0)
        delta_draws = []
        if min_len > 0:
            for s in range(min_len):
                diffs = []
                for i in range(n):
                    if len(mu0_draws_all[i]) > s and len(mu1_draws_all[i]) > s:
                        diffs.append(mu1_draws_all[i][s] - mu0_draws_all[i][s])
                if diffs:
                    delta_draws.append(float(np.mean(diffs)))
        else:
            # Fallback to posterior means if draws unavailable
            mu0_means = np.array([np.mean(draws) for draws in mu0_draws_all])
            mu1_means = np.array([np.mean(draws) for draws in mu1_draws_all])
            delta_draws = [float(np.mean(mu1_means - mu0_means))]
        
        # Record calibration samples
        self._record_calibration_samples(X)
        
        return _delta_summary(np.asarray(delta_draws), self.alpha)
    
    def _record_calibration_samples(self, X: np.ndarray):
        """Record calibration samples for diagnostics."""
        X = _as_2d(X)
        n = X.shape[0]
        k = min(self.calibration_max_samples, n)
        
        if k <= 0 or len(self._discount_cache) < n:
            self._calibration_payload = []
            return
        
        idx = np.linspace(0, n - 1, k).astype(int) if k < n else np.arange(n)
        
        payload = []
        for i in idx:
            entry = self._discount_cache[i]
            x_row = X[i]
            payload.append({
                "x1": float(x_row[0]),
                "x2": float(x_row[1]),
                "a_mean": entry["mean"],
                "a_ci_low": entry["ci_low"],
                "a_ci_high": entry["ci_high"],
                "compatibility": entry["compatibility"],
            })
        
        self._calibration_payload = payload
    
    def get_calibration_payload(self) -> list:
        """Return calibration diagnostics."""
        return list(self._calibration_payload)
    
    @staticmethod
    def _imbalance_probability(n0_eff: float, n1_eff: float) -> float:
        """Biased coin allocation probability."""
        n0_eff = max(float(n0_eff), 1e-6)
        n1_eff = max(float(n1_eff), 1e-6)
        total = n0_eff + n1_eff
        g0 = np.clip(n0_eff / total, 1e-6, 1.0 - 1e-6)
        g1 = np.clip(n1_eff / total, 1e-6, 1.0 - 1e-6)
        inv_g0 = 1.0 / g0
        inv_g1 = 1.0 / g1
        denom = inv_g0 + inv_g1 - 2.0
        if denom <= 1e-8:
            return 0.5
        pi = (inv_g0 - 1.0) / denom
        return float(np.clip(pi, 0.0, 1.0))


class CAHB_PP_IPD(CAHBPowerPrior):
    """
    CAHB-PP-IPD: Individual-level historical data access.
    
    Uses full historical data {X_h, Y_h} for borrowing.
    """
    method_label = "CAHB-PP-IPD"


class CAHB_PP_SLD(CAHBPowerPrior):
    """
    CAHB-PP-SLD: Summary-level historical data.

    Uses summary statistics instead of individual-level data.
    Precomputes stratum-specific summaries for efficiency.
    """
    method_label = "CAHB-PP-SLD"

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self._hist_summary = self._build_summary_table()

    def _build_summary_table(self) -> dict:
        """Build summary statistics table by strata."""
        summary = {}
        if self.X_h.shape[0] == 0:
            return summary
        
        # Assuming X2 (binary) and X4 (3-level) are discrete
        x2 = self.X_h[:, 1].astype(int)
        x4 = self.X_h[:, 3].astype(int)
        y = self.Y_h
        
        # Global statistics
        global_var = float(max(np.var(y, ddof=1), 1e-6)) if y.size > 1 else 1.0
        global_mean = float(np.mean(y)) if y.size else 0.0
        self._hist_global = (global_mean, global_var, float(y.size))
        
        # Stratum-specific statistics
        for lvl2 in (0, 1):
            for lvl4 in (0, 1, 2):
                mask = (x2 == lvl2) & (x4 == lvl4)
                count = int(mask.sum())
                if count == 0:
                    continue
                mean = float(np.mean(y[mask]))
                variance = float(max(np.var(y[mask], ddof=1), 1e-6)) if count > 1 else global_var
                summary[(lvl2, lvl4)] = (mean, variance, float(count))
        
        return summary

    def _theta_h(self, X_eval: np.ndarray) -> np.ndarray:
        """Override to use summary statistics."""
        X_eval = _as_2d(X_eval)
        m = X_eval.shape[0]
        theta_h = np.zeros(m)
        
        for i in range(m):
            x_i = X_eval[i]
            x2 = int(np.clip(round(x_i[1]), 0, 1))
            x4 = int(np.clip(round(x_i[3]), 0, 2))
            
            if (x2, x4) in self._hist_summary:
                theta_h[i] = self._hist_summary[(x2, x4)][0]  # mean
            else:
                theta_h[i] = self._hist_global[0]  # global mean
        
        return theta_h


__all__ = ["AllocationResult", "KBCD", "CAHB", "CAHB_PP_IPD", "CAHB_PP_SLD"]
