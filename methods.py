"""
methods.py

Implementation of the covariate-adaptive methods used in the CAHB-UIP
simulation study.  The code mirrors the notation in the CAHB-UIP paper and
supplementary materials as well as the original authors' R reference
implementation (`utils.R`).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import warnings

import numpy as np
from scipy.stats import norm
from scipy.interpolate import CubicSpline

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


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


if HAS_NUMBA:

    @njit(cache=True, fastmath=True)
    def _gaussian_kernel_weights_nb(x_new_vec: np.ndarray, X_ref: np.ndarray, h: np.ndarray) -> np.ndarray:
        n, p = X_ref.shape
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            acc = 0.0
            for j in range(p):
                diff = (X_ref[i, j] - x_new_vec[j]) / h[j]
                acc += diff * diff
            out[i] = np.exp(-0.5 * acc)
        return out
else:
    _gaussian_kernel_weights_nb = None



def _gaussian_kernel_weights(x_new: np.ndarray, X_ref: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Product Gaussian kernel weights."""
    if X_ref.shape[0] == 0:
        return np.zeros(0)
    x_new = _as_2d(x_new)
    if HAS_NUMBA and x_new.shape[0] == 1:
        try:
            X_ref64 = np.asarray(X_ref, dtype=np.float64)
            h64 = np.asarray(h, dtype=np.float64)
            x_vec = np.asarray(x_new[0], dtype=np.float64)
            return _gaussian_kernel_weights_nb(x_vec, X_ref64, h64)
        except Exception:
            pass
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
        pi = (inv_g1 - 1.0) / denom
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
        # λ2 baseline (paper: 300 log n)
        self.lambda_proj = float(priors.get("cahb_lambda", 300.0))
        self.lambda_quantile = float(priors.get("cahb_lambda_quantile", 0.10))
        self.lambda_trunc_default = float(priors.get("cahb_lambda_trunc", 0.0))
        # 1/γ^2 with default γ = √3 ⇒ ~0.333
        self.invgam2 = float(priors.get("cahb_invgam2", 1.0 / max(self.gamma ** 2, 1e-8)))

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
        """
        Euclidean projection onto the nonnegative L1-ball of radius `radius`.

        Ported from R Codes/simplex.py (Duchi et al., 2008).
        """
        v = np.asarray(v, dtype=float)
        v = np.maximum(v, 0.0)
        if not np.isfinite(radius) or radius <= 0.0:
            return v
        n = v.shape[0]
        if v.sum() <= radius or n == 0:
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho_idx = np.nonzero(u * np.arange(1, n + 1) > (cssv - radius))[0]
        if rho_idx.size == 0:
            return np.zeros_like(v)
        rho = rho_idx[-1]
        theta = (cssv[rho] - radius) / (rho + 1.0)
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
        lambda_proj: Optional[float] = None,
        lambda_trunc: Optional[float] = None,
        return_raw: bool = False,
    ):
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
        
        # Allow caller to override lambda settings (used for Algorithm 2 tuning)
        lam_trunc_val = self.lambda_trunc_default if lambda_trunc is None else float(lambda_trunc)
        tau_thresholded = tau_raw.copy()
        tau_thresholded[tau_thresholded <= lam_trunc_val] = 0.0
        
        n = kernel_matrix.shape[1]
        lam_proj_val = self.lambda_proj if lambda_proj is None else float(lambda_proj)
        radius = lam_proj_val
        if np.isfinite(radius):
            radius = radius * max(np.log(max(n, 1)), 1.0)
        tau_projected = self._project_nonnegative_l1(tau_thresholded, radius)
        if return_raw:
            return tau_projected, tau_raw
        return tau_projected

    def _mu0_no_borrow(
        self,
        kernel_matrix: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """Replicates mu0.no.est.fn for Algorithm 2 (λ1 tuning)."""
        n = len(Y)
        zeros = np.zeros(n)
        phi0_init = 1.0
        mu0 = self._m_opt_mu0(kernel_matrix, Y, Z, zeros, phi0_init, zeros)
        return mu0

    def _coordinate_updates(
        self,
        kernel_matrix: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        theta0: np.ndarray,
        lambda_proj: float,
        lambda_trunc: float,
    ) -> Optional[dict]:
        if (1.0 - Z).sum() < 2:
            return None
        n = len(Y)
        tau = np.zeros(n)
        phi0 = max(np.std(Y[Z == 0], ddof=1), 1.0)
        mu0_prev = None
        phi0_prev = None
        tau_prev = None
        tau_raw_latest = np.zeros(n)
        for iter_idx in range(self.max_iter):
            mu0 = self._m_opt_mu0(kernel_matrix, Y, Z, tau, phi0, theta0)
            phi0 = self._opt_phi0(Y, Z, mu0)
            tau_new, tau_raw = self._m_opt_tau(
                kernel_matrix,
                Y,
                Z,
                mu0,
                theta0,
                lambda_proj=lambda_proj,
                lambda_trunc=lambda_trunc,
                return_raw=True,
            )
            if mu0_prev is not None and phi0_prev is not None and tau_prev is not None:
                err_mu = np.mean((mu0 - mu0_prev) ** 2)
                err_tau = np.mean((tau_new - tau_prev) ** 2)
                err_phi0 = (phi0 - phi0_prev) ** 2
                if max(err_mu, err_tau, err_phi0) < 1e-5:
                    tau = tau_new
                    tau_raw_latest = tau_raw
                    break
            mu0_prev = mu0
            phi0_prev = phi0
            tau_prev = tau
            tau = tau_new
            tau_raw_latest = tau_raw
        return {
            "mu0": mu0,
            "tau": tau,
            "tau_raw": tau_raw_latest,
            "phi0": phi0,
        }

    def _select_lambda_trunc(
        self,
        kernel_matrix: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> float:
        """Implements Algorithm 2 (Table 1) to choose λ1."""
        if self.lambda_quantile <= 0.0 or not np.isfinite(self.lambda_quantile):
            return self.lambda_trunc_default
        if (1.0 - Z).sum() < 3:
            return self.lambda_trunc_default
        theta0_true = self._mu0_no_borrow(kernel_matrix, Y, Z)
        tuning_fit = self._coordinate_updates(
            kernel_matrix,
            Y,
            Z,
            theta0_true,
            lambda_proj=self.lambda_proj,
            lambda_trunc=0.0,
        )
        if tuning_fit is None or tuning_fit.get("tau_raw") is None:
            return self.lambda_trunc_default
        tau_raw = tuning_fit["tau_raw"]
        tau_raw = tau_raw[np.isfinite(tau_raw)]
        if tau_raw.size == 0:
            return self.lambda_trunc_default
        try:
            lam_val = float(np.quantile(tau_raw, self.lambda_quantile))
        except ValueError:
            lam_val = self.lambda_trunc_default
        return max(lam_val, self.lambda_trunc_default)

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

        lambda_trunc = self._select_lambda_trunc(kernel_matrix, Y_curr, Z_curr)
        coord_updates = self._coordinate_updates(
            kernel_matrix,
            Y_curr,
            Z_curr,
            theta0,
            lambda_proj=self.lambda_proj,
            lambda_trunc=lambda_trunc,
        )
        if coord_updates is None:
            return None
        mu0 = coord_updates["mu0"]
        tau = coord_updates["tau"]
        phi0 = coord_updates["phi0"]
        tau_raw = coord_updates["tau_raw"]

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
            "tau_raw": tau_raw,
            "phi0": phi0,
            "phi0_sq": phi0_sq,
            "diff_sq": diff_sq,
            "lambda_trunc": lambda_trunc,
        }

    def _tau_at_x(self, context: dict, x_eval: np.ndarray) -> float:
        weights = self._kernel_vector(x_eval, context["X"], context["cov_inv"], context["log_norm"])
        numerator = np.dot(context["sZs"], weights)
        if numerator <= 1e-8:
            return 0.0
        denominator = np.dot(context["sZs"] * context["diff_sq"], weights) + self.invgam2
        return float(np.clip(numerator / max(denominator, 1e-8), 0.0, 1e6))

    def _R_n_at_x(self, context: dict, x_eval: np.ndarray) -> float:
        tau_val = self._tau_at_x(context, x_eval)
        return float(np.clip(1.0 + tau_val, 1.0, 1e4))

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
        """Helper for local kernel regressions (shared with CAHB-UIP variants)."""
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
# CAHB-UIP: Variance-aware Unit Information Prior Borrowing
# ==============================================================================


class CAHBUIPBase(BorrowingMethod):
    """Variance-aware covariate-adjusted borrowing with the UIP framework."""

    method_label = "CAHB-UIP"

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        self.name = self.method_label
        self.alpha0 = float(priors.get("ig_shape", 0.01))
        self.beta0 = float(priors.get("ig_scale", 0.01))
        self.gamma_alpha = float(priors.get("uip_gamma_alpha", 2.0))
        self.coord_tol = float(priors.get("uip_coord_tol", 1e-4))
        self.coord_max_iter = int(priors.get("uip_coord_iter", 50))
        self.post_gibbs_iter = int(priors.get("post_gibbs_iter", 400))
        self.post_gibbs_burn = int(priors.get("post_gibbs_burn", 200))
        self.beta1_prior_scale = float(priors.get("beta1_prior_scale", 1e6))
        self._amount_cache = []
        self._calibration_payload = []
        try:
            import config as _cfg  # type: ignore

            self.calibration_max_samples = int(getattr(_cfg, "CALIBRATION_MAX_SAMPLES", 32))
        except ImportError:
            self.calibration_max_samples = 32

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _hist_local_stats(self, x_eval: np.ndarray) -> LocalStats:
        return self._hist_stats(x_eval)

    def _center_covariates(self, X: np.ndarray) -> np.ndarray:
        """Optionally center concurrent covariates before post-trial inference."""
        return _as_2d(X)

    @staticmethod
    def _design_matrix(X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        intercept = np.ones((X.shape[0], 1), dtype=float)
        return np.hstack([intercept, X])

    @staticmethod
    def _stable_inverse(matrix: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
        dim = matrix.shape[0]
        work = matrix + np.eye(dim) * ridge
        try:
            return np.linalg.inv(work)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(work)

    def _historical_summary(self, feature_dim: int, X_curr: np.ndarray) -> Optional[dict]:
        if self.X_h.size == 0:
            return None
        Phi_h = self._design_matrix(self.X_h)
        if Phi_h.shape[1] != feature_dim:
            if Phi_h.shape[1] > feature_dim:
                Phi_h = Phi_h[:, :feature_dim]
            else:
                pad = np.zeros((Phi_h.shape[0], feature_dim - Phi_h.shape[1]))
                Phi_h = np.hstack([Phi_h, pad])
        y_h = np.asarray(self.Y_h, dtype=float)
        n_h = y_h.shape[0]
        if n_h == 0:
            return None
        X_curr = _as_2d(X_curr)
        if X_curr.size == 0:
            return None
        bandwidth = _silverman_bandwidth(np.vstack([X_curr, _as_2d(self.X_h)]))
        weights_raw = []
        for x_h in self.X_h:
            k_vals = _gaussian_kernel_weights(x_h, X_curr, bandwidth)
            weights_raw.append(float(np.mean(k_vals)))
        weights_raw = np.asarray(weights_raw, dtype=float)
        total = float(np.sum(weights_raw[np.isfinite(weights_raw)]))
        if not np.isfinite(total) or total <= 1e-8:
            weights_raw = np.ones(n_h, dtype=float)
            total = float(weights_raw.sum())
        lam = weights_raw / total
        weighted = lam[:, None] * Phi_h
        S_h = Phi_h.T @ weighted
        q_h = Phi_h.T @ (lam * y_h)
        beta_hat = self._stable_inverse(S_h) @ q_h
        W_h = total
        return {
            "Phi_h": Phi_h,
            "y_h": y_h,
            "lambda": lam,
            "S_h": S_h,
            "q_h": q_h,
            "beta_hat": beta_hat,
            "W_h": W_h,
        }

    def _historical_ss(self, hist_summary: dict, beta0: np.ndarray) -> float:
        lam = hist_summary.get("lambda")
        Phi_h = hist_summary.get("Phi_h")
        y_h = hist_summary.get("y_h")
        if lam is None or Phi_h is None or y_h is None:
            return 0.0
        resid = y_h - Phi_h @ beta0
        return float(np.dot(lam, resid ** 2))

    def _full_data_gibbs(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        n_iter: int,
        burn_in: int,
    ) -> Optional[dict]:
        from scipy.stats import invgamma

        X_raw = _as_2d(X)
        X_centered = self._center_covariates(X_raw)
        design_all = self._design_matrix(X_centered)
        feature_dim = design_all.shape[1]
        hist_summary = self._historical_summary(feature_dim, X_raw)
        if hist_summary is None:
            return None

        ctrl_mask = Z == 0
        trt_mask = Z == 1
        X0 = design_all[ctrl_mask]
        X1 = design_all[trt_mask]
        y0 = Y[ctrl_mask]
        y1 = Y[trt_mask]
        if X0.shape[0] == 0 or X1.shape[0] == 0:
            return None

        # Initial values via least squares
        def _ols(design, response):
            if design.shape[0] == 0:
                return np.zeros(feature_dim)
            try:
                return np.linalg.lstsq(design, response, rcond=None)[0]
            except np.linalg.LinAlgError:
                return np.zeros(feature_dim)

        beta0 = _ols(X0, y0)
        beta1 = _ols(X1, y1)
        resid0 = y0 - X0 @ beta0
        sigma2_0c = float(max(np.var(resid0, ddof=1), 1e-3))
        sigma2_0h = max(self._historical_ss(hist_summary, beta0), 1e-3)
        sigma2_1 = float(max(np.var(y1 - X1 @ beta1, ddof=1), 1e-3))
        M = max(hist_summary["W_h"], 1e-6)

        ridge = 1e-8 * np.eye(feature_dim)
        prior_reg = np.eye(feature_dim) / max(self.beta1_prior_scale, 1e-8)
        prior_reg[0, 0] = 0.0  # leave intercept noninformative (centered covariates)

        kept_beta0 = []
        kept_beta1 = []
        kept_M = []
        delta_draws = []

        phi_all = design_all
        burn = max(0, int(burn_in))
        total_iter = max(n_iter, burn + 10)

        for it in range(total_iter):
            # beta0 update
            prec0 = (X0.T @ X0) / max(sigma2_0c, 1e-8)
            prec_hist = (M / max(sigma2_0h, 1e-8)) * hist_summary["S_h"]
            V0_inv = prec0 + prec_hist + ridge
            V0 = self._stable_inverse(V0_inv)
            V0 = 0.5 * (V0 + V0.T)
            mean0 = V0 @ (
                (X0.T @ y0) / max(sigma2_0c, 1e-8) + (M / max(sigma2_0h, 1e-8)) * hist_summary["q_h"]
            )
            beta0 = np.random.multivariate_normal(mean0, V0)

            # sigma0c^2 update
            resid0 = y0 - X0 @ beta0
            shape_c = self.alpha0 + X0.shape[0] / 2.0
            scale_c = self.beta0 + 0.5 * np.dot(resid0, resid0)
            sigma2_0c = float(invgamma.rvs(shape_c, scale=max(scale_c, 1e-8)))

            # sigma0h^2 update
            ss_h = self._historical_ss(hist_summary, beta0)
            shape_h = self.alpha0 + hist_summary["W_h"] / 2.0
            scale_h = self.beta0 + 0.5 * ss_h
            sigma2_0h = float(invgamma.rvs(shape_h, scale=max(scale_h, 1e-8)))

            # M update
            diff = beta0 - hist_summary["beta_hat"]
            quad = float(diff.T @ (hist_summary["S_h"] @ diff))
            rate_M = (self.gamma_alpha / max(hist_summary["W_h"], 1e-8)) + 0.5 * quad / max(sigma2_0h, 1e-8)
            shape_M = self.gamma_alpha + feature_dim / 2.0
            M = float(np.random.gamma(shape_M, 1.0 / max(rate_M, 1e-8)))

            # beta1 update
            XtX = X1.T @ X1
            post_prec = XtX + prior_reg
            V1 = self._stable_inverse(post_prec)
            V1 = 0.5 * (V1 + V1.T)
            mean1 = V1 @ (X1.T @ y1)
            beta1 = np.random.multivariate_normal(mean1, sigma2_1 * V1)

            # sigma1^2 update
            resid1 = y1 - X1 @ beta1
            shape_t = self.alpha0 + X1.shape[0] / 2.0
            scale_t = self.beta0 + 0.5 * np.dot(resid1, resid1)
            sigma2_1 = float(invgamma.rvs(shape_t, scale=max(scale_t, 1e-8)))

            if it >= burn:
                kept_beta0.append(beta0.copy())
                kept_beta1.append(beta1.copy())
                kept_M.append(M)
                delta_draws.append(float(np.mean(phi_all @ (beta1 - beta0))))

        if not kept_beta0:
            return None

        kept_beta0 = np.asarray(kept_beta0)
        kept_beta1 = np.asarray(kept_beta1)
        kept_M = np.asarray(kept_M)
        delta_draws = np.asarray(delta_draws, dtype=float)

        return {
            "beta0": kept_beta0,
            "beta1": kept_beta1,
            "M_draws": kept_M,
            "delta_draws": delta_draws,
        }

    @staticmethod
    def _weighted_ss(stats: LocalStats, target_mean: float) -> float:
        if stats.weight <= 1e-8 or not np.isfinite(stats.mean):
            return 0.0
        variance = stats.variance if np.isfinite(stats.variance) and stats.variance > 0 else 1.0
        diff = stats.mean - target_mean
        return float(stats.weight * (variance + diff * diff))

    def _posterior_mode_variance(self, stats: LocalStats, target_mean: float) -> float:
        base = stats.variance if np.isfinite(stats.variance) and stats.variance > 0 else 1.0
        if stats.weight <= 1e-8:
            return float(max(base, 1e-6))
        ss = self._weighted_ss(stats, target_mean)
        alpha_post = self.alpha0 + stats.weight / 2.0
        beta_post = self.beta0 + ss / 2.0
        denom = max(alpha_post + 1.0, 1e-8)
        return float(max(beta_post / denom, 1e-6))

    def _amount_prior_rate(self, weight: float) -> float:
        weight = max(float(weight), 1e-6)
        return self.gamma_alpha / weight

    def _posterior_mode_amount(self, mu0: float, hist_mean: float, sigma2_hist: float, beta_prior: float) -> float:
        if not np.isfinite(beta_prior):
            return 0.0
        alpha_post = self.gamma_alpha + 0.5
        diff_sq = (mu0 - hist_mean) ** 2
        beta_post = beta_prior + diff_sq / (2.0 * max(sigma2_hist, 1e-8))
        if alpha_post <= 1.0:
            return float(alpha_post / max(beta_post, 1e-8))
        return float((alpha_post - 1.0) / max(beta_post, 1e-8))

    @staticmethod
    def _cecss_gain(W0: float, Wh: float, sigma2_ctrl: float, sigma2_hist: float, amount: float) -> float:
        if W0 <= 1e-8 or Wh <= 1e-8 or amount <= 1e-10:
            return 1.0
        denom = max(W0 * max(sigma2_hist, 1e-8), 1e-8)
        try:
            import config as _cfg

            max_rn = float(getattr(_cfg, "MAX_ESS", 1e4))
        except Exception:
            max_rn = 1e4
        gain = 1.0 + amount * max(sigma2_ctrl, 1e-8) / denom
        return float(np.clip(gain, 1.0, max_rn))

    # ------------------------------------------------------------------
    # Local MAP approximation and Gibbs sampler
    # ------------------------------------------------------------------

    def _coordinate_ascent_at_x(
        self,
        x_eval: np.ndarray,
        X_curr: np.ndarray,
        Y_curr: np.ndarray,
        Z_curr: np.ndarray,
    ) -> Optional[dict]:
        x_eval = np.asarray(x_eval, dtype=float).ravel()
        X_curr = _as_2d(X_curr)
        Y_curr = np.asarray(Y_curr, dtype=float)
        Z_curr = np.asarray(Z_curr, dtype=int)

        ctrl_stats = self._control_stats(X_curr, Y_curr, Z_curr, x_eval)
        if ctrl_stats.weight <= 1e-8 or not np.isfinite(ctrl_stats.mean):
            return None

        hist_stats = self._hist_local_stats(x_eval)
        borrow_enabled = hist_stats.weight > 1e-8 and np.isfinite(hist_stats.mean)
        if not borrow_enabled:
            hist_stats = LocalStats(
                weight=0.0,
                mean=float(ctrl_stats.mean),
                variance=float(ctrl_stats.variance if np.isfinite(ctrl_stats.variance) else 1.0),
            )

        trt_stats = self._treated_stats(X_curr, Y_curr, Z_curr, x_eval)

        mu0 = float(ctrl_stats.mean)
        sigma2_0c = max(ctrl_stats.variance if np.isfinite(ctrl_stats.variance) else 1.0, 1e-6)
        sigma2_0h = max(hist_stats.variance if np.isfinite(hist_stats.variance) else sigma2_0c, 1e-6)
        amount = float(hist_stats.weight) if borrow_enabled else 0.0
        mu1 = float(trt_stats.mean) if trt_stats.weight > 1e-8 and np.isfinite(trt_stats.mean) else mu0
        sigma2_1 = max(
            trt_stats.variance if trt_stats.weight > 1e-8 and np.isfinite(trt_stats.variance) else sigma2_0c,
            1e-6,
        )
        beta_M = self._amount_prior_rate(hist_stats.weight) if borrow_enabled else np.inf

        for _ in range(self.coord_max_iter):
            mu0_prev = mu0
            sigma2_0c_prev = sigma2_0c
            sigma2_0h_prev = sigma2_0h
            amount_prev = amount

            precision = ctrl_stats.weight / max(sigma2_0c_prev, 1e-8)
            if borrow_enabled:
                precision += amount_prev / max(sigma2_0h_prev, 1e-8)
            if precision > 1e-8:
                weighted_mean = (
                    ctrl_stats.weight * ctrl_stats.mean / max(sigma2_0c_prev, 1e-8)
                    + (amount_prev * hist_stats.mean / max(sigma2_0h_prev, 1e-8) if borrow_enabled else 0.0)
                ) / precision
                mu0 = float(weighted_mean)
            else:
                mu0 = float(ctrl_stats.mean)

            sigma2_0c = self._posterior_mode_variance(ctrl_stats, mu0)
            if borrow_enabled:
                sigma2_0h = self._posterior_mode_variance(hist_stats, mu0)
                amount = self._posterior_mode_amount(mu0, hist_stats.mean, sigma2_0h, beta_M)
            else:
                sigma2_0h = sigma2_0c
                amount = 0.0

            if trt_stats.weight > 1e-8 and np.isfinite(trt_stats.mean):
                mu1 = float(trt_stats.mean)
                sigma2_1 = self._posterior_mode_variance(trt_stats, mu1)
            else:
                mu1 = mu0
                sigma2_1 = sigma2_0c

            deltas = [
                abs(mu0 - mu0_prev),
                abs(amount - amount_prev),
                abs(np.log(max(sigma2_0c, 1e-6)) - np.log(max(sigma2_0c_prev, 1e-6))),
                abs(np.log(max(sigma2_0h, 1e-6)) - np.log(max(sigma2_0h_prev, 1e-6))),
            ]
            if max(deltas) < self.coord_tol:
                break

        return {
            "mu0": mu0,
            "mu1": mu1,
            "sigma2_0c": sigma2_0c,
            "sigma2_0h": sigma2_0h,
            "sigma2_1": sigma2_1,
            "M": max(amount, 0.0),
            "W0": float(ctrl_stats.weight),
            "Wh": float(hist_stats.weight),
            "W1": float(trt_stats.weight),
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_allocation_prob(self, X_curr, Y_curr, Z_curr, X_new) -> AllocationResult:
        X_curr = _as_2d(X_curr)
        Y_curr = np.asarray(Y_curr, dtype=float)
        Z_curr = np.asarray(Z_curr, dtype=int)
        X_new = np.asarray(X_new, dtype=float).ravel()

        if len(X_curr) < 2:
            return AllocationResult(pi_treatment=0.5, diagnostics={"R_n": 1.0, "M": 0.0})

        ca = self._coordinate_ascent_at_x(X_new, X_curr, Y_curr, Z_curr)
        if ca is None:
            return AllocationResult(pi_treatment=0.5, diagnostics={"R_n": 1.0, "M": 0.0})

        R_n = self._cecss_gain(ca["W0"], ca["Wh"], ca["sigma2_0c"], ca["sigma2_0h"], ca["M"])
        n0_eff = R_n * max(ca["W0"], 1e-8)
        pi = self._imbalance_probability(n0_eff, ca["W1"])
        diagnostics = {
            "R_n": R_n,
            "M": ca["M"],
            "mu0": ca["mu0"],
            "sigma2_0c": ca["sigma2_0c"],
            "sigma2_0h": ca["sigma2_0h"],
        }
        return AllocationResult(pi_treatment=pi, diagnostics=diagnostics)

    def estimate_treatment_effect(self, X, Y, Z) -> Tuple[float, float, float, float]:
        X = _as_2d(X)
        Y = np.asarray(Y, dtype=float)
        Z = np.asarray(Z, dtype=int)

        if (Z == 0).sum() < 2 or (Z == 1).sum() < 2:
            delta = float(Y[Z == 1].mean() - Y[Z == 0].mean())
            return delta, np.nan, np.nan, 0.5

        gibbs_result = self._full_data_gibbs(
            X,
            Y,
            Z,
            n_iter=self.post_gibbs_iter,
            burn_in=self.post_gibbs_burn,
        )
        if gibbs_result is None or gibbs_result["delta_draws"].size == 0:
            delta = float(Y[Z == 1].mean() - Y[Z == 0].mean())
            return delta, np.nan, np.nan, 0.5

        delta_draws = gibbs_result["delta_draws"]
        M_draws = gibbs_result["M_draws"]
        if M_draws.size > 0:
            mean_M = float(np.mean(M_draws))
            ci_low = float(np.quantile(M_draws, self.alpha / 2.0))
            ci_high = float(np.quantile(M_draws, 1.0 - self.alpha / 2.0))
        else:
            mean_M = 0.0
            ci_low = np.nan
            ci_high = np.nan

        amount_summary = {"mean": mean_M, "ci_low": ci_low, "ci_high": ci_high, "gain": np.nan}
        self._amount_cache = [amount_summary for _ in range(X.shape[0])]
        self._record_calibration_samples(X)
        return _delta_summary(delta_draws, self.alpha)

    def _record_calibration_samples(self, X: np.ndarray):
        X = _as_2d(X)
        n = X.shape[0]
        k = min(self.calibration_max_samples, n)

        if k <= 0 or len(self._amount_cache) < n:
            self._calibration_payload = []
            return

        idx = np.linspace(0, n - 1, k).astype(int) if k < n else np.arange(n)
        payload = []
        for i in idx:
            entry = self._amount_cache[i]
            x_row = X[i]
            payload.append(
                {
                    "x1": float(x_row[0]),
                    "x2": float(x_row[1]),
                    "x3": float(x_row[2]) if x_row.size > 2 else 0.0,
                    "x4": float(x_row[3]) if x_row.size > 3 else 0.0,
                    "M_mean": entry["mean"],
                    "M_ci_low": entry["ci_low"],
                    "M_ci_high": entry["ci_high"],
                    "R_n": entry["gain"],
                }
            )
        self._calibration_payload = payload

    def get_calibration_payload(self) -> list:
        return list(self._calibration_payload)


class CAHB_UIP_IPD(CAHBUIPBase):
    """CAHB-UIP implementation using individual-level historical data."""

    method_label = "CAHB-UIP-IPD"


class CAHB_UIP_SLD(CAHBUIPBase):
    """CAHB-UIP implementation that consumes summary-level historical data."""

    method_label = "CAHB-UIP-SLD"

    def __init__(self, historical_data: dict, scenario_params: dict, priors: dict):
        super().__init__(historical_data, scenario_params, priors)
        y = np.asarray(self.Y_h, dtype=float)
        self._sld_n = float(len(y))
        self._summary_weight = 1.0
        psi_shape = self._design_matrix(self.X_h).shape[1]
        self._summary_mean = 0.0
        self._summary_var = 1.0
        self._summary_var_mu = 1.0
        self._summary_beta = np.zeros(psi_shape)
        self._summary_bar_psi = np.zeros(psi_shape)
        if self._summary_weight <= 1e-8:
            pass
        else:
            psi = self._design_matrix(self.X_h)
            XtX = psi.T @ psi
            ridge = 1e-8 * np.eye(XtX.shape[0])
            XtX_reg = XtX + ridge
            beta_hat = np.linalg.lstsq(XtX_reg, psi.T @ y, rcond=None)[0]
            resid = y - psi @ beta_hat
            dof = max(len(y) - psi.shape[1], 1)
            sigma2_hat = float(np.dot(resid, resid) / dof)
            sigma2_hat = max(sigma2_hat, 1e-6)
            bar_psi = psi.mean(axis=0)
            XtX_inv = np.linalg.pinv(XtX_reg)
            var_mu = float(bar_psi @ XtX_inv @ bar_psi) * sigma2_hat
            self._summary_mean = float(bar_psi @ beta_hat)
            self._summary_var = sigma2_hat
            self._summary_var_mu = max(var_mu, 1e-8)
            self._summary_beta = beta_hat
            self._summary_bar_psi = bar_psi
        self._current_center = None

    def _hist_local_stats(self, x_eval: np.ndarray) -> LocalStats:  # pylint: disable=unused-argument
        return LocalStats(
            weight=self._summary_weight,
            mean=self._summary_mean,
            variance=self._summary_var,
        )

    def _center_covariates(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if X.size == 0:
            return X
        center = np.mean(X, axis=0, keepdims=True)
        self._current_center = center
        return X - center

    def _historical_summary(self, feature_dim: int, X_curr: np.ndarray) -> Optional[dict]:
        if self._sld_n <= 1e-8:
            return None
        iu = 1.0 / max(self._sld_n * self._summary_var_mu, 1e-8)
        S_h = np.zeros((feature_dim, feature_dim))
        S_h[0, 0] = iu
        q_h = np.zeros(feature_dim)
        q_h[0] = iu * self._summary_mean
        beta_hat = np.zeros(feature_dim)
        beta_hat[0] = self._summary_mean
        return {
            "Phi_h": None,
            "y_h": None,
            "lambda": None,
            "S_h": S_h,
            "q_h": q_h,
            "beta_hat": beta_hat,
            "W_h": 1.0,
        }

    def _historical_ss(self, hist_summary: dict, beta0: np.ndarray) -> float:
        diff = beta0[0] - self._summary_mean
        return float(self._summary_var_mu + diff ** 2)


__all__ = ["AllocationResult", "KBCD", "CAHB", "CAHB_UIP_IPD", "CAHB_UIP_SLD"]
