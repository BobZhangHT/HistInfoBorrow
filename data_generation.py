"""
data_generation.py

Data Generating Mechanisms (DGMs) for CAHB-PP Simulation Study

This module implements the covariate and outcome generation processes described
in Section 3.1 of the CAHB-PP manuscript. It supports both:
    1. Historical data generation (control-only, single-arm trials)
    2. Current trial data generation (adaptive two-arm randomized trials)

The DGMs feature:
    - Four baseline covariates (X1, X2, X3, X4) with different distributions
    - Covariate-dependent treatment effects (heterogeneous treatment effects)
    - Heteroscedastic outcome variances depending on covariates
    - Historical bias mechanisms (scenarios S1-S4)

Mathematical Details (Section 3.1):
-----------------------------------
Outcome Model:
    Y_{zi} | X_i ~ N(mu_z(X_i), sigma^2_z(X_i))
    
    where z = 0 (control) or z = 1 (treatment)

Control arm mean (current trial):
    mu_0(x) = b(x)^T * beta_0 + 0.75 * X1 * X3
    
    b(x) = (1, X1, X2, X3, I(X4=1), I(X4=2))^T
    beta_0 = (0, 0.8, 0.5, -0.5, 0.3, -0.2)^T

Treatment effect (heterogeneous):
    tau(x) = tau_0 + 0.5 * X1 - 0.5 * I(X2=1)
    
    where tau_0 is the base treatment effect (0 or 0.4 in simulations)

Treatment arm mean:
    mu_1(x) = mu_0(x) + tau(x)

Variance functions:
    sigma^2_0c(x) = exp(0.2 + 0.4*X2 + 0.3*I(X4=2))  # Current control
    sigma^2_1(x) = exp(0.2 + 0.2*X1)                  # Treatment arm
    sigma^2_0h(x) = kappa * exp(0.2 + 0.2*X2)        # Historical control

Historical data model:
    Y_0h | X ~ N(mu_0(X) + Delta_0(X), sigma^2_0h(X))
    
    where Delta_0(X) represents historical bias (scenario-dependent)

References:
    CAHB-PP manuscript Section 3.1 (references/CAHB_PP(2).pdf)
"""

import numpy as np
from typing import Callable, Tuple

def generate_covariates(n: int, p: int = 4) -> np.ndarray:
    """
    Generates baseline covariates for trial subjects (Section 3.1).
    
    Four covariates with different distributions to capture diverse covariate types:
        X1 ~ Normal(0, 1)           - Continuous, symmetric
        X2 ~ Bernoulli(0.5)         - Binary indicator
        X3 ~ Uniform(-1, 1)         - Continuous, bounded
        X4 ~ Categorical{0,1,2}     - Multi-level factor
             P(X4=0) = 0.4
             P(X4=1) = 0.4
             P(X4=2) = 0.2
    
    These covariates are used to construct the design vector b(x) and appear in
    both the outcome mean and variance functions.
    
    Args:
        n: Number of subjects to generate
        p: Number of covariates (fixed at 4 for this study)
    
    Returns:
        Covariate matrix of shape (n, p) where each row is one subject's covariates
    
    Notes:
        - Independence assumed between covariates
        - Same distribution for historical and current trial subjects
        - Random seed should be set externally for reproducibility
    
    Example:
        >>> np.random.seed(123)
        >>> X = generate_covariates(n=5)
        >>> X.shape
        (5, 4)
    """
    X = np.zeros((n, p))
    X[:, 0] = np.random.normal(0, 1, n)                      # X1: Standard normal
    X[:, 1] = np.random.binomial(1, 0.5, n)                  # X2: Binary
    X[:, 2] = np.random.uniform(-1, 1, n)                    # X3: Uniform
    X[:, 3] = np.random.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2])  # X4: Categorical
    return X

def _get_b_x(X: np.ndarray) -> np.ndarray:
    """
    Constructs the design matrix b(x) for the outcome model (Section 3.1).
    
    Design vector components:
        b(x) = (1, X1, X2, X3, I(X4=1), I(X4=2))^T
    
    This 6-dimensional vector includes:
        - Intercept (1)
        - Three continuous/binary covariates (X1, X2, X3)
        - Dummy coding for categorical X4 (reference category X4=0)
    
    Args:
        X: Covariate matrix of shape (n, 4)
    
    Returns:
        Design matrix of shape (n, 6)
    
    Notes:
        - X4 uses effect coding with X4=0 as reference
        - Used in computing mu_0(x) = b(x)^T * beta_0 + interaction term
    """
    n = X.shape[0]
    b_x = np.ones((n, 6))
    b_x[:, 1] = X[:, 0]                         # X1
    b_x[:, 2] = X[:, 1]                         # X2
    b_x[:, 3] = X[:, 2]                         # X3
    b_x[:, 4] = (X[:, 3] == 1).astype(float)    # Indicator: X4=1
    b_x[:, 5] = (X[:, 3] == 2).astype(float)    # Indicator: X4=2
    return b_x

def get_true_means(X: np.ndarray, tau_0: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes true outcome means for control and treatment arms (Section 3.1).
    
    Control arm mean:
        mu_0(x) = b(x)^T * beta_0 + 0.75 * X1 * X3
        
        where beta_0 = (0, 0.8, 0.5, -0.5, 0.3, -0.2)^T
    
    Treatment effect (Conditional Average Treatment Effect, CATE):
        tau(x) = tau_0 + 0.5 * X1 - 0.5 * I(X2=1)
        
        - tau_0: Base treatment effect (marginal component)
        - Heterogeneity via X1 (continuous modifier)
        - Heterogeneity via X2 (binary modifier)
    
    Treatment arm mean:
        mu_1(x) = mu_0(x) + tau(x)
    
    Marginal Average Treatment Effect (MATE):
        E[tau(X)] = E[tau_0 + 0.5*X1 - 0.5*I(X2=1)]
                  = tau_0 + 0.5*E[X1] - 0.5*P(X2=1)
                  = tau_0 + 0.5*0 - 0.5*0.5
                  = tau_0 - 0.25
    
    Args:
        X: Covariate matrix of shape (n, 4)
        tau_0: Base treatment effect parameter (0 for Type I error, 0.4 for power)
    
    Returns:
        Tuple of (mu_0, tau, mu_1), each of shape (n,):
            - mu_0: True control outcome means
            - tau: True conditional treatment effects (CATE)
            - mu_1: True treatment outcome means
    
    Notes:
        - Includes X1*X3 interaction in mu_0 for additional complexity
        - Treatment effect heterogeneity ensures different subjects benefit differently
        - Both positive (X1) and negative (X2) treatment effect modifiers included
    """
    # Regression coefficients for control mean (chosen to create realistic structure)
    beta_0 = np.array([0, 0.8, 0.5, -0.5, 0.3, -0.2])
    b_x = _get_b_x(X)
    
    # Control arm mean: linear predictor + interaction
    mu_0 = b_x @ beta_0 + 0.75 * X[:, 0] * X[:, 2]
    
    # Conditional treatment effect (heterogeneous)
    tau = tau_0 + 0.5 * X[:, 0] - 0.5 * (X[:, 1] == 1).astype(float)
    
    # Treatment arm mean
    mu_1 = mu_0 + tau
    
    return mu_0, tau, mu_1

def get_true_variances(X: np.ndarray, kappa: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes true heteroscedastic outcome variances (Section 3.1).
    
    The outcome variances depend on covariates to create heteroscedasticity:
    
    Current control arm variance:
        sigma^2_0c(x) = exp(0.2 + 0.4*X2 + 0.3*I(X4=2))
        
        - Higher variance for X2=1 (binary covariate)
        - Higher variance for X4=2 subgroup
        - Base variance: exp(0.2) ≈ 1.22
    
    Treatment arm variance:
        sigma^2_1(x) = exp(0.2 + 0.2*X1)
        
        - Variance increases with X1
        - Same base variance as control
    
    Historical control arm variance:
        sigma^2_0h(x) = kappa * exp(0.2 + 0.2*X2)
        
        - Scaled by kappa (variance inflation factor)
        - kappa = 1.0: Same structure as current (Scenario S1)
        - kappa = 1.3: Inflated variance (Scenarios S2, S3, S4)
    
    Args:
        X: Covariate matrix of shape (n, 4)
        kappa: Historical variance inflation factor
            - kappa = 1.0: No variance mismatch
            - kappa > 1.0: Historical data has larger variance
    
    Returns:
        Tuple of (sigma_0_c_sq, sigma_1_sq, sigma_0_h_sq), each of shape (n,):
            - sigma_0_c_sq: Current control arm variances
            - sigma_1_sq: Treatment arm variances
            - sigma_0_h_sq: Historical control arm variances
    
    Notes:
        - Exponential link ensures positive variances
        - Different covariate dependencies in each arm creates complexity
        - Variance heterogeneity motivates covariate-adaptive methods
    """
    # Current trial control arm variance
    sigma_0_c_sq = np.exp(0.2 + 0.4 * X[:, 1] + 0.3 * (X[:, 3] == 2).astype(float))
    
    # Current trial treatment arm variance
    sigma_1_sq = np.exp(0.2 + 0.2 * X[:, 0])
    
    # Historical trial control arm variance (scaled by kappa)
    sigma_0_h_sq = kappa * np.exp(0.2 + 0.2 * X[:, 1])
    
    return sigma_0_c_sq, sigma_1_sq, sigma_0_h_sq

def generate_historical_data(n_h: int, scenario_params: dict) -> dict:
    """
    Generates historical control arm data (Section 3.1-3.2).
    
    Historical trials provide control-arm-only data that may or may not be
    compatible with the current trial control arm. Compatibility is controlled
    by the scenario-specific historical bias function Delta_0(x).
    
    Historical outcome model:
        Y_0h | X ~ N(mu_0(X) + Delta_0(X), sigma^2_0h(X))
    
    where:
        - mu_0(X): Current trial control mean (base truth)
        - Delta_0(X): Historical bias function (scenario-dependent)
        - sigma^2_0h(X): Historical variance (may differ from current via kappa)
    
    Scenario-specific historical biases (Section 3.2):
        S1: Delta_0(x) = 0        (perfect compatibility)
        S2: Delta_0(x) = 0        (compatible means, variance mismatch only)
        S3: Delta_0(x) = 0.4      (constant bias)
        S4: Delta_0(x) = 0.6*I(X4=2)  (subgroup-specific bias)
    
    Args:
        n_h: Number of historical subjects
        scenario_params: Dictionary containing:
            - 'Delta_0_func': Function x -> Delta_0(x) defining historical bias
            - 'kappa': Variance inflation factor
            - Other scenario metadata
    
    Returns:
        Dictionary containing:
            - 'X_h': Historical covariates, shape (n_h, 4)
            - 'Y_h': Historical outcomes, shape (n_h,)
    
    Notes:
        - Historical data assumed to be control-arm only (single-arm trial)
        - Bias Delta_0(x) represents population shift, measurement shift, etc.
        - Methods must adaptively borrow based on local compatibility
    
    Example:
        >>> scenario = {'Delta_0_func': lambda x: 0.0, 'kappa': 1.0}
        >>> hist_data = generate_historical_data(n_h=100, scenario_params=scenario)
        >>> hist_data['X_h'].shape
        (100, 4)
    """
    # Generate historical subject covariates
    X_h = generate_covariates(n_h)
    
    # Compute base control means (current trial truth)
    mu_0_base, _, _ = get_true_means(X_h, tau_0=0.0)
    
    # Apply scenario-specific historical bias
    Delta_0 = np.array([scenario_params['Delta_0_func'](x) for x in X_h])
    mu_0_h = mu_0_base + Delta_0
    
    # Get historical variances (may be inflated via kappa)
    _, _, sigma_0_h_sq = get_true_variances(X_h, scenario_params['kappa'])
    
    # Generate historical outcomes
    Y_h = np.random.normal(mu_0_h, np.sqrt(sigma_0_h_sq))
    
    return {'X_h': X_h, 'Y_h': Y_h}

def generate_concurrent_outcomes(X: np.ndarray, Z: np.ndarray, 
                                tau_0: float, kappa: float = 1.0) -> np.ndarray:
    """
    Generates outcomes for current trial subjects (batch generation).
    
    This function is useful for generating outcomes for a set of subjects all at
    once, e.g., after allocations are determined. For sequential (adaptive)
    randomization, use generate_single_outcome() instead.
    
    Outcome model:
        Y_i | X_i, Z_i ~ N(mu_{Z_i}(X_i), sigma^2_{Z_i}(X_i))
    
    where:
        - Z_i = 0: Control arm -> Y_i ~ N(mu_0(X_i), sigma^2_0c(X_i))
        - Z_i = 1: Treatment arm -> Y_i ~ N(mu_1(X_i), sigma^2_1(X_i))
    
    Args:
        X: Covariate matrix of shape (n, 4)
        Z: Treatment assignments of shape (n,), values in {0, 1}
        tau_0: Base treatment effect (0 for null, 0.4 for alternative)
        kappa: Variance inflation factor (typically 1.0 for current trial,
               only used if comparing variance structures)
    
    Returns:
        Outcome vector of shape (n,)
    
    Notes:
        - No historical bias applied (current trial data)
        - Outcomes depend on both covariate and treatment assignment
        - Heteroscedasticity: variance differs across subjects
    
    Example:
        >>> X = generate_covariates(n=100)
        >>> Z = np.random.binomial(1, 0.5, 100)
        >>> Y = generate_concurrent_outcomes(X, Z, tau_0=0.4)
        >>> Y.shape
        (100,)
    """
    n = X.shape[0]
    Y = np.zeros(n)
    
    # Compute true means for both arms
    mu_0, _, mu_1 = get_true_means(X, tau_0)
    
    # Compute true variances for both arms
    sigma_0_c_sq, sigma_1_sq, _ = get_true_variances(X, kappa=kappa)
    
    # Identify subjects in each arm
    idx_ctrl = np.where(Z == 0)[0]
    idx_treat = np.where(Z == 1)[0]
    
    # Generate outcomes separately for each arm
    if len(idx_ctrl) > 0:
        Y[idx_ctrl] = np.random.normal(
            mu_0[idx_ctrl], 
            np.sqrt(sigma_0_c_sq[idx_ctrl])
        )
    
    if len(idx_treat) > 0:
        Y[idx_treat] = np.random.normal(
            mu_1[idx_treat], 
            np.sqrt(sigma_1_sq[idx_treat])
        )
    
    return Y

def generate_single_outcome(X_new: np.ndarray, z_new: int, 
                          tau_0: float, kappa: float = 1.0) -> float:
    """
    Generates outcome for a single subject (for sequential randomization).
    
    This function is called during adaptive trials to generate the outcome for
    each newly enrolled subject immediately after their treatment assignment is
    determined. The outcome is then used to update the adaptive allocation
    probabilities for subsequent subjects.
    
    Outcome model:
        Y | X, Z ~ N(mu_Z(X), sigma^2_Z(X))
    
    Sequential trial flow:
        1. Subject i arrives with covariates X_i
        2. Adaptive allocation rule determines Z_i based on past data
        3. This function generates Y_i | X_i, Z_i
        4. (X_i, Z_i, Y_i) added to cumulative dataset
        5. Repeat for next subject
    
    Args:
        X_new: Covariate vector for the new subject, shape (1, 4) or (4,)
        z_new: Treatment assignment for the new subject (0 = control, 1 = treatment)
        tau_0: Base treatment effect parameter
        kappa: Variance inflation factor (typically 1.0 for current trial)
    
    Returns:
        Scalar outcome value for the new subject
    
    Notes:
        - Called n - n_init times per trial (after burn-in)
        - Outcome observed immediately (no delayed outcomes)
        - No historical bias (Delta_0 = 0 for current trial)
    
    Example (adaptive randomization):
        >>> X_new = generate_covariates(n=1)
        >>> pi_1 = adaptive_allocation_rule(X_new, past_data)
        >>> z_new = np.random.binomial(1, pi_1)
        >>> y_new = generate_single_outcome(X_new, z_new, tau_0=0.4)
        >>> # Update past_data with (X_new, z_new, y_new)
    """
    # Ensure X_new is 2D
    X_new = X_new.reshape(1, -1)
    
    # Compute true mean and variance for this subject
    mu_0, _, mu_1 = get_true_means(X_new, tau_0)
    sigma_0_c_sq, sigma_1_sq, _ = get_true_variances(X_new, kappa=kappa)
    
    # Generate outcome based on treatment assignment
    if int(z_new) == 1:
        # Treatment arm
        return float(np.random.normal(mu_1[0], np.sqrt(sigma_1_sq[0])))
    else:
        # Control arm
        return float(np.random.normal(mu_0[0], np.sqrt(sigma_0_c_sq[0])))
