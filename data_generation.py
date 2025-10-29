"""
data_generation.py

Implements the Data-Generating Mechanisms (DGMs) from Sec 3.1.
Generates covariates and outcomes for both historical and current trials.
"""

import numpy as np

def generate_covariates(n, p=4):
    """
    Generates the p=4 baseline covariates (Sec 3.1).
    X1 ~ N(0,1)
    X2 ~ Bernoulli(0.5)
    X3 ~ Unif[-1, 1]
    X4 ~ Categorical{0,1,2} with p=(0.4, 0.4, 0.2)
    
    Args:
        n (int): Number of subjects.
    
    Returns:
        np.ndarray: An (n, p) array of covariates.
    """
    X = np.zeros((n, p))
    X[:, 0] = np.random.normal(0, 1, n)
    X[:, 1] = np.random.binomial(1, 0.5, n)
    X[:, 2] = np.random.uniform(-1, 1, n)
    X[:, 3] = np.random.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2])
    return X

def _get_b_x(X):
    """Helper to create the design vector b(x) from Sec 3.1."""
    n = X.shape[0]
    b_x = np.ones((n, 6))
    b_x[:, 1] = X[:, 0]  # X1
    b_x[:, 2] = X[:, 1]  # X2
    b_x[:, 3] = X[:, 2]  # X3
    b_x[:, 4] = (X[:, 3] == 1).astype(float) # 1{X4=1}
    b_x[:, 5] = (X[:, 3] == 2).astype(float) # 1{X4=2}
    return b_x

def get_true_means(X, tau_0):
    """
    Calculates the true covariate-dependent means (Sec 3.1).
    
    Args:
        X (np.ndarray): (n, p) covariate array.
        tau_0 (float): Base treatment effect.
        
    Returns:
        tuple: (mu_0, tau, mu_1)
            mu_0 (np.ndarray): True control mean.
            tau (np.ndarray): True treatment effect (CATE).
            mu_1 (np.ndarray): True treated mean.
    """
    beta_0 = np.array([0, 0.8, 0.5, -0.5, 0.3, -0.2])
    b_x = _get_b_x(X)
    
    mu_0 = b_x @ beta_0 + 0.75 * X[:, 0] * X[:, 2] # mu_0(x) = b(x)T*beta_0 + 0.75*X1*X3
    tau = tau_0 + 0.5 * X[:, 0] - 0.5 * (X[:, 1] == 1) # tau(x)
    mu_1 = mu_0 + tau # mu_1(x)
    
    return mu_0, tau, mu_1

def get_true_variances(X, kappa):
    """
    Calculates the true heteroscedastic variances (Sec 3.1).
    
    Args:
        X (np.ndarray): (n, p) covariate array.
        kappa (float): Historical variance factor.
        
    Returns:
        tuple: (sigma_0_c_sq, sigma_1_sq, sigma_0_h_sq)
    """
    sigma_0_c_sq = np.exp(0.2 + 0.4 * X[:, 1] + 0.3 * (X[:, 3] == 2))
    sigma_1_sq = np.exp(0.2 + 0.2 * X[:, 0])
    sigma_0_h_sq = kappa * np.exp(0.2 + 0.2 * X[:, 1])
    
    return sigma_0_c_sq, sigma_1_sq, sigma_0_h_sq

def generate_historical_data(n_h, scenario_params):
    """
    Generates a full historical dataset (Sec 3.1).
    
    Args:
        n_h (int): Number of historical subjects.
        scenario_params (dict): Specific scenario config.
        
    Returns:
        dict: {'X_h': np.ndarray, 'Y_h': np.ndarray}
    """
    X_h = generate_covariates(n_h)
    
    # Get base means
    mu_0_base, _, _ = get_true_means(X_h, tau_0=0.0)
    
    # Apply historical shift (Sec 3.2)
    Delta_0 = np.array([scenario_params['Delta_0_func'](x) for x in X_h])
    mu_0_h = mu_0_base + Delta_0
    
    # Get historical variance
    _, _, sigma_0_h_sq = get_true_variances(X_h, scenario_params['kappa'])
    
    # Generate outcomes
    Y_h = np.random.normal(mu_0_h, np.sqrt(sigma_0_h_sq))
    
    return {'X_h': X_h, 'Y_h': Y_h}

def generate_concurrent_outcomes(X, Z, tau_0, kappa=1.0):
    """
    Generates outcomes for a fully enrolled concurrent trial.
    
    Args:
        X (np.ndarray): (n, p) covariates for concurrent subjects.
        Z (np.ndarray): (n,) treatment assignments (0 or 1).
        tau_0 (float): Base treatment effect for this run.
        kappa (float): kappa for variance generation (default 1.0, unused for concurrent).

    Returns:
        np.ndarray: (n,) array of outcomes Y.
    """
    n = X.shape[0]
    Y = np.zeros(n)
    
    mu_0, _, mu_1 = get_true_means(X, tau_0)
    sigma_0_c_sq, sigma_1_sq, _ = get_true_variances(X, kappa=kappa)
    
    # Find subjects in each arm
    idx_ctrl = np.where(Z == 0)[0]
    idx_treat = np.where(Z == 1)[0]
    
    # Generate outcomes based on assigned arm
    Y[idx_ctrl] = np.random.normal(mu_0[idx_ctrl], np.sqrt(sigma_0_c_sq[idx_ctrl]))
    Y[idx_treat] = np.random.normal(mu_1[idx_treat], np.sqrt(sigma_1_sq[idx_treat]))
    
    return Y

def generate_single_outcome(X_new, z_new, tau_0, kappa=1.0):
    """
    Generates the outcome for a single subject sequentially.

    Args:
        X_new (np.ndarray): (1, p) covariate vector for the subject.
        z_new (int): Treatment assignment (0 = control, 1 = treatment).
        tau_0 (float): Current trial base treatment effect.
        kappa (float): Variance inflation factor for historical mismatch.

    Returns:
        float: Simulated outcome.
    """
    X_new = X_new.reshape(1, -1)
    mu_0, _, mu_1 = get_true_means(X_new, tau_0)
    sigma_0_c_sq, sigma_1_sq, _ = get_true_variances(X_new, kappa=kappa)
    if int(z_new) == 1:
        return float(np.random.normal(mu_1[0], np.sqrt(sigma_1_sq[0])))
    return float(np.random.normal(mu_0[0], np.sqrt(sigma_0_c_sq[0])))
