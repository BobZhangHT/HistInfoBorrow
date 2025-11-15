"""
config.py

Configuration file for the CAHB-PP simulation study.

This module centralizes all simulation parameters, scenario definitions, prior
specifications, and file paths for the Covariate-Adjusted Historical Borrowing
with Power Prior (CAHB-PP) simulation study as described in the manuscript.

The simulation framework evaluates four covariate-adaptive randomization methods:
    - CAHB-PP-IPD: Proposed method with local power prior discount (IPD access)
    - CAHB-PP-SLD: Proposed method with summary-level discount
    - CAHB: Jin et al. (2023) - SIM paper
    - KBCD: Jiang et al. (2018) - kernel-based biased coin design

References:
    - CAHB-PP manuscript (references/CAHB_PP(2).pdf)
    - Jin et al. (2023): Statistics in Medicine (references/2023_SIM_CAHB.pdf)
    - Jiang et al. (2018): KBCD paper (references/KBCD.pdf)
"""

import os
import numpy as np


def _env_flag(var_name: str, default: str = "0") -> bool:
    """Parses boolean-like environment variables (\"1\", \"true\", \"on\")."""
    return os.environ.get(var_name, default).strip().lower() in {"1", "true", "yes", "on"}

# =============================================================================
# Core Simulation Settings
# =============================================================================

# Number of Monte Carlo replicates per scenario (Section 3.5 of manuscript)
# Recommended: 1000 for publication results, reduce for testing
FULL_RUN_REPLICATES = 1000

# Fast demonstration mode (smaller replication count, identical settings otherwise)
FAST_DEMO = _env_flag("CAHB_FAST_DEMO", "0")
FAST_DEMO_REPLICATES = int(os.environ.get("CAHB_FAST_DEMO_REPS", "50"))
FAST_DEMO_REPLICATES = max(5, FAST_DEMO_REPLICATES)

if FAST_DEMO:
    N_REPLICATES = FAST_DEMO_REPLICATES
else:
    N_REPLICATES = FULL_RUN_REPLICATES

# Parallel computation settings
# N_JOBS controls joblib parallelization:
#   -1: Use all available CPU cores
#   -2: Use all cores except one
#   N: Use exactly N cores
N_JOBS = -1

# Enable checkpointing to avoid re-running completed simulations
# Cache stored in CACHE_DIR; delete directory to force fresh run
USE_CACHE = True

# Memory management: maximum memory per worker process (in MB)
# Adjust based on available RAM to prevent memory overflow
# None = no limit (use with caution)
MAX_MEMORY_PER_JOB = 2000  # 2GB per worker

# =============================================================================
# Trial Design Parameters (Section 3.4)
# =============================================================================

# Burn-in phase: initial subjects enrolled with balanced allocation
# before adaptive randomization begins
N_INIT = 40

# Sequential allocation monitoring grid (for allocation diagnostics figure)
SEQ_MONITOR_START = N_INIT
SEQ_MONITOR_STEP = 20

# Allocation probability during burn-in (0.5 = balanced 1:1 randomization)
ALLOC_BURN_IN = 0.5

# Batch size for parallel processing to manage memory
# Larger batches = faster but more memory usage
BATCH_SIZE = 100

# =============================================================================
# Method Comparison Configuration (Section 3.3)
# =============================================================================

# Methods to evaluate in the simulation study
# Each method name must correspond to a class in methods.py
METHODS_TO_RUN = [
    'CAHB_PP_IPD',  # Proposed method with IPD-driven discounting
    'CAHB_PP_SLD',  # Proposed method using summary-level historical borrowing
    'CAHB',         # Jin et al. (2023) - baseline borrowing method
    'KBCD',         # Jiang et al. (2018) - no borrowing benchmark
]

# Method display names for tables and figures
METHOD_LABELS = {
    'CAHB_PP_IPD': 'CAHB-PP-IPD',
    'CAHB_PP_SLD': 'CAHB-PP-SLD',
    'CAHB': 'CAHB (Jin et al. 2023)',
    'KBCD': 'KBCD (Jiang et al. 2018)',
}

# =============================================================================
# File and Directory Structure
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache directory for checkpointing simulation replicates
CACHE_DIR = os.path.join(BASE_DIR, '.simulation_cache')

# Results directory structure
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
RAW_DATA_DIR = os.path.join(RESULTS_DIR, 'raw_data')

# =============================================================================
# Prior Hyperparameters (Section 2.5)
# =============================================================================

PRIORS = {
    # Inverse-Gamma priors for variance parameters
    # Chosen to be weakly informative
    'variance_ig_a': 2.0,     # IG shape parameter (alpha)
    'variance_ig_b': 1.0,     # IG scale parameter (beta)
    
    # Power prior discount parameter a(x) ~ Beta(a_beta_a, a_beta_b)
    # Beta(1,1) = Uniform[0,1] prior for CAHB-PP
    'a_beta_a': 1.0,          # Beta prior shape parameter (alpha)
    'a_beta_b': 1.0,          # Beta prior shape parameter (beta)
    'bridge_schedule': (0.0, 0.5, 1.0),
    'bridge_samples': 64,
    'bridge_burn_in': 32,
    'bridge_thin': 1,
    'bridge_step_size': 0.12,
    
    # CAHB tuning parameter (Jin et al. 2023, Section 4)
    # Controls borrowing strength via compatibility measure
    'cahb_gamma': np.sqrt(3.0)  # gamma = sqrt(3)
}

# =============================================================================
# Evaluation Metrics Configuration (Section 3.5)
# =============================================================================

# Confidence interval significance level (two-sided)
ALPHA = 0.05

# Posterior probability threshold for declaring treatment success
# Trial deemed successful if Pr(delta > 0 | Data) > DECISION_THRESHOLD
DECISION_THRESHOLD = 0.975

# Calibration diagnostics: per-replicate subsample size for discount summaries
CALIBRATION_MAX_SAMPLES = 32

# Effective sample size constraints (for numerical stability)
MIN_ESS = 1e-6      # Minimum effective sample size
MAX_ESS = 1e6       # Maximum effective sample size (clip to prevent overflow)

# =============================================================================
# Simulation Scenario Definitions (Sections 3.1-3.2)
# =============================================================================

def get_scenario_definitions():
    """
    Generates the full factorial grid of simulation scenarios.
    
    The simulation study evaluates 32 scenarios resulting from the factorial
    combination of:
        - Current trial sample sizes: n ∈ {200, 400}
        - Historical trial sample sizes: n_h ∈ {400, 800}
        - Base treatment effects: tau_0 ∈ {0.0, 0.4}
        - Data generating mechanisms: S1, S2, S3, S4
    
    Scenario Definitions (Section 3.2):
    -----------------------------------
    S1: Ideal case - no historical bias, homoscedastic variance
        - Delta_0(x) = 0 (perfect historical-current compatibility)
        - kappa = 1.0 (identical variance structure)
        - Tests borrowing under best conditions
    
    S2: Variance heterogeneity - no mean shift but variance mismatch
        - Delta_0(x) = 0 (compatible means)
        - kappa = 1.3 (historical controls have inflated variance)
        - Tests robustness to variance mismatch
    
    S3: Constant mean shift with variance heterogeneity
        - Delta_0(x) = 0.4 (constant historical bias)
        - kappa = 1.3 (variance mismatch)
        - Tests ability to detect and discount incompatible historical data
    
    S4: Covariate-dependent historical bias
        - Delta_0(x) = 0.6 * I(X4 = 2) (bias only for specific subgroup)
        - kappa = 1.3 (variance mismatch)
        - Tests local borrowing adaptivity
    
    Returns:
        list[dict]: List of scenario dictionaries, each containing:
            - id: Unique scenario identifier (0 to 31)
            - name: Human-readable scenario name
            - n: Current trial sample size
            - n_h: Historical trial sample size
            - tau_0: Base treatment effect (0 = Type I error, 0.4 = Power)
            - kappa: Historical variance inflation factor
            - Delta_0_func: Function x -> Delta_0(x) defining historical bias
            - scenario_type: Scenario label (S1, S2, S3, S4)
    
    Notes:
        - tau_0 = 0.0: scenarios for evaluating Type I error control
        - tau_0 = 0.4: scenarios for evaluating power
        - Each scenario replicated N_REPLICATES times
    """
    
    # Factorial design grid
    n_list = [200, 400]              # Current trial sample sizes
    n_h_list = [400, 800]            # Historical trial sample sizes  
    tau_0_list = [0.0, 0.4]          # Base treatment effects
    
    # Data-generating mechanism specifications (Section 3.2)
    scenario_params = {
        'S1': {
            'Delta_0_func': lambda x: 0.0,           # No historical bias
            'kappa': 1.0,                             # Homoscedastic variance
            'description': 'Ideal: no bias, homoscedastic'
        },
        'S2': {
            'Delta_0_func': lambda x: 0.0,           # No historical bias
            'kappa': 1.3,                             # Variance heterogeneity
            'description': 'Variance mismatch only'
        },
        'S3': {
            'Delta_0_func': lambda x: 0.4,           # Constant mean shift
            'kappa': 1.3,                             # Variance heterogeneity
            'description': 'Constant bias + variance mismatch'
        },
        'S4': {
            'Delta_0_func': lambda x: 0.6 * (x[3] == 2),  # Subgroup-specific bias
            'kappa': 1.3,                                   # Variance heterogeneity
            'description': 'Local bias (X4=2 subgroup)'
        }
    }
    
    scenarios = []
    scenario_id = 0
    
    # Generate full factorial combination
    for n in n_list:
        for n_h in n_h_list:
            for tau_0 in tau_0_list:
                for s_name in ['S1', 'S2', 'S3', 'S4']:  # Ordered for consistency
                    s_params = scenario_params[s_name]
                    
                    scenarios.append({
                        'id': scenario_id,
                        'name': f"n={n}_nh={n_h}_tau0={tau_0}_{s_name}",
                        'n': n,
                        'n_h': n_h,
                        'tau_0': tau_0,
                        'kappa': s_params['kappa'],
                        'Delta_0_func': s_params['Delta_0_func'],
                        'scenario_type': s_name,
                        'description': s_params['description']
                    })
                    scenario_id += 1
    
    return scenarios


# =============================================================================
# Derived Parameters (computed from above settings)
# =============================================================================

def get_total_simulations():
    """
    Computes the total number of simulation runs.
    
    Returns:
        int: Total number of (scenario × method × replicate) combinations
    """
    n_scenarios = len(get_scenario_definitions())
    n_methods = len(METHODS_TO_RUN)
    return n_scenarios * n_methods * N_REPLICATES


# =============================================================================
# Validation Functions
# =============================================================================

def validate_config():
    """
    Validates configuration parameters and prints warnings if needed.
    
    Checks for:
        - Reasonable parameter ranges
        - Directory creation capability
        - Memory settings vs available RAM
    """
    import warnings
    
    # Check replicate count
    if N_REPLICATES < 100:
        warnings.warn(
            f"N_REPLICATES={N_REPLICATES} is low for reliable results. "
            "Consider N_REPLICATES >= 1000 for publication.",
            UserWarning
        )
    
    # Check if results directory is writable
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
    except OSError as e:
        warnings.warn(f"Cannot create results directory: {e}", UserWarning)
    
    # Validate prior parameters
    if PRIORS['a_beta_a'] <= 0 or PRIORS['a_beta_b'] <= 0:
        raise ValueError("Beta prior parameters must be positive")
    
    # Print configuration summary
    n_sims = get_total_simulations()
    print(f"Configuration validated. Total simulations: {n_sims:,}")
    
    return True
