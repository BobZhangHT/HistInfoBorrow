"""
config.py

Configuration file for the CAHB-PP simulation study.

This file defines all static parameters, scenario definitions,
file paths, and settings for the simulation runs, as described
in Section 3 of the CAHB_PP(2).pdf paper.
"""

import os
import numpy as np

# --- Core Simulation Settings ---
N_REPLICATES = 1000  # Number of simulation replicates per scenario (Sec 3.5)
N_JOBS = -1          # Number of parallel cores to use (-1 = all available)
USE_CACHE = True     # Enable/disable checkpointing/caching

# --- Trial Conduct Settings (Sec 3.4) ---
N_INIT = 40          # Burn-in sample size
ALLOC_BURN_IN = 0.5  # 1:1 allocation for burn-in phase

# --- Methods to Compare (Sec 3.3) ---
METHODS_TO_RUN = [
    'CAHB_PP', # Proposed method
    'CAHB',    # Jin et al. (2023)
    'KBCD',    # Jiang et al. (2018)
    'rMAP_KBCD' # Schmidli et al. (2014) + KBCD
]

# --- File/Directory Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '.simulation_cache')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')

# --- Prior Hyperparameters (Sec 3.4) ---
PRIORS = {
    'variance_ig_a': 2.0,  # Inverse-Gamma alpha for variance priors
    'variance_ig_b': 1.0,  # Inverse-Gamma beta for variance priors
    'a_beta_a': 1.0,     # Beta(1,1) prior for power prior discount a(x)
    'a_beta_b': 1.0,
    'rmap_weight': 0.1   # Weight for the robust (vague) component in rMAP
}

# --- Evaluation Settings (Sec 3.5) ---
ALPHA = 0.05               # Significance level for CIs
DECISION_THRESHOLD = 0.975 # Posterior prob threshold for success Pr(delta > 0 | D)

def get_scenario_definitions():
    """
    Defines the grid of simulation scenarios based on Sec 3.1 and 3.2.

    Returns:
        list[dict]: A list of scenario parameter dictionaries.
    """
    
    # Base parameters from Sec 3.1
    n_list = [200, 400]           # Current trial sample size
    n_h_list = [400, 800]         # Historical trial sample size
    tau_0_list = [0.0, 0.4]       # Base treatment effect (for Type I error / power)
    
    # Parameters for Scenarios S1-S4 (Sec 3.2)
    scenario_params = {
        'S1': {'Delta_0_func': lambda x: 0.0, 'kappa': 1.0},
        'S2': {'Delta_0_func': lambda x: 0.0, 'kappa': 1.3}, # Also 0.7, paper uses 0.7, 1.3. Using 1.3 for brevity.
        'S3': {'Delta_0_func': lambda x: 0.4, 'kappa': 1.3}, # Also 0.7
        'S4': {'Delta_0_func': lambda x: 0.6 * (x[3] == 2), 'kappa': 1.3},
    }

    scenarios = []
    scenario_id = 0
    
    # Create the full factorial design
    for n in n_list:
        for n_h in n_h_list:
            for tau_0 in tau_0_list:
                for s_name, s_params in scenario_params.items():
                    scenarios.append({
                        'id': scenario_id,
                        'name': f"n={n}_nh={n_h}_tau0={tau_0}_{s_name}",
                        'n': n,
                        'n_h': n_h,
                        'tau_0': tau_0,
                        'kappa': s_params['kappa'],
                        'Delta_0_func': s_params['Delta_0_func']
                    })
                    scenario_id += 1
                    
    return scenarios
