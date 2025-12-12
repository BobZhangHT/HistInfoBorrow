# BRAVE: Bayesian Robust Adaptive Variance-Aware Design

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation study code for BRAVE (Bayesian Robust Adaptive Variance-Aware), a variance-aware adaptive randomization design that borrows historical control information while guarding against incompatibility through a robust mixture prior framework.

## Overview

Four covariate-adaptive allocation rules are implemented in `methods.py`:

1. **BRAVE-IPD** (proposed): variance-aware borrowing with individual-patient historical data using robust mixture prior.  
2. **BRAVE-SLD** (proposed): same BRAVE framework using only study-level summaries such as historical sample size, mean, and variance.  
3. **CAHB** (Jin et al., 2023): commensurability prior on concurrent-control precision.  
4. **KBCD** (Jiang et al., 2018): kernel-based biased coin design without borrowing.

Key model components:

- Outcomes follow a normal model with arm-specific means and variances. The control mean uses a linear basis on the covariates, and the treatment mean adds a heterogeneous treatment effect that increases with the first covariate and decreases with the third.
- BRAVE separates historical and concurrent control variances and uses a Gamma-distributed borrowing-amount parameter M(X) to decide how much information to pull from the historical controls at each covariate profile.
- The information ratio R_n(X) guiding randomization grows when the concurrent control data are sparse or noisy relative to the historical control, and shrinks when compatibility is poor.
- BRAVE uses a robust Beta-Bernoulli mixture prior with compatibility weight w_L(X) to guard against incompatibility.
- Parameters for the control mean, control variances, and borrowing amount are updated by coordinate ascent, followed by a Gibbs sampler to draw posterior treatment-effect curves.

## Installation

- Python 3.8+ (tested on Python 3.9)
- Recommended: 8+ GB RAM for parallel runs

```bash
git clone https://github.com/yourusername/BRAVE.git
cd BRAVE
python -m venv venv
venv\Scripts\activate  # PowerShell
pip install -r requirements.txt
```

## Running Simulations

- Default (fast demo): `python main.py` uses the current `config.py` setting `FAST_DEMO=1` with `FAST_DEMO_REPLICATES=10` for quick sanity checks. Results are cached under `simulation_cache/` and can be resumed automatically.  
- Force full study (1,000 reps): `python main.py --full` or set `FAST_DEMO=0` in `config.py`.  
- Force demo regardless of config: `python main.py --demo`.  
- Reset cache: `python main.py --reset` (or delete `simulation_cache/`).

## Simulation Design (matches `config.py`)

- Current sample size: 200  
- Historical sample sizes: 400 or 800  
- Base treatment effect: 0.0 for Type I error or 0.4 for Power  
- Variance inflation for historical control: 0.7, 1.0, or 1.3 depending on scenario  
- Total scenarios: 24 (all combinations of historical size, base effect, scenario type, and variance factor)

Scenario-specific historical bias:

- S1 (ideal): No historical bias, variance factor 1.0.  
- S2 (variance mismatch): No historical bias, variance factor 0.7 or 1.3.  
- S3 (constant bias): Constant mean shift of 0.4, variance factor 0.7 or 1.3.  
- S4 (subgroup bias): Mean shift of 0.6 when the fourth covariate exceeds 1, variance factor 1.3.

## Outputs

### Final Inference Stage Metrics (CSV)

Tables (`results/tables/`): 
- `final_inference_metrics.csv`: Comprehensive evaluation metrics including:
  - **Estimation accuracy**: Bias, RMSE
  - **Uncertainty quantification**: 95% CrI Coverage, CI Width
  - **Hypothesis testing**: Type I Error (τ₀ = 0), Power (τ₀ = 0.4)
  - **BRAVE-specific**: Posterior mean M and w_L for BRAVE-IPD and BRAVE-SLD
- `estimation_metrics.tex`, `type_i_error_power.tex`: LaTeX-formatted tables for manuscript

### Adaptive Allocation Stage Plots

Plots (`results/plots/`): 

**Individual scenario plots** (per scenario × κ combination):
- `traj_alloc_*.pdf`: Allocation trajectory plots showing averaged allocation ratio to treatment arms for all 4 designs
- `density_rn_*.pdf`: Density plots of R_n(X) for all 4 designs (KBCD fixed at R_n = 1.0)
- `density_M_*.pdf`: Density plots of M(X) for BRAVE-IPD and BRAVE-SLD
- `density_wL_*.pdf`: Density plots of w_L(x) for BRAVE-IPD and BRAVE-SLD

**Combined grid plots**:
- `combined_6x4_grid_nh400.pdf`: 6×4 grid showing all scenarios (rows) × 4 plot types (columns) for n_h=400
- `combined_6x4_grid_nh800.pdf`: 6×4 grid showing all scenarios (rows) × 4 plot types (columns) for n_h=800
  - Columns: (1) Allocation trajectory, (2) R_n density, (3) M density, (4) w_L density
  - Rows: 6 scenario × κ combinations (S1 κ=1.0, S2 κ=0.7, S2 κ=1.3, S3 κ=0.7, S3 κ=1.3, S4 κ=1.3)

**Cache**: `simulation_cache/` stores replicate checkpoints for resumability.

## Configuration Notes (`config.py`)

- **Simulation settings**: `FAST_DEMO`, `FAST_DEMO_REPLICATES`, `FULL_RUN_REPLICATES`
- **Parallelism and memory**: `N_JOBS`, `MAX_MEMORY_PER_JOB`, `BATCH_SIZE`, `USE_CACHE`
- **BRAVE priors and Gibbs settings**: `PRIORS['brave_gamma_a0']`, `brave_gamma_b0`, `brave_amount_shape`, `uip_coord_iter`, `post_gibbs_iter`, etc.
- **CAHB settings**: `PRIORS['cahb_gamma']`, `cahb_lambda`, `cahb_lambda_quantile`
- **Scenario grid**: `get_scenario_definitions()` controls historical sample sizes, base treatment effects, variance factors (κ), and historical bias functions
- **Evaluation metrics**: `ALPHA` (significance level), `DECISION_THRESHOLD` (posterior probability threshold for success)

## Project Structure

```
BRAVE/
├── config.py               # Simulation configuration and scenarios
├── data_generation.py       # Data generating mechanisms (DGMs)
├── methods.py              # Implementation of BRAVE-IPD, BRAVE-SLD, CAHB, KBCD
├── analysis.py             # Results processing, visualization, and evaluation metrics
├── main.py                 # Main orchestration script
├── single_test.py          # Deterministic single-replication harness
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Citation

```bibtex
@article{brave2024,
  title   = {BRAVE: Bayesian Robust Adaptive Variance-Aware Design for Covariate-Adaptive Randomization with Historical Borrowing},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2024}
}
```

## License

This project is licensed under the MIT License. See `LICENSE`.
