# CAHB-UIP: Covariate-Adjusted Historical Borrowing with Unit Information Prior

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation study code for Covariate-Adjusted Historical Borrowing with Unit Information Prior (CAHB-UIP), a variance-aware adaptive randomization design that borrows historical control information while guarding against incompatibility.

## Overview

Four covariate-adaptive allocation rules are implemented in `methods.py`:

1. **CAHB-UIP-IPD** (proposed): variance-aware borrowing with individual-patient historical data.  
2. **CAHB-UIP-SLD** (proposed): same UIP framework using only study-level summaries such as historical sample size, mean, and variance.  
3. **CAHB** (Jin et al., 2023): commensurability prior on concurrent-control precision.  
4. **KBCD** (Jiang et al., 2018): kernel-based biased coin design without borrowing.

Key model components:

- Outcomes follow a normal model with arm-specific means and variances. The control mean uses a linear basis on the covariates, and the treatment mean adds a heterogeneous treatment effect that increases with the first covariate and decreases with the third.
- CAHB-UIP separates historical and concurrent control variances and uses a Gamma-distributed borrowing-amount parameter to decide how much information to pull from the historical controls at each covariate profile.
- The CECCS gain guiding randomization grows when the concurrent control data are sparse or noisy relative to the historical control, and shrinks when compatibility is poor.
- Parameters for the control mean, control variances, and borrowing amount are updated by coordinate ascent, followed by a Gibbs sampler to draw posterior treatment-effect curves.

## Installation

- Python 3.8+ (tested on Python 3.9)
- Recommended: 8+ GB RAM for parallel runs

```bash
git clone https://github.com/yourusername/CAHB-UIP.git
cd CAHB-UIP
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

- Tables (`results/tables/`): `estimation_metrics.csv`, `estimation_metrics.tex`, `type_i_error_power.tex` (Bias, RMSE, Coverage, CI width, Type I error, Power, allocation rate).  
- Plots (`results/plots/`): Boxplots of the CECCS gain and borrowing amount by each combination of current and historical sample sizes, plus subgroup borrowing diagnostics for S4 (`rn_box_*.pdf`, `m_box_*.pdf`, `s4_borrowing_box.pdf`).  
- Cache: `simulation_cache/` stores replicate checkpoints for resumability.

## Configuration Notes (`config.py`)

- `FAST_DEMO`, `FAST_DEMO_REPLICATES`, `FULL_RUN_REPLICATES`
- Parallelism and memory: `N_JOBS`, `MAX_MEMORY_PER_JOB`, `BATCH_SIZE`, `USE_CACHE`
- Priors and Gibbs settings: `PRIORS['uip_gamma_alpha']`, `uip_coord_iter`, `post_gibbs_iter`, etc.
- Scenario grid: `get_scenario_definitions()` controls historical sample sizes, base treatment effects, variance factors, and historical bias functions.

## Project Structure

```
CAHB-UIP/
├── config.py               # Simulation configuration and scenarios
├── data_generation.py      # Data generating mechanisms (DGMs)
├── methods.py              # Implementation of CAHB-UIP, CAHB, KBCD
├── analysis.py             # Results processing and visualization
├── main.py                 # Main orchestration script
├── single_test.py          # Deterministic single-replication harness
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Citation

```bibtex
@article{cahb-uip2024,
  title   = {Covariate-Adjusted Historical Borrowing with Unit Information Prior},
  author  = {[Authors]},
  journal = {[Journal]},
  year    = {2024}
}
```

## License

This project is licensed under the MIT License. See `LICENSE`.
