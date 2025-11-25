# CAHB-UIP: Covariate-Adjusted Historical Borrowing with Unit Information Prior

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation study code for **Covariate-Adjusted Historical Borrowing with Unit Information Prior (CAHB-UIP)**—a variance-aware adaptive randomization design that borrows historical control information while guarding against incompatibility.

## Overview

Four covariate-adaptive allocation rules are implemented in `methods.py`:

1. **CAHB-UIP-IPD** (proposed): variance-aware borrowing with individual-patient historical data.  
2. **CAHB-UIP-SLD** (proposed): same UIP framework using only study-level summaries \((n_h, \hat{\mu}_h, V_h)\).  
3. **CAHB** (Jin et al., 2023): commensurability prior on concurrent-control precision.  
4. **KBCD** (Jiang et al., 2018): kernel-based biased coin design without borrowing.

Key model components follow the manuscript:

$$
Y_{z} \mid X \sim \mathcal{N}\!\big(\mu_z(X),\, \sigma_z^2(X)\big), \quad z \in \{0,1\}
$$

$$
\mu_0(x) = b(x)^\top \beta_0,\qquad \tau(x) = \tau_0 + 0.5x_1 - 0.5x_3,\qquad \mu_1(x) = \mu_0(x) + \tau(x)
$$

CAHB-UIP introduces a Gamma-distributed amount parameter \(M(x)\) and separates the historical and concurrent control variances. The local CECCS gain driving randomization is

$$
R_n(x) = 1 + \frac{M(x)\,\sigma^2_{0,c}(x)}{W_0(x)\,\sigma^2_{0,h}(x)},
$$

where \(W_0(x)\) is the kernel-weighted effective sample size for the current control. Coordinate-ascent updates for \(\mu_0(x)\), \(\sigma^2_{0,c}(x)\), \(\sigma^2_{0,h}(x)\), and \(M(x)\) are followed by a Gibbs sampler for posterior draws of \(\{\mu_0(x), \mu_1(x)\}\).

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

- **Default (fast demo):** `python main.py` uses the current `config.py` setting `FAST_DEMO=1` with `FAST_DEMO_REPLICATES=10` for quick sanity checks. Results are cached under `simulation_cache/` and can be resumed automatically.  
- **Force full study (1,000 reps):** `python main.py --full` or set `FAST_DEMO=0` in `config.py`.  
- **Force demo regardless of config:** `python main.py --demo`.  
- **Reset cache:** `python main.py --reset` (or delete `simulation_cache/`).

## Simulation Design (matches `config.py`)

- Current sample size: \(n = 200\)  
- Historical sample sizes: \(n_h \in \{400, 800\}\)  
- Base treatment effect: \(\tau_0 \in \{0.0\ \text{(Type I)},\ 0.4\ \text{(Power)}\}\)  
- Variance inflation for historical control: \(\kappa \in \{0.7, 1.0, 1.3\}\) depending on scenario  
- Total scenarios: **24** (all combinations of \(n_h\), \(\tau_0\), scenario type, and \(\kappa\))

Scenario-specific historical bias \(\Delta_0(x)\):

- **S1 (ideal):** \(\Delta_0(x)=0,\ \kappa=1.0\)  
- **S2 (variance mismatch):** \(\Delta_0(x)=0,\ \kappa\in\{0.7,1.3\}\)  
- **S3 (constant bias):** \(\Delta_0(x)=0.4,\ \kappa\in\{0.7,1.3\}\)  
- **S4 (subgroup bias):** \(\Delta_0(x)=0.6\,\mathbb{1}\{x_4>1\},\ \kappa=1.3\)

## Outputs

- **Tables (`results/tables/`):** `estimation_metrics.csv`, `estimation_metrics.tex`, `type_i_error_power.tex` (Bias, RMSE, Coverage, CI width, Type I error, Power, allocation rate).  
- **Plots (`results/plots/`):** Boxplots of \(R_n(\mathbf{X})\) and \(M(\mathbf{X})\) by \((n, n_h)\), plus subgroup borrowing diagnostics for S4 (`rn_box_*.pdf`, `m_box_*.pdf`, `s4_borrowing_box.pdf`).  
- **Cache:** `simulation_cache/` stores replicate checkpoints for resumability.

## Configuration Notes (`config.py`)

- `FAST_DEMO`, `FAST_DEMO_REPLICATES`, `FULL_RUN_REPLICATES`
- Parallelism and memory: `N_JOBS`, `MAX_MEMORY_PER_JOB`, `BATCH_SIZE`, `USE_CACHE`
- Priors and Gibbs settings: `PRIORS['uip_gamma_alpha']`, `uip_coord_iter`, `post_gibbs_iter`, etc.
- Scenario grid: `get_scenario_definitions()` controls \(n_h\), \(\tau_0\), \(\kappa\), and \(\Delta_0(\cdot)\).

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
