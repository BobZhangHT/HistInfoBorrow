# CAHB-UIP: Covariate-Adjusted Historical Borrowing with Unit Information Prior

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation study code for **"Covariate-Adjusted Historical Borrowing with Unit Information Prior (CAHB-UIP)"** — a variance-aware adaptive randomization method for clinical trials with historical control information.

## Overview

This repository implements the sequential design described in the CAHB-UIP manuscript. Four covariate-adaptive allocation rules are compared:

1. **CAHB-UIP-IPD** *(Proposed)* — Variance-aware borrowing that uses individual-patient historical data and the variance-aware unit information prior.
2. **CAHB-UIP-SLD** *(Proposed)* — Same framework but consuming only study-level summaries of the historical controls.
3. **CAHB (Jin et al., 2023)** — Baseline covariate-adjusted borrowing with commensurability priors.
4. **KBCD (Jiang et al., 2018)** — Kernel-based biased coin design with no historical borrowing.

### Key Innovation

CAHB-UIP decouples the concurrent-control variance from the historical-control variance and introduces a **Gamma-distributed amount parameter** \(M(x)\) that measures how many *units of normalized Fisher information* can be borrowed at covariate profile \(x\). The framework:

- Updates \( \mu_0(x), \sigma^2_{0,c}(x), \sigma^2_{0,h}(x), M(x) \) via the variance-aware coordinate-ascent Algorithm 1.
- Computes the local CECCS gain \(R_n(x) = 1 + M(x)\sigma^2_{0,c}(x)/(W_0(x)\sigma^2_{0,h}(x))\) and drives adaptive allocation through Algorithm 2.
- Runs a Gibbs sampler (Algorithm 3) for final inference, providing posterior draws of the individualized treatment effect functions.

The SLD variant swaps the IPD kernel-weighted quantities \(W_h, \bar{Y}_h, SS_h\) with the global summary triple \((n_h, \hat{\mu}_h, V_h)\) while retaining the same conjugate updates.

## Installation

### Prerequisites

- Python 3.8 or higher (tested on Python 3.9)
- 8+ GB RAM recommended for parallel simulations

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CAHB-UIP.git
cd CAHB-UIP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Complete Simulation Study

```bash
python main.py
```

This runs 1,000 replicates per (scenario × method) combination, executes in parallel with all available CPU cores, checkpoints intermediate results, and creates publication-ready tables/plots under `results/`.

### Fast Demo Mode

```bash
python main.py --demo
python main.py --demo --reset  # delete cache first
```

Environment variables `CAHB_FAST_DEMO=1` and `CAHB_FAST_DEMO_REPS=<N>` provide the same switches. Use `python main.py --full` to force the publication-sized run even when demo flags are set.

### Resume / Reset

- Rerun `python main.py` to resume from cached replicates.
- Remove `.simulation_cache/` (or use `--reset`) to start from scratch.

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

## Simulation Design

The simulation evaluates **32 scenarios** produced by the factorial combination:

- Current sample size (`n`): 200, 400
- Historical sample size (`n_h`): 400, 800
- Base treatment effect (`tau_0`): 0.0 (Type I error) or 0.4 (Power)
- Generating mechanisms: S1 (ideal), S2 (variance mismatch), S3 (constant bias), S4 (subgroup bias)

### Evaluation Metrics

- **Estimation**: Bias, RMSE, Coverage, CI Width
- **Decision**: Type I error (`tau_0 = 0`), Power (`tau_0 = 0.4`)
- **Allocation**: Treatment allocation rate and CECCS trajectories

## Methods Implemented

### CAHB-UIP-IPD (Proposed)

Implements Algorithms 1–3 from the CAHB-UIP paper. The method:

- Fits local kernel statistics for \(W_0, \bar{Y}_0, W_h, \bar{Y}_h, SS\) at each covariate profile.
- Runs the variance-aware coordinate-ascent updates for \(\mu_0, \sigma^2_{0,c}, \sigma^2_{0,h}, M\) until convergence.
- Computes the CECCS gain \(R_n(x)\) to bias future assignments toward the informationally weaker arm.
- Performs a local Gibbs sampler for \(\{\mu_0(x), \mu_1(x)\}\) to obtain posterior draws for inference.

### CAHB-UIP-SLD (Proposed)

Applies the same UIP machinery when only study-level historical summaries are available. The kernel-based historical sufficient statistics are replaced with the global triple \((n_h, \hat{\mu}_h, V_h)\), yielding a conjugate Gaussian/Gamma update without IPD access.

### CAHB (Jin et al., 2023)

Baseline covariate-adjusted borrowing with commensurability priors that operate on the concurrent control precision.

### KBCD (Jiang et al., 2018)

Kernel-based biased coin design that balances local sample sizes without borrowing historical data.

## Output Files

### Tables (`results/tables/`)

- `simulation_summary.csv` — Full metric dump.
- `estimation_metrics.csv/.tex` — Bias/RMSE/Coverage summaries.
- `type_i_error_power.tex` — Scenario-level decision summaries.

### Plots (`results/plots/`)

- `type_power_curves.pdf` — Type I error & power trajectories.
- `allocation_dynamics.pdf` — Average treatment allocation and CECCS gain trajectories.
- `calibration_discount.pdf` — Posterior summaries of the UIP amount parameter \(M(x)\) across covariates.

All figures are PDF (300 DPI, Times New Roman) and ready for publication.

## Configuration Highlights

Key parameters live in `config.py`:

- `N_REPLICATES`, `N_JOBS`, `USE_CACHE`
- `METHODS_TO_RUN` and `METHOD_LABELS`
- `PRIORS` (variance IG hyperparameters, UIP Gamma prior, Gibbs settings)
- Scenario definitions and bias functions (`delta0_*`)

## Citation

```bibtex
@article{cahb-uip2024,
  title={Covariate-Adjusted Historical Borrowing with Unit Information Prior},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).
