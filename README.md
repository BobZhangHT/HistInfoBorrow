# CAHB-PP: Covariate-Adjusted Historical Borrowing with Power Prior

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation study code for **"Covariate-Adjusted Historical Borrowing with Power Prior (CAHB-PP)"** - a novel adaptive randomization method for clinical trials with historical control data.

## Overview

This repository implements the simulation study evaluating four covariate-adaptive randomization methods:

1. **CAHB-PP-IPD** *(Proposed)*: Bridge-sampled local power-prior discounting using individual-level historical data
2. **CAHB-PP-SLD** *(Proposed)*: Summary-level discounting for sites without IPD access
3. **CAHB**: Covariate-adjusted historical borrowing (Jin et al., 2023)
4. **KBCD**: Kernel-based biased coin design - no borrowing benchmark (Jiang et al., 2018)

### Key Innovation

CAHB-PP introduces a **local, data-driven discount parameter** (x) ∈ [0,1] learned via bridge sampling + Metropolis–Hastings updates. The sampler adaptively tunes borrowing strength based on the compatibility between historical and current control data at each covariate value x, automatically down-weighting incompatible sources and preserving subgroup robustness.

## Installation

### Prerequisites

- Python 3.8 or higher (tested on Python 3.9)
- 8+ GB RAM recommended for parallel simulations

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CAHB-PP.git
cd CAHB-PP

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Complete Simulation Study

`ash
python main.py
`

This will:
- Run 1,000 replicates per (scenario × method) combination
- Execute in parallel using all available CPU cores
- Cache results for resumability
- Generate tables and plots in 
esults/

### Fast Demo Mode

Enable a lightweight run (default 50 replicates) without editing the code:

```bash
python main.py --demo
```

Add `--reset` to delete `.simulation_cache/` before running (works for both demo and full modes):

```bash
python main.py --demo --reset   # fast mode, starting fresh
python main.py --reset          # full run from scratch
```

Alternatively, set an environment variable before running:

```bash
# Linux/macOS
export CAHB_FAST_DEMO=1
python main.py

# Windows PowerShell
$env:CAHB_FAST_DEMO = 1
python main.py
```

Override the replicate count via `CAHB_FAST_DEMO_REPS` (e.g., set to 10 for smoke tests). Use `python main.py --full` to force the publication-size run even if a demo flag/environment variable is set.

### Resume Interrupted Run

Simply rerun - the checkpoint system will skip completed replicates:
`ash
python main.py
`

### Force Fresh Run

`ash
# Linux/macOS
rm -rf .simulation_cache/
python main.py

# Windows PowerShell
Remove-Item -Recurse -Force .simulation_cache
python main.py
`

## Project Structure

```
CAHB-PP/
├── config.py               # Simulation configuration and scenarios
├── data_generation.py      # Data generating mechanisms (DGMs)
├── methods.py              # Implementation of all methods
├── analysis.py             # Results processing and visualization
├── main.py                 # Main orchestration script
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT License
└── README.md               # This file
```

## Simulation Design

The simulation evaluates **32 scenarios** from a full factorial design:
- **Current sample size** (`n`): 200, 400
- **Historical sample size** (`n_h`): 400, 800
- **Base treatment effect** (`tau_0`): 0.0 (Type I error), 0.4 (Power)
- **Data generating mechanisms**: S1 (ideal), S2 (variance mismatch), S3 (constant bias), S4 (subgroup-specific bias)

### Evaluation Metrics

- **Estimation**: Bias, RMSE, Coverage, CI Width
- **Decision**: Type I Error (`tau_0 = 0`), Power (`tau_0 = 0.4`)
- **Allocation**: Treatment allocation rate

## Methods Implemented

### CAHB-PP-IPD (Proposed)
Bridge sampling + Metropolis–Hastings draws (x) using the full historical individual patient data and the local compatibility likelihood. The sampled discounts feed both allocation (R_n(x)) and inference to provide fully adaptive borrowing.

### CAHB-PP-SLD (Proposed)
Uses the same bridge sampler but replaces the IPD likelihood with summary-level strata (defined by key covariates), enabling trials to borrow from partners that can share only aggregated information.

### CAHB (Jin et al., 2023)
Sample size inflation R_n(x) based on posterior precision with compatibility measure τ_n(x); no stochastic discounting.

### KBCD (Jiang et al., 2018)
No-borrowing benchmark using kernel-based covariate-adaptive allocation.

## Output Files

### Tables (`results/tables/`)
- `simulation_summary.csv` – Full metric dump
- `estimation_metrics.csv / .tex` – Publication-ready Bias/RMSE/Coverage/CI-width table
- `type_i_error_power.tex` – Scenario-level decision summaries

### Plots (`results/plots/`)
- `type_power_curves.pdf` – Type I error & power vs. enrolled sample size (two subplots with 95% CIs)
- `allocation_dynamics.pdf` – Average treatment allocation and `R_n(X)` trajectories under adaptive assignment
- `calibration_discount.pdf` – Discount calibration (`a(x)`) vs. sample size across scenarios

All plots are publication-ready (PDF, 300 DPI, Times New Roman font).

## Configuration

Key parameters in `config.py`:
- `N_REPLICATES = 1000` - Number of replicates per scenario
- `N_JOBS = -1` - CPU cores (-1 = all available)
- `USE_CACHE = True` - Enable checkpointing
- `METHODS_TO_RUN` - List of methods to compare

## Citation

If you use this code in your research, please cite:

```bibtex
@article{cahb-pp2024,
  title={Covariate-Adjusted Historical Borrowing with Power Prior for Adaptive Randomization},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
**Last Updated**: October 2025
