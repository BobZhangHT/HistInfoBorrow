# CAHB-PP: Covariate-Adjusted Historical Borrowing with Power Prior

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Simulation study code for **"Covariate-Adjusted Historical Borrowing with Power Prior (CAHB-PP)"** - a novel adaptive randomization method for clinical trials with historical control data.

## Overview

This repository implements the simulation study evaluating four covariate-adaptive randomization methods:

1. **CAHB-PP** (Proposed): Covariate-adjusted borrowing with local power prior discounting
2. **CAHB**: Covariate-adjusted historical borrowing (Jin et al., 2023)
3. **KBCD**: Kernel-based biased coin design - no borrowing benchmark (Jiang et al., 2018)
4. **rMAP-KBCD**: Robust MAP prior with KBCD allocation (Schmidli et al., 2014)

### Key Innovation

CAHB-PP introduces a **local, data-driven discount parameter** `a(x) ∈ [0,1]` that adapts borrowing strength based on compatibility between historical and current control data at each covariate value `x`. This enables automatic downweighting of incompatible historical data and robustness to subgroup-specific bias.

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

```bash
python main.py
```

This will:
- Run 1,000 replicates per (scenario × method) combination
- Execute in parallel using all available CPU cores
- Cache results for resumability
- Generate tables and plots in `results/`

### Quick Test

For testing, reduce replicates in `config.py`:
```python
N_REPLICATES = 100  # Instead of 1000
```

### Resume Interrupted Run

Simply rerun - the checkpoint system will skip completed replicates:
```bash
python main.py
```

### Force Fresh Run

```bash
# Linux/macOS
rm -rf .simulation_cache/
python main.py

# Windows PowerShell
Remove-Item -Recurse -Force .simulation_cache
python main.py
```

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

### CAHB-PP (Proposed)
Local discount parameter `a(x)` learned from data via power prior framework. Automatically downweights incompatible historical data.

### CAHB (Jin et al., 2023)
Sample size inflation `R_n(x)` based on posterior precision with compatibility measure `τ_n(x)`.

### KBCD (Jiang et al., 2018)
No-borrowing benchmark using kernel-based covariate-adaptive allocation.

### rMAP-KBCD (Schmidli et al., 2014)
Global borrowing benchmark using robust MAP prior with mixture of informative and vague components.

## Output Files

### Tables (`results/tables/`)
- `simulation_summary.csv` - Full results table
- `simulation_summary.tex` - LaTeX table for manuscript
- `type_i_error_power.tex` - Type I error and power results

### Plots (`results/plots/`)
- `rmse_vs_n.pdf` - RMSE vs. sample size
- `power_vs_n.pdf` - Power vs. sample size
- `type_i_error.pdf` - Type I error comparison
- `coverage.pdf` - Coverage probability

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
