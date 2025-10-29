# CAHB-PP Simulation Study

This repository implements the simulation study described in **“Covariate-Adjusted Hybrid Bayesian Power Prior (CAHB-PP)”** and reproduces the methodological comparisons from the supporting papers:

- Jin et al. (2023) – CAHB
- Jiang et al. (2018) – KBCD
- Schmidli et al. (2014) – Robust MAP

The code base mirrors the notation in the CAHB-PP manuscript (see `references/CAHB_PP(2).pdf`) and is organised around three core modules:

| Module | Role |
| ------ | ---- |
| `data_generation.py` | Implements the data-generating mechanisms from Sec. 3.1, including sequential outcome simulation for adaptive randomisation. |
| `methods.py` | Contains the four allocation/analysis strategies from Sec. 3.3 with borrowing factors and power-prior discounts matching the cited papers. |
| `analysis` | Processes replicate-level results (Sec. 3.5) and produces publication-ready tables/plots. |

## Simulation Scenarios

The grid defined in `config.py` reproduces Scenarios S1–S4 (Sec. 3.2):

- Concurrent sample sizes: `n ∈ {200, 400}`
- Historical sample sizes: `n_h ∈ {400, 800}`
- Marginal treatment effects: `τ₀ ∈ {0.0, 0.4}`
- Heterogeneity parameters (`Δ₀(x)`, `κ`) follow the scenario definitions in the paper.

Each scenario is replicated `N_REPLICATES = 1000` times by default. The simulation is fully response-adaptive: after the burn-in phase (`N_INIT = 40`) outcomes are generated immediately and fed back into the allocation rules.

## Implemented Methods

- **KBCD** (Jiang et al., 2018): Covariate-adaptive biased coin design without historical borrowing.
- **CAHB** (Jin et al., 2023): Local borrowing via posterior precision `τₙ(x)` and residual variance `φ₀²`.
- **CAHB-PP** (proposed): Extends CAHB with a local power-prior discount `a(x)` informed by compatibility between historical and concurrent controls.
- **rMAP-KBCD** (Schmidli et al., 2014 + KBCD): Applies a robust MAP prior to the control mean while using KBCD-style allocation.

Each implementation follows the notation and equations in Sec. 2–3 of the respective papers; see in-code comments for equation references.

## Running the Study

1. **Install dependencies**
   ```bash
   pip install numpy scipy pandas matplotlib scikit-learn joblib tqdm
   ```
2. **Launch the simulation**
   ```bash
   python main.py
   ```
   The joblib configuration can be adjusted via `config.N_JOBS`.
3. **Inspect results**
   - Aggregated metrics: `results/tables/simulation_summary.csv`
   - LaTeX table: `results/tables/simulation_summary.tex`
   - Plots (RMSE, power, allocation rates): `results/plots/*.pdf`

All artefacts are ignored by Git through the provided `.gitignore`.

## Reproducing or Extending

- Modify `config.py` to change scenario grids, replicate counts, or prior hyperparameters.
- Extend `methods.py` by subclassing `BaseMethod` and adding the class name to `config.METHODS_TO_RUN`.
- Additional evaluation metrics or visualisations can be added in `analysis`.
