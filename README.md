# RADISH

Reference implementation for **Robust Adaptive Discrepancy-Informed Shrinkage for Historical Borrowing in Clinical Trials** by Hengtao Zhang, Yuanke Qu, and Huaqing Jin.

RADISH is a covariate-adaptive randomized design that borrows information from external controls only where historical and concurrent controls are locally compatible. A single Gaussian product kernel is used throughout planning, sequential allocation, and final estimation. Sequential allocation uses the centered excess-surprisal score $D=(-\log\kappa-1)_+$. Final analysis combines the local score with a locally gated marginal safeguard, so a trial-wide discrepancy strengthens discounting only where local conflict is already present. As conflict increases, the design moves toward the concurrent-only KBCD baseline.

## Installation

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/BobZhangHT/HistInfoBorrow.git
cd HistInfoBorrow
python -m pip install -r requirements.txt
```

The pure-Python implementation works without compilation. An optional C backend accelerates the main simulation loops:

```bash
python build_c.py
```

On Windows with MinGW, use `python build_c.py --mingw`. Add `--self-test` to run the centered Stage-II and Stage-III Python/C parity checks immediately after compilation. The build creates `radish_core.dll`, `radish_core.so`, or `radish_core.dylib` beside `methods_c.py`. `main.py` detects the library automatically and otherwise uses `methods.py`. Set `RADISH_BACKEND=python` to force the reference implementation or `RADISH_BACKEND=c` to require the compiled backend.

## Synthetic simulation study

The manuscript's primary experiment is a compact three-by-two conflict-and-precision design. For `X1 in {0,1}`, the historical shift is `b(X1) = (b0,b1)`. Every nonzero-conflict setting differs from the concurrent response surface in both strata, with a larger shift in the `X1=1` stratum. Historical outcome standard deviation is `sigma_H in {0.5,3.0}`; concurrent and historical sample sizes are both 200.

| Scenario | `(b0,b1)` | Historical SD | Interpretation |
|:--|:--:|--:|:--|
| S1 | `(0,0)` | 0.5 | Compatible, precise history |
| S2 | `(0,0)` | 3.0 | Compatible, noisy history |
| S3 | `(0.25,0.75)` | 0.5 | Moderate local conflict, precise history |
| S4 | `(0.25,0.75)` | 3.0 | Moderate local conflict, noisy history |
| S5 | `(1.5,2.5)` | 0.5 | Severe local conflict, precise history |
| S6 | `(1.5,2.5)` | 3.0 | Severe local conflict, noisy history |

Run the paper experiments from the repository root:

```bash
# Fast end-to-end check using the paper's six-scenario design
python main.py --mode six --reps 2 --jobs 4
python main.py --mode six --reps 2 --validate-only

# Main six-scenario experiment: 36,000 fits
python main.py --mode six --jobs 8

# Five-point local-conflict path used for the local-borrowing figure: 30,000 fits
python main.py --mode local_path --jobs 8

# Weak/strong global conflict crossed with weak/strong local heterogeneity: 24,000 fits
python main.py --mode global_local --jobs 8

# Validate existing formal outputs without rerunning simulations
python main.py --mode six --validate-only
python main.py --mode local_path --validate-only
python main.py --mode global_local --validate-only
```

The `local_path` mode uses five precise-history shifts, `(0,0)`, `(0.10,0.30)`, `(0.25,0.75)`, `(0.50,1.50)`, and `(1,3)`, to display how borrowing changes across covariate strata. The `global_local` mode uses `b(X1)=g+ell*s(X1)`, where `E{s(X1)}=0`, and crosses weak/strong values of `g` and `ell`. Thus every diagnostic cell remains covariate dependent while marginal and local conflict vary separately.

Additional modes retain broader sensitivity analyses. `full` is the earlier four-by-two local-conflict grid and is the parent run for the paired `hist_size` analysis; `precision` and `bias` provide focused parameter sweeps. The historical-size workflow requires a validated default-seed `results/full/raw_results.csv`, reuses its `N_H=200` rows, and computes the paired `N_H=400` extension. Its pairing contract is written to `results/hist_size/hist_size_pairing_manifest.json`.

All modes use seed 2026 by default. `--seed` changes it except for the optimized `hist_size` workflow, whose stored parent result has no seed column. `--reps` supports inexpensive checks. Outputs are written under `results/<mode>/` as `raw_results.csv`, `metrics.csv`, `plots/`, and `tables/`; generated results are excluded from Git.

## HORIZON-FIT semi-synthetic study

The semi-synthetic study uses restricted HORIZON and FIT participant-level data, which are not distributed in this repository. After obtaining access under the applicable data-use agreements, place the merged file at `real_data/dat_merge.csv`; the required schema is documented in `real_data/real_setup.py`. The scripts use the empirical covariate geometry and a HORIZON-calibrated response surface rather than reporting a new treatment-effect analysis. The formal grid contains `N=200`, a 20-participant burn-in, historical sizes `{30,350}`, conflict levels `{0,.16,.32,.64,1}`, three methods, two effect settings, and 1000 replications per cell.

```bash
python real_data/real_setup.py
python real_data/real_run.py --reps 1000 --jobs 6
python real_data/real_run.py --validate-only
python real_data/real_figures.py
```

The experiment contains 60,000 fits. Every nonzero historical shift depends on `X1`, with stronger conflict in the `X1=1` stratum while preserving the prespecified marginal severity scale. The integrated validator checks the grid, replication counts, identifiers, finite outputs, and writes a SHA-256 manifest before figures are generated.

## Methods and implementation

| Method | Implementation | Historical borrowing |
|:--|:--|:--|
| RADISH | `methods.py::RADISH` | Local conflict discount with a gated marginal safeguard |
| CAHB | `methods.py::CAHB` | Variance-ratio borrowing |
| KBCD | `methods.py::KBCD` | Concurrent-only baseline |

`methods.py` is the reference implementation. `methods_c.py` exposes matching C-backed classes through `ctypes`, and `c_src/radish_core.c` contains the native kernels. The production Python and C paths both use centered Stage-II allocation by default; the legacy uncentered map remains available only for explicit historical-reproducibility checks. `analysis.py` and `real_data/real_figures.py` generate the publication figures.

## Repository layout

```text
.
|-- main.py                         synthetic simulation entry point
|-- config.py                       scenarios and runtime settings
|-- methods.py                      pure-Python implementations
|-- methods_c.py                    optional C-backed implementations
|-- analysis.py                     synthetic tables and figures
|-- historical_size_analysis.py    historical-size sensitivity analysis
|-- build_c.py                      native-library build helper
|-- c_src/radish_core.c             C kernels
|-- real_data/                      restricted-data reproduction scripts
|-- requirements.txt
`-- LICENSE
```

Generated outputs, native binaries, restricted data, manuscripts, and the local `archive/` directory are excluded by `.gitignore`.

## Reproducibility checks

```bash
python -B -c "import config, methods, analysis, historical_size_analysis; import real_data.real_scenarios, real_data.real_run, real_data.real_figures"
python build_c.py --self-test
python main.py --mode six --reps 2 --jobs 4
python main.py --mode six --reps 2 --validate-only
python real_data/real_run.py --validate-only
```

- Command-line runners fix the default seed.
- Joblib uses process-level parallelism and each worker restricts BLAS libraries to one thread.
- The C backend is optional; use `RADISH_BACKEND=python` to validate the reference implementation.
- The manuscript's formal numerical claims are based only on result files that pass the corresponding validator.

## Citation

```bibtex
@unpublished{zhang2026radish,
  title  = {RADISH: Robust Adaptive Discrepancy-Informed Shrinkage for
            Historical Borrowing in Clinical Trials},
  author = {Zhang, Hengtao and Qu, Yuanke and Jin, Huaqing},
  year   = {2026},
  note   = {Manuscript under review}
}
```

## License

The source code is released under the [MIT License](LICENSE). HORIZON and FIT data remain governed by their original data-use agreements and are not covered by this license.
