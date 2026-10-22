# RADISH

Reference implementation for **Robust Adaptive Discrepancy-Informed Shrinkage for Historical Borrowing in Clinical Trials** by Hengtao Zhang, Yuanke Qu, and Huaqing Jin.

RADISH is a covariate-adaptive randomized design that borrows information from external controls only where historical and concurrent controls are locally compatible. A single Gaussian product kernel is used throughout planning, sequential allocation, and final estimation. Sequential allocation uses the centered excess-surprisal score $D=(-\log\kappa-1)_+$. Final analysis combines the local score with a locally gated marginal safeguard, so a trial-wide discrepancy strengthens discounting only where local conflict is already present. The reported standard error also uses a concurrent-only variance floor to protect finite-sample coverage when terminal weights are estimated from the outcomes. As conflict increases, the design moves toward the concurrent-only KBCD baseline.

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

The manuscript's primary experiment is a compact four-by-two conflict-and-precision design. For `X1 in {0,1}`, the stratum-specific historical shifts are `(b(0),b(1))`. Every nonzero-conflict setting differs from the concurrent response surface in both strata, with a larger shift in the `X1=1` stratum. Historical outcome standard deviation is `sigma_H in {0.5,3.0}`; concurrent and historical sample sizes are both 200.

| Scenario | `(b(0),b(1))` | Historical SD | Interpretation |
|:--|:--:|--:|:--|
| S1 | `(0,0)` | 0.5 | Compatible, precise history |
| S2 | `(0,0)` | 3.0 | Compatible, noisy history |
| S3 | `(0.04,0.12)` | 0.5 | Small local conflict, precise history |
| S4 | `(0.04,0.12)` | 3.0 | Small local conflict, noisy history |
| S5 | `(0.25,0.75)` | 0.5 | Moderate local conflict, precise history |
| S6 | `(0.25,0.75)` | 3.0 | Moderate local conflict, noisy history |
| S7 | `(1.5,2.5)` | 0.5 | Severe local conflict, precise history |
| S8 | `(1.5,2.5)` | 3.0 | Severe local conflict, noisy history |

Run the paper experiments from the repository root:

```bash
# Fast end-to-end check using the paper's eight-scenario design
python main.py --mode six --reps 2 --jobs 4
python main.py --mode six --reps 2 --validate-only

# Main eight-scenario experiment: 48,000 fits
python main.py --mode six --jobs 8

# Nine-point local-conflict path used for the local-borrowing figure: 54,000 fits
python main.py --mode local_path --jobs 8

# Weak/strong marginal shift crossed with weak/strong local heterogeneity: 24,000 fits
python main.py --mode global_local --jobs 8

# Validate existing formal outputs without rerunning simulations
python main.py --mode six --validate-only
python main.py --mode local_path --validate-only
python main.py --mode global_local --validate-only
```

The `local_path` mode uses `b(1) in {0,0.001,0.003,0.01,0.03,0.10,0.30,1,3}` with `b(0)=b(1)/3`. Its positive levels are spaced by approximately half a decade, resolving both behavior near compatibility and the return toward no borrowing. The `global_local` mode uses `b(X1)=g+ell*s(X1)`, where `E{s(X1)}=0`, and crosses weak/strong values of `g` and `ell`. Thus every diagnostic cell remains covariate dependent while marginal and local conflict vary separately.

Additional modes retain broader sensitivity analyses. `full` uses the primary eight-cell grid; `precision` and `bias` provide focused parameter sweeps. The `hist_size` workflow independently generates nested `N_H=200` and `N_H=400` histories while sharing separate current-covariate, allocation-uniform, and outcome-error streams within each paired replication. Its pairing contract is written to `results/hist_size/hist_size_pairing_manifest.json`.

All modes use seed 2026 by default, and `--seed` changes it consistently. `--reps` supports inexpensive checks. Outputs are written under `results/<mode>/` as `raw_results.csv`, `metrics.csv`, `plots/`, and `tables/`; generated results are excluded from Git.

## HORIZON-FIT semi-synthetic study

The semi-synthetic study uses restricted HORIZON and FIT participant-level data, which are not distributed in this repository. After obtaining access under the applicable data-use agreements, place the merged file at `real_data/dat_merge.csv`; the required schema is documented in `real_data/real_setup.py`. The scripts use the empirical covariate geometry and a HORIZON-calibrated response surface rather than reporting a new treatment-effect analysis. The primary experiment fixes `N=200`, a 20-participant burn-in, and `N_H=350`, then independently crosses weak/strong target-population mean shift with weak/strong `X1`-dependent heterogeneity. Three methods, two effect settings, and 1000 paired replications produce 24,000 fits. A complementary seven-point localized-conflict path uses `(b(0),b(1))/sigma_C=4*xi*(-1,(1-p1)/p1)` for `xi in {0,0.003162,0.01,0.031623,0.1,0.316228,1}`, where `p1=P_C(X1=1)`. The positive levels are evenly spaced on the log scale; the two shifts have opposite signs and average to zero, so every positive conflict is covariate dependent and invisible to a marginal mean comparison. The same path at `N_H=30` supplies the historical-size sensitivity analysis.

```bash
python real_data/real_setup.py
python real_data/real_run.py --design global-local --reps 1000 --jobs 6
python real_data/real_run.py --design global-local --validate-only
python real_data/real_figures.py --design global-local

# Seven-point conflict-by-history-size sensitivity analysis
python real_data/real_run.py --design xi --reps 1000 --jobs 6
python real_data/real_run.py --design xi --validate-only
python real_data/real_figures.py --design xi
```

The primary shift is `b(X1)/sigma_C = g + ell*(X1-p1)/sqrt(p1*(1-p1))`, so `g` is the target-population mean shift and `ell` is the across-stratum standard deviation. Every primary cell is covariate dependent. The integrated validators check the grids, replication counts, identifiers, finite outputs, and write SHA-256 manifests before figures are generated.

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
