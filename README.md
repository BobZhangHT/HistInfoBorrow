# RADISH: Robust Adaptive Discrepancy-Informed Shrinkage for Historical Borrowing in Clinical Trials

Reference implementation and reproduction code for the manuscript

> **RADISH: Robust Adaptive Discrepancy-Informed Shrinkage for Historical
> Borrowing in Clinical Trials.**
> Hengtao Zhang, Yuanke Qu, and Huaqing Jin.

RADISH is a covariate-adaptive randomisation design that borrows from
external historical controls through a local prior-data-conflict (PDC)
discount. Borrowing is information-ratio driven and degrades gracefully
to a non-borrowing baseline when the historical and concurrent cohorts
disagree. The repository reproduces every figure and table in the paper:
the 3×2 bias-by-precision Monte Carlo factorial (§4) and the
HORIZON–FIT real-data application (§5).

---

## 1. Installation

```bash
git clone https://github.com/BobZhangHT/HistInfoBorrow.git
cd HistInfoBorrow
pip install -r requirements.txt
```

Requirements: Python ≥ 3.9, `numpy`, `scipy`, `pandas`, `joblib`, `matplotlib`.

### Optional C backend (≈ 5–8× faster)

```bash
python build_c.py        # MSVC on Windows; gcc/clang on Linux/macOS
```

Produces `radish_core.{dll,so}`. `main.py` auto-detects the library and
falls back to the pure-Python implementation if it is absent. Set
`RADISH_BACKEND=python` to force the Python path.

---

## 2. Reproducing the simulation study (§4)

The simulation uses a 3 (bias) × 2 (precision) factorial in
`config.py::SCENARIOS`:

|              | High precision (σ_H = 0.5) | Low precision (σ_H = 3.0) |
|:------------:|:--------------------------:|:-------------------------:|
| No bias (b=0)    | B1                     | B2                    |
| Moderate (b=0.5) | B3                     | B4                    |
| Large (b=2)      | B5                     | B6                    |

```bash
# Quick smoke test (10 reps, ≈ 2 min on 4 cores)
python main.py --mode demo --jobs 4

# Full paper grid (1000 reps × 6 scenarios × 2 effects × 3 methods)
python main.py --mode full      --jobs 8

# Supplementary precision-gradient sweep (Fig. P1–P4)
python main.py --mode precision --jobs 8

# Supplementary bias-gradient sweep (Theorem 1 verification)
python main.py --mode bias      --jobs 8
```

Outputs land under `results/<mode>/`:

```
results/<mode>/
├── raw_results.csv      # per-trial draw, with allocation + diagnostics
├── metrics.csv          # bias / RMSE / coverage / power, per cell
├── tables/              # publication LaTeX + CSV
└── plots/               # six paper figures (vector PDF)
```

Reproducing the paper's wall-clock numbers requires `--mode full
--jobs N`; running times scale roughly linearly in `--jobs`.

---

## 3. Reproducing the real-data application (§5)

The real-data application is anchored to two non-redistributable
trials:

- **HORIZON Pivotal Fracture Trial** (zoledronic acid; current trial)
- **Fracture Intervention Trial — FIT** vertebral and clinical
  sub-cohorts (historical controls)

Both must be obtained from the original investigators or NHLBI BioLINCC
under their respective data-use agreements. **The data are not
included in this repository.** Once obtained:

1. Merge HORIZON and FIT into a single subject-level table named
   `real_data/dat_merge.csv` with the columns expected by
   `real_data/real_setup.py` (see the docstring there for the schema).
2. Run the pipeline:

```bash
python real_data/real_setup.py          # fit working model, dump real_params.pkl
python real_data/real_run.py            # (ξ × η × method × effect × rep) sweep
python real_data/real_figures.py        # F1–F3 + Table 1
```

The ξ (bias) × η (precision/sample-size) factorial mirrors the
synthetic design but with HORIZON-fitted subgroup coefficients and
FIT-KDE covariate distributions; see `real_data/real_scenarios.py`.

---

## 4. Methods

| Method        | File / class                | Borrowing rule                     |
|---------------|-----------------------------|------------------------------------|
| **RADISH**    | `methods.py::RADISH`        | PDC-adaptive: `R_n = 1 + Nh/(τ_H² · Nc · exp(D_PDC))` |
| CAHB          | `methods.py::CAHB`          | Variance-ratio with τ(x) coordinate ascent |
| KBCD          | `methods.py::KBCD`          | No borrowing (`R_n ≡ 1`) — baseline |

All three share the linearised ATE estimator with a sandwich variance
defined in §3 of the paper. C-backed counterparts live in
`methods_c.py` and load `radish_core.{dll,so}` via `ctypes`.

---

## 5. Repository layout

```
.
├── config.py             # Hyper-parameters + scenario grids
├── methods.py            # KBCD, CAHB, RADISH (pure Python reference)
├── methods_c.py          # ctypes bindings to radish_core
├── c_src/radish_core.c   # C kernel (Stage I / II hot loops)
├── build_c.py            # Compiler-agnostic shared-library build
├── main.py               # Monte Carlo simulation runner
├── analysis.py           # Publication figures + LaTeX tables
├── real_data/            # Real-data application code (§5)
│   ├── real_setup.py     #   fit working model on HORIZON / FIT
│   ├── real_scenarios.py #   ξ × η factorial definition
│   ├── real_run.py       #   simulation sweep on real-anchored DGP
│   └── real_figures.py   #   F1–F3 + Table 1
├── requirements.txt
├── LICENSE
└── README.md
```

Generated artefacts (`results/`, `real_data/*.csv`, `real_data/*.pkl`,
`real_data/figures/`, build outputs) are ignored by `.gitignore`.

---

## 6. Citation

```bibtex
@unpublished{zhang2026radish,
  title  = {RADISH: Robust Adaptive Discrepancy-Informed Shrinkage for
            Historical Borrowing in Clinical Trials},
  author = {Zhang, Hengtao and Qu, Yuanke and Jin, Huaqing},
  year   = {2026},
  note   = {Manuscript under review.}
}
```

Correspondence: Huaqing Jin (`huaqingjin@mail.tsinghua.edu.cn`).

---

## 7. License

Code is released under the [MIT License](LICENSE). The HORIZON and FIT
datasets are governed by their respective data-use agreements and are
not covered by this license.
