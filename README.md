# RADISH: Robust Adaptive Discrepancy-Informed Shrinkage for Historical Borrowing

A variance-aware covariate-adaptive randomisation design that dynamically borrows
from external historical controls while guarding against prior-data conflict.

## Quick Start

```bash
pip install numpy scipy pandas joblib
python main.py                          # demo (10 reps, ~2 min)
python main.py --mode full --jobs 4     # full (500 reps)
```

Results → `results/{demo,full}/{raw_results.csv, metrics.csv, plots/, tables/}`

## Simulation Design (3×2 factorial)

|           | High precision (σ_H = 0.5) | Low precision (σ_H = 2.0) |
|-----------|---------------------------|--------------------------|
| **No bias** (b=0)   | B1: ideal borrowing       | B2: noisy but unbiased   |
| **Mod bias** (b=0.5)| B3: CAHB fails, RADISH ✓  | B4: CAHB fails, RADISH ✓ |
| **Large bias** (b=2) | B5: all detect conflict   | B6: all detect conflict  |

## Methods

| Method | Borrowing | Key mechanism |
|--------|-----------|---------------|
| **RADISH** | PDC-adaptive | D = −log(p_n) → exp(−κD) discount |
| CAHB | Variance-ratio | τ(x) coordinate ascent |
| KBCD | None | Baseline (R_n ≡ 1) |

## Files

```
├── config.py      # Parameters + 6 scenarios
├── methods.py     # KBCD, CAHB, RADISH + linearized ATE
├── main.py        # Simulation runner (parallel, subgroup tracking)
├── analysis.py    # Diagnostic plots
└── README.md
```

## License

MIT
