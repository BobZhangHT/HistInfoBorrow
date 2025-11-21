"""
single_test.py

Run a single replication for a chosen scenario across multiple methods.
All methods see the same pre-generated covariate sequence and potential
outcomes. CAHB-UIP allocation uses the variance-aware coordinate-ascent
updates implemented in methods.py, while the final inference still uses
the Gibbs sampler inside each method's `estimate_treatment_effect`.
"""

import argparse
import time
from typing import Dict, List

import numpy as np

import config
import data_generation
import methods
from main import run_single_simulation  # retained for reference if needed


def _build_shared_sequences(scenario: Dict, seed: int):
    """Generate shared historical data and potential outcomes."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    hist_data = data_generation.generate_historical_data(
        scenario["n_h"], scenario
    )

    n_total = scenario["n"]
    covariates = data_generation.generate_covariates(n_total)
    mu0, _, mu1 = data_generation.get_true_means(covariates, scenario["tau_0"])
    sigma0_sq, sigma1_sq, _ = data_generation.get_true_variances(
        covariates, scenario["kappa"]
    )

    epsilon0 = rng.normal(size=n_total)
    epsilon1 = rng.normal(size=n_total)
    potential_y0 = mu0 + np.sqrt(sigma0_sq) * epsilon0
    potential_y1 = mu1 + np.sqrt(sigma1_sq) * epsilon1
    return hist_data, covariates, potential_y0, potential_y1


def _simulate_single_method(
    method_name: str,
    scenario: Dict,
    hist_data: Dict,
    covariates: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    seed: int,
):
    """Run one adaptive trial using pre-generated sequences."""
    rng = np.random.default_rng(seed)
    MethodClass = getattr(methods, method_name)
    hist_copy = {
        "X_h": np.array(hist_data["X_h"], copy=True),
        "Y_h": np.array(hist_data["Y_h"], copy=True),
    }
    method = MethodClass(hist_copy, scenario, config.PRIORS)

    n_total = scenario["n"]
    burn = config.N_INIT

    X_curr = covariates[:burn].copy()
    Z_curr = rng.binomial(1, config.ALLOC_BURN_IN, burn)
    Y_curr = np.where(Z_curr == 1, y1[:burn], y0[:burn])

    for idx in range(burn, n_total):
        x_new = covariates[idx].reshape(1, -1)
        alloc = method.get_allocation_prob(X_curr, Y_curr, Z_curr, x_new)
        pi = float(np.clip(alloc.pi_treatment, 0.0, 1.0))
        z_new = rng.binomial(1, pi)
        y_new = y1[idx] if z_new else y0[idx]
        X_curr = np.vstack([X_curr, x_new])
        Y_curr = np.append(Y_curr, y_new)
        Z_curr = np.append(Z_curr, z_new)

    delta_hat, ci_low, ci_high, prob = method.estimate_treatment_effect(
        X_curr, Y_curr, Z_curr
    )
    return {
        "delta_hat": float(delta_hat),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "prob_gt_0": float(prob),
        "n_treated": int(Z_curr.sum()),
        "n_control": int((1 - Z_curr).sum()),
    }


def run_single_scenario(
    scenario_id: int,
    replicate_id: int,
    methods_to_test: List[str],
    seed: int,
):
    scenarios = config.get_scenario_definitions()
    scenario = scenarios[scenario_id]
    print("=" * 80)
    print("SINGLE REPLICATION TEST".center(80))
    print("=" * 80)
    print(f"Scenario: {scenario['name']}")
    print(f"  id={scenario['id']}, type={scenario.get('scenario_type','NA')}")
    print(f"  n={scenario['n']}, n_h={scenario['n_h']}, tau_0={scenario['tau_0']}, "
          f"kappa={scenario.get('kappa', 1.0)}")
    print(f"Replicate: {replicate_id}")
    print(f"Methods: {', '.join(methods_to_test)}\n")

    hist_data, covariates, y0, y1 = _build_shared_sequences(scenario, seed)
    print("Step 1: Generated historical data.")
    print("Step 2: Generated trial covariate & potential outcome sequences.\n")

    print("Step 3: Running methods...\n")
    for method_name in methods_to_test:
        start = time.perf_counter()
        try:
            results = _simulate_single_method(
                method_name,
                scenario,
                hist_data,
                covariates,
                y0,
                y1,
                seed + hash(method_name) % 10000,
            )
            elapsed = time.perf_counter() - start
            print(f"Testing {method_name}...")
            ci_str = f"[{results['ci_low']:.4f}, {results['ci_high']:.4f}]"
            print(
                f"  √ {method_name}: delta={results['delta_hat']:.4f}, "
                f"CI={ci_str}, time={elapsed:.2f}s, "
                f"n1={results['n_treated']}, n0={results['n_control']}"
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"  x {method_name} failed after {elapsed:.2f}s: {exc}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-replication comparison")
    parser.add_argument("--scenario", type=int, default=0, help="Scenario id")
    parser.add_argument("--replicate", type=int, default=0, help="Replicate id")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=config.METHODS_TO_RUN,
        help="Subset of methods to test",
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    args = parser.parse_args()

    run_single_scenario(args.scenario, args.replicate, args.methods, args.seed)



