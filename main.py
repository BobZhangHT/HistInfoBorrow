import argparse
"""
main.py

Main Orchestration Script for CAHB-UIP Simulation Study

This script coordinates the complete simulation workflow:
    1. Configuration loading and validation
    2. Parallel execution of simulation replicates
    3. Checkpoint/cache management for resumability
    4. Progress tracking with detailed status updates
    5. Results aggregation and analysis

Key Features:
-------------
- **Parallel Computation**: Uses joblib for multi-core execution
- **Memory Management**: Batch processing to prevent memory overflow
- **Checkpointing**: Automatic caching of completed replicates
- **Resumability**: Continues from last checkpoint after interruption
- **Progress Tracking**: Real-time progress bars via tqdm

Simulation Workflow:
--------------------
For each (scenario, method, replicate) combination:
    1. Generate historical data
    2. Initialize method with historical data
    3. Run adaptive trial:
        a. Burn-in phase: balanced 1:1 randomization
        b. Adaptive phase: method-specific allocation probabilities
    4. Final analysis: estimate treatment effect and inference
    5. Cache results

Usage:
------
    python main.py

Configuration:
--------------
Modify config.py to change:
    - Number of replicates (N_REPLICATES)
    - Parallel workers (N_JOBS)
    - Caching behavior (USE_CACHE)
    - Memory limits (MAX_MEMORY_PER_JOB)

Output:
-------
- results/tables/*.csv, *.tex: Performance metric tables
- results/plots/*.pdf: Publication-quality figures
- .simulation_cache/: Cached simulation results (for resumability)

References:
    CAHB-UIP manuscript Section 3 (Simulation Study Design)
"""

import argparse
import os
import shutil
import sys
import gc
import warnings
from contextlib import contextmanager
from typing import Dict, List, Optional

import numpy as np
import joblib
from joblib import parallel
from tqdm import tqdm

# =============================================================================
# Project Module Imports
# =============================================================================

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import config
    import data_generation
    import methods
    import analysis
except ImportError as e:
    print(f"ERROR: Could not import project modules: {e}")
    print("Ensure all .py files are in the same directory.")
    sys.exit(1)

AllocationResult = getattr(methods, "AllocationResult", None)


# =============================================================================
# Helpers
# =============================================================================

@contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to patch joblib to report into a tqdm progress bar.
    """
    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


# =============================================================================
# Single Simulation Function
# =============================================================================

def run_single_simulation(scenario: Dict, method_name: str, replicate_id: int) -> Optional[Dict]:
    """
    Executes a single simulation replicate for one scenario and method.
    
    This function is the core computation unit that gets parallelized and cached.
    It simulates a complete adaptive randomized trial from start to finish.
    
    Trial Simulation Process:
    --------------------------
    1. Set random seed for reproducibility
    2. Generate historical data (control-only)
    3. Instantiate the randomization method
    4. Burn-in phase: enroll n_init subjects with balanced allocation
    5. Adaptive phase: enroll remaining subjects with method-specific allocation
        - For each new subject:
            a. Generate covariates
            b. Compute allocation probability
            c. Randomize treatment assignment
            d. Generate outcome
            e. Update cumulative dataset
    6. Final analysis: estimate treatment effect and compute inference
    7. Return performance metrics
    
    Args:
        scenario: Dictionary with scenario parameters:
            - id: Scenario identifier
            - name: Human-readable name
            - n: Current trial sample size
            - n_h: Historical trial sample size
            - tau_0: Base treatment effect
            - kappa: Variance inflation factor
            - Delta_0_func: Historical bias function
        method_name: Name of the method class (must exist in methods.py)
        replicate_id: Replicate number (0 to N_REPLICATES-1)
    
    Returns:
        Dictionary containing:
            - scenario_id, scenario_name, method, replicate_id
            - n, n_h, tau_0: Scenario parameters
            - delta_hat: Treatment effect estimate
            - ci_low, ci_high: 95% CI bounds
            - prob_gt_0: P(delta > 0 | Data)
            - n_treated: Number assigned to treatment
            - n_total: Total sample size
        
        Returns None if simulation fails (caught exceptions)
    
    Notes:
        - Uses deterministic seeding for reproducibility
        - Seed depends on (scenario_id, method_index, replicate_id)
        - Failures are logged but do not crash the entire study
        - Memory is explicitly released after completion
    """
    
    # ==== 1. Set Random Seed for Reproducibility ====
    
    # Construct unique seed from simulation parameters
    # This ensures different scenarios/methods/replicates get different seeds
    # but the same combination always gets the same seed
    try:
        method_idx = config.METHODS_TO_RUN.index(method_name)
    except ValueError:
        method_idx = 0
    
    seed = (scenario['id'] * 100000) + (method_idx * 10000) + replicate_id
    np.random.seed(seed)
    
    # ==== 2. Generate Historical Data ====
    
    try:
        hist_data = data_generation.generate_historical_data(
            n_h=scenario['n_h'],
            scenario_params=scenario
        )
    except Exception as e:
        warnings.warn(f"Failed to generate historical data for {scenario['name']}, "
                     f"method {method_name}, rep {replicate_id}: {e}")
        return None
    
    # ==== 3. Instantiate Method ====
    
    try:
        # Get method class from methods module
        MethodClass = getattr(methods, method_name)
        method_instance = MethodClass(
            historical_data=hist_data,
            scenario_params=scenario,
            priors=config.PRIORS
        )
    except AttributeError:
        warnings.warn(f"Method {method_name} not found in methods module.")
        return None
    except Exception as e:
        warnings.warn(f"Failed to instantiate method {method_name}: {e}")
        return None

    # ==== 4. Trial Simulation ====
    
    # 4a. Burn-in Phase: Balanced Allocation
    # ---------------------------------------
    # First n_init subjects receive balanced 1:1 randomization
    # This provides initial data for adaptive methods to "learn" from
    
    X_curr = data_generation.generate_covariates(config.N_INIT)
    Z_curr = np.random.binomial(1, config.ALLOC_BURN_IN, config.N_INIT)
    
    # Generate burn-in outcomes
    Y_curr = np.array([
        data_generation.generate_single_outcome(
            X_new=X_curr[i].reshape(1, -1),
            z_new=Z_curr[i],
            tau_0=scenario['tau_0'],
            kappa=scenario['kappa']
        )
        for i in range(config.N_INIT)
    ])
    
    # 4b. Adaptive Allocation Phase
    # ------------------------------
    # Remaining subjects receive adaptive allocation based on method
    
    n_adaptive = scenario['n'] - config.N_INIT
    monitor_start = getattr(config, "SEQ_MONITOR_START", config.N_INIT)
    monitor_step = max(1, int(getattr(config, "SEQ_MONITOR_STEP", 20)))
    allocation_path = []
    
    for _ in range(n_adaptive):
        # Generate new subject's covariates
        X_new = data_generation.generate_covariates(n=1)
        
        # Get allocation probability from the method
        # This is where methods differ (CAHB-UIP variants vs CAHB vs KBCD)
        try:
            alloc_output = method_instance.get_allocation_prob(
                X_curr=X_curr,
                Y_curr=Y_curr,
                Z_curr=Z_curr,
                X_new=X_new
            )
            diag = {}
            if AllocationResult is not None and isinstance(alloc_output, AllocationResult):
                pi_1 = float(np.clip(alloc_output.pi_treatment, 0.0, 1.0))
                diag = dict(getattr(alloc_output, "diagnostics", {}) or {})
            else:
                pi_1 = float(np.clip(float(alloc_output), 0.0, 1.0))
        except Exception as e:
            # If allocation fails, default to balanced
            warnings.warn(f"Allocation failed for {method_name}, using balanced: {e}")
            pi_1 = 0.5
            diag = {}
        
        # Guard diagnostics
        if not np.isfinite(pi_1):
            pi_1 = 0.5
        diag_Rn = float(diag.get("R_n", 1.0)) if isinstance(diag, dict) else 1.0
        diag_discount = np.nan
        if isinstance(diag, dict):
            if "M" in diag and np.isfinite(diag["M"]):
                diag_discount = float(diag["M"])
            elif "mean" in diag and np.isfinite(diag["mean"]):
                diag_discount = float(diag["mean"])
        max_rn_allowed = float(getattr(config, "MAX_ESS", 1e6))
        if not np.isfinite(diag_Rn):
            diag_Rn = 1.0
        diag_Rn = float(np.clip(diag_Rn, 0.0, max_rn_allowed))
        
        # Randomize treatment assignment
        Z_new = np.random.binomial(1, pi_1)
        
        # Generate outcome
        Y_new = data_generation.generate_single_outcome(
            X_new=X_new,
            z_new=Z_new,
            tau_0=scenario['tau_0'],
            kappa=scenario['kappa']
        )
        
        # Update cumulative dataset
        X_curr = np.vstack([X_curr, X_new])
        Z_curr = np.append(Z_curr, Z_new)
        Y_curr = np.append(Y_curr, Y_new)
        
        total_enrolled = len(Z_curr)
        if total_enrolled >= monitor_start and ((total_enrolled - monitor_start) % monitor_step == 0):
            prop_treated = float(np.sum(Z_curr) / total_enrolled)
            x_vals = X_new.reshape(-1)
            allocation_path.append({
                "sample_size": int(total_enrolled),
                "prop_treated": prop_treated,
                "R_n": diag_Rn,
                "M": diag_discount,
                "x1": float(x_vals[0]),
                "x2": float(x_vals[1]),
                "x3": float(x_vals[2]),
                "x4": float(x_vals[3]),
            })
    
    # ==== 5. Final Analysis ====
    
    try:
        delta_hat, ci_low, ci_high, prob_gt_0 = method_instance.estimate_treatment_effect(
            X=X_curr,
            Y=Y_curr,
            Z=Z_curr
        )
    except Exception as e:
        # Estimation can fail (e.g., singular matrices with small samples)
        warnings.warn(f"Estimation failed for {scenario['name']}, method {method_name}, "
                     f"rep {replicate_id}: {e}")
        delta_hat, ci_low, ci_high, prob_gt_0 = np.nan, np.nan, np.nan, np.nan

    try:
        calibration_samples = method_instance.get_calibration_payload()
        if calibration_samples is None:
            calibration_samples = []
    except Exception:
        calibration_samples = []
    
    # ==== 6. Compile Results ====
    
    result = {
        # Identifiers
        'scenario_id': scenario['id'],
        'scenario_name': scenario['name'],
        'scenario_type': scenario.get('scenario_type', 'Unknown'),
        'method': method_name,
        'replicate_id': replicate_id,
        
        # Scenario parameters
        'n': scenario['n'],
        'n_h': scenario['n_h'],
        'tau_0': scenario['tau_0'],
        'kappa': scenario.get('kappa', np.nan),
        
        # Estimates
        'delta_hat': delta_hat,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'prob_gt_0': prob_gt_0,
        
        # Allocation summary
        'n_total': scenario['n'],
        'n_treated': int(np.sum(Z_curr)),
        'n_control': int(np.sum(1 - Z_curr)),
        'allocation_path': allocation_path,
        'calibration_samples': calibration_samples,
    }
    
    # ==== 7. Memory Cleanup ====
    
    # Explicitly delete large objects to free memory
    del X_curr, Y_curr, Z_curr, method_instance, hist_data
    gc.collect()
    
    return result


# =============================================================================
# Main Orchestration Function
# =============================================================================

def main(demo_override: Optional[bool] = None, clear_cache: bool = False):
    """
    Main entry point for the simulation study.
    
    Coordinates the complete workflow:
        1. Validate configuration
        2. Setup directories
        3. Initialize caching
        4. Create task list
        5. Execute tasks in parallel with progress tracking
        6. Process and save results
        7. Generate tables and plots
    
    Progress Tracking:
        - Real-time progress bar showing completed/total tasks
        - Estimated time remaining
        - Current task details
    
    Checkpointing:
        - Completed tasks cached to disk
        - Interrupted runs can resume from last checkpoint
        - Cache location: .simulation_cache/
    
    Error Handling:
        - Individual task failures logged but don't stop study
        - Final check: warn if too many failures
        - Continue with valid results even if some tasks fail
    """
    
    # Determine run mode (full vs fast demo)
    demo_mode = config.FAST_DEMO if demo_override is None else bool(demo_override)
    config.FAST_DEMO = demo_mode
    config.N_REPLICATES = config.FAST_DEMO_REPLICATES if demo_mode else config.FULL_RUN_REPLICATES

    if clear_cache and os.path.isdir(config.CACHE_DIR):
        print(f"Clearing cache directory for fresh run: {config.CACHE_DIR}")
        shutil.rmtree(config.CACHE_DIR, ignore_errors=True)

    print("=" * 80)
    print("CAHB-UIP SIMULATION STUDY".center(80))
    print("=" * 80)
    print(f"Mode: {'FAST-DEMO' if demo_mode else 'FULL'} (N_REPLICATES={config.N_REPLICATES})")
    print()
    
    # ==== 1. Validate Configuration ====
    
    print("Step 1: Validating configuration...")
    try:
        config.validate_config()
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        sys.exit(1)
    
    print()
    
    # ==== 2. Setup Directories ====
    
    print("Step 2: Creating directories...")
    directories = [
        config.CACHE_DIR,
        config.RESULTS_DIR,
        config.PLOTS_DIR,
        config.TABLES_DIR
    ]
    
    if hasattr(config, 'RAW_DATA_DIR'):
        directories.append(config.RAW_DATA_DIR)
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  √ {directory}")
    
    print()
    
    # ==== 3. Setup Caching ====
    
    print("Step 3: Initializing checkpoint system...")
    
    # Create memory-mapped cache with size limits
    try:
        memory = joblib.Memory(
            location=config.CACHE_DIR,
            verbose=0,
            bytes_limit=None  # Supported in newer joblib
        )
    except TypeError:
        memory = joblib.Memory(
            location=config.CACHE_DIR,
            verbose=0
        )
    
    # Wrap simulation function with caching if enabled
    if config.USE_CACHE:
        cached_run_single = memory.cache(run_single_simulation)
        print(f"  √ Checkpointing enabled (cache: {config.CACHE_DIR})")
        print("    √ Completed replicates will be cached")
        print("    √ Interrupted runs can resume from checkpoint")
    else:
        cached_run_single = run_single_simulation
        print("  √ Checkpointing DISABLED (set USE_CACHE=True to enable)")
    
    print()
    
    # ==== 4. Load Scenarios and Create Task List ====
    
    print("Step 4: Loading simulation scenarios...")
    scenarios = config.get_scenario_definitions()
    
    print(f"  √ Loaded {len(scenarios)} scenarios")
    print(f"  √ Comparing {len(config.METHODS_TO_RUN)} methods:")
    for method in config.METHODS_TO_RUN:
        print(f"      - {method}")
    print(f"  √ Running {config.N_REPLICATES} replicates per (scenario x method)")
    if getattr(config, "FAST_DEMO", False):
        print(f"    √ FAST_DEMO active (use '--full' or unset flag for {config.FULL_RUN_REPLICATES} replicates)")
    
    # Create task list: all (scenario, method, replicate) combinations
    print("\nStep 5: Creating task queue...")
    
    tasks = []
    for scenario in scenarios:
        for method_name in config.METHODS_TO_RUN:
            for rep_id in range(config.N_REPLICATES):
                # Use joblib.delayed to defer execution
                task = joblib.delayed(cached_run_single)(scenario, method_name, rep_id)
                tasks.append(task)
    
    n_tasks = len(tasks)
    print(f"  √ Created {n_tasks:,} tasks")
    print(f"  √ Using {config.N_JOBS} parallel workers")
    if hasattr(config, 'MAX_MEMORY_PER_JOB') and config.MAX_MEMORY_PER_JOB:
        print(f"  √ Memory limit: {config.MAX_MEMORY_PER_JOB} MB per worker")
    
    print()
    
    # ==== 5. Execute Tasks in Parallel ====
    
    print("=" * 80)
    print("RUNNING SIMULATIONS".center(80))
    print("=" * 80)
    print()
    
    # Configure joblib Parallel with memory management
    # Limit BLAS threads in workers to avoid oversubscription/memory spikes
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    parallel_kwargs = {
        'n_jobs': config.N_JOBS,
        'verbose': 0,  # Suppress joblib's own progress (we use tqdm)
        'backend': 'loky',  # Use loky backend for better memory management
        'batch_size': 1,    # Small batches to cap per-worker memory
        'pre_dispatch': 'n_jobs',  # Do not queue more than workers
    }
    
    # Add memory limit if specified
    if hasattr(config, 'MAX_MEMORY_PER_JOB') and config.MAX_MEMORY_PER_JOB:
        # Convert MB to bytes
        max_memory_bytes = config.MAX_MEMORY_PER_JOB * 1024 * 1024
        # Note: joblib's max_nbytes applies per worker
        parallel_kwargs['max_nbytes'] = max_memory_bytes
    
    progress_bar = tqdm(
        total=n_tasks,
        desc="Simulating",
        unit="task",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )

    try:
        with tqdm_joblib(progress_bar):
            results = joblib.Parallel(**parallel_kwargs)(tasks)
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user (Ctrl+C).")
        print("Partial results may be cached. Rerun to resume from checkpoint.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR during parallel execution: {e}")
        print("Check logs for details.")
        sys.exit(1)
    
    print()
    print("=" * 80)
    
    # ==== 6. Filter and Validate Results ====
    
    print("\nStep 6: Processing results...")
    
    # Remove None results (failed tasks)
    n_total = len(results)
    results = [r for r in results if r is not None]
    n_valid = len(results)
    n_failed = n_total - n_valid
    
    print(f"  √ Total tasks: {n_total:,}")
    print(f"  √ Successful: {n_valid:,} ({100*n_valid/n_total:.1f}%)")
    
    if n_failed > 0:
        print(f"  √ Failed: {n_failed:,} ({100*n_failed/n_total:.1f}%)")
        
        # Warn if failure rate is high
        if n_failed / n_total > 0.1:
            warnings.warn(
                f"High failure rate ({100*n_failed/n_total:.1f}%). "
                "Check error messages above for details.",
                UserWarning
            )
    
    if n_valid == 0:
        print("\n ERROR: All simulations failed. Cannot generate results.")
        print("Check configuration and error messages.")
        sys.exit(1)
    
    print()
    
    # ==== 7. Aggregate Results ====
    
    print("Step 7: Computing performance metrics...")
    
    try:
        analysis_outputs = analysis.process_results(results)
        summary_df = analysis_outputs.get('summary')
        n_summary = len(summary_df) if summary_df is not None else 0
        print(f"  √ Aggregated {n_summary} (scenario x method) combinations")
        print(f"  √ Metrics computed: Bias, RMSE, Coverage, Type I Error, Power")
    except Exception as e:
        print(f"\n ERROR during results processing: {e}")
        sys.exit(1)
    
    print()
    
    # ==== 8. Generate Tables ====
    
    print("Step 8: Generating tables...")
    
    try:
        analysis.generate_tables(analysis_outputs)
    except Exception as e:
        print(f" Warning: Table generation failed: {e}")
    
    print()
    
    # ==== 9. Generate Plots ====
    
    print("Step 9: Generating plots...")
    
    try:
        analysis.generate_plots(analysis_outputs)
    except Exception as e:
        print(f" Warning: Plot generation failed: {e}")
    
    print()
    
    # ==== 10. Summary ====
    
    print("=" * 80)
    print("SIMULATION COMPLETE".center(80))
    print("=" * 80)
    print()
    print("Results saved to:")
    print(f"  √ Tables: {config.TABLES_DIR}")
    print(f"  √ Plots:  {config.PLOTS_DIR}")
    print(f"  √ Cache:  {config.CACHE_DIR} (for resumability)")
    print()
    print("Next steps:")
    print("  1. Review tables in results/tables/")
    print("  2. Review plots in results/plots/")
    print("  3. Incorporate into manuscript")
    print()
    print("To rerun with fresh data:")
    print(f"  rm -rf {config.CACHE_DIR}")
    print(f"  python main.py")
    print()
    print(" Done!")
    print("=" * 80)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CAHB-UIP simulation study")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in fast demonstration mode (small number of replicates)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the cache directory before running (applies to demo and full modes)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full run even if FAST_DEMO env/config is enabled",
    )
    args = parser.parse_args()
    if args.demo and args.full:
        parser.error("Cannot specify both --demo and --full")
    override = True if args.demo else False if args.full else None
    main(demo_override=override, clear_cache=args.reset)





