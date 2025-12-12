"""
analysis.py

Results Analysis and Visualization for the BRAVE Simulation Study

This module processes raw simulation results and generates publication-ready
outputs including:
    1. Performance metric tables (CSV and LaTeX formats)
    2. High-quality PDF plots meeting Statistics in Medicine standards
    3. Comprehensive evaluation metrics (Section 3.5)

Evaluation Metrics (Section 3.5 of manuscript):
-----------------------------------------------
1. Estimation Performance:
   - Bias: E[delta_hat - delta_true]
   - RMSE: sqrt(E[(delta_hat - delta_true)^2])
   - Coverage: P(delta_true in CI_95%)
   - CI Width: Average width of 95% confidence intervals

2. Decision Error Rates:
   - Type I Error: P(reject H0 | H0 true) when tau_0 = 0
   - Power: P(reject H0 | H1 true) when tau_0 = 0.4
   - Decision threshold: P(delta > 0 | Data) > 0.975

3. Allocation Performance:
   - Treatment allocation rate: proportion assigned to treatment
   - Allocation balance across covariate subgroups

Output Specifications:
----------------------
- Figures: PDF format, 300 DPI, Times New Roman font
- Tables: CSV for data, LaTeX for manuscript insertion
- Follows Statistics in Medicine figure/table guidelines

References:
    BRAVE manuscript Section 3.5 (Evaluation Metrics)
"""

import os
import warnings
from typing import List
import pandas as pd
import numpy as np

# Configure matplotlib for publication-quality PDFs
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Import project configuration
try:
    import config
    import data_generation
except ImportError as e:
    warnings.warn(f"Could not import project modules: {e}")


# =============================================================================
# Style Configuration for Publication-Quality Figures
# =============================================================================

def configure_plot_style():
    """
    Configures matplotlib and seaborn for Statistics in Medicine publication standards.
    
    Guidelines:
        - Font: Times New Roman, 10pt for body, 12pt for titles
        - Resolution: 300 DPI minimum
        - Format: PDF (vector graphics)
        - Colors: Colorblind-friendly palette
        - Grids: Light, unobtrusive
    """
    # Set publication style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    
    # Matplotlib rcParams
    plt.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'legend.title_fontsize': 10,
        
        # Figure settings
        'figure.figsize': (7, 5),
        'figure.dpi': 100,  # Screen display
        'savefig.dpi': 300,  # Publication quality
        'savefig.format': 'pdf',
        'savefig.bbox': 'tight',
        
        # Line and marker settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'patch.linewidth': 0.5,
        
        # Grid settings
        'axes.grid': True,
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'grid.alpha': 0.6,
        
        # Axes settings
        'axes.linewidth': 0.8,
        'axes.edgecolor': 'black',
        'axes.labelpad': 4.0,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': 'gray',
        'legend.fancybox': False,
    })
    
    # Define colorblind-friendly palette
    # Based on Wong (2011) Nature Methods palette
    colors = {
        'KBCD': '#009E73',      # Bluish green
        'CAHB': '#56B4E9',      # Sky blue
        'BRAVE_IPD': '#E69F00', # Orange
        'BRAVE_SLD': '#D55E00', # Vermilion
    }
    
    return colors


# Default method label mapping (used if config lacks entries or for legacy aliases)
DEFAULT_METHOD_LABELS = {
    'BRAVE_IPD': 'BRAVE-IPD',
    'BRAVE_SLD': 'BRAVE-SLD',
    'CAHB': 'CAHB',
    'KBCD': 'KBCD',
}


# =============================================================================
# True Treatment Effect Calculation
# =============================================================================

def get_true_delta(tau_0: float) -> float:
    """
    Computes the true marginal average treatment effect (MATE).
    
    From Section 3.1, the conditional average treatment effect is:
        tau(x) = tau_0 + 0.5*X1 - 0.5*I(X2=1)
    
    Marginal expectation:
        E[tau(X)] = E[tau_0 + 0.5*X1 - 0.5*I(X2=1)]
                  = tau_0 + 0.5*E[X1] - 0.5*P(X2=1)
                  = tau_0 + 0.5*0 - 0.5*0.5
                  = tau_0 - 0.25
    
    Args:
        tau_0: Base treatment effect parameter
    
    Returns:
        True marginal average treatment effect
    """
    return tau_0


# =============================================================================
# Results Processing
# =============================================================================

def process_results(results_list: list) -> dict:
    """
    Processes raw simulation results into aggregated performance metrics and
    supporting data structures required for downstream figures.
    """
    outputs = {
        'summary': pd.DataFrame(),
        'raw': pd.DataFrame(),
        'allocation_path': pd.DataFrame(),
        'calibration': pd.DataFrame(),
    }

    if not results_list:
        warnings.warn("No results to process. Returning empty structures.")
        return outputs

    df = pd.DataFrame(results_list)
    outputs['raw'] = df.copy()

    n_total = len(df)
    df = df[df['delta_hat'].notna()]
    n_valid = len(df)
    if n_valid < n_total:
        warnings.warn(f"Removed {n_total - n_valid} failed replicates ({100*(n_total-n_valid)/n_total:.1f}%)")

    if df.empty:
        warnings.warn("All replicates failed. Cannot compute metrics.")
        return outputs

    df['true_delta'] = df['tau_0'].apply(get_true_delta)
    df['error'] = df['delta_hat'] - df['true_delta']
    df['abs_error'] = np.abs(df['error'])
    df['sq_error'] = df['error'] ** 2
    df['coverage'] = ((df['true_delta'] >= df['ci_low']) &
                      (df['true_delta'] <= df['ci_high']))
    df['ci_width'] = df['ci_high'] - df['ci_low']
    threshold = config.DECISION_THRESHOLD
    df['is_success'] = df['prob_gt_0'] > threshold
    df['alloc_rate_treatment'] = df['n_treated'] / df['n_total']

    key_cols = ['method', 'tau_0', 'n', 'n_h', 'scenario_type', 'kappa']
    key_cols = [col for col in key_cols if col in df.columns]
    groupby_cols = key_cols.copy()
    if 'scenario_name' in df.columns:
        groupby_cols = ['scenario_name'] + groupby_cols

    agg_funcs = {
        'error': 'mean',
        'abs_error': 'mean',
        'sq_error': lambda x: np.sqrt(np.mean(x)),
        'coverage': 'mean',
        'ci_width': 'mean',
        'alloc_rate_treatment': 'mean',
        'is_success': 'mean',
        'n_total': 'mean',
        'post_M_mean': 'mean',
        'post_wL_mean': 'mean',
        'replicate_id': 'count'
    }

    summary = df.groupby(groupby_cols, as_index=False).agg(agg_funcs).rename(columns={
        'error': 'Bias',
        'abs_error': 'MAE',
        'sq_error': 'RMSE',
        'coverage': 'Coverage',
        'ci_width': 'CI_Width',
        'alloc_rate_treatment': 'Alloc_Rate_Treatment',
        'is_success': 'Success_Rate',
        'replicate_id': 'N_Replicates'
    })

    type_key = key_cols.copy()
    type_i_df = (
        df[df['tau_0'] == 0.0]
        .groupby(type_key, as_index=False)['is_success']
        .mean()
        .rename(columns={'is_success': 'Type_I_Error'})
    ) if type_key else pd.DataFrame()
    power_df = (
        df[df['tau_0'] == 0.4]
        .groupby(type_key, as_index=False)['is_success']
        .mean()
        .rename(columns={'is_success': 'Power'})
    ) if type_key else pd.DataFrame()
    if not type_i_df.empty:
        summary = pd.merge(summary, type_i_df, on=type_key, how='left')
    if not power_df.empty:
        summary = pd.merge(summary, power_df, on=type_key, how='left')
    summary = summary.sort_values(['scenario_name', 'method']).reset_index(drop=True)
    outputs['summary'] = summary

    # Allocation trajectories -------------------------------------------------
    allocation_records = []
    for row in results_list:
        path = row.get('allocation_path') or []
        for point in path:
            sample_size = point.get('sample_size')
            if sample_size is None:
                continue
            allocation_records.append({
                'scenario_id': row.get('scenario_id'),
                'scenario_name': row.get('scenario_name'),
                'scenario_type': row.get('scenario_type', 'Unknown'),
                'kappa': row.get('kappa', np.nan),
                'method': row.get('method'),
                'tau_0': row.get('tau_0'),
                'n': row.get('n'),
                'n_h': row.get('n_h'),
                'replicate_id': row.get('replicate_id'),
                'sample_size': sample_size,
                'prop_treated': point.get('prop_treated'),
                'R_n': point.get('R_n'),
                'M': point.get('M'),
                'w_L': point.get('w_L'),
                'gamma': point.get('gamma'),
                'x1': point.get('x1'),
                'x2': point.get('x2'),
                'x3': point.get('x3'),
                'x4': point.get('x4'),
            })

    if allocation_records:
        outputs['allocation_path'] = pd.DataFrame(allocation_records)

    # Calibration payload -----------------------------------------------------
    calibration_records = []
    for row in results_list:
        calib = row.get('calibration_samples') or []
        for entry in calib:
            calibration_records.append({
                'scenario_id': row.get('scenario_id'),
                'scenario_name': row.get('scenario_name'),
                'scenario_type': row.get('scenario_type', 'Unknown'),
                'kappa': row.get('kappa', np.nan),
                'method': row.get('method'),
                'tau_0': row.get('tau_0'),
                'n': row.get('n'),
                'n_h': row.get('n_h'),
                'replicate_id': row.get('replicate_id'),
                'sample_size': row.get('n'),
                'x1': entry.get('x1'),
                'x2': entry.get('x2'),
                'x3': entry.get('x3'),
                'x4': entry.get('x4'),
                'M_mean': entry.get('M_mean'),
                'M_ci_low': entry.get('M_ci_low'),
                'M_ci_high': entry.get('M_ci_high'),
                'R_n': entry.get('R_n'),
                'w_L': entry.get('w_L'),
                'gamma': entry.get('gamma'),
            })

    if calibration_records:
        outputs['calibration'] = pd.DataFrame(calibration_records)

    return outputs


# =============================================================================
# Table Generation
# =============================================================================

def generate_tables(results_obj: dict):
    """
    Generates publication-ready tables in CSV and LaTeX formats.
    
    Creates:
        1. estimation_metrics.(csv|tex): Publication-ready estimation summaries
        2. type_i_error_power.tex: Focused table for error/power results
    
    Args:
        results_obj: Output dictionary returned by process_results()
    
    Output Files:
        - results/tables/estimation_metrics.csv
        - results/tables/estimation_metrics.tex
        - results/tables/type_i_error_power.tex
    """
    summary_df = results_obj.get('summary', pd.DataFrame())

    if summary_df.empty:
        warnings.warn("Empty summary DataFrame. No tables generated.")
        return
        
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    
    # Final inference stage metrics (Section 3.4)
    # Includes: Bias, RMSE, Type I Error, Power, 95% CrI Coverage, CI Width
    # Also includes posterior M and w_L for BRAVE methods (NaN for others)
    estimation_cols = [
        'scenario_name', 'method', 'tau_0', 'n', 'n_h', 'scenario_type', 'kappa',
        'Bias', 'RMSE', 'Coverage', 'CI_Width', 'Alloc_Rate_Treatment',
        'Type_I_Error', 'Power', 'post_M_mean', 'post_wL_mean'
    ]
    estimation_cols = [c for c in estimation_cols if c in summary_df.columns]
    estimation_df = summary_df[estimation_cols].copy()
    est_csv = os.path.join(config.TABLES_DIR, 'final_inference_metrics.csv')
    estimation_df.to_csv(est_csv, index=False, float_format='%.4f')
    print(f"[+] Final inference metrics table saved: {est_csv}")

    # Publication-ready LaTeX table
    display_df = estimation_df.copy()
    method_labels = DEFAULT_METHOD_LABELS.copy()
    if hasattr(config, 'METHOD_LABELS'):
        method_labels.update(getattr(config, 'METHOD_LABELS', {}))
    display_df['method_label'] = display_df['method'].map(method_labels).fillna(display_df['method'])

    try:
        tex_pivot = display_df.pivot_table(
            index=['scenario_name', 'tau_0'],
            columns='method_label',
            values=['Bias', 'RMSE', 'Coverage', 'CI_Width']
        )
        tex_path = os.path.join(config.TABLES_DIR, 'estimation_metrics.tex')
        with open(tex_path, 'w') as f:
            f.write("% Estimation metrics table (Bias/RMSE/Coverage/CI Width)\n")
            tex_pivot.to_latex(
                f,
                float_format='%.3f',
                na_rep='-',
                bold_rows=True,
                multicolumn_format='c',
                escape=False
            )
        print(f"[+] LaTeX estimation table saved: {tex_path}")
    except Exception as e:
        warnings.warn(f"Failed to build LaTeX estimation table: {e}")
    
    # ==== Type I Error and Power Table (Focused) ====
    
    if 'Type_I_Error' in summary_df.columns and 'Power' in summary_df.columns:
        # Extract Type I Error (tau_0 = 0) and Power (tau_0 = 0.4)
        type_i = summary_df[summary_df['tau_0'] == 0.0][['scenario_name', 'method', 'Type_I_Error']].dropna()
        power = summary_df[summary_df['tau_0'] == 0.4][['scenario_name', 'method', 'Power']].dropna()
        
        # Merge
        error_power = pd.merge(
            type_i, power, 
            on=['scenario_name', 'method'], 
            how='outer'
        )
        
        # Pivot for cleaner presentation
        try:
            ep_pivot = error_power.pivot(
                index='scenario_name',
                columns='method',
                values=['Type_I_Error', 'Power']
            )
            
            ep_tex_path = os.path.join(config.TABLES_DIR, 'type_i_error_power.tex')
            with open(ep_tex_path, 'w') as f:
                f.write("% Type I Error and Power Results\n")
                f.write("% Nominal alpha = 0.05 for Type I Error\n\n")
                ep_pivot.to_latex(
                    f,
                    float_format='%.3f',
                    bold_rows=True,
                    multicolumn_format='c',
                    escape=False
                )
            print(f"[+] Type I Error/Power table saved: {ep_tex_path}")
            
        except Exception as e:
            warnings.warn(f"Could not create Type I Error/Power pivot table: {e}")
    
    print(f"[+] All tables saved to: {config.TABLES_DIR}")


# =============================================================================
# Plot Generation
# =============================================================================

def generate_plots(results_obj: dict):
    """
    Generate evaluation metrics plots for adaptive allocation stage and final inference stage.
    
    Adaptive allocation stage plots:
    1. Averaged trajectory curves for allocation ratio (all 4 designs)
    2. Averaged density plots for R_n(X) (all 4 designs, KBCD fixed at 1)
    3. Averaged density plots for M(X) (BRAVE-IPD and BRAVE-SLD)
    4. Averaged density plots for w_L(x) (BRAVE-IPD and BRAVE-SLD)
    5. Individual plots per scenario
    6. Combined 6×4 grid plots for n_h=400 and 800
    """
    alloc_df = results_obj.get('allocation_path', pd.DataFrame())
    if alloc_df.empty:
        warnings.warn("Empty allocation diagnostics. No plots generated.")
        return

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    colors = configure_plot_style()
    method_labels = DEFAULT_METHOD_LABELS.copy()
    if hasattr(config, 'METHOD_LABELS'):
        method_labels.update(getattr(config, 'METHOD_LABELS', {}))

    scenario_order = [
        ('S1', 1.0, r'S1 $\kappa$=1.0'),
        ('S2', 0.7, r'S2 $\kappa$=0.7'),
        ('S2', 1.3, r'S2 $\kappa$=1.3'),
        ('S3', 0.7, r'S3 $\kappa$=0.7'),
        ('S3', 1.3, r'S3 $\kappa$=1.3'),
        ('S4', 1.3, r'S4 $\kappa$=1.3'),
    ]
    label_order = [s[2] for s in scenario_order]
    label_map = {(stype, float(kappa)): label for stype, kappa, label in scenario_order}

    def _label(method: str) -> str:
        return method_labels.get(method, method)

    def _scenario_label(row: pd.Series) -> str:
        stype = row.get('scenario_type')
        kappa_val = row.get('kappa')
        try:
            kappa_key = float(kappa_val)
        except Exception:
            kappa_key = kappa_val
        return label_map.get((stype, kappa_key), f"{stype} $\\kappa$={kappa_val}")

    alloc_df = alloc_df.copy()
    # Backfill n_h if missing
    if 'n_h' not in alloc_df.columns or alloc_df['n_h'].isna().all():
        raw_df = results_obj.get('raw', pd.DataFrame())
        if not raw_df.empty and 'scenario_id' in alloc_df.columns:
            nh_map = raw_df[['scenario_id', 'n_h']].drop_duplicates()
            alloc_df = alloc_df.merge(nh_map, on='scenario_id', how='left', suffixes=('', '_raw'))
            alloc_df['n_h'] = alloc_df['n_h'].fillna(alloc_df.get('n_h_raw'))
            if 'n_h_raw' in alloc_df.columns:
                alloc_df = alloc_df.drop(columns=['n_h_raw'])
        if alloc_df['n_h'].isna().any() and 'scenario_name' in alloc_df.columns:
            extracted = alloc_df['scenario_name'].str.extract(r'nh=([0-9]+)')[0]
            alloc_df['n_h'] = alloc_df['n_h'].fillna(pd.to_numeric(extracted, errors='coerce'))

    alloc_df['scenario_label'] = alloc_df.apply(_scenario_label, axis=1)
    alloc_df = alloc_df[alloc_df['scenario_label'].isin(label_order)]
    
    # Ensure KBCD has R_n = 1.0
    alloc_df.loc[alloc_df['method'] == 'KBCD', 'R_n'] = 1.0

    # Design order for all plots
    design_order = ['BRAVE_IPD', 'BRAVE_SLD', 'CAHB', 'KBCD']
    brave_methods = ['BRAVE_IPD', 'BRAVE_SLD']
    palette_all = {m: colors.get(m, 'gray') for m in design_order}

    def _label(method: str) -> str:
        return method_labels.get(method, method)
    
    def _safe_filename(label: str) -> str:
        """Convert scenario label to safe filename."""
        return label.replace(' ', '_').replace('$', '').replace('\\', '').replace('=', '').replace('{', '').replace('}', '')

    # =====================================================================
    # 1. Allocation Trajectory Plots (all 4 designs)
    # =====================================================================
    def _plot_allocation_trajectories():
        """Generate averaged trajectory curves for allocation ratio (all 4 designs)."""
        df = alloc_df.dropna(subset=['prop_treated', 'sample_size']).copy()
        if df.empty:
            warnings.warn("No allocation trajectory data available.")
            return
        
        df['scenario_label'] = df.apply(_scenario_label, axis=1)
        
        # Average across replicates for each (scenario, method, sample_size, n_h) combination
        traj_agg = (
            df.groupby(['scenario_label', 'scenario_type', 'kappa', 'n_h', 'method', 'sample_size'], as_index=False)
            .agg(prop_treated_mean=('prop_treated', 'mean'))
        )
        
        # Individual plots per scenario
        for n_h_val in sorted(df['n_h'].dropna().unique()):
            for scen_label in label_order:
                sub = traj_agg[(traj_agg['n_h'] == n_h_val) & (traj_agg['scenario_label'] == scen_label)]
                if sub.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(7, 5))
                for method in design_order:
                    method_data = sub[sub['method'] == method]
                    if method_data.empty:
                        continue
                    ax.plot(
                        method_data['sample_size'],
                        method_data['prop_treated_mean'],
                        color=palette_all.get(method, 'gray'),
                        label=_label(method),
                        linewidth=2.0
                    )
                ax.set_ylabel('Allocation ratio to treatment', fontsize=11)
                ax.set_xlabel('Sample size', fontsize=11)
                ax.set_ylim(0, 1)
                ax.set_title(f"{scen_label}, $n_h$={int(n_h_val)}", fontsize=12)
                ax.legend(title='Method', fontsize=9)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                safe_label = _safe_filename(scen_label)
                out = os.path.join(config.PLOTS_DIR, f"traj_alloc_{safe_label}_nh{int(n_h_val)}.pdf")
                fig.savefig(out)
                plt.close(fig)
                print(f"[+] Saved allocation trajectory: {out}")
    
    _plot_allocation_trajectories()

    # =====================================================================
    # 2. Density Plots for R_n(X) (all 4 designs, averaged across replications)
    # =====================================================================
    def _plot_rn_density():
        """Generate averaged density plots for R_n(X) across all designs."""
        df = alloc_df.dropna(subset=['R_n']).copy()
        if df.empty:
            warnings.warn("No R_n data available.")
            return
        
        df['scenario_label'] = df.apply(_scenario_label, axis=1)
        
        # Individual plots per scenario
        for n_h_val in sorted(df['n_h'].dropna().unique()):
            for scen_label in label_order:
                sub = df[(df['n_h'] == n_h_val) & (df['scenario_label'] == scen_label)]
                if sub.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(7, 5))
                for method in design_order:
                    vals = sub[sub['method'] == method]['R_n'].dropna()
                    if vals.empty:
                        continue
                    
                    # For KBCD, R_n should be exactly 1.0
                    if method == 'KBCD':
                        ax.axvline(1.0, color=palette_all.get(method, 'gray'), 
                                 linestyle='--', linewidth=2.0, label=_label(method))
                    elif np.nanstd(vals) < 1e-6:
                        ax.axvline(np.nanmean(vals), color=palette_all.get(method, 'gray'),
                                 linestyle='--', linewidth=2.0, label=_label(method))
                    else:
                        sns.kdeplot(vals, ax=ax, label=_label(method), 
                                  color=palette_all.get(method, 'gray'), 
                                  clip=(0, None), linewidth=2.0)
                
                ax.set_xlabel(r'$R_n(\mathbf{X})$', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.set_title(f"$R_n$ density - {scen_label}, $n_h$={int(n_h_val)}", fontsize=12)
                ax.legend(title='Method', fontsize=9)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                safe_label = _safe_filename(scen_label)
                out = os.path.join(config.PLOTS_DIR, f"density_rn_{safe_label}_nh{int(n_h_val)}.pdf")
                fig.savefig(out)
                plt.close(fig)
                print(f"[+] Saved R_n density: {out}")
    
    _plot_rn_density()

    # =====================================================================
    # 3. Density Plots for M(X) (BRAVE-IPD and BRAVE-SLD only)
    # =====================================================================
    def _plot_M_density():
        """Generate averaged density plots for M(X) (BRAVE methods only)."""
        df = alloc_df.dropna(subset=['M']).copy()
        if df.empty:
            warnings.warn("No M data available.")
            return
        
        df = df[df['method'].isin(brave_methods)]
        if df.empty:
            warnings.warn("No BRAVE method data for M.")
            return
        
        df['scenario_label'] = df.apply(_scenario_label, axis=1)
        
        # Individual plots per scenario
        for n_h_val in sorted(df['n_h'].dropna().unique()):
            for scen_label in label_order:
                sub = df[(df['n_h'] == n_h_val) & (df['scenario_label'] == scen_label)]
                if sub.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(7, 5))
                for method in brave_methods:
                    vals = sub[sub['method'] == method]['M'].dropna()
                    if vals.empty:
                        continue
                    
                    if np.nanstd(vals) < 1e-6:
                        ax.axvline(np.nanmean(vals), color=palette_all.get(method, 'gray'),
                                 linestyle='--', linewidth=2.0, label=_label(method))
                    else:
                        sns.kdeplot(vals, ax=ax, label=_label(method),
                                  color=palette_all.get(method, 'gray'),
                                  clip=(0, None), linewidth=2.0)
                
                ax.set_xlabel(r'$M(\mathbf{X})$', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.set_title(f"$M$ density - {scen_label}, $n_h$={int(n_h_val)}", fontsize=12)
                ax.legend(title='Method', fontsize=9)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                safe_label = _safe_filename(scen_label)
                out = os.path.join(config.PLOTS_DIR, f"density_M_{safe_label}_nh{int(n_h_val)}.pdf")
                fig.savefig(out)
                plt.close(fig)
                print(f"[+] Saved M density: {out}")
    
    _plot_M_density()

    # =====================================================================
    # 4. Density Plots for w_L(x) (BRAVE-IPD and BRAVE-SLD only)
    # =====================================================================
    def _plot_wL_density():
        """Generate averaged density plots for w_L(x) (BRAVE methods only)."""
        df = alloc_df.dropna(subset=['w_L']).copy()
        if df.empty:
            warnings.warn("No w_L data available.")
            return
        
        df = df[df['method'].isin(brave_methods)]
        if df.empty:
            warnings.warn("No BRAVE method data for w_L.")
            return
        
        df['scenario_label'] = df.apply(_scenario_label, axis=1)
        
        # Individual plots per scenario
        for n_h_val in sorted(df['n_h'].dropna().unique()):
            for scen_label in label_order:
                sub = df[(df['n_h'] == n_h_val) & (df['scenario_label'] == scen_label)]
                if sub.empty:
                    continue
                
                fig, ax = plt.subplots(figsize=(7, 5))
                for method in brave_methods:
                    vals = sub[sub['method'] == method]['w_L'].dropna()
                    if vals.empty:
                        continue
                    
                    if np.nanstd(vals) < 1e-6:
                        ax.axvline(np.nanmean(vals), color=palette_all.get(method, 'gray'),
                                 linestyle='--', linewidth=2.0, label=_label(method))
                    else:
                        sns.kdeplot(vals, ax=ax, label=_label(method),
                                  color=palette_all.get(method, 'gray'),
                                  clip=(0, 1), linewidth=2.0)
                
                ax.set_xlabel(r'$w_L(\mathbf{x})$', fontsize=11)
                ax.set_ylabel('Density', fontsize=11)
                ax.set_title(f"$w_L$ density - {scen_label}, $n_h$={int(n_h_val)}", fontsize=12)
                ax.legend(title='Method', fontsize=9)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                safe_label = _safe_filename(scen_label)
                out = os.path.join(config.PLOTS_DIR, f"density_wL_{safe_label}_nh{int(n_h_val)}.pdf")
                fig.savefig(out)
                plt.close(fig)
                print(f"[+] Saved w_L density: {out}")
    
    _plot_wL_density()

    # =====================================================================
    # 5. Combined 6×4 Grid Plots for n_h=400 and 800
    # Each row = scenario, Each column = plot type (allocation, Rn, M, w_L)
    # =====================================================================
    def _plot_combined_grids():
        """Generate combined 6×4 grid plots: 6 scenarios × 4 plot types."""
        scenarios = [(stype, kappa, label) for stype, kappa, label in scenario_order]
        plot_types = ['allocation', 'Rn', 'M', 'w_L']
        
        for n_h_val in [400, 800]:
            # Check if we have data for this n_h
            df_nh = alloc_df[alloc_df['n_h'] == n_h_val].copy()
            if df_nh.empty:
                continue
            
            df_nh['scenario_label'] = df_nh.apply(_scenario_label, axis=1)
            
            # Create 6×4 grid: rows = scenarios, cols = plot types
            fig, axes = plt.subplots(len(scenarios), len(plot_types),
                                   figsize=(3.5 * len(plot_types), 2.5 * len(scenarios)),
                                   sharex=False, sharey=False)
            axes = np.atleast_2d(axes)
            
            for i, (stype, kappa, scen_label) in enumerate(scenarios):
                scen_sub = df_nh[(df_nh['scenario_type'] == stype) &
                                (df_nh['kappa'].astype(float) == float(kappa))]
                
                if scen_sub.empty:
                    continue
                
                # Column 0: Allocation trajectory
                ax = axes[i, 0]
                traj_df = scen_sub.dropna(subset=['prop_treated', 'sample_size']).copy()
                if not traj_df.empty:
                    traj_agg = (
                        traj_df.groupby(['method', 'sample_size'], as_index=False)
                        .agg(prop_treated_mean=('prop_treated', 'mean'))
                    )
                    for method in design_order:
                        method_data = traj_agg[traj_agg['method'] == method]
                        if not method_data.empty:
                            ax.plot(method_data['sample_size'], method_data['prop_treated_mean'],
                                  color=palette_all.get(method, 'gray'), label=_label(method),
                                  linewidth=1.5, alpha=0.8)
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Alloc. ratio', fontsize=8)
                    if i == len(scenarios) - 1:
                        ax.set_xlabel('Sample size', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.set_title('Allocation', fontsize=9)
                        ax.legend(fontsize=6, loc='upper right', ncol=2)
                
                # Column 1: R_n density
                ax = axes[i, 1]
                rn_df = scen_sub.dropna(subset=['R_n']).copy()
                if not rn_df.empty:
                    for method in design_order:
                        vals = rn_df[rn_df['method'] == method]['R_n'].dropna()
                        if vals.empty:
                            continue
                        if method == 'KBCD':
                            ax.axvline(1.0, color=palette_all.get(method, 'gray'),
                                     linestyle='--', linewidth=1.5, alpha=0.8)
                        elif np.nanstd(vals) < 1e-6:
                            ax.axvline(np.nanmean(vals), color=palette_all.get(method, 'gray'),
                                     linestyle='--', linewidth=1.5, alpha=0.8)
                        else:
                            sns.kdeplot(vals, ax=ax, color=palette_all.get(method, 'gray'),
                                      fill=False, clip=(0, None), linewidth=1.5, alpha=0.8)
                    ax.set_ylabel('Density', fontsize=8)
                    if i == len(scenarios) - 1:
                        ax.set_xlabel(r'$R_n$', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.set_title(r'$R_n$ Density', fontsize=9)
                
                # Column 2: M density (BRAVE only)
                ax = axes[i, 2]
                m_df = scen_sub.dropna(subset=['M']).copy()
                m_df = m_df[m_df['method'].isin(brave_methods)]
                if not m_df.empty:
                    for method in brave_methods:
                        vals = m_df[m_df['method'] == method]['M'].dropna()
                        if vals.empty:
                            continue
                        if np.nanstd(vals) < 1e-6:
                            ax.axvline(np.nanmean(vals), color=palette_all.get(method, 'gray'),
                                     linestyle='--', linewidth=1.5, alpha=0.8)
                        else:
                            sns.kdeplot(vals, ax=ax, color=palette_all.get(method, 'gray'),
                                      fill=False, clip=(0, None), linewidth=1.5, alpha=0.8)
                    ax.set_ylabel('Density', fontsize=8)
                    if i == len(scenarios) - 1:
                        ax.set_xlabel(r'$M$', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.set_title(r'$M$ Density', fontsize=9)
                
                # Column 3: w_L density (BRAVE only)
                ax = axes[i, 3]
                wl_df = scen_sub.dropna(subset=['w_L']).copy()
                wl_df = wl_df[wl_df['method'].isin(brave_methods)]
                if not wl_df.empty:
                    for method in brave_methods:
                        vals = wl_df[wl_df['method'] == method]['w_L'].dropna()
                        if vals.empty:
                            continue
                        if np.nanstd(vals) < 1e-6:
                            ax.axvline(np.nanmean(vals), color=palette_all.get(method, 'gray'),
                                     linestyle='--', linewidth=1.5, alpha=0.8)
                        else:
                            sns.kdeplot(vals, ax=ax, color=palette_all.get(method, 'gray'),
                                      fill=False, clip=(0, 1), linewidth=1.5, alpha=0.8)
                    ax.set_ylabel('Density', fontsize=8)
                    if i == len(scenarios) - 1:
                        ax.set_xlabel(r'$w_L$', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    if i == 0:
                        ax.set_title(r'$w_L$ Density', fontsize=9)
                
                # Add scenario label on the leftmost axis
                axes[i, 0].set_ylabel(scen_label, fontsize=9)
            
            fig.suptitle(f'Adaptive Allocation Stage Metrics ($n_h$={int(n_h_val)})', 
                         fontsize=12, y=0.995)
            fig.tight_layout(rect=[0.05, 0, 1, 0.99])
            out = os.path.join(config.PLOTS_DIR, f"combined_6x4_grid_nh{int(n_h_val)}.pdf")
            fig.savefig(out)
            plt.close(fig)
            print(f"[+] Saved combined 6×4 grid: {out}")
    
    _plot_combined_grids()

    print(f"[+] All plots saved to: {config.PLOTS_DIR}")

# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    print("analysis.py - Results processing module")
    print("This module is typically called from main.py")
    print("\nTo test, run: python main.py")
