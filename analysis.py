"""
analysis.py

Results Analysis and Visualization for CAHB-PP Simulation Study

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
    CAHB-PP manuscript Section 3.5 (Evaluation Metrics)
"""

import os
import warnings
import pandas as pd
import numpy as np

# Configure matplotlib for publication-quality PDFs
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

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
        'KBCD': '#009E73',          # Bluish green
        'CAHB': '#56B4E9',          # Sky blue
        'CAHB_PP_IPD': '#E69F00',   # Orange
        'CAHB_PP_SLD': '#D55E00',   # Vermilion
    }
    
    return colors


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
    return tau_0 - 0.25


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

    groupby_cols = ['scenario_name', 'method', 'tau_0', 'n', 'n_h', 'scenario_type']
    groupby_cols = [col for col in groupby_cols if col in df.columns]

    agg_funcs = {
        'error': 'mean',
        'abs_error': 'mean',
        'sq_error': lambda x: np.sqrt(np.mean(x)),
        'coverage': 'mean',
        'ci_width': 'mean',
        'alloc_rate_treatment': 'mean',
        'is_success': 'mean',
        'n_total': 'mean',
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

    type_i_df = (
        df[df['tau_0'] == 0.0]
        .groupby(['scenario_name', 'method'], as_index=False)['is_success']
        .mean()
        .rename(columns={'is_success': 'Type_I_Error'})
    )
    power_df = (
        df[df['tau_0'] == 0.4]
        .groupby(['scenario_name', 'method'], as_index=False)['is_success']
        .mean()
        .rename(columns={'is_success': 'Power'})
    )
    summary = pd.merge(summary, type_i_df, on=['scenario_name', 'method'], how='left')
    summary = pd.merge(summary, power_df, on=['scenario_name', 'method'], how='left')
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
                'method': row.get('method'),
                'tau_0': row.get('tau_0'),
                'n': row.get('n'),
                'replicate_id': row.get('replicate_id'),
                'sample_size': sample_size,
                'prop_treated': point.get('prop_treated'),
                'R_n': point.get('R_n'),
                'a_mean': point.get('a_mean'),
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
                'method': row.get('method'),
                'tau_0': row.get('tau_0'),
                'n': row.get('n'),
                'replicate_id': row.get('replicate_id'),
                'sample_size': row.get('n'),
                'x1': entry.get('x1'),
                'x2': entry.get('x2'),
                'a_mean': entry.get('a_mean'),
                'a_ci_low': entry.get('a_ci_low'),
                'a_ci_high': entry.get('a_ci_high'),
                'compatibility': entry.get('compatibility'),
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
        1. simulation_summary.csv: Full results table
        2. estimation_metrics.(csv|tex): Publication-ready estimation summaries
        3. type_i_error_power.tex: Focused table for error/power results
    
    Args:
        results_obj: Output dictionary returned by process_results()
    
    Output Files:
        - results/tables/simulation_summary.csv
        - results/tables/estimation_metrics.csv
        - results/tables/estimation_metrics.tex
        - results/tables/type_i_error_power.tex
    """
    summary_df = results_obj.get('summary', pd.DataFrame())

    if summary_df.empty:
        warnings.warn("Empty summary DataFrame. No tables generated.")
        return
        
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    
    # ==== CSV Output (Full Table) ====
    csv_path = os.path.join(config.TABLES_DIR, 'simulation_summary.csv')
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Full summary table saved: {csv_path}")
    
    estimation_cols = ['scenario_name', 'scenario_type', 'n', 'n_h', 'tau_0', 'method',
                       'Bias', 'RMSE', 'Coverage', 'CI_Width']
    estimation_cols = [c for c in estimation_cols if c in summary_df.columns]
    estimation_df = summary_df[estimation_cols].copy()
    est_csv = os.path.join(config.TABLES_DIR, 'estimation_metrics.csv')
    estimation_df.to_csv(est_csv, index=False, float_format='%.4f')
    print(f"✓ Estimation metrics table saved: {est_csv}")

    # Publication-ready LaTeX table
    display_df = estimation_df.copy()
    if hasattr(config, 'METHOD_LABELS'):
        display_df['method_label'] = display_df['method'].map(config.METHOD_LABELS).fillna(display_df['method'])
    else:
        display_df['method_label'] = display_df['method']

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
        print(f"✓ LaTeX estimation table saved: {tex_path}")
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
            print(f"✓ Type I Error/Power table saved: {ep_tex_path}")
            
        except Exception as e:
            warnings.warn(f"Could not create Type I Error/Power pivot table: {e}")
    
    print(f"✓ All tables saved to: {config.TABLES_DIR}")


# =============================================================================
# Plot Generation
# =============================================================================

def generate_plots(results_obj: dict):
    """Generate required publication-quality figures."""
    summary_df = results_obj.get('summary', pd.DataFrame())
    if summary_df.empty:
        warnings.warn("Empty summary DataFrame. No plots generated.")
        return

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    colors = configure_plot_style()

    available_methods = summary_df['method'].unique().tolist()
    if hasattr(config, 'METHODS_TO_RUN'):
        methods = [m for m in config.METHODS_TO_RUN if m in available_methods]
    else:
        methods = available_methods
    if not methods:
        methods = available_methods

    def _label(method: str) -> str:
        if hasattr(config, 'METHOD_LABELS'):
            return config.METHOD_LABELS.get(method, method)
        return method

    def _aggregate_rate(df: pd.DataFrame, rate_col: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        grouped = (
            df.groupby(['n', 'method'])
            .apply(lambda g: pd.Series({
                'value': np.average(g[rate_col], weights=g['N_Replicates']),
                'N_total': g['N_Replicates'].sum()
            }))
            .reset_index()
        )
        grouped['ci'] = 1.96 * np.sqrt(
            np.clip(grouped['value'] * (1 - grouped['value']) / np.maximum(grouped['N_total'], 1), 0, 1)
        )
        grouped['lower'] = np.clip(grouped['value'] - grouped['ci'], 0.0, 1.0)
        grouped['upper'] = np.clip(grouped['value'] + grouped['ci'], 0.0, 1.0)
        return grouped

    type_df = _aggregate_rate(summary_df[summary_df['tau_0'] == 0.0], 'Type_I_Error')
    power_df = _aggregate_rate(summary_df[summary_df['tau_0'] == 0.4], 'Power')

    if not type_df.empty or not power_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        panels = [
            (type_df, "Type I Error vs Sample Size", "Type I Error", 0.05),
            (power_df, "Power vs Sample Size", "Power", 0.8),
        ]
        for ax, (df_rate, title, ylabel, ref_line) in zip(axes, panels):
            if df_rate.empty:
                ax.set_visible(False)
                continue
            for method in methods:
                data = df_rate[df_rate['method'] == method].sort_values('n')
                if data.empty:
                    continue
                ax.plot(
                    data['n'], data['value'],
                    color=colors.get(method, 'gray'),
                    marker='o',
                    label=_label(method)
                )
                ax.fill_between(
                    data['n'], data['lower'], data['upper'],
                    color=colors.get(method, 'gray'), alpha=0.2
                )
            ax.set_xlabel('Enrolled Sample Size (n)')
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.axhline(ref_line, color='gray', linestyle='--', linewidth=1)
        handles, labels = [], []
        for axis in axes:
            h, l = axis.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        path_type = os.path.join(config.PLOTS_DIR, 'type_power_curves.pdf')
        fig.savefig(path_type)
        plt.close(fig)
        print(f"? Type I/Power curves saved: {path_type}")

    alloc_df = results_obj.get('allocation_path', pd.DataFrame())
    if not alloc_df.empty:
        alloc_group = (
            alloc_df.groupby(['method', 'sample_size'])
            .agg(
                prop_mean=('prop_treated', 'mean'),
                prop_sd=('prop_treated', 'std'),
                rn_mean=('R_n', 'mean'),
                rn_sd=('R_n', 'std'),
                count=('replicate_id', 'nunique')
            )
            .reset_index()
        )
        alloc_group['prop_sd'] = alloc_group['prop_sd'].fillna(0.0)
        alloc_group['rn_sd'] = alloc_group['rn_sd'].fillna(0.0)
        alloc_group['prop_ci'] = 1.96 * alloc_group['prop_sd'] / np.sqrt(np.maximum(alloc_group['count'], 1))
        alloc_group['rn_ci'] = 1.96 * alloc_group['rn_sd'] / np.sqrt(np.maximum(alloc_group['count'], 1))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
        for method in methods:
            data = alloc_group[alloc_group['method'] == method].sort_values('sample_size')
            if data.empty:
                continue
            label = _label(method)
            color = colors.get(method, 'gray')
            axes[0].plot(data['sample_size'], data['prop_mean'], color=color, label=label)
            axes[0].fill_between(
                data['sample_size'],
                np.clip(data['prop_mean'] - data['prop_ci'], 0.0, 1.0),
                np.clip(data['prop_mean'] + data['prop_ci'], 0.0, 1.0),
                color=color, alpha=0.2
            )
            axes[1].plot(data['sample_size'], data['rn_mean'], color=color, label=label)
            axes[1].fill_between(
                data['sample_size'],
                np.maximum(data['rn_mean'] - data['rn_ci'], 0.0),
                data['rn_mean'] + data['rn_ci'],
                color=color, alpha=0.2
            )
        axes[0].set_xlabel('Enrolled Sample Size (n)')
        axes[0].set_ylabel('Avg. treatment proportion')
        axes[0].set_ylim(0, 1)
        axes[0].set_title('(a) Allocation proportion')
        axes[1].set_xlabel('Enrolled Sample Size (n)')
        axes[1].set_ylabel('Avg. R_n(X)')
        axes[1].set_title('(b) Effective sample size multiplier')
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        alloc_path = os.path.join(config.PLOTS_DIR, 'allocation_dynamics.pdf')
        fig.savefig(alloc_path)
        plt.close(fig)
        print(f"? Allocation dynamics plot saved: {alloc_path}")

    calib_df = results_obj.get('calibration', pd.DataFrame())
    if not calib_df.empty:
        scenario_types = sorted(calib_df['scenario_type'].dropna().unique())
        if not scenario_types:
            scenario_types = ['All']
            calib_df['scenario_type'] = 'All'
        n_types = len(scenario_types)
        ncols = 2
        nrows = int(np.ceil(n_types / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, scenario in enumerate(scenario_types):
            ax = axes.flat[idx]
            subset = calib_df[calib_df['scenario_type'] == scenario]
            if subset.empty:
                ax.set_visible(False)
                continue
            agg = subset.groupby(['method', 'sample_size'], as_index=False).agg({
                'a_mean': 'mean',
                'a_ci_low': 'mean',
                'a_ci_high': 'mean'
            })
            for method in methods:
                data = agg[agg['method'] == method].sort_values('sample_size')
                if data.empty:
                    continue
                label = _label(method)
                color = colors.get(method, 'gray')
                yerr = np.vstack([
                    np.clip(data['a_mean'] - data['a_ci_low'], 0.0, 1.0),
                    np.clip(data['a_ci_high'] - data['a_mean'], 0.0, 1.0)
                ])
                ax.errorbar(
                    data['sample_size'], data['a_mean'],
                    yerr=yerr,
                    fmt='o-', color=color, label=label
                )
            ax.set_title(scenario)
            ax.set_xlabel('Enrolled Sample Size (n)')
            ax.set_ylabel('Average a(x)')
            ax.set_ylim(0, 1)
        total_axes = nrows * ncols
        for idx in range(len(scenario_types), total_axes):
            axes.flat[idx].set_visible(False)
        handles = []
        labels = []
        for method in methods:
            handles.append(Line2D([0], [0], color=colors.get(method, 'gray'), marker='o', label=_label(method)))
            labels.append(_label(method))
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=min(len(handles), 4), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        calib_path = os.path.join(config.PLOTS_DIR, 'calibration_discount.pdf')
        fig.savefig(calib_path)
        plt.close(fig)
        print(f"? Calibration plot saved: {calib_path}")

    print(f"? All plots saved to: {config.PLOTS_DIR}")

# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    print("analysis.py - Results processing module")
    print("This module is typically called from main.py")
    print("\nTo test, run: python main.py")

