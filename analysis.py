"""
analysis.py

Results Analysis and Visualization for CAHB-UIP Simulation Study

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
    CAHB-UIP manuscript Section 3.5 (Evaluation Metrics)
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
        'CAHB_UIP_IPD': '#E69F00',  # Orange
        'CAHB_UIP_SLD': '#D55E00',  # Vermilion
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
    
    estimation_cols = [
        'scenario_name', 'method', 'tau_0', 'n', 'n_h', 'scenario_type', 'kappa',
        'Bias', 'RMSE', 'Coverage', 'CI_Width', 'Alloc_Rate_Treatment',
        'Type_I_Error', 'Power'
    ]
    estimation_cols = [c for c in estimation_cols if c in summary_df.columns]
    estimation_df = summary_df[estimation_cols].copy()
    est_csv = os.path.join(config.TABLES_DIR, 'estimation_metrics.csv')
    estimation_df.to_csv(est_csv, index=False, float_format='%.4f')
    print(f"[+] Estimation metrics table saved: {est_csv}")

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
    """Generate 2x2 grid boxplots for R_n(X) and M(X) diagnostics."""
    alloc_df = results_obj.get('allocation_path', pd.DataFrame())
    if alloc_df.empty:
        warnings.warn("Empty allocation diagnostics. No plots generated.")
        return

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    colors = configure_plot_style()
    method_labels = getattr(config, 'METHOD_LABELS', {})

    scenario_order = [
        ('S1', 1.0, r'S1 $\\kappa$=1.0'),
        ('S2', 0.7, r'S2 $\\kappa$=0.7'),
        ('S2', 1.3, r'S2 $\\kappa$=1.3'),
        ('S3', 0.7, r'S3 $\\kappa$=0.7'),
        ('S3', 1.3, r'S3 $\\kappa$=1.3'),
        ('S4', 1.3, r'S4 $\\kappa$=1.3'),
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

    def _prepare_metric(metric_col: str, method_pool: List[str]) -> pd.DataFrame:
        metric_df = alloc_df.dropna(subset=[metric_col]).copy()
        if metric_df.empty:
            return metric_df
        group_cols = ['n', 'n_h', 'scenario_type', 'kappa', 'tau_0', 'method', 'replicate_id', 'scenario_label']
        metric_df = metric_df.groupby(group_cols, as_index=False)[metric_col].median()
        metric_df['scenario_label'] = pd.Categorical(metric_df['scenario_label'], categories=label_order, ordered=True)
        metric_df = metric_df[metric_df['method'].isin(method_pool)]
        return metric_df

    def _method_handles(methods_for_plot: List[str]) -> List[Line2D]:
        handles = []
        for m in methods_for_plot:
            handles.append(
                Line2D([0], [0], color=colors.get(m, 'gray'), marker='s', linestyle='', label=_label(m))
            )
        return handles

    def _draw_box(ax, data: pd.DataFrame, metric_col: str, methods_for_plot: List[str], ylabel: str):
        if data.empty:
            ax.set_visible(False)
            return []
        method_order = [m for m in methods_for_plot if m in data['method'].unique()]
        if not method_order:
            ax.set_visible(False)
            return []
        palette = {m: colors.get(m, 'gray') for m in method_order}
        sns.boxplot(
            data=data,
            x='scenario_label',
            y=metric_col,
            hue='method',
            order=label_order,
            hue_order=method_order,
            palette=palette,
            ax=ax,
            linewidth=0.8,
            fliersize=2.5,
        )
        ax.set_xlabel('Simulation scenario')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', labelrotation=25)
        if ax.legend_:
            ax.legend_.remove()
        return method_order

    def _combination_order(df: pd.DataFrame) -> List[tuple]:
        preferred = [(200, 400), (200, 800), (400, 400), (400, 800)]
        observed = list({(int(row.n), int(row.n_h)) for row in df[['n', 'n_h']].dropna().itertuples(index=False)})
        combos = [c for c in preferred if c in observed]
        for combo in sorted(observed):
            if combo not in combos:
                combos.append(combo)
        return combos

    def _plot_metric(metric_df: pd.DataFrame, methods_for_plot: List[str], metric_col: str, ylabel: str, prefix: str):
        if metric_df.empty:
            warnings.warn(f"No data available for {metric_col} plotting.")
            return

        combos = _combination_order(metric_df)
        if not combos:
            warnings.warn(f"No (n, n_h) combinations found for {metric_col}.")
            return

        n_panels = len(combos)
        ncols = 2
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=True)
        axes = np.atleast_2d(axes)
        axes_flat = axes.flatten()
        legend_methods = [m for m in methods_for_plot if m in metric_df['method'].unique()]
        for ax, (n_val, nh_val) in zip(axes_flat, combos):
            subset = metric_df[(metric_df['n'] == n_val) & (metric_df['n_h'] == nh_val)]
            _draw_box(ax, subset, metric_col, methods_for_plot, ylabel)
            ax.set_title(rf"n={int(n_val)}, $n_h$={int(nh_val)}")
        for ax in axes_flat[len(combos):]:
            ax.set_visible(False)
        if legend_methods:
            handles = _method_handles(legend_methods)
            fig.legend(handles, [_label(m) for m in legend_methods], loc='lower center', ncol=min(len(handles), 4), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        grid_path = os.path.join(config.PLOTS_DIR, f"{prefix}_grid.pdf")
        fig.savefig(grid_path)
        plt.close(fig)
        print(f"[+] Saved {prefix} grid: {grid_path}")

        for n_val, nh_val in combos:
            subset = metric_df[(metric_df['n'] == n_val) & (metric_df['n_h'] == nh_val)]
            fig_single, ax_single = plt.subplots(figsize=(7, 5))
            used_methods = _draw_box(ax_single, subset, metric_col, methods_for_plot, ylabel)
            if used_methods:
                handles = _method_handles(used_methods)
                fig_single.legend(handles, [_label(m) for m in used_methods], loc='upper right', frameon=False)
            fig_single.tight_layout()
            sub_path = os.path.join(config.PLOTS_DIR, f"{prefix}_n{int(n_val)}_nh{int(nh_val)}.pdf")
            fig_single.savefig(sub_path)
            plt.close(fig_single)
            print(f"[+] Saved subplot: {sub_path}")

    all_methods = getattr(config, 'METHODS_TO_RUN', [])
    rn_methods = [m for m in all_methods if m in alloc_df['method'].unique()]
    m_methods = [m for m in all_methods if m.startswith('CAHB_UIP') and m in alloc_df['method'].unique()]

    rn_df = _prepare_metric('R_n', rn_methods)
    m_df = _prepare_metric('M', m_methods)

    _plot_metric(rn_df, rn_methods, 'R_n', r'Median $R_n(\\mathbf{X})$', 'rn_box')
    _plot_metric(m_df, m_methods, 'M', r'Median $M(\\mathbf{X})$', 'm_box')

    print(f"[+] All plots saved to: {config.PLOTS_DIR}")

# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    print("analysis.py - Results processing module")
    print("This module is typically called from main.py")
    print("\nTo test, run: python main.py")

