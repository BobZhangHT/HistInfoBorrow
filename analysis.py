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
        'CAHB_PP': '#E69F00',      # Orange
        'CAHB': '#56B4E9',          # Sky blue
        'KBCD': '#009E73',          # Bluish green
        'rMAP_KBCD': '#CC79A7'      # Reddish purple
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

def process_results(results_list: list) -> pd.DataFrame:
    """
    Processes raw simulation results into aggregated performance metrics.
    
    Takes the list of individual replicate results and computes:
        - Bias, RMSE, coverage for each scenario × method combination
        - Type I error rates (when tau_0 = 0)
        - Power (when tau_0 = 0.4)
        - Allocation rates
    
    Args:
        results_list: List of dictionaries, one per simulation replicate
                      Each dict contains: delta_hat, ci_low, ci_high, prob_gt_0, etc.
    
    Returns:
        DataFrame with aggregated metrics indexed by (scenario, method)
    
    Notes:
        - Filters out failed replicates (NaN results)
        - Computes summary statistics across replicates
        - Handles missing data gracefully
    """
    if not results_list:
        warnings.warn("No results to process. Returning empty DataFrame.")
        return pd.DataFrame()

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(results_list)
    
    # Remove failed replicates
    n_total = len(df)
    df = df[df['delta_hat'].notna()]
    n_valid = len(df)
    if n_valid < n_total:
        warnings.warn(f"Removed {n_total - n_valid} failed replicates ({100*(n_total-n_valid)/n_total:.1f}%)")
    
    if df.empty:
        warnings.warn("All replicates failed. Cannot compute metrics.")
        return pd.DataFrame()
    
    # Calculate true treatment effect for each row
    df['true_delta'] = df['tau_0'].apply(get_true_delta)
    
    # ==== Performance Metrics ====
    
    # Bias: E[delta_hat - delta_true]
    df['error'] = df['delta_hat'] - df['true_delta']
    df['abs_error'] = np.abs(df['error'])
    df['sq_error'] = df['error'] ** 2
    
    # Coverage: I(delta_true in CI)
    df['coverage'] = ((df['true_delta'] >= df['ci_low']) & 
                      (df['true_delta'] <= df['ci_high']))
    
    # CI Width
    df['ci_width'] = df['ci_high'] - df['ci_low']
    
    # Decision: Success if P(delta > 0 | Data) > threshold
    threshold = config.DECISION_THRESHOLD
    df['is_success'] = df['prob_gt_0'] > threshold
    
    # Allocation rate to treatment
    df['alloc_rate_treatment'] = df['n_treated'] / df['n_total']
    
    # ==== Aggregate by Scenario and Method ====
    
    groupby_cols = ['scenario_name', 'method', 'tau_0', 'n', 'n_h', 'scenario_type']
    
    # Remove scenario_type if not present (for backwards compatibility)
    groupby_cols = [col for col in groupby_cols if col in df.columns]
    
    agg_funcs = {
        'error': 'mean',                           # Bias
        'abs_error': 'mean',                       # MAE
        'sq_error': lambda x: np.sqrt(np.mean(x)), # RMSE
        'coverage': 'mean',                        # Coverage probability
        'ci_width': 'mean',                        # Average CI width
        'alloc_rate_treatment': 'mean',            # Average allocation rate
        'is_success': 'mean',                      # Success rate
        'n_total': 'mean',                         # Verify sample size
        'replicate_id': 'count'                    # Number of replicates
    }
    
    summary = df.groupby(groupby_cols, as_index=False).agg(agg_funcs)
    
    # Rename columns for clarity
    summary = summary.rename(columns={
        'error': 'Bias',
        'abs_error': 'MAE',
        'sq_error': 'RMSE',
        'coverage': 'Coverage',
        'ci_width': 'CI_Width',
        'alloc_rate_treatment': 'Alloc_Rate_Treatment',
        'is_success': 'Success_Rate',
        'replicate_id': 'N_Replicates'
    })
    
    # ==== Compute Type I Error and Power ====
    
    # Type I Error: tau_0 = 0 (null hypothesis true)
    type_i_df = df[df['tau_0'] == 0.0].groupby(['scenario_name', 'method'], as_index=False)['is_success'].mean()
    type_i_df = type_i_df.rename(columns={'is_success': 'Type_I_Error'})
    
    # Power: tau_0 = 0.4 (alternative hypothesis true)
    power_df = df[df['tau_0'] == 0.4].groupby(['scenario_name', 'method'], as_index=False)['is_success'].mean()
    power_df = power_df.rename(columns={'is_success': 'Power'})
    
    # Merge Type I Error and Power into summary
    summary = pd.merge(summary, type_i_df, on=['scenario_name', 'method'], how='left')
    summary = pd.merge(summary, power_df, on=['scenario_name', 'method'], how='left')
    
    # ==== Final Cleanup ====
    
    # Sort by scenario and method
    summary = summary.sort_values(['scenario_name', 'method']).reset_index(drop=True)
    
    return summary


# =============================================================================
# Table Generation
# =============================================================================

def generate_tables(summary_df: pd.DataFrame):
    """
    Generates publication-ready tables in CSV and LaTeX formats.
    
    Creates:
        1. simulation_summary.csv: Full results table
        2. simulation_summary.tex: LaTeX table for manuscript
        3. type_i_error_power.tex: Focused table for error/power results
    
    Args:
        summary_df: Aggregated results DataFrame from process_results()
    
    Output Files:
        - results/tables/simulation_summary.csv
        - results/tables/simulation_summary.tex
        - results/tables/type_i_error_power.tex
    """
    if summary_df.empty:
        warnings.warn("Empty summary DataFrame. No tables generated.")
        return
        
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    
    # ==== CSV Output (Full Table) ====
    csv_path = os.path.join(config.TABLES_DIR, 'simulation_summary.csv')
    summary_df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"✓ Full summary table saved: {csv_path}")
    
    # ==== LaTeX Output (Main Results) ====
    
    # Select key columns for main table
    main_cols = ['scenario_name', 'method', 'tau_0', 'Bias', 'RMSE', 
                 'Coverage', 'Type_I_Error', 'Power', 'Alloc_Rate_Treatment']
    
    # Filter to existing columns
    main_cols = [col for col in main_cols if col in summary_df.columns]
    
    tex_df = summary_df[main_cols].copy()
    
    # Create multi-index for better LaTeX formatting
    try:
        # Pivot to get methods as columns
        tex_pivot = tex_df.pivot_table(
            index=['scenario_name', 'tau_0'],
            columns='method',
            values=['Bias', 'RMSE', 'Coverage', 'Type_I_Error', 'Power']
        )
        
        # Reorder method columns according to config
        if hasattr(config, 'METHODS_TO_RUN'):
            tex_pivot = tex_pivot.reindex(columns=config.METHODS_TO_RUN, level='method')
        
        # Save LaTeX table
        tex_path = os.path.join(config.TABLES_DIR, 'simulation_summary.tex')
        with open(tex_path, 'w') as f:
            f.write("% LaTeX table for CAHB-PP simulation results\n")
            f.write("% Generated automatically by analysis.py\n\n")
            tex_pivot.to_latex(
                f, 
                float_format='%.3f',
                na_rep='-',
                bold_rows=True,
                multicolumn_format='c',
                escape=False,
                column_format='ll' + 'r' * (len(tex_pivot.columns))
            )
        print(f"✓ LaTeX summary table saved: {tex_path}")
        
    except Exception as e:
        # Fallback: save flat table if pivot fails
        warnings.warn(f"Could not create pivot table for LaTeX: {e}. Saving flat table.")
        tex_path = os.path.join(config.TABLES_DIR, 'simulation_summary.tex')
        tex_df.to_latex(
            tex_path, 
            index=False, 
            float_format='%.3f',
            na_rep='-',
            escape=False
        )
        print(f"✓ LaTeX summary table (flat) saved: {tex_path}")
    
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

def generate_plots(summary_df: pd.DataFrame):
    """
    Generates publication-quality PDF plots for manuscript.
    
    Creates the following plots:
        1. RMSE vs sample size (by scenario)
        2. Power vs sample size (by scenario)
        3. Type I Error comparison (bar plot)
        4. Coverage probability (by scenario)
        5. Allocation rate to treatment (by method)
    
    All plots use:
        - Colorblind-friendly palettes
        - 300 DPI resolution
        - PDF vector format
        - Times New Roman font
    
    Args:
        summary_df: Aggregated results DataFrame
    
    Output Files:
        - results/plots/rmse_vs_n.pdf
        - results/plots/power_vs_n.pdf
        - results/plots/type_i_error.pdf
        - results/plots/coverage.pdf
        - results/plots/allocation_rate.pdf
    """
    if summary_df.empty:
        warnings.warn("Empty summary DataFrame. No plots generated.")
        return
        
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    # Configure plot style
    colors = configure_plot_style()
    
    # Extract scenario type if available
    if 'scenario_type' in summary_df.columns:
        summary_df['scenario_base'] = summary_df['scenario_type']
    else:
        summary_df['scenario_base'] = summary_df['scenario_name'].str.extract(r'(S\d+)$')[0]
    
    scenarios = sorted(summary_df['scenario_base'].dropna().unique())
    methods = config.METHODS_TO_RUN if hasattr(config, 'METHODS_TO_RUN') else summary_df['method'].unique()
    
    # ==== Plot 1: RMSE vs Sample Size ====
    
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4*len(scenarios), 4), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]
    
    for ax, scenario in zip(axes, scenarios):
        df_s = summary_df[summary_df['scenario_base'] == scenario]
        
        for method in methods:
            df_m = df_s[df_s['method'] == method].sort_values('n')
            
            if df_m.empty:
                continue
            
            for n_h in sorted(df_m['n_h'].unique()):
                df_nh = df_m[df_m['n_h'] == n_h]
                
                label = f"{method} (n_h={n_h})"
                marker = 'o' if n_h == min(df_m['n_h'].unique()) else 's'
                
                ax.plot(
                    df_nh['n'], df_nh['RMSE'],
                    label=label,
                    color=colors.get(method, 'gray'),
                    marker=marker,
                    linestyle='-',
                    linewidth=1.5
                )
        
        ax.set_title(f"Scenario {scenario}", fontweight='bold')
        ax.set_xlabel("Current Sample Size (n)")
        if ax == axes[0]:
            ax.set_ylabel("RMSE")
        ax.grid(True, linestyle=':', alpha=0.6)
    
    # Add legend outside plot area
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), 
               frameon=True, edgecolor='gray')
    fig.suptitle("Root Mean Squared Error vs. Sample Size", fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    
    rmse_path = os.path.join(config.PLOTS_DIR, 'rmse_vs_n.pdf')
    fig.savefig(rmse_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ RMSE plot saved: {rmse_path}")
    
    # ==== Plot 2: Power vs Sample Size ====
    
    power_df = summary_df[summary_df['tau_0'] == 0.4].copy()
    
    if not power_df.empty and 'Power' in power_df.columns:
        fig, axes = plt.subplots(1, len(scenarios), figsize=(4*len(scenarios), 4), sharey=True)
        if len(scenarios) == 1:
            axes = [axes]
        
        for ax, scenario in zip(axes, scenarios):
            df_s = power_df[power_df['scenario_base'] == scenario]
            
            for method in methods:
                df_m = df_s[df_s['method'] == method].sort_values('n')
                
                if df_m.empty:
                    continue
                
                for n_h in sorted(df_m['n_h'].unique()):
                    df_nh = df_m[df_m['n_h'] == n_h]
                    
                    label = f"{method} (n_h={n_h})"
                    marker = 'o' if n_h == min(df_m['n_h'].unique()) else 's'
                    
                    ax.plot(
                        df_nh['n'], df_nh['Power'],
                        label=label,
                        color=colors.get(method, 'gray'),
                        marker=marker,
                        linestyle='-',
                        linewidth=1.5
                    )
            
            ax.set_title(f"Scenario {scenario}", fontweight='bold')
            ax.set_xlabel("Current Sample Size (n)")
            if ax == axes[0]:
                ax.set_ylabel("Power")
            ax.set_ylim([0, 1.05])
            ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (0.8)')
            ax.grid(True, linestyle=':', alpha=0.6)
        
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5),
                   frameon=True, edgecolor='gray')
        fig.suptitle("Statistical Power vs. Sample Size (tau_0 = 0.4)", fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout(rect=[0, 0, 0.85, 1])
        
        power_path = os.path.join(config.PLOTS_DIR, 'power_vs_n.pdf')
        fig.savefig(power_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Power plot saved: {power_path}")
    
    # ==== Plot 3: Type I Error ====
    
    type_i_df = summary_df[summary_df['tau_0'] == 0.0].copy()
    
    if not type_i_df.empty and 'Type_I_Error' in type_i_df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Prepare data for grouped bar plot
        plot_df = type_i_df.groupby(['scenario_base', 'method'])['Type_I_Error'].mean().reset_index()
        
        # Pivot for easier plotting
        plot_pivot = plot_df.pivot(index='scenario_base', columns='method', values='Type_I_Error')
        
        # Reorder methods
        plot_pivot = plot_pivot[methods] if all(m in plot_pivot.columns for m in methods) else plot_pivot
        
        # Create grouped bar plot
        plot_pivot.plot(
            kind='bar',
            ax=ax,
            color=[colors.get(m, 'gray') for m in plot_pivot.columns],
            edgecolor='black',
            linewidth=0.5,
            width=0.8
        )
        
        # Add nominal alpha line
        ax.axhline(y=config.ALPHA, color='red', linestyle='--', linewidth=2, label=f'Nominal α = {config.ALPHA}')
        
        ax.set_title("Type I Error Control by Scenario", fontsize=14, fontweight='bold')
        ax.set_xlabel("Scenario", fontweight='bold')
        ax.set_ylabel("Type I Error Rate", fontweight='bold')
        ax.set_ylim([0, max(config.ALPHA * 2, plot_pivot.max().max() * 1.2)])
        ax.legend(title='Method', frameon=True, edgecolor='gray')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        plt.xticks(rotation=0)
        
        fig.tight_layout()
        
        type_i_path = os.path.join(config.PLOTS_DIR, 'type_i_error.pdf')
        fig.savefig(type_i_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Type I Error plot saved: {type_i_path}")
    
    # ==== Plot 4: Coverage Probability ====
    
    if 'Coverage' in summary_df.columns:
        fig, axes = plt.subplots(1, len(scenarios), figsize=(4*len(scenarios), 4), sharey=True)
        if len(scenarios) == 1:
            axes = [axes]
        
        for ax, scenario in zip(axes, scenarios):
            df_s = summary_df[summary_df['scenario_base'] == scenario]
            
            for method in methods:
                df_m = df_s[df_s['method'] == method].sort_values('n')
                
                if df_m.empty:
                    continue
                
                # Average over n_h for clarity
                df_avg = df_m.groupby('n')['Coverage'].mean().reset_index()
                
                ax.plot(
                    df_avg['n'], df_avg['Coverage'],
                    label=method,
                    color=colors.get(method, 'gray'),
                    marker='o',
                    linestyle='-',
                    linewidth=1.5
                )
            
            ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_title(f"Scenario {scenario}", fontweight='bold')
            ax.set_xlabel("Current Sample Size (n)")
            if ax == axes[0]:
                ax.set_ylabel("Coverage Probability")
            ax.set_ylim([0.85, 1.0])
            ax.grid(True, linestyle=':', alpha=0.6)
        
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.12, 0.5),
                   frameon=True, edgecolor='gray')
        fig.suptitle("95% Confidence Interval Coverage", fontsize=14, fontweight='bold', y=1.02)
        fig.tight_layout(rect=[0, 0, 0.88, 1])
        
        coverage_path = os.path.join(config.PLOTS_DIR, 'coverage.pdf')
        fig.savefig(coverage_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ Coverage plot saved: {coverage_path}")
    
    print(f"✓ All plots saved to: {config.PLOTS_DIR}")


# =============================================================================
# Main Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    print("analysis.py - Results processing module")
    print("This module is typically called from main.py")
    print("\nTo test, run: python main.py")

