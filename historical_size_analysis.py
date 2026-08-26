"""Analysis for the paired historical-sample-size sensitivity study."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from analysis import (
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_ORDER,
    PUB_RC,
    SCENARIO_DISPLAY,
)


plt.rcParams.update(PUB_RC)


def _prepare(df):
    out = df.copy()
    out["error"] = out["estimated_delta"] - out["true_delta"]
    out["sq_error"] = out["error"] ** 2
    out["coverage"] = (
        (out["ci_low"] <= out["true_delta"])
        & (out["true_delta"] <= out["ci_high"])
    ).astype(float)
    out["width"] = out["ci_high"] - out["ci_low"]
    return out


def _paired_mean(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    a = np.asarray(a)[mask]
    b = np.asarray(b)[mask]
    if len(a) == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    diff = b - a
    estimate = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else np.nan
    return estimate, se, estimate - 1.96 * se, estimate + 1.96 * se, len(diff)


def _paired_rmse(sq_a, sq_b):
    mask = np.isfinite(sq_a) & np.isfinite(sq_b)
    sq_a = np.asarray(sq_a)[mask]
    sq_b = np.asarray(sq_b)[mask]
    mse_a = float(np.mean(sq_a))
    mse_b = float(np.mean(sq_b))
    rmse_a = np.sqrt(mse_a)
    rmse_b = np.sqrt(mse_b)
    delta = rmse_b - rmse_a
    if len(sq_a) > 1 and rmse_a > 0 and rmse_b > 0:
        influence = ((sq_b - mse_b) / (2.0 * rmse_b)
                     - (sq_a - mse_a) / (2.0 * rmse_a))
        se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    else:
        se = np.nan
    return rmse_a, rmse_b, delta, se, delta - 1.96 * se, delta + 1.96 * se


def paired_history_size_comparison(df):
    data = _prepare(df)
    sizes = sorted(int(x) for x in data["n_historical"].unique())
    if len(sizes) != 2:
        raise ValueError("historical-size analysis currently expects exactly two N_H values")
    n_low, n_high = sizes
    keys = ["scenario", "effect_type", "method", "rep"]
    low = data[data.n_historical == n_low]
    high = data[data.n_historical == n_high]
    paired = low.merge(high, on=keys, suffixes=("_low", "_high"), validate="one_to_one")

    rows = []
    mean_metrics = {
        "Bias": "error",
        "Rejection": "rejected",
        "Coverage": "coverage",
        "Width": "width",
        "Allocation": "allocation_ratio",
        "Mean_W": "mean_W",
        "Mean_Rn": "mean_Rn",
        "Mean_Dpdc": "mean_Dpdc",
    }
    for (scenario, effect, method), group in paired.groupby(
            ["scenario", "effect_type", "method"], sort=False):
        row = {
            "Scenario": scenario,
            "Effect": effect,
            "Method": method,
            "N_H_Low": n_low,
            "N_H_High": n_high,
            "N_Paired": len(group),
        }
        rmse_low, rmse_high, delta, se, ci_low, ci_high = _paired_rmse(
            group["sq_error_low"].to_numpy(), group["sq_error_high"].to_numpy())
        row.update({
            f"RMSE_NH{n_low}": rmse_low,
            f"RMSE_NH{n_high}": rmse_high,
            "Delta_RMSE": delta,
            "SE_Delta_RMSE": se,
            "CI95_L_Delta_RMSE": ci_low,
            "CI95_U_Delta_RMSE": ci_high,
        })
        for label, column in mean_metrics.items():
            a = group[f"{column}_low"].to_numpy(dtype=float)
            b = group[f"{column}_high"].to_numpy(dtype=float)
            change, change_se, change_l, change_u, n_pair = _paired_mean(a, b)
            finite_a = a[np.isfinite(a)]
            finite_b = b[np.isfinite(b)]
            row.update({
                f"{label}_NH{n_low}": float(np.mean(finite_a)) if len(finite_a) else np.nan,
                f"{label}_NH{n_high}": float(np.mean(finite_b)) if len(finite_b) else np.nan,
                f"Delta_{label}": change,
                f"SE_Delta_{label}": change_se,
                f"CI95_L_Delta_{label}": change_l,
                f"CI95_U_Delta_{label}": change_u,
                f"N_Paired_{label}": n_pair,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def method_contrasts(df):
    """Paired RADISH contrasts; positive gain/reduction favors RADISH."""
    data = _prepare(df)
    rows = []
    keys = ["scenario", "effect_type", "n_historical", "rep"]
    radish = data[data.method == "RADISH"]
    for comparator in ["KBCD", "CAHB"]:
        other = data[data.method == comparator]
        paired = other.merge(radish, on=keys, suffixes=("_other", "_radish"),
                             validate="one_to_one")
        for (scenario, effect, n_historical), group in paired.groupby(
                ["scenario", "effect_type", "n_historical"], sort=False):
            rmse_other, rmse_radish, _, _, _, _ = _paired_rmse(
                group["sq_error_other"].to_numpy(),
                group["sq_error_radish"].to_numpy())
            # Positive values favor RADISH for RMSE and interval width.
            rmse_gain = rmse_other - rmse_radish
            mse_o = np.mean(group["sq_error_other"])
            mse_r = np.mean(group["sq_error_radish"])
            if rmse_other > 0 and rmse_radish > 0:
                influence = ((group["sq_error_other"].to_numpy() - mse_o) / (2 * rmse_other)
                             - (group["sq_error_radish"].to_numpy() - mse_r) / (2 * rmse_radish))
                rmse_gain_se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
            else:
                rmse_gain_se = np.nan
            rejection_gain, rejection_se, rejection_l, rejection_u, _ = _paired_mean(
                group["rejected_other"].to_numpy(dtype=float),
                group["rejected_radish"].to_numpy(dtype=float))
            width_reduction, width_se, width_l, width_u, _ = _paired_mean(
                group["width_radish"].to_numpy(dtype=float),
                group["width_other"].to_numpy(dtype=float))
            coverage_gain, coverage_se, coverage_l, coverage_u, _ = _paired_mean(
                group["coverage_other"].to_numpy(dtype=float),
                group["coverage_radish"].to_numpy(dtype=float))
            rows.append({
                "Scenario": scenario,
                "Effect": effect,
                "N_Historical": int(n_historical),
                "Comparator": comparator,
                "RMSE_Comparator": rmse_other,
                "RMSE_RADISH": rmse_radish,
                "RMSE_Gain_RADISH": rmse_gain,
                "SE_RMSE_Gain": rmse_gain_se,
                "CI95_L_RMSE_Gain": rmse_gain - 1.96 * rmse_gain_se,
                "CI95_U_RMSE_Gain": rmse_gain + 1.96 * rmse_gain_se,
                "Rejection_Gain_RADISH": rejection_gain,
                "SE_Rejection_Gain": rejection_se,
                "CI95_L_Rejection_Gain": rejection_l,
                "CI95_U_Rejection_Gain": rejection_u,
                "Width_Reduction_RADISH": width_reduction,
                "SE_Width_Reduction": width_se,
                "CI95_L_Width_Reduction": width_l,
                "CI95_U_Width_Reduction": width_u,
                "Coverage_Gain_RADISH": coverage_gain,
                "SE_Coverage_Gain": coverage_se,
                "CI95_L_Coverage_Gain": coverage_l,
                "CI95_U_Coverage_Gain": coverage_u,
            })
    return pd.DataFrame(rows)


def advantage_change(df):
    """Change in RADISH's advantage when N_H increases.

    Positive values favor a larger historical cohort for RMSE gain, rejection
    gain under the alternative, width reduction, and coverage gain.
    """
    data = _prepare(df)
    sizes = sorted(int(x) for x in data["n_historical"].unique())
    if len(sizes) != 2:
        raise ValueError("advantage-change analysis expects exactly two N_H values")
    n_low, n_high = sizes
    index = ["scenario", "effect_type", "rep"]
    rows = []
    for comparator in ["KBCD", "CAHB"]:
        cells = {}
        for method in [comparator, "RADISH"]:
            for n_historical in sizes:
                key = f"{method}_{n_historical}"
                cells[key] = data[
                    (data.method == method)
                    & (data.n_historical == n_historical)
                ].set_index(index)
        common = cells[f"{comparator}_{n_low}"].index
        for frame in cells.values():
            common = common.intersection(frame.index)
        for scenario, effect in common.droplevel("rep").unique():
            group_index = [idx for idx in common if idx[0] == scenario and idx[1] == effect]
            c_low = cells[f"{comparator}_{n_low}"].loc[group_index]
            c_high = cells[f"{comparator}_{n_high}"].loc[group_index]
            r_low = cells[f"RADISH_{n_low}"].loc[group_index]
            r_high = cells[f"RADISH_{n_high}"].loc[group_index]

            sq_cl = c_low.sq_error.to_numpy()
            sq_ch = c_high.sq_error.to_numpy()
            sq_rl = r_low.sq_error.to_numpy()
            sq_rh = r_high.sq_error.to_numpy()
            rmse_cl, rmse_ch = np.sqrt(np.mean(sq_cl)), np.sqrt(np.mean(sq_ch))
            rmse_rl, rmse_rh = np.sqrt(np.mean(sq_rl)), np.sqrt(np.mean(sq_rh))
            gain_low = rmse_cl - rmse_rl
            gain_high = rmse_ch - rmse_rh
            delta_rmse_gain = gain_high - gain_low
            influence = (
                (sq_ch - np.mean(sq_ch)) / (2 * rmse_ch)
                - (sq_rh - np.mean(sq_rh)) / (2 * rmse_rh)
                - (sq_cl - np.mean(sq_cl)) / (2 * rmse_cl)
                + (sq_rl - np.mean(sq_rl)) / (2 * rmse_rl)
            )
            se_rmse = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))

            rejection_did = ((r_high.rejected.to_numpy() - c_high.rejected.to_numpy())
                             - (r_low.rejected.to_numpy() - c_low.rejected.to_numpy()))
            width_did = ((c_high.width.to_numpy() - r_high.width.to_numpy())
                         - (c_low.width.to_numpy() - r_low.width.to_numpy()))
            coverage_did = ((r_high.coverage.to_numpy() - c_high.coverage.to_numpy())
                            - (r_low.coverage.to_numpy() - c_low.coverage.to_numpy()))

            def summarize(values):
                estimate = float(np.mean(values))
                se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                return estimate, se, estimate - 1.96 * se, estimate + 1.96 * se

            rej, rej_se, rej_l, rej_u = summarize(rejection_did)
            wid, wid_se, wid_l, wid_u = summarize(width_did)
            cov, cov_se, cov_l, cov_u = summarize(coverage_did)
            rows.append({
                "Scenario": scenario,
                "Effect": effect,
                "Comparator": comparator,
                "N_H_Low": n_low,
                "N_H_High": n_high,
                "N_Paired": len(group_index),
                "RMSE_Gain_Low": gain_low,
                "RMSE_Gain_High": gain_high,
                "Delta_RMSE_Gain": delta_rmse_gain,
                "SE_Delta_RMSE_Gain": se_rmse,
                "CI95_L_Delta_RMSE_Gain": delta_rmse_gain - 1.96 * se_rmse,
                "CI95_U_Delta_RMSE_Gain": delta_rmse_gain + 1.96 * se_rmse,
                "Delta_Rejection_Gain": rej,
                "SE_Delta_Rejection_Gain": rej_se,
                "CI95_L_Delta_Rejection_Gain": rej_l,
                "CI95_U_Delta_Rejection_Gain": rej_u,
                "Delta_Width_Reduction": wid,
                "SE_Delta_Width_Reduction": wid_se,
                "CI95_L_Delta_Width_Reduction": wid_l,
                "CI95_U_Delta_Width_Reduction": wid_u,
                "Delta_Coverage_Gain": cov,
                "SE_Delta_Coverage_Gain": cov_se,
                "CI95_L_Delta_Coverage_Gain": cov_l,
                "CI95_U_Delta_Coverage_Gain": cov_u,
            })
    return pd.DataFrame(rows)


def _plot_sensitivity(comparison, out_path):
    sizes = sorted({int(comparison.N_H_Low.iloc[0]), int(comparison.N_H_High.iloc[0])})
    power = comparison[(comparison.Method == "RADISH") & (comparison.Effect == "Power")].copy()
    scenario_order = list(config.get_mode_scenarios("hist_size"))
    power["ScenarioIndex"] = pd.Categorical(
        power.Scenario, categories=scenario_order, ordered=True)
    power = power.sort_values("ScenarioIndex")
    labels = [SCENARIO_DISPLAY.get(s, s.split("_")[0])
              for s in power.Scenario]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(6.83, 5.25), sharex=True)
    panels = [
        ("RMSE", "RMSE", "(a) RMSE"),
        ("Rejection", "Power", "(b) Power"),
        ("Width", "CI width", "(c) Confidence-interval width"),
        ("Mean_W", "Mean borrowing weight", "(d) Borrowing weight"),
    ]
    for ax, (prefix, ylabel, title) in zip(axes.flat, panels):
        for n_h, marker, linestyle, alpha in zip(
                sizes, ["o", "s"], ["--", "-"], [0.58, 1.0]):
            ax.plot(x, power[f"{prefix}_NH{n_h}"],
                    color=METHOD_COLORS["RADISH"], marker=marker,
                    linestyle=linestyle, alpha=alpha, linewidth=1.8,
                    label=fr"$N_H={n_h}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.25)
        if prefix == "Rejection":
            ax.set_ylim(0.80, 1.01)
        elif prefix == "Mean_W":
            ax.set_ylim(0, 0.92)
    for ax in axes[-1, :]:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Scenario")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_conflict_robustness(comparison, out_path):
    null = comparison[comparison.Effect == "Null"].copy()
    sizes = sorted({int(null.N_H_Low.iloc[0]), int(null.N_H_High.iloc[0])})
    scenarios = ["B3_mdBias_highPrec", "B4_mdBias_lowPrec"]
    scenario_titles = ["S5: precise history", "S6: noisy history"]
    fig, axes = plt.subplots(2, 2, figsize=(6.83, 5.25), sharex="col")
    x = np.arange(len(METHOD_ORDER))
    width = 0.34
    method_colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    hatches = [None, "//"]
    alphas = [0.62, 1.0]

    for col, (scenario, scenario_title) in enumerate(zip(scenarios, scenario_titles)):
        subset = null[null.Scenario == scenario].set_index("Method")
        for row, (prefix, ylabel, reference) in enumerate([
                ("Rejection", "Type I error", 0.05),
                ("Coverage", "Coverage", 0.95)]):
            ax = axes[row, col]
            for j, (n_h, hatch, alpha) in enumerate(zip(sizes, hatches, alphas)):
                values = [subset.loc[m, f"{prefix}_NH{n_h}"] for m in METHOD_ORDER]
                bars = ax.bar(x + (j - 0.5) * width, values, width=width,
                              color=method_colors, alpha=alpha,
                              edgecolor="black", linewidth=0.5,
                              hatch=hatch, label=fr"$N_H={n_h}$")
                ax.bar_label(bars, fmt="%.3f", padding=1.5, fontsize=6.5,
                             rotation=90)
            ax.axhline(reference, color="black", linestyle="--", linewidth=0.8)
            ax.set_ylabel(ylabel)
            panel = chr(ord("a") + row * 2 + col)
            ax.set_title(f"({panel}) {scenario_title}", loc="left")
            ax.yaxis.grid(True, alpha=0.3)
            ax.xaxis.grid(False)
            if row == 0:
                ax.set_ylim(0, 0.38)
            else:
                ax.set_ylim(0.60, 1.01)
            ax.set_xticks(x, METHOD_ORDER)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def run_analysis(df, out_dir):
    out_dir = Path(out_dir)
    table_dir = out_dir / "tables"
    plot_dir = out_dir / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    comparison = paired_history_size_comparison(df)
    contrasts = method_contrasts(df)
    advantage = advantage_change(df)
    comparison.to_csv(table_dir / "paired_history_size_comparison.csv", index=False)
    contrasts.to_csv(table_dir / "radish_method_contrasts.csv", index=False)
    advantage.to_csv(table_dir / "radish_advantage_change.csv", index=False)
    _plot_sensitivity(comparison, plot_dir / "fig_historical_size_sensitivity.pdf")
    _plot_conflict_robustness(
        comparison, plot_dir / "fig_historical_size_robustness.pdf")

    print("  [ok] paired_history_size_comparison.csv")
    print("  [ok] radish_method_contrasts.csv")
    print("  [ok] radish_advantage_change.csv")
    print("  [ok] fig_historical_size_sensitivity.pdf")
    print("  [ok] fig_historical_size_robustness.pdf")
    return comparison, contrasts, advantage


if __name__ == "__main__":
    root = config.get_mode_output_dir("hist_size")
    frame = pd.read_csv(root / "raw_results.csv")
    run_analysis(frame, root)
