"""Generate publication figures and tables for the real-data study."""
from __future__ import annotations

import pickle
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import (
    COL2_W,
    DIV_CMAP,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METHOD_ORDER,
    PUB_RC,
    SEQ_CMAP,
)
from real_data.real_scenarios import (
    ETA_GRID,
    INTERIM_START,
    N_CURRENT,
    N_REPS,
    XI_GRID,
    XI_REGIMES,
    n_H_of_eta,
)

PARAMS_PKL = ROOT / "real_data" / "real_params.pkl"
RUN_CSV = ROOT / "real_data" / "real_run_results.csv"
FIG_DIR = ROOT / "real_data" / "figures"
TABLE1_CSV = ROOT / "real_data" / "table1_scenarios.csv"
TABLE2_CSV = ROOT / "real_data" / "table2_operating_characteristics.csv"
TABLE2_TEX = ROOT / "real_data" / "table2_operating_characteristics.tex"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(PUB_RC)

ALPHA = 0.05
DELTA_POWER = 0.14
SUBGROUP_COLORS = ["#B64342", "#E9A6A1", "#767676", "#0F4D92"]
METHOD_STYLES = {"KBCD": "--", "CAHB": "-", "RADISH": "-"}
def _load_params():
    with PARAMS_PKL.open("rb") as handle:
        return pickle.load(handle)


def _generate_historical(params, xi, eta, seed):
    from real_data.real_run import _gen_historical

    rng = np.random.default_rng(seed)
    return _gen_historical(params, eta, xi, rng)


def summarize_results(raw, alpha=ALPHA):
    """Compute operating characteristics for every design cell."""
    rows = []
    group_cols = ["xi", "eta", "method"]
    for keys, group in raw.groupby(group_cols, sort=True):
        xi, eta, method = keys
        null = group[group["effect"] == "Null"].copy()
        power = group[group["effect"] == "Power"].copy()
        if len(null) != N_REPS or len(power) != N_REPS:
            raise ValueError(f"unexpected replication count in {(xi, eta, method)}")
        if not np.allclose(power["delta_true"], DELTA_POWER):
            raise ValueError("power alternative does not match DELTA_POWER")

        for sample in (null, power):
            sample["se"] = (sample["hi"] - sample["lo"]) / (2 * 1.959964)
            sample["zabs"] = np.abs(sample["est"]) / sample["se"].clip(1e-12)

        critical = float(np.quantile(null["zabs"], 1 - alpha))
        delta = DELTA_POWER
        errors = power["est"] - delta
        abs_errors = np.abs(errors)
        sq_errors = errors**2

        calibrated_power = float((power["zabs"] > critical).mean())
        type1 = float(null["rejected"].mean())
        coverage = float(((power["lo"] <= delta) & (power["hi"] >= delta)).mean())
        mae = float(abs_errors.mean())
        rmse = float(np.sqrt(sq_errors.mean()))
        mse_se = float(sq_errors.std(ddof=1) / np.sqrt(len(sq_errors)))

        rows.append(
            dict(
                xi=float(xi),
                eta=float(eta),
                method=method,
                n_H=int(group["n_H"].iloc[0]),
                n_rep_h0=len(null),
                n_rep_h1=len(power),
                mae_h1=mae,
                mae_h1_se=float(abs_errors.std(ddof=1) / np.sqrt(len(abs_errors))),
                rmse_h1=rmse,
                rmse_h1_se=mse_se / (2 * rmse) if rmse > 0 else 0.0,
                bias_h1=float(errors.mean()),
                bias_h1_se=float(power["est"].std(ddof=1) / np.sqrt(len(power))),
                power=calibrated_power,
                power_se=float(
                    np.sqrt(calibrated_power * (1 - calibrated_power) / len(power))
                ),
                type1_raw=type1,
                type1_se=float(np.sqrt(type1 * (1 - type1) / len(null))),
                coverage_h1=coverage,
                coverage_h1_se=float(
                    np.sqrt(coverage * (1 - coverage) / len(power))
                ),
                alloc_overall=float(power["alloc_overall"].mean()),
                alloc_se=float(
                    power["alloc_overall"].std(ddof=1) / np.sqrt(len(power))
                ),
                mean_W=float(power["mean_W"].mean()),
                mean_W_se=float(power["mean_W"].std(ddof=1) / np.sqrt(len(power))),
            )
        )

    summary = pd.DataFrame(rows)
    baseline = (
        summary[summary["method"] == "KBCD"][["xi", "eta", "rmse_h1"]]
        .rename(columns={"rmse_h1": "rmse_kbcd"})
    )
    summary = summary.merge(baseline, on=["xi", "eta"], how="left", validate="many_to_one")
    summary["ess"] = (summary["rmse_kbcd"] / summary["rmse_h1"]) ** 2
    return summary


def make_T1(params):
    """Write the fitted model and executed simulation design."""
    rows = [
        dict(
            Component="Current outcome model",
            Symbol=r"$Y_C=m_0(X)+\delta Z+\varepsilon_C$",
            Meaning="HORIZON-calibrated response surface",
            Values=(
                f"sigma_C=sqrt(0.11)={np.sqrt(0.11):.4f}; "
                "delta=0 (null) or 0.14 (power)"
            ),
        ),
        dict(
            Component="Historical-control model",
            Symbol=r"$Y_H=m_0(X)+b_\theta+\varepsilon_H$",
            Meaning="Same response surface plus a location discrepancy",
            Values=r"$b_\theta=2\xi\sigma_C$; $\sigma_H=\sigma_C$",
        ),
        dict(
            Component="Subgroup distributions",
            Symbol=r"$P(X_1,X_2)$",
            Meaning="Empirical HORIZON/FIT cohort proportions",
            Values=(
                "current="
                + np.array2string(params["sg_dist_curr"], precision=3)
                + "; history="
                + np.array2string(params["sg_dist_hist"], precision=3)
            ),
        ),
        dict(
            Component="Continuous covariates",
            Symbol=r"$(X_3,X_4)\mid(X_1,X_2)$",
            Meaning="Subgroup-specific KDEs",
            Values="HORIZON KDE for current; FIT KDE for history",
        ),
        dict(
            Component="Conflict grid",
            Symbol=r"$\xi$",
            Meaning=r"$b_\theta=2\xi\sigma_C$",
            Values="{" + ", ".join(f"{x:g}" for x in XI_GRID) + "}",
        ),
        dict(
            Component="Historical-size grid",
            Symbol=r"$\eta$ and $n_H$",
            Meaning="eta changes the historical-control count",
            Values=(
                "eta={"
                + ", ".join(f"{e:g}" for e in ETA_GRID)
                + "}; n_H={"
                + ", ".join(str(n_H_of_eta(e)) for e in ETA_GRID)
                + "}"
            ),
        ),
        dict(
            Component="Trial geometry",
            Symbol=r"$(N,t_{\rm burn})$",
            Meaning="Current trial and equal-randomization burn-in",
            Values=(
                f"N={N_CURRENT}; burn-in={INTERIM_START}; "
                f"{N_REPS} replications per cell"
            ),
        ),
        dict(
            Component="CAHB comparator",
            Symbol=r"$(\gamma,\lambda_2)$",
            Meaning="CAHB-specific tuning from Jin et al. (2023)",
            Values=r"$\gamma=\sqrt{3}$; $\lambda_2=300\log N$",
        ),
    ]
    table = pd.DataFrame(rows)
    table.to_csv(TABLE1_CSV, index=False, encoding="utf-8")
    md = [
        "# Real-data-calibrated simulation design",
        "",
        "| Component | Symbol | Meaning | Values |",
        "|---|---|---|---|",
    ]
    for _, row in table.iterrows():
        md.append("| " + " | ".join(str(row[c]) for c in table.columns) + " |")
    TABLE1_CSV.with_suffix(".md").write_text(
        "\n".join(md) + "\n", encoding="utf-8"
    )
    print(f"[T1] -> {TABLE1_CSV}")


def make_F1(params):
    """Plot representative outcome distributions and the design grid."""
    corners = [
        (0.0, 0.0, "compatible, small history", "#8BCF8B", 11111),
        (0.0, 1.0, "compatible, large history", "#0F4D92", 11112),
        (1.0, 0.0, "severe conflict, small history", "#E9A6A1", 11113),
        (1.0, 1.0, "severe conflict, large history", "#B64342", 11114),
    ]
    samples = []
    for xi, eta, label, color, seed in corners:
        _, outcomes = _generate_historical(params, xi, eta, seed)
        samples.append((label, color, outcomes))

    all_outcomes = np.concatenate([entry[-1] for entry in samples])
    x_grid = np.linspace(
        np.quantile(all_outcomes, 0.002), np.quantile(all_outcomes, 0.998), 400
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.25))
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.90, bottom=0.18, wspace=0.30
    )

    ax = axes[0]
    for label, color, outcomes in samples:
        density = gaussian_kde(outcomes)
        ax.plot(
            x_grid,
            density(x_grid),
            color=color,
            lw=2.0,
            label=f"{label} ($n_H={len(outcomes)}$)",
        )
    ax.set_xlabel("Standardized 24-month hip BMD")
    ax.set_ylabel("Density")
    ax.set_title("(a) Representative historical-control distributions")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", alpha=0.20, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    regime_colors = ("#EAF4EA", "#EAF1F8", "#FFF3DA", "#F7E6E6")
    regime_x = (0.006, 0.08, 0.43, 0.93)
    for (name, lo, hi), color, xpos in zip(XI_REGIMES, regime_colors, regime_x):
        ax.axvspan(lo, hi, color=color, alpha=0.65, zorder=0)
        display_name = name.replace(" conflict", "") if name.startswith(("Moderate", "Severe")) else name
        ax.text(xpos, max(n_H_of_eta(e) for e in ETA_GRID) + 16, display_name,
                ha="center", va="bottom", fontsize=6.8, color="0.30")
    for eta in ETA_GRID:
        n_h = n_H_of_eta(eta)
        ax.scatter(
            XI_GRID,
            [n_h] * len(XI_GRID),
            s=34,
            color="#4A7AB2",
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

    sigma_c = float(np.sqrt(0.11))
    empirical_xi = np.maximum(np.abs(params["bias_per_sg"]) / (2 * sigma_c), 0)
    n_equiv = params["n_h_per_sg"] / np.maximum(
        params["sg_dist_hist"], 1e-12
    )
    for k, (x_value, y_value) in enumerate(zip(empirical_xi, n_equiv), start=1):
        ax.scatter(
            x_value,
            y_value,
            marker="*",
            s=140,
            color=SUBGROUP_COLORS[k - 1],
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )
        ax.annotate(
            f"sg{k}",
            (x_value, y_value),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.set_xscale("symlog", base=2, linthresh=0.01, linscale=0.8)
    ax.set_xlim(-0.0025, 1.08)
    ax.set_ylim(10, 388)
    ax.set_xticks(XI_GRID)
    ax.set_xticklabels([f"{x:g}" for x in XI_GRID], fontsize=8)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.set_yticks([n_H_of_eta(e) for e in ETA_GRID])
    ax.set_xlabel(r"Conflict $\xi$ (base-2 scale for $\xi>0$)")
    ax.set_ylabel(r"Historical controls $n_H$")
    ax.set_title("(b) Conflict-by-history-size design grid")
    ax.grid(alpha=0.16, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    out = FIG_DIR / "F1_scenario_exploratory.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[F1] -> {out}")


def _set_xi_axis(ax):
    """Use base-2 log spacing for positive xi while retaining xi=0."""
    ax.set_xscale("symlog", base=2, linthresh=0.01, linscale=0.8)
    ax.set_xlim(-0.0025, 1.08)
    ax.set_xticks(XI_GRID)
    ax.set_xticklabels([f"{x:g}" for x in XI_GRID], fontsize=7.5)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)


def _regime_bands(ax, label=False):
    colors = ("#EAF4EA", "#EAF1F8", "#FFF3DA", "#F7E6E6")
    label_x = (0.006, 0.08, 0.43, 0.93)
    for (name, lo, hi), color, xpos in zip(XI_REGIMES, colors, label_x):
        ax.axvspan(lo, hi, color=color, alpha=0.62, zorder=0)
        if label:
            display_name = (
                name.replace(" conflict", "")
                if name.startswith(("Moderate", "Severe")) else name
            )
            ax.annotate(
                display_name,
                xy=(xpos, 0.985),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=6.3,
                color="0.28",
            )


def _heatmap_grid(df, value_col):
    return (
        df.pivot_table(index="eta", columns="xi", values=value_col)
        .reindex(index=ETA_GRID, columns=XI_GRID)
        .values
    )


def make_F2(df):
    """Plot the conflict path at the largest historical sample size."""
    d = df[np.isclose(df["eta"], max(ETA_GRID))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(COL2_W * 1.72, 3.05))
    fig.subplots_adjust(
        left=0.065, right=0.99, top=0.82, bottom=0.22, wspace=0.29
    )
    panels = [
        ("mae_h1", "mae_h1_se", "Mean absolute error", "(a) Estimation error"),
        ("power", "power_se", "Size-calibrated power", "(b) Power"),
        (
            "alloc_overall",
            "alloc_se",
            "Proportion assigned to treatment",
            "(c) Allocation ratio",
        ),
    ]
    for j, (col, se_col, ylabel, title) in enumerate(panels):
        ax = axes[j]
        _regime_bands(ax, label=(j == 0))
        for method in METHOD_ORDER:
            dm = d[d["method"] == method].sort_values("xi")
            x = dm["xi"].to_numpy(float)
            y = dm[col].to_numpy(float)
            se = dm[se_col].to_numpy(float)
            lo, hi = y - 1.96 * se, y + 1.96 * se
            if col in {"power", "alloc_overall"}:
                lo, hi = np.clip(lo, 0.0, 1.0), np.clip(hi, 0.0, 1.0)
            ax.fill_between(
                x, lo, hi, color=METHOD_COLORS[method], alpha=0.12,
                linewidth=0, zorder=1,
            )
            ax.plot(
                x, y, color=METHOD_COLORS[method],
                linestyle=METHOD_STYLES[method],
                marker=METHOD_MARKERS[method], linewidth=1.8,
                markersize=4.8, zorder=3,
            )
        if col == "alloc_overall":
            ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1.0)
        _set_xi_axis(ax)
        ax.set_xlabel(r"Conflict $\xi$ (base-2 scale for $\xi>0$)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [
        Line2D(
            [0], [0], color=METHOD_COLORS[m], linestyle=METHOD_STYLES[m],
            marker=METHOD_MARKERS[m], linewidth=1.8, markersize=5,
            label=METHOD_LABELS[m],
        )
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=3,
        bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5,
    )
    out = FIG_DIR / "F2_marginal_effects.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[F2] -> {out}")


def make_F3(df):
    """Plot full conflict-by-history-size diagnostics for borrowing methods."""
    methods = ["CAHB", "RADISH"]
    nx, ny = len(XI_GRID), len(ETA_GRID)
    extent = [-0.5, nx - 0.5, -0.5, ny - 0.5]
    xtick = [f"{x:g}" for x in XI_GRID]
    ytick = [f"$\\eta$={e:g}\n($n_H$={n_H_of_eta(e)})" for e in ETA_GRID]
    keys = ("ess", "type1_raw", "mean_W")
    grids = {
        (method, key): _heatmap_grid(df[df["method"] == method], key)
        for method in methods for key in keys
    }
    ess_dev = max(
        np.nanmax(np.abs(grids[(method, "ess")] - 1.0))
        for method in methods
    )
    size_dev = max(
        np.nanmax(np.abs(grids[(method, "type1_raw")] - 0.05))
        for method in methods
    )
    w_max = max(np.nanmax(grids[(method, "mean_W")]) for method in methods)
    ess_dev, size_dev, w_max = (
        max(ess_dev, 0.05), max(size_dev, 0.01), max(w_max, 0.05)
    )
    columns = [
        (
            "ess", "Relative efficiency vs KBCD", DIV_CMAP,
            TwoSlopeNorm(
                vmin=max(0.0, 1.0 - ess_dev), vcenter=1.0,
                vmax=1.0 + ess_dev,
            ),
            "{:.2f}", r"$(\mathrm{RMSE}_{\mathrm{KBCD}}/\mathrm{RMSE})^2$",
        ),
        (
            "type1_raw", "Type I error", DIV_CMAP,
            TwoSlopeNorm(
                vmin=max(0.0, 0.05 - size_dev), vcenter=0.05,
                vmax=0.05 + size_dev,
            ),
            "{:.3f}", r"Rejection rate under $H_0$",
        ),
        (
            "mean_W", r"Mean borrowing weight $\bar W$", SEQ_CMAP,
            plt.Normalize(vmin=0.0, vmax=w_max), "{:.2f}", r"$\bar W$",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(COL2_W * 1.68, 6.45))
    fig.subplots_adjust(
        left=0.11, right=0.93, top=0.93, bottom=0.09,
        wspace=0.27, hspace=0.24,
    )
    images = []
    for row, method in enumerate(methods):
        for col, (key, title, cmap, norm, fmt, cbar_label) in enumerate(columns):
            ax = axes[row, col]
            grid = grids[(method, key)]
            image = ax.imshow(
                grid, origin="lower", cmap=cmap, norm=norm,
                aspect="auto", extent=extent,
            )
            images.append((col, image, cbar_label))
            cmap_obj = plt.get_cmap(cmap)
            for i in range(ny):
                for k in range(nx):
                    value = grid[i, k]
                    if not np.isfinite(value):
                        continue
                    rgba = cmap_obj(norm(value))
                    luminance = (
                        0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    )
                    ax.text(
                        k, i, fmt.format(value), ha="center", va="center",
                        fontsize=6.7,
                        color="white" if luminance < 0.52 else "black",
                    )
            ax.set_xticks(np.arange(nx))
            ax.set_xticklabels(xtick, fontsize=7.5)
            ax.set_yticks(np.arange(ny))
            ax.set_yticklabels(ytick if col == 0 else [], fontsize=7.3)
            if row == 1:
                ax.set_xlabel(r"Conflict $\xi$ (combined grid)")
            if col == 0:
                ax.set_ylabel(
                    f"{method}\nHistorical precision",
                    color=METHOD_COLORS[method], fontsize=9.5,
                )
            if row == 0:
                ax.set_title(title, fontsize=10)
    for col in range(3):
        image, label = images[col][1], images[col][2]
        cbar = fig.colorbar(
            image, ax=axes[:, col], fraction=0.035, pad=0.025,
            location="right",
        )
        cbar.set_label(label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    out = FIG_DIR / "F3_xi_eta_interaction.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[F3] -> {out}")


def make_T2(summary):
    """Write representative points from all four conflict regimes."""
    anchors = [
        (0.0, "Exact compatibility"),
        (0.08, "Low conflict"),
        (0.64, "Moderate conflict"),
        (1.0, "Severe conflict"),
    ]
    rows = []
    for xi, regime in anchors:
        cell = summary[
            np.isclose(summary["xi"], xi) & np.isclose(summary["eta"], 1.0)
        ]
        for method in METHOD_ORDER:
            row = cell[cell["method"] == method].iloc[0]
            rows.append(
                dict(
                    Regime=regime,
                    xi=xi,
                    n_H=int(row["n_H"]),
                    Method=method,
                    MAE=row["mae_h1"],
                    RMSE=row["rmse_h1"],
                    Calibrated_power=row["power"],
                    Type_I_error=row["type1_raw"],
                    Coverage=row["coverage_h1"],
                    Allocation=row["alloc_overall"],
                    Mean_W=row["mean_W"],
                    Replications=int(row["n_rep_h1"]),
                )
            )
    table = pd.DataFrame(rows)
    table.to_csv(TABLE2_CSV, index=False, float_format="%.4f")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Real-data-calibrated operating characteristics at "
            r"$n_H=350$ for representative conflict levels.}"
        ),
        r"\label{tab:real_operating}",
        r"\small",
        r"\begin{tabular*}{0.96\linewidth}{@{\extracolsep{\fill}}llrrrrrrr@{}}",
        r"\toprule",
        (
            r"Regime & Method & MAE & RMSE & Power & Type I & Coverage & "
            r"Allocation & $\overline W$ \\"
        ),
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Regime']} & {row['Method']} & {row['MAE']:.3f} & "
            f"{row['RMSE']:.3f} & {row['Calibrated_power']:.3f} & "
            f"{row['Type_I_error']:.3f} & {row['Coverage']:.3f} & "
            f"{row['Allocation']:.3f} & {row['Mean_W']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table}"])
    TABLE2_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[T2] -> {TABLE2_CSV}")
    print(f"[T2] -> {TABLE2_TEX}")


def main():
    params = _load_params()
    raw = pd.read_csv(RUN_CSV)
    summary = summarize_results(raw)
    make_T1(params)
    make_F1(params)
    make_F2(summary)
    make_F3(summary)
    make_T2(summary)


if __name__ == "__main__":
    main()
