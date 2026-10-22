"""Generate figures and tables for the HORIZON-FIT semi-synthetic study."""
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
GLOBAL_LOCAL_CSV = ROOT / "real_data" / "real_global_local_results.csv"
GLOBAL_LOCAL_SUMMARY_CSV = ROOT / "real_data" / "global_local_summary.csv"
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
        if len(null) != len(power) or len(null) == 0:
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
                alloc_x1_0=float(power["alloc_x1_0"].mean()),
                alloc_x1_1=float(power["alloc_x1_1"].mean()),
                alloc_x1_0_se=float(
                    power["alloc_x1_0"].std(ddof=1) / np.sqrt(len(power))
                ),
                alloc_x1_1_se=float(
                    power["alloc_x1_1"].std(ddof=1) / np.sqrt(len(power))
                ),
                mean_W=float(power["mean_W"].mean()),
                mean_W_se=float(power["mean_W"].std(ddof=1) / np.sqrt(len(power))),
                mean_W_x1_0=float(power["mean_W_x1_0"].mean()),
                mean_W_x1_1=float(power["mean_W_x1_1"].mean()),
                mean_W_x1_0_se=float(
                    power["mean_W_x1_0"].std(ddof=1) / np.sqrt(len(power))
                ),
                mean_W_x1_1_se=float(
                    power["mean_W_x1_1"].std(ddof=1) / np.sqrt(len(power))
                ),
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


def summarize_global_local_results(raw, alpha=ALPHA):
    """Summarize the mean-shift by local-heterogeneity diagnostic."""
    rows = []
    for (regime, method), group in raw.groupby(["regime", "method"], sort=False):
        null = group[group["effect"] == "Null"].copy()
        power = group[group["effect"] == "Power"].copy()
        if len(null) != len(power) or len(null) == 0:
            raise ValueError(f"unbalanced null/power replications in {(regime, method)}")
        for sample in (null, power):
            sample["se"] = (sample["hi"] - sample["lo"]) / (2 * 1.959964)
            sample["zabs"] = np.abs(sample["est"]) / sample["se"].clip(1e-12)
        critical = float(np.quantile(null["zabs"], 1 - alpha))
        errors = power["est"] - DELTA_POWER
        abs_errors = np.abs(errors)
        row = dict(
            regime=regime,
            regime_label=group["regime_label"].iloc[0],
            method=method,
            g_sd=float(group["g_sd"].iloc[0]),
            ell_sd=float(group["ell_sd"].iloc[0]),
            b0_sd=float(group["b0_sd"].iloc[0]),
            b1_sd=float(group["b1_sd"].iloc[0]),
            n_H=int(group["n_H"].iloc[0]),
            n_rep=len(null),
            mae=float(abs_errors.mean()),
            mae_se=float(abs_errors.std(ddof=1) / np.sqrt(len(abs_errors))),
            bias=float(errors.mean()),
            bias_se=float(errors.std(ddof=1) / np.sqrt(len(errors))),
            power=float((power["zabs"] > critical).mean()),
            type1=float(null["rejected"].mean()),
            alloc_x1_0=float(power["alloc_x1_0"].mean()),
            alloc_x1_1=float(power["alloc_x1_1"].mean()),
            mean_W_x1_0=float(power["mean_W_x1_0"].mean()),
            mean_W_x1_1=float(power["mean_W_x1_1"].mean()),
        )
        for key in ("alloc_x1_0", "alloc_x1_1", "mean_W_x1_0", "mean_W_x1_1"):
            row[f"{key}_se"] = float(
                power[key].std(ddof=1) / np.sqrt(len(power))
            )
        row["power_se"] = float(
            np.sqrt(row["power"] * (1 - row["power"]) / len(power))
        )
        row["type1_se"] = float(
            np.sqrt(row["type1"] * (1 - row["type1"]) / len(null))
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _global_local_order():
    return ["GW_LW", "GS_LW", "GW_LS", "GS_LS"]


def _global_local_labels(summary):
    base = summary.drop_duplicates("regime").set_index("regime")
    names = ["Weak/weak", "Strong/weak", "Weak/strong", "Strong/strong"]
    return [
        f"{name}\n({base.loc[regime, 'b0_sd']:.2f}, "
        f"{base.loc[regime, 'b1_sd']:.2f})"
        for name, regime in zip(names, _global_local_order())
    ]


def _global_local_factor_labels():
    return [
        "Mean weak\nLocal weak",
        "Mean strong\nLocal weak",
        "Mean weak\nLocal strong",
        "Mean strong\nLocal strong",
    ]


def make_global_local_F4(summary, suffix=""):
    """Plot overall performance across the four conflict structures."""
    order = _global_local_order()
    labels = _global_local_factor_labels()
    x = np.arange(len(order), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.1), sharex=True)
    panels = [
        ("mae", "mae_se", "Mean absolute error", "(a) Estimation error\nLower is better"),
        ("power", "power_se", "Calibrated power (%)", "(b) Calibrated power\nHigher is better"),
        ("type1", "type1_se", "Type I error (%)", "(c) False-positive rate\nTarget: 5%"),
        ("bias", "bias_se", "Bias", "(d) Estimation bias\nCloser to 0 is better"),
    ]
    for ax, (metric, se_metric, ylabel, title) in zip(axes.flat, panels):
        for i_method, method in enumerate(METHOD_ORDER):
                dm = summary[summary["method"] == method].set_index("regime").loc[order]
                y = dm[metric].to_numpy(float)
                se = dm[se_metric].to_numpy(float)
                if metric in {"power", "type1"}:
                    y, se = 100.0 * y, 100.0 * se
                lo, hi = y - 1.96 * se, y + 1.96 * se
                if metric in {"power", "type1"}:
                    lo, hi = np.clip(lo, 0, 100), np.clip(hi, 0, 100)
                xpos = x + 0.15 * (i_method - 1)
                ax.errorbar(xpos, y, yerr=[y - lo, hi - y],
                            color=METHOD_COLORS[method], linestyle="none",
                            marker=METHOD_MARKERS[method], markersize=5,
                            capsize=2, linewidth=1.2)
        if metric == "type1":
            n_rep = int(summary["n_rep"].iloc[0])
            half = 100.0 * 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / n_rep)
            ax.axhspan(max(0, 100 * ALPHA - half), 100 * ALPHA + half,
                       color="0.80", alpha=0.35, zorder=0)
            ax.axhline(100 * ALPHA, color="0.35", linestyle=":", linewidth=1)
            ax.text(0.02, 0.94, "Target = 5%", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.2, color="0.30")
        if metric == "bias":
            ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
            ax.text(0.98, 0.51, "No bias", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.2, color="0.30")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, labels, fontsize=7.2)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1, 0].set_xlabel("Conflict structure")
    axes[1, 1].set_xlabel("Conflict structure")
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linestyle=METHOD_STYLES[m],
               marker=METHOD_MARKERS[m], linewidth=1.7, markersize=5,
               label=METHOD_LABELS[m]) for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.12,
                        wspace=0.27, hspace=0.34)
    out = FIG_DIR / f"F1_global_local{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[global-local F4] -> {out}")


def make_global_local_F5(summary, suffix=""):
    """Show how borrowing and allocation adapt within each X1 region."""
    order = _global_local_order()
    factor_labels = _global_local_factor_labels()
    x = np.arange(len(order), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.25), sharex=False)
    panels = [
        ("mean_W_x1_0", "mean_W_x1_0_se", "Mean final borrowing weight",
         r"(a) Borrowing, $X_1=0$", 0.0),
        ("mean_W_x1_1", "mean_W_x1_1_se", "Mean final borrowing weight",
         r"(b) Borrowing, $X_1=1$", 0.0),
        ("alloc_x1_0", "alloc_x1_0_se", "Proportion assigned to treatment",
         r"(c) Treatment allocation, $X_1=0$", 0.5),
        ("alloc_x1_1", "alloc_x1_1_se", "Proportion assigned to treatment",
         r"(d) Treatment allocation, $X_1=1$", 0.5),
    ]
    for ax, (metric, se_metric, ylabel, title, reference) in zip(axes.flat, panels):
        for i_method, method in enumerate(METHOD_ORDER):
            dm = summary[summary["method"] == method].set_index("regime").loc[order]
            y = dm[metric].to_numpy(float)
            se = dm[se_metric].to_numpy(float)
            xpos = x + 0.15 * (i_method - 1)
            ax.errorbar(
                xpos, y, yerr=1.96 * se,
                color=METHOD_COLORS[method], linestyle="none",
                marker=METHOD_MARKERS[method], linewidth=1.4,
                markersize=5.2, capsize=2, zorder=3,
            )
        ax.axhline(reference, color="0.35", linestyle="--", linewidth=1.1, zorder=0)
        if metric.startswith("mean_W"):
            ax.set_ylim(-0.04, 0.55)
            ax.text(0.98, 0.96, "Dashed: no borrowing", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.7, color="0.35")
        else:
            ax.set_ylim(0.48, 0.63)
            ax.text(0.98, 0.96, "Dashed: equal randomization", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.7, color="0.35")
        ax.set_title(title, fontsize=9.2)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, factor_labels, fontsize=6.7)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5, 0.91,
        r"Local adaptation: compare borrowing and treatment allocation within each covariate region",
        ha="center", va="center", fontsize=8.2,
    )
    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m], linestyle=METHOD_STYLES[m],
               marker=METHOD_MARKERS[m], linewidth=1.7, markersize=5,
               label=METHOD_LABELS[m]) for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.11,
                        wspace=0.27, hspace=0.40)
    out = FIG_DIR / f"F2_global_local{suffix}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[global-local F5] -> {out}")


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
            Symbol=r"$Y_H=m_0(X)+b(X_1)+\varepsilon_H$",
            Meaning="Same response surface; conflict varies with X1",
            Values=(r"$b(X_1)/\sigma_C=g+\ell(X_1-p_1)/"
                    r"\sqrt{p_1(1-p_1)}$; $p_1=P_C(X_1=1)$; "
                    r"$\sigma_H=\sigma_C$"),
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
            Component="Primary conflict grid",
            Symbol=r"$(g,\ell)$",
            Meaning="Mean shift crossed with local heterogeneity",
            Values=r"$g\in\{0,2\}$; $\ell\in\{0.1,1.493\}$",
        ),
        dict(
            Component="Primary historical size",
            Symbol=r"$N_H$",
            Meaning="Fixed to isolate the two conflict axes",
            Values="350",
        ),
        dict(
            Component="Sensitivity grid",
            Symbol=r"$(\xi,N_H)$",
            Meaning="Localized conflict path and historical precision",
            Values=("xi={" + ", ".join(f"{x:g}" for x in XI_GRID)
                    + "}; N_H={" + ", ".join(str(n_H_of_eta(e)) for e in ETA_GRID)
                    + "}"),
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
        "# HORIZON-FIT semi-synthetic simulation design",
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


def make_F1(df):
    """Compare efficiency and calibration across both historical sizes."""
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.1), sharex=True)
    fig.subplots_adjust(
        left=0.09, right=0.985, top=0.88, bottom=0.11,
        wspace=0.26, hspace=0.30,
    )
    eta_order = sorted(df["eta"].unique())
    for col, eta in enumerate(eta_order):
        d = df[np.isclose(df["eta"], eta)]
        n_h = int(d["n_H"].iloc[0])
        for row, (metric, se_metric, ylabel) in enumerate([
            ("ess", None, "Relative efficiency vs KBCD"),
            ("type1_raw", "type1_se", "Type I error"),
        ]):
            ax = axes[row, col]
            _regime_bands(ax, label=(row == 0), log_x=True)
            for method in METHOD_ORDER:
                dm = d[d["method"] == method].sort_values("xi")
                x_raw = dm["xi"].to_numpy(float)
                x = np.where(np.isclose(x_raw, 0.0), 1e-3, x_raw)
                y = dm[metric].to_numpy(float)
                if se_metric is not None:
                    se = dm[se_metric].to_numpy(float)
                    ax.fill_between(
                        x, np.clip(y - 1.96 * se, 0, 1),
                        np.clip(y + 1.96 * se, 0, 1),
                        color=METHOD_COLORS[method], alpha=0.10, linewidth=0,
                    )
                ax.plot(
                    x, y, color=METHOD_COLORS[method],
                    linestyle=METHOD_STYLES[method],
                    marker=METHOD_MARKERS[method], linewidth=1.8,
                    markersize=4.8,
                )
            if metric == "ess":
                ax.axhline(1.0, color="0.35", linestyle=":", linewidth=1.0)
            else:
                mc_halfwidth = 1.96 * np.sqrt(ALPHA * (1 - ALPHA) / N_REPS)
                ax.axhspan(ALPHA - mc_halfwidth, ALPHA + mc_halfwidth,
                           color="0.80", alpha=0.35, zorder=0)
                ax.axhline(ALPHA, color="0.35", linestyle=":", linewidth=1.0)
                ax.set_ylim(0, max(0.22, 1.08 * d[metric].max()))
            if row == 0:
                y_lo, y_hi = ax.get_ylim()
                ax.set_ylim(y_lo, y_hi + 0.14 * (y_hi - y_lo))
            _set_xi_axis(ax)
            ax.set_ylabel(ylabel if col == 0 else "")
            if row == 0:
                ax.set_title(rf"$N_H={n_h}$", fontsize=10)
            else:
                ax.set_xlabel(r"Conflict $\xi$ (log scale)")
            panel_letter = chr(ord("a") + row * len(eta_order) + col)
            ax.text(-0.11, 1.04, f"({panel_letter})", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=9.5,
                    fontweight="bold", clip_on=False)
            ax.grid(axis="y", alpha=0.22, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[m],
               linestyle=METHOD_STYLES[m], marker=METHOD_MARKERS[m],
               linewidth=1.8, markersize=5, label=METHOD_LABELS[m])
        for m in METHOD_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.995), frameon=False, fontsize=8.5)

    out = FIG_DIR / "F1_scenario_exploratory.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[F1] -> {out}")


def _set_xi_axis(ax):
    """Use the common logarithmic display for the localized-conflict path."""
    _set_xi_log_axis(ax)


def _set_xi_log_axis(ax):
    """Use a true log axis, with xi=0 shown at a labeled display anchor."""
    display_grid = np.asarray(
        [1e-3 if np.isclose(x, 0.0) else x for x in XI_GRID], dtype=float
    )
    ax.set_xscale("log", base=10)
    ax.set_xlim(1e-3, 1.08 * max(XI_GRID))
    tick_values = np.asarray([display_grid[0], 0.01, 0.10, 1.0])
    ax.set_xticks(tick_values)
    labels = ["0\ncompatible", "0.01", "0.1", "1"]
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)


def _regime_bands(ax, label=False, log_x=False):
    colors = ("#EAF4EA", "#EAF1F8", "#FFF3DA", "#F7E6E6")
    if log_x:
        label_x = tuple(np.sqrt(max(lo, 1e-3) * hi)
                        for _, lo, hi in XI_REGIMES)
        label_x = (1.12 * label_x[0],) + label_x[1:]
        label_y = (0.985,) * len(XI_REGIMES)
    else:
        label_x = (0.065, 0.16, 0.53, 0.935)
        label_y = (0.985, 0.89, 0.985, 0.985)
    for (name, lo, hi), color, xpos, ypos in zip(
        XI_REGIMES, colors, label_x, label_y
    ):
        plot_lo = max(lo, 1e-3) if log_x else lo
        ax.axvspan(plot_lo, hi, color=color, alpha=0.62, zorder=0)
        if label:
            display_name = (
                "Compatible/\nminimal"
                if log_x and name == "Compatible/minimal"
                else name
            )
            ax.annotate(
                display_name,
                xy=(xpos, ypos),
                xycoords=("data", "axes fraction"),
                ha="center",
                va="top",
                fontsize=5.8 if log_x else 6.0,
                linespacing=0.85,
                color="0.28",
            )


def _heatmap_grid(df, value_col):
    return (
        df.pivot_table(index="eta", columns="xi", values=value_col)
        .reindex(index=ETA_GRID, columns=XI_GRID)
        .values
    )


def make_F2(df):
    """Plot the compatible-to-high localized-conflict path at large n_H."""
    d = df[np.isclose(df["eta"], max(ETA_GRID))].copy()
    fig, axes = plt.subplots(1, 3, figsize=(COL2_W, 3.35))
    fig.subplots_adjust(
        left=0.075, right=0.985, top=0.80, bottom=0.18, wspace=0.34,
    )
    panels = [
        ("mae_h1", "mae_h1_se", "Mean absolute error",
         "(a) Estimation error\nLower is better"),
        ("power", "power_se", "Calibrated power",
         "(b) Calibrated power\nHigher is better"),
        (
            "alloc_overall",
            "alloc_se",
            "Proportion assigned to treatment",
            "(c) Allocation ratio\n0.5 = equal randomization",
        ),
    ]
    for j, (col, se_col, ylabel, title) in enumerate(panels):
        ax = axes.flat[j]
        for method in METHOD_ORDER:
            dm = d[d["method"] == method].sort_values("xi")
            x_raw = dm["xi"].to_numpy(float)
            # xi=0 is plotted at 1e-3 only as a labeled display anchor; the
            # underlying compatible result and all calculations remain xi=0.
            x = np.where(np.isclose(x_raw, 0.0), 1e-3, x_raw)
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
        # Reserve a slim header band for the conflict-regime labels so that
        # they remain visually separate from curves near the upper boundary.
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(y_lo, y_hi + 0.14 * (y_hi - y_lo))
        _regime_bands(ax, label=True, log_x=True)
        _set_xi_log_axis(ax)
        ax.set_xlabel(r"Conflict $\xi$ (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5)
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
    fig, axes = plt.subplots(2, 3, figsize=(COL2_W, 4.80))
    fig.subplots_adjust(
        left=0.11, right=0.93, top=0.91, bottom=0.13,
        wspace=0.30, hspace=0.34,
    )
    images = []
    for row, method in enumerate(methods):
        for col, (key, title, cmap, norm, fmt, cbar_label) in enumerate(columns):
            ax = axes[row, col]
            grid = grids[(method, key)]
            image = ax.pcolormesh(
                np.arange(nx + 1) - 0.5,
                np.arange(ny + 1) - 0.5,
                grid,
                cmap=cmap,
                norm=norm,
                shading="flat",
            )
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])
            ax.set_box_aspect(1.0)
            images.append((col, image, cbar_label))
            ax.set_xticks(np.arange(nx))
            ax.set_xticklabels(xtick, fontsize=6.8, rotation=45, ha="right")
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


def make_FS4(summary):
    """Plot stratum-specific borrowing and allocation along the xi path."""
    d = summary[np.isclose(summary["eta"], 1.0)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.15), sharex=True)
    fig.subplots_adjust(
        left=0.10, right=0.98, top=0.84, bottom=0.12,
        wspace=0.25, hspace=0.30,
    )
    panels = [
        ("mean_W_x1_0", "mean_W_x1_0_se", r"(a) Borrowing weight: $X_1=0$"),
        ("mean_W_x1_1", "mean_W_x1_1_se", r"(b) Borrowing weight: $X_1=1$"),
        ("alloc_x1_0", "alloc_x1_0_se", r"(c) Treatment allocation: $X_1=0$"),
        ("alloc_x1_1", "alloc_x1_1_se", r"(d) Treatment allocation: $X_1=1$"),
    ]
    handles = []
    for ax, (value, se, title) in zip(axes.flat, panels):
        _regime_bands(ax, label=(ax is axes[0, 0] or ax is axes[0, 1]),
                      log_x=True)
        for method in METHOD_ORDER:
            dm = d[d["method"] == method].sort_values("xi")
            x_raw = dm["xi"].to_numpy(float)
            x = np.where(np.isclose(x_raw, 0.0), 1e-3, x_raw)
            y = dm[value].to_numpy(float)
            err = 1.959964 * dm[se].to_numpy(float)
            line = ax.errorbar(
                x, y, yerr=err,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linestyle=METHOD_STYLES[method],
                linewidth=1.5, markersize=4.2, capsize=2.0,
                label=METHOD_LABELS[method], zorder=3,
            )
            if ax is axes[0, 0]:
                handles.append(line)
        if value.startswith("mean_W"):
            ax.set_ylim(-0.025, 0.56)
            ax.axhline(0.0, color="#555555", linewidth=0.7, zorder=1)
            ax.set_ylabel("Mean final borrowing weight")
        else:
            ax.set_ylim(0.475, 0.625)
            ax.axhline(0.5, color="#555555", linewidth=0.9,
                       linestyle=":", zorder=1)
            ax.set_ylabel("Proportion assigned to treatment")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        _set_xi_axis(ax)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    for ax in axes[1]:
        ax.set_xlabel(r"Conflict $\xi$ (log scale)")
    fig.legend(
        handles=handles, loc="upper center", ncol=3,
        bbox_to_anchor=(0.54, 0.99), frameon=False, fontsize=8.5,
    )
    out = FIG_DIR / "FS4_stratum_diagnostics.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FS4] -> {out}")


def make_T2(summary):
    """Write four representative conflict levels at the larger historical size."""
    anchors = [
        (0.0, "Exact compatibility"),
        (0.01, "Low conflict"),
        (0.10, "Medium conflict"),
        (1.0, "High conflict"),
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
            r"\caption{HORIZON--FIT semi-synthetic operating characteristics at "
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


def main(design="xi", input_csv=None):
    if design == "global-local":
        path = GLOBAL_LOCAL_CSV if input_csv is None else Path(input_csv)
        raw = pd.read_csv(path)
        summary = summarize_global_local_results(raw)
        suffix = "" if path == GLOBAL_LOCAL_CSV else "_pilot"
        summary_path = (
            GLOBAL_LOCAL_SUMMARY_CSV if not suffix else
            GLOBAL_LOCAL_SUMMARY_CSV.with_name(
                f"{GLOBAL_LOCAL_SUMMARY_CSV.stem}{suffix}.csv"
            )
        )
        summary.to_csv(summary_path, index=False)
        print(f"[global-local summary] -> {summary_path}")
        make_global_local_F4(summary, suffix=suffix)
        make_global_local_F5(summary, suffix=suffix)
        return
    params = _load_params()
    raw = pd.read_csv(RUN_CSV)
    summary = summarize_results(raw)
    make_T1(params)
    make_F1(summary)
    make_F2(summary)
    make_FS4(summary)
    make_T2(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--design", choices=("xi", "global-local"), default="xi"
    )
    parser.add_argument("--input", type=Path, default=None)
    cli = parser.parse_args()
    main(cli.design, cli.input)
