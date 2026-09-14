"""Publication-quality figures and tables for all simulation modes.

Factorial modes produce allocation, interval, calibrated-power, borrowing, and
precision diagnostics. Dedicated paths handle the local-conflict,
global-by-local, precision, bias, and historical-size studies. Figures are
written as publication-quality vector PDFs.
"""
from __future__ import annotations
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

try:
    import config
except ImportError:
    config = None

# ═══════════════════════════════════════════════════════════════════
# Publication-quality global style
# ═══════════════════════════════════════════════════════════════════
# Publication style is defined here and reused by the other existing analysis
# scripts.  Keeping it in an established module avoids an extra helper entry
# point while preserving one source of truth for all manuscript figures.
COL1_W, COL15_W, COL2_W = 3.27, 5.00, 6.83
METHOD_ORDER = ["KBCD", "CAHB", "RADISH"]
METHOD_COLORS = {
    "KBCD": "#767676",
    "CAHB": "#B64342",
    "RADISH": "#0F4D92",
}
METHOD_MARKERS = {"KBCD": "o", "CAHB": "s", "RADISH": "^"}
METHOD_LABELS = {
    "KBCD": "KBCD (no borrowing)",
    "CAHB": "CAHB",
    "RADISH": "RADISH (proposed)",
}
SEQ_CMAP, DIV_CMAP = "viridis", "RdBu_r"
PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8.5,
    "figure.titlesize": 10,
    "mathtext.fontset": "stix",
    "mathtext.rm": "STIXGeneral",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.5",
    "legend.fancybox": False,
    "legend.borderpad": 0.4,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
}

plt.rcParams.update(PUB_RC)

EFFECT_DISPLAY = {"Null": r"$\Delta = 0$ (Null)", "Power": r"$\Delta = 0.5$ (Power)"}

# Publication labels use one scenario prefix throughout.  Internal keys are
# retained so previously generated result files remain readable.
SCENARIO_DISPLAY = {
    "B1_noBias_highPrec": "S1",
    "B2_noBias_lowPrec": "S2",
    "B3_mdBias_highPrec": "S3",
    "B4_mdBias_lowPrec": "S4",
    "B5_lgBias_highPrec": "S5",
    "B6_lgBias_lowPrec": "S6",
    "L1_lowBias_highPrec": "L1",
    "L2_lowBias_lowPrec": "L2",
}


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="pdf")
    plt.close(fig)


def _fmt_shift(v):
    """Compact %g-style label for a single historical shift."""
    return f"{float(v):g}"


def _b_local_label(meta):
    """Return "(b0,b1)" for a scenario dict that defines local shifts."""
    bl = meta.get("b_local") if isinstance(meta, dict) else None
    try:
        return f"({_fmt_shift(bl[0])},{_fmt_shift(bl[1])})"
    except (TypeError, IndexError, KeyError, ValueError):
        return None


def _scen_short_factorial(s):
    """Three-line scenario label: index, historical shift(s), noise SD.

    The middle line reports the LOCAL stratum-wise shifts whenever the
    scenario defines them, because the primary grid keeps the X1 = 0 stratum
    exactly compatible; the marginal shift b_theta would average that
    structure away.  Scenarios without b_local (the precision and bias grids)
    keep the legacy marginal-shift rendering.
    """
    if s in SCENARIO_DISPLAY:
        return SCENARIO_DISPLAY[s]
    if config:
        pools = (getattr(config, "PRIMARY_SCENARIOS", {}),
                 getattr(config, "DEMO_SCENARIOS", {}), config.SCENARIOS)
        for pool in pools:
            if s in pool:
                sc = pool[s]
                idx = s.split("_")[0]
                loc = _b_local_label(sc)
                if loc is not None:
                    return f"{idx}\n$b={loc}$\n$\\sigma_H={sc['sigma_h']}$"
                return f"{idx}\n$b={sc['b_theta']}$\n$\\sigma_H={sc['sigma_h']}$"
    return s


def _diag_se(frame, name):
    """Standard error of a per-replication diagnostic column (skipna)."""
    v = frame.get(name, pd.Series(dtype=float))
    return v.std(ddof=1, skipna=True) / np.sqrt(max(len(v.dropna()), 1))


def _aggregate(df, scenarios):
    df = df.copy()
    td = df["true_delta"]
    df["coverage"] = ((df.ci_low <= td) & (td <= df.ci_high)).astype(int)
    df["width"] = df.ci_high - df.ci_low
    df["sq_err"] = (df.estimated_delta - td) ** 2
    se = df["width"] / (2 * 1.959964)
    valid_se = np.isfinite(se) & (se != 0)
    df["z_abs"] = np.nan
    df.loc[valid_se, "z_abs"] = (
        np.abs(df.loc[valid_se, "estimated_delta"]) / se.loc[valid_se]
    )

    alpha = config.ALPHA if config is not None else 0.05
    calibrated = {}
    for sc in scenarios:
        for mt in METHOD_ORDER:
            cell = df[(df.scenario == sc) & (df.method == mt)]
            null_z = cell.loc[cell.effect_type == "Null", "z_abs"]
            power_z = cell.loc[cell.effect_type == "Power", "z_abs"]
            null_z = null_z[np.isfinite(null_z)]
            power_z = power_z[np.isfinite(power_z)]
            if null_z.empty or power_z.empty:
                calibrated[(sc, mt)] = (np.nan, np.nan)
                continue
            critical = np.nanquantile(null_z, 1 - alpha)
            rejection = float((power_z > critical).mean())
            rejection_se = float(np.sqrt(
                rejection * (1 - rejection) / len(power_z)
            ))
            calibrated[(sc, mt)] = (rejection, rejection_se)

    rows = []
    for sc in scenarios:
        for ef in ["Null", "Power"]:
            for mt in METHOD_ORDER:
                s = df[(df.scenario == sc) & (df.effect_type == ef) & (df.method == mt)]
                if s.empty: continue
                rejection_calibrated, rejection_calibrated_se = (
                    calibrated[(sc, mt)] if ef == "Power" else (np.nan, np.nan)
                )
                rows.append(dict(
                    Scenario=sc, Effect=ef, Method=mt, n=len(s),
                    Bias=s.estimation_bias.mean(),
                    Bias_se=s.estimation_bias.std(ddof=1)/np.sqrt(len(s)),
                    RMSE=np.sqrt(s.sq_err.mean()),
                    Rejection=s.rejected.mean(),
                    Rejection_se=np.sqrt(s.rejected.mean()*(1-s.rejected.mean())/len(s)),
                    Rejection_ci95=1.96*np.sqrt(s.rejected.mean()*(1-s.rejected.mean())/len(s)),
                    Rejection_calibrated=rejection_calibrated,
                    Rejection_calibrated_se=rejection_calibrated_se,
                    Rejection_calibrated_ci95=1.96*rejection_calibrated_se,
                    Coverage=s.coverage.mean(),
                    Width=s.width.mean(),
                    Width_se=s.width.std(ddof=1)/np.sqrt(len(s)),
                    Alloc=s.allocation_ratio.mean(),
                    Mean_W=s.get("mean_W", pd.Series(dtype=float)).mean(skipna=True),
                    Mean_W_se=_diag_se(s, "mean_W"),
                    # Stratum-wise Stage-III weights (covariate-local borrowing)
                    Mean_W_x1_0=s.get("mean_W_x1_0", pd.Series(dtype=float)).mean(skipna=True),
                    Mean_W_x1_0_se=_diag_se(s, "mean_W_x1_0"),
                    Mean_W_x1_1=s.get("mean_W_x1_1", pd.Series(dtype=float)).mean(skipna=True),
                    Mean_W_x1_1_se=_diag_se(s, "mean_W_x1_1"),
                    Mean_Rn=s.get("mean_Rn", pd.Series(dtype=float)).mean(skipna=True),
                    Mean_Dpdc=s.get("mean_Dpdc", pd.Series(dtype=float)).mean(skipna=True),
                ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Relative efficiency against the concurrent-only baseline, expressed as
    # an equivalent trial size: ESS = (RMSE_KBCD / RMSE_method)^2.  A value of
    # 1.4 means the design attains the accuracy KBCD would need a 40% larger
    # trial to reach; a value below 1 means borrowing has actively cost
    # accuracy.  Plotting this rather than the raw RMSE puts the borrowing
    # gain and the borrowing harm on opposite sides of a single reference
    # line, which the zero-based RMSE scale cannot show.
    base = (out[out.Method == "KBCD"]
            .set_index(["Scenario", "Effect"])["RMSE"])
    key = list(zip(out.Scenario, out.Effect))
    out["RMSE_KBCD"] = [base.get(k, np.nan) for k in key]
    out["ESS"] = (out["RMSE_KBCD"] / out["RMSE"]) ** 2
    return out


# Regime layouts for the retained six-cell subset and primary eight-cell grid.
_REGIME_LAYOUTS = {
    6: [("Compatible", 0, 2), ("Moderate conflict", 2, 4),
        ("Severe conflict", 4, 6)],
    8: [("Compatible", 0, 2), ("Low conflict", 2, 4),
        ("Moderate conflict", 4, 6), ("Severe conflict", 6, 8)],
}


def _regime_bands(ax, scenarios, label=True, names=None):
    """Shade conflict regimes for either the six- or eight-cell grid.

    names optionally overrides the regime captions, which the three-column
    borrowing figure needs because the full captions would run together on a
    panel that is about a third of a page wide.
    """
    regimes = _REGIME_LAYOUTS.get(len(scenarios))
    if regimes is None:
        return
    for k, (name, lo, hi) in enumerate(regimes):
        if names:
            name = names.get(name, name)
        if k % 2 == 1:
            ax.axvspan(lo - 0.5, hi - 0.5, color="0.5", alpha=0.07, zorder=0)
        if lo > 0:
            ax.axvline(lo - 0.5, color="0.75", lw=0.6, ls="-", zorder=1)
        if label:
            ax.annotate(name, xy=((lo + hi) / 2 - 0.5, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.8, color="0.35")


def _regime_bands_at_positions(ax, centers, scenarios, label=True):
    """Shade conflict regimes when scenario centers are not unit-spaced."""
    regimes = _REGIME_LAYOUTS.get(len(scenarios))
    if regimes is None or len(centers) < 2:
        return
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    for k, (name, lo, hi) in enumerate(regimes):
        if k % 2 == 1:
            ax.axvspan(edges[lo], edges[hi], color="0.5", alpha=0.07, zorder=0)
        if lo > 0:
            ax.axvline(edges[lo], color="0.75", lw=0.6, zorder=1)
        if label:
            ax.annotate(name, xy=((centers[lo] + centers[hi - 1]) / 2, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5.8, color="0.35")

def _grouped_bar(ax, agg, scenarios, metric, ylabel, ylim=None,
                 ref_line=None, ref_label=None,
                 short_fn=_scen_short_factorial,
                 err_metric=None, value_fontsize=None, value_fmt="{:.3f}",
                 value_methods=None):
    """Grouped bar chart. Set value_fontsize to a number (e.g. 7) to overlay
    numeric values on bars; default is None (no overlay) for clean figures.
    value_methods restricts the overlay to the named methods, which keeps a
    crowded axis readable when one series is a constant reference."""
    x = np.arange(len(scenarios)); w = 0.27
    for j, mt in enumerate(METHOD_ORDER):
        ms = agg[agg.Method == mt]
        vals, errs = [], []
        for sc in scenarios:
            row = ms[ms.Scenario == sc]
            vals.append(float(row[metric].values[0]) if len(row) else np.nan)
            errs.append(float(row[err_metric].values[0]) if (err_metric and len(row)) else 0.0)
        bars = ax.bar(x + (j-1)*w, vals, w, label=METHOD_LABELS[mt],
                      color=METHOD_COLORS[mt], alpha=0.88,
                      edgecolor="white", linewidth=0.7,
                      yerr=errs if err_metric else None,
                      capsize=2.5, error_kw=dict(ecolor="0.25", lw=0.8))
        if value_fontsize and (value_methods is None or mt in value_methods):
            for bar, v in zip(bars, vals):
                if not np.isfinite(v): continue
                yoff = max(abs(v)*0.012, 0.003)
                # CAHB and RADISH efficiency labels can be nearly equal; give
                # the RADISH label a small vertical offset to prevent overlap.
                stagger = 0.04 if mt == "RADISH" and v >= 0 else 0.0
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (yoff + stagger if v >= 0 else -yoff),
                        value_fmt.format(v),
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=value_fontsize)
    if ref_line is not None:
        ax.axhline(ref_line, color="0.25", ls="--", lw=1.1,
                   label=ref_label or f"{ref_line}")
    ax.set_xticks(x)
    ax.set_xticklabels([short_fn(s) for s in scenarios], fontsize=7.3)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.margins(x=0.02)


def _shared_method_legend(fig, loc="upper center", ncol=3, y=1.02):
    handles = [mpatches.Patch(facecolor=METHOD_COLORS[m], alpha=0.88,
                              label=METHOD_LABELS[m], edgecolor="white") for m in METHOD_ORDER]
    fig.legend(handles=handles, loc=loc, ncol=ncol,
               bbox_to_anchor=(0.5, y), frameon=True)


# ═══════════════════════════════════════════════════════════════════
# CONFLICT-BY-PRECISION FIGURES
# ═══════════════════════════════════════════════════════════════════

def plot_figure1_allocation(df, scen_order, out_dir):
    df = df[df.scenario.isin(scen_order)].copy()
    if df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 3.0), sharey=True)
    n_m = len(METHOD_ORDER); w = 0.30; gap = 0.95

    for ax, eff in zip(axes, ["Null", "Power"]):
        sub = df[df.effect_type == eff]
        positions, data_list, colors = [], [], []
        tick_pos, tick_lab = [], []; xc = 0.0
        for sc in scen_order:
            for j, mt in enumerate(METHOD_ORDER):
                vals = sub[(sub.scenario == sc) & (sub.method == mt)]["allocation_ratio"].dropna()
                if len(vals) == 0: continue
                positions.append(xc + j*(w+0.04))
                data_list.append(vals.values)
                colors.append(METHOD_COLORS[mt])
            tick_pos.append(xc + (n_m-1)*(w+0.04)/2)
            tick_lab.append(_scen_short_factorial(sc))
            xc += n_m*(w+0.04) + gap
        if not data_list: continue
        _regime_bands_at_positions(ax, tick_pos, scen_order)
        bp = ax.boxplot(data_list, positions=positions, widths=w,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=1.4),
                        whiskerprops=dict(lw=0.8),
                        capprops=dict(lw=0.8),
                        boxprops=dict(lw=0.7))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(colors[i]); patch.set_alpha(0.78)
            patch.set_edgecolor("0.25")
        ax.axhline(0.5, color="0.4", ls=":", lw=1.0)
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=7.3)
        ax.set_title(EFFECT_DISPLAY[eff], pad=22)
        ax.set_ylim(0.42, 0.72)
    axes[0].set_ylabel("Allocation Ratio to Treatment Arm")
    _shared_method_legend(fig, y=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, out_dir / "plots" / "fig1_allocation_ratio.pdf")
    print("  [ok] Figure 1: allocation ratio")


def plot_figure3_ci(df, scen_order, out_dir):
    agg = _aggregate(df, scen_order)
    if agg.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.4))
    for col, eff in enumerate(["Null", "Power"]):
        sub = agg[agg.Effect == eff]
        scen = [s for s in scen_order if s in sub.Scenario.values]
        ax = axes[0, col]
        _regime_bands(ax, scen)
        _grouped_bar(ax, sub, scen, "Width", "CI Width",
                     err_metric="Width_se")
        ax.set_title(f"CI Width: {EFFECT_DISPLAY[eff]}", pad=22)
        ax = axes[1, col]
        _regime_bands(ax, scen, label=False)
        _grouped_bar(ax, sub, scen, "Coverage", "Coverage Probability",
                     ref_line=0.95, ref_label="nominal 95%",
                     ylim=(0.55, 1.02))
        ax.set_title(f"Coverage: {EFFECT_DISPLAY[eff]}")
    _shared_method_legend(fig, y=0.96)
    fig.suptitle("Confidence Interval Width and Coverage", y=1.00)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    _save(fig, out_dir / "plots" / "fig3_ci_width_coverage.pdf")
    print("  [ok] Figure 3: CI width / coverage")


def plot_figure4_testing(df, scen_order, out_dir):
    agg = _aggregate(df, scen_order)
    if agg.empty: return
    n_rep = int(agg["n"].max()) if "n" in agg else 1000
    # Binomial 95% reference band for a nominal 0.05 rate at this M.
    half = 1.959964 * np.sqrt(0.05 * 0.95 / max(n_rep, 1))
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 3.1))
    for ax, eff, ttl in zip(axes, ["Null", "Power"],
                            ["Type I Error  ($\\Delta=0$)",
                             "Calibrated Power  ($\\Delta=0.5$)"]):
        sub = agg[agg.Effect == eff]
        scen = [s for s in scen_order if s in sub.Scenario.values]
        _regime_bands(ax, scen)
        if eff == "Null":
            # Shade the band a correctly calibrated test should fall in, so
            # that "inside the band" is a visual judgement rather than an
            # arithmetic one.
            ax.axhspan(0.05 - half, 0.05 + half, color="0.45", alpha=0.13,
                       zorder=0, label="binomial 95% band")
            _grouped_bar(ax, sub, scen, "Rejection", "Rejection rate",
                         ref_line=0.05, ref_label=r"$\alpha = 0.05$",
                         err_metric="Rejection_ci95",
                         ylim=(0, max(0.28, sub.Rejection.max() * 1.18)))
        else:
            # Zoom to the occupied range: on a 0-1 axis the B1 borrowing gain
            # is compressed into a few pixels and reads as "no difference".
            lo = float(np.nanmin(sub.Rejection_calibrated.values))
            _grouped_bar(ax, sub, scen, "Rejection_calibrated", "Power",
                         err_metric="Rejection_calibrated_ci95",
                         ylim=(max(0.0, lo - 0.10), 1.02))
        ax.set_title(ttl, pad=14)
    _shared_method_legend(fig, y=1.07)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "fig4_typeI_power.pdf")
    print("  [ok] Figure 4: type I / power")


# (method, column, fill alpha, hatch) drawn left to right inside each scenario
# group: solid = X1 = 0 stratum, hatched = X1 = 1 stratum.  KBCD is omitted
# because its Stage-III weight is 0/1 by construction.
_LOCAL_WEIGHT_SERIES = [
    ("RADISH", "Mean_W_x1_0", 0.85, None),
    ("RADISH", "Mean_W_x1_1", 0.55, "///"),
    ("CAHB",   "Mean_W_x1_0", 0.85, None),
    ("CAHB",   "Mean_W_x1_1", 0.55, "///"),
]

# Compact regime captions for the three-column borrowing figure.
_REGIME_SHORT = {"Compatible": "Comp.", "Low conflict": "Low",
                 "Moderate conflict": "Mod.", "Severe conflict": "Severe"}


def _local_weight_bars(ax, agg, scenarios, width=0.19):
    """Grouped bars of the mean Stage-III weight within each covariate stratum."""
    x = np.arange(len(scenarios))
    n = len(_LOCAL_WEIGHT_SERIES)
    offs = (np.arange(n) - (n - 1) / 2.0) * (width + 0.02)
    for off, (mt, col, alpha, hatch) in zip(offs, _LOCAL_WEIGHT_SERIES):
        vals = []
        for sc in scenarios:
            row = agg[(agg.Method == mt) & (agg.Scenario == sc)]
            vals.append(float(row[col].values[0])
                        if (len(row) and col in agg.columns) else np.nan)
        ax.bar(x + off, vals, width, color=METHOD_COLORS[mt], alpha=alpha,
               hatch=hatch,
               edgecolor="0.25" if hatch else "white",
               linewidth=0.5 if hatch else 0.7)
    # Headroom above the unit interval so the stratum key never covers a bar.
    ax.set_ylim(0, 1.24)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    # Colour already encodes the method, so the in-panel legend only names the
    # two strata through their fill style.
    strata = [
        mpatches.Patch(facecolor="0.4", alpha=0.85, edgecolor="white",
                       linewidth=0.7, label=r"$X_1=0$"),
        mpatches.Patch(facecolor="0.4", alpha=0.55, edgecolor="0.25",
                       linewidth=0.5, hatch="///", label=r"$X_1=1$"),
    ]
    ax.legend(handles=strata, loc="upper center", ncol=2, fontsize=6.0,
              handlelength=1.0, handletextpad=0.3, borderpad=0.25,
              columnspacing=0.8, bbox_to_anchor=(0.5, 1.0))


def _plot_figure5_mechanism_legacy(df, scen_order, out_dir):
    """Legacy borrowing-mechanism panel retained for reproducibility.

    Panel (a) is the effective information gain, panel (b) the covariate-local
    borrowing profile -- the mean Stage-III weight inside each stratum, which
    is where the conflict actually lives -- and panel (c) the conflict index
    that drives the local decline.  The single mean weight is reported in the
    operating-characteristics table, where it can be read against the
    stratum-wise values plotted here.
    """
    agg = _aggregate(df, scen_order)
    if agg.empty: return
    # use Power runs (richer signal); pattern is similar under Null
    sub = agg[agg.Effect == "Power"]
    scen = [s for s in scen_order if s in sub.Scenario.values]

    fig, axes = plt.subplots(1, 3, figsize=(COL2_W, 2.9))

    # (a) Effective information gain.
    ax = axes[0]
    _regime_bands(ax, scen, names=_REGIME_SHORT)
    _grouped_bar(ax, sub, scen, "Mean_Rn",
                 r"Mean information ratio $\overline{R_n(x)}$",
                 ref_line=1.0, ref_label=r"$R_n=1$ (no borrowing)",
                 ylim=(0.95, max(1.6, sub.Mean_Rn.max()*1.10)))
    ax.set_ylabel(r"Mean information ratio $\overline{R_n(x)}$", fontsize=7.6)
    ax.set_title(r"Effective information gain", pad=12, fontsize=8)
    ax.tick_params(axis="both", labelsize=6.8)

    # (b) Covariate-local borrowing profile.
    ax = axes[1]
    _regime_bands(ax, scen, names=_REGIME_SHORT)
    _local_weight_bars(ax, sub, scen)
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels([_scen_short_factorial(s) for s in scen], fontsize=6.8)
    ax.set_ylabel(r"Mean borrowing weight $\overline{W}$", fontsize=7.6)
    ax.set_title("Local borrowing profile", pad=12, fontsize=8)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.margins(x=0.02)

    # (c) Conflict index.  Defined for RADISH only (CAHB has no HCC statistic
    # and KBCD never borrows), so this panel carries a single series.
    ax = axes[2]
    _regime_bands(ax, scen, label=False)
    r = sub[sub.Method == "RADISH"]
    vals = [float(r[r.Scenario == s]["Mean_Dpdc"].values[0]) if len(r[r.Scenario == s])
            else np.nan for s in scen]
    ax.bar(np.arange(len(scen)), vals, 0.55, color=METHOD_COLORS["RADISH"],
           alpha=0.88, edgecolor="white", linewidth=0.7)
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v * 1.12, f"{v:.1f}", ha="center", va="bottom", fontsize=6.0)
    ax.set_yscale("log")
    vmax = float(np.nanmax(vals)) if np.isfinite(vals).any() else np.nan
    if np.isfinite(vmax) and vmax > 0:
        ax.set_ylim(top=vmax * 2.6)
    ax.set_xticks(np.arange(len(scen)))
    ax.set_xticklabels([_scen_short_factorial(s) for s in scen], fontsize=6.8)
    ax.set_ylabel(r"Mean conflict index $\overline{D_{\mathrm{HCC}}(x)}$",
                  fontsize=7.6)
    ax.set_title(r"Conflict detection (RADISH)", pad=12, fontsize=8)
    ax.tick_params(axis="y", labelsize=6.8)
    ax.margins(x=0.02)

    _shared_method_legend(fig, y=1.06)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "fig5_borrowing_diagnostics.pdf")
    print("  [ok] Figure 5: borrowing diagnostics")


def plot_figure5_borrowing_diagnostics(df, scen_order, out_dir):
    """CAHB-style operating-characteristic display for local borrowing.

    The precise-history path isolates increasing covariate-dependent conflict.
    Outcome panels show the consequence; stratum panels show where borrowing
    and allocation change.
    """
    scenario_meta = {}
    if config:
        candidates = (
            config.SCENARIOS,
            getattr(config, "LOCAL_PATH_SCENARIOS", {}),
            config.get_mode_scenarios("full"),
        )
        scenario_meta = next(
            (pool for pool in candidates if all(sc in pool for sc in scen_order)),
            {},
        )
    precise = []
    for sc in scen_order:
        meta = scenario_meta.get(sc, {})
        if meta.get("sigma_h") == 0.5:
            precise.append(sc)
    if not precise:
        return
    agg = _aggregate(df, precise)
    power = agg[agg.Effect == "Power"]
    x = np.array([scenario_meta[s]["b_local"][1] for s in precise])

    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.15), sharex=True)

    ax = axes[0, 0]
    for mt in METHOD_ORDER:
        vals, errs = [], []
        for sc in precise:
            s = df[(df.scenario == sc) & (df.effect_type == "Power") &
                   (df.method == mt)]
            ae = np.abs(s.estimated_delta - s.true_delta)
            vals.append(ae.mean())
            errs.append(1.96 * ae.std(ddof=1) / np.sqrt(max(len(ae), 1)))
        ax.errorbar(x, vals, yerr=errs, color=METHOD_COLORS[mt],
                    marker=METHOD_MARKERS[mt], linestyle="-", capsize=2,
                    label=METHOD_LABELS[mt])
    ax.set_ylabel(r"Mean absolute error $|\widehat\Delta-\Delta|$")

    ax = axes[0, 1]
    for mt in METHOD_ORDER:
        r = power[power.Method == mt].set_index("Scenario")
        vals = np.array([r.loc[s, "Rejection_calibrated"] for s in precise])
        errs = 1.96 * np.array([r.loc[s, "Rejection_calibrated_se"] for s in precise])
        ax.errorbar(x, vals, yerr=errs, color=METHOD_COLORS[mt],
                    marker=METHOD_MARKERS[mt], linestyle="-", capsize=2)
    ax.set_ylabel("Calibrated power")
    ax.set_ylim(0.0, 1.01)

    ax = axes[1, 0]
    for mt in ("CAHB", "RADISH"):
        r = power[power.Method == mt].set_index("Scenario")
        for k, ls in ((0, "-"), (1, "--")):
            vals = np.array([r.loc[s, f"Mean_W_x1_{k}"] for s in precise])
            errs = 1.96 * np.array([r.loc[s, f"Mean_W_x1_{k}_se"] for s in precise])
            ax.errorbar(x, vals, yerr=errs, color=METHOD_COLORS[mt],
                        marker=METHOD_MARKERS[mt], linestyle=ls, capsize=2)
    ax.axhline(0, color="0.45", linewidth=0.8, linestyle=":")
    ax.set_ylabel(r"Mean final borrowing weight $\overline{W}_N^F$")
    ax.set_ylim(-0.02, 0.58)

    ax = axes[1, 1]
    for mt in ("CAHB", "RADISH"):
        for k, ls in ((0, "-"), (1, "--")):
            vals, errs = [], []
            for sc in precise:
                s = df[(df.scenario == sc) & (df.effect_type == "Power") &
                       (df.method == mt)][f"alloc_x1_{k}"].dropna()
                vals.append(s.mean())
                errs.append(1.96 * s.std(ddof=1) / np.sqrt(max(len(s), 1)))
            ax.errorbar(x, vals, yerr=errs, color=METHOD_COLORS[mt],
                        marker=METHOD_MARKERS[mt], linestyle=ls, capsize=2)
    ax.axhline(0.5, color=METHOD_COLORS["KBCD"], linewidth=1.0,
               linestyle=":")
    ax.set_ylabel("Proportion assigned to treatment")
    ax.set_ylim(0.47, 0.62)

    for label, ax in zip("abcd", axes.ravel()):
        ax.set_title(f"({label})", loc="left", pad=5, fontweight="bold")
        ax.set_xticks(x)
        span = max(float(np.ptp(x)), 1.0)
        ax.set_xlim(float(np.min(x)) - 0.04 * span,
                    float(np.max(x)) + 0.04 * span)
    for ax in axes[1, :]:
        ax.set_xlabel(r"Local conflict in $X_1=1$ stratum, $b_1$")

    method_handles = [plt.Line2D([], [], color=METHOD_COLORS[m],
                      marker=METHOD_MARKERS[m], label=METHOD_LABELS[m])
                      for m in METHOD_ORDER]
    stratum_handles = [
        plt.Line2D([], [], color="0.25", linestyle="-", label=r"$X_1=0$ lower conflict"),
        plt.Line2D([], [], color="0.25", linestyle="--", label=r"$X_1=1$ higher conflict"),
    ]
    fig.legend(handles=method_handles + stratum_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir / "plots" / "fig5_borrowing_diagnostics.pdf")
    print("  [ok] Figure 5: local borrowing operating characteristics")


def plot_global_local_quadrants(df, scen_order, out_dir):
    """Compact diagnostic for the orthogonal global-by-local experiment."""
    labels = [r"$G^-L^-$", r"$G^+L^-$", r"$G^-L^+$", r"$G^+L^+$"]
    x = np.arange(len(scen_order))
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.0), sharex=True)

    ax = axes[0, 0]
    for method in ("CAHB", "RADISH"):
        rows = df[(df.effect_type == "Power") & (df.method == method)]
        for k, ls in ((0, "-"), (1, "--")):
            vals = [rows[rows.scenario == sc][f"mean_W_x1_{k}"].mean()
                    for sc in scen_order]
            ax.plot(x, vals, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                    linestyle=ls)
    ax.set_ylabel(r"Mean final borrowing weight")

    ax = axes[0, 1]
    for method in ("CAHB", "RADISH"):
        rows = df[(df.effect_type == "Power") & (df.method == method)]
        for k, ls in ((0, "-"), (1, "--")):
            vals = [rows[rows.scenario == sc][f"alloc_x1_{k}"].mean()
                    for sc in scen_order]
            ax.plot(x, vals, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                    linestyle=ls)
    ax.axhline(0.5, color="0.45", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Proportion assigned to treatment")

    agg = _aggregate(df, scen_order)
    power = agg[agg.Effect == "Power"]
    ax = axes[1, 0]
    for method in METHOD_ORDER:
        r = power[power.Method == method].set_index("Scenario")
        ax.plot(x, [r.loc[sc, "RMSE"] for sc in scen_order],
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method])
    ax.set_ylabel("RMSE")

    ax = axes[1, 1]
    for method in METHOD_ORDER:
        r = power[power.Method == method].set_index("Scenario")
        ax.plot(x, [r.loc[sc, "Rejection_calibrated"] for sc in scen_order],
                color=METHOD_COLORS[method], marker=METHOD_MARKERS[method])
    ax.set_ylabel("Calibrated power")

    for label, ax in zip("abcd", axes.ravel()):
        ax.set_title(f"({label})", loc="left", pad=5, fontweight="bold")
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    method_handles = [plt.Line2D([], [], color=METHOD_COLORS[m],
                      marker=METHOD_MARKERS[m], label=METHOD_LABELS[m])
                      for m in METHOD_ORDER]
    stratum_handles = [
        plt.Line2D([], [], color="0.25", linestyle="-", label=r"$X_1=0$"),
        plt.Line2D([], [], color="0.25", linestyle="--", label=r"$X_1=1$"),
    ]
    fig.legend(handles=method_handles + stratum_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, fontsize=7.2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, out_dir / "plots" / "fig_global_local_quadrants.pdf")
    print("  [ok] Global-by-local four-quadrant diagnostic")


def plot_figure6_precision_map(df, scen_order, out_dir):
    """2-D map of borrowing weight over local conflict and historical noise."""
    if config is None: return
    rows = []
    for sc in scen_order:
        primary = getattr(config, "PRIMARY_SCENARIOS", getattr(config, "DEMO_SCENARIOS", {}))
        meta = primary.get(sc, config.SCENARIOS.get(sc, {}))
        # The x coordinate is the LOCAL conflict b1 whenever the grid defines
        # it: the marginal shift would plot a cell whose X1 = 0 stratum is
        # exactly compatible at the same coordinate as a uniformly shifted one.
        bl = meta.get("b_local") if isinstance(meta, dict) else None
        b = float(bl[1]) if bl is not None else meta.get("b_theta")
        sh = meta.get("sigma_h")
        for mt in ["CAHB", "RADISH"]:
            s = df[(df.scenario==sc)&(df.effect_type=="Power")&(df.method==mt)]
            if s.empty or "mean_W" not in s: continue
            rows.append(dict(b_loc=b, sigma_h=sh, method=mt,
                             W=s.mean_W.mean(skipna=True)))
    pdat = pd.DataFrame(rows).dropna()
    if pdat.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 3.0), sharey=True)
    # Build coordinate axes
    bvals = sorted(pdat.b_loc.unique())
    svals = sorted(pdat.sigma_h.unique())
    for ax, mt in zip(axes, ["CAHB", "RADISH"]):
        Z = np.full((len(svals), len(bvals)), np.nan)
        for _, r in pdat[pdat.method == mt].iterrows():
            i = svals.index(r.sigma_h); j = bvals.index(r.b_loc)
            Z[i, j] = r.W
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap=SEQ_CMAP,
                       vmin=0, vmax=max(0.4, np.nanmax(Z)*1.05))
        ax.set_xticks(range(len(bvals))); ax.set_xticklabels([f"{b:g}" for b in bvals])
        ax.set_yticks(range(len(svals))); ax.set_yticklabels([f"{s:g}" for s in svals])
        ax.set_xlabel(r"Local conflict  $b_1$")
        if mt == "CAHB": ax.set_ylabel(r"Historical noise SD  $\sigma_H$")
        ax.set_title(f"{mt}: mean $W(x)$")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                if np.isfinite(Z[i,j]):
                    col = "white" if Z[i,j] > 0.18 else "black"
                    ax.text(j, i, f"{Z[i,j]:.2f}", ha="center", va="center",
                            fontsize=9, color=col)
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.02)
    cbar.set_label("Mean borrowing weight $\\overline{W}$")
    fig.suptitle("Precision–Bias Borrowing Map  ($\\Delta = 0.5$)", y=1.02)
    _save(fig, out_dir / "plots" / "fig6_precision_bias_map.pdf")
    print("  [ok] Figure 6: precision–bias borrowing map")


# ═══════════════════════════════════════════════════════════════════
# PRECISION-GRADIENT FIGURES (mode="precision")
# ═══════════════════════════════════════════════════════════════════

def _gather_precision(df, scenarios):
    """Long-format frame with σ_H sensitivity."""
    rows = []
    for sc, meta in scenarios.items():
        for ef in ["Null", "Power"]:
            for mt in METHOD_ORDER:
                s = df[(df.scenario==sc) & (df.effect_type==ef) & (df.method==mt)]
                if s.empty: continue
                td = s.true_delta.iloc[0]
                cov = ((s.ci_low <= td) & (td <= s.ci_high)).astype(int)
                bias = s.estimation_bias
                rmse = np.sqrt(((s.estimated_delta - td)**2).mean())
                rej = s.rejected.mean()
                rej_se = np.sqrt(rej*(1-rej)/max(len(s),1))
                W = s.mean_W if "mean_W" in s else pd.Series([np.nan])
                Rn = s.mean_Rn if "mean_Rn" in s else pd.Series([np.nan])
                rows.append(dict(
                    bias_label=meta.get("_grid_bias", ""),
                    b_theta=meta["b_theta"],
                    sigma_h=meta["sigma_h"],
                    effect=ef, method=mt, n=len(s),
                    bias=bias.mean(),
                    bias_se=bias.std(ddof=1)/np.sqrt(len(s)),
                    rmse=rmse,
                    rejection=rej,
                    rejection_se=rej_se,
                    coverage=cov.mean(),
                    width=(s.ci_high - s.ci_low).mean(),
                    width_se=(s.ci_high - s.ci_low).std(ddof=1)/np.sqrt(len(s)),
                    mean_W=W.mean(skipna=True),
                    mean_W_se=W.std(ddof=1, skipna=True)/np.sqrt(max(W.dropna().shape[0],1)),
                    mean_Rn=Rn.mean(skipna=True),
                ))
    return pd.DataFrame(rows)


def _line_panel(ax, dfp, ycol, ylabel, *, ycol_se=None,
                ylim=None, ref=None, ref_label=None, log_x=True,
                show_legend=True):
    for mt in METHOD_ORDER:
        d = dfp[dfp.method == mt].sort_values("sigma_h")
        if d.empty: continue
        y = d[ycol].values; x = d["sigma_h"].values
        ax.plot(x, y, marker=METHOD_MARKERS[mt],
                color=METHOD_COLORS[mt], label=METHOD_LABELS[mt],
                lw=1.9, mec="white", mew=0.7)
        if ycol_se and ycol_se in d.columns:
            se = d[ycol_se].values
            ax.fill_between(x, y - 1.96*se, y + 1.96*se,
                            color=METHOD_COLORS[mt], alpha=0.14, lw=0)
    if ref is not None:
        ax.axhline(ref, color="0.25", ls="--", lw=1.0,
                   label=ref_label or f"{ref}")
    if log_x: ax.set_xscale("log")
    ax.set_xlabel(r"Historical noise SD  $\sigma_H$  (log scale)")
    ax.set_ylabel(ylabel)
    if ylim is not None: ax.set_ylim(*ylim)
    if show_legend:
        ax.legend(loc="best")
    # Show actual σ_H values on x-axis
    xvals = sorted(dfp.sigma_h.unique())
    ax.set_xticks(xvals)
    ax.set_xticklabels([f"{v:g}" for v in xvals])
    ax.minorticks_off()


def plot_precision_W(dfp, out_dir):
    """Fig P1: borrowing weight W vs σ_H — RADISH's signature plot."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 3.0), sharey=True)
    for ax, blab, ttl in zip(axes, ["unbiased", "modBias"],
                              [r"Unbiased history  $b_\theta = 0$",
                               r"Moderately biased  $b_\theta = 0.5$"]):
        d = dfp[(dfp.bias_label == blab) & (dfp.effect == "Power")]
        _line_panel(ax, d, "mean_W", r"Mean borrowing weight  $\overline{W(x)}$",
                    ycol_se="mean_W_se", ylim=(-0.02, max(0.55, d.mean_W.max()*1.15)),
                    show_legend=(blab=="unbiased"))
        ax.set_title(ttl)
    fig.suptitle("Borrowing Weight as a Function of Historical Precision", y=1.03)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "figP1_borrowing_weight_vs_sigmaH.pdf")
    print("  [ok] Figure P1: W vs σ_H")


def plot_precision_Rn(dfp, out_dir):
    """Fig P2: information ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 3.0), sharey=True)
    for ax, blab, ttl in zip(axes, ["unbiased", "modBias"],
                              [r"Unbiased  $b_\theta = 0$",
                               r"Moderately biased  $b_\theta = 0.5$"]):
        d = dfp[(dfp.bias_label == blab) & (dfp.effect == "Power")]
        _line_panel(ax, d, "mean_Rn", r"Mean information ratio  $\overline{R_n(x)}$",
                    ref=1.0, ref_label="$R_n = 1$",
                    ylim=(0.95, max(2.5, d.mean_Rn.max()*1.10)),
                    show_legend=(blab=="unbiased"))
        ax.set_title(ttl)
    fig.suptitle("Information Ratio as a Function of Historical Precision", y=1.03)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "figP2_information_ratio_vs_sigmaH.pdf")
    print("  [ok] Figure P2: R_n vs σ_H")


def plot_precision_estimation(dfp, out_dir):
    """Fig P3: RMSE & coverage vs σ_H, both bias regimes."""
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.4))
    for col, blab in enumerate(["unbiased", "modBias"]):
        ttl = r"$b_\theta = 0$" if blab == "unbiased" else r"$b_\theta = 0.5$"
        d_pow = dfp[(dfp.bias_label == blab) & (dfp.effect == "Power")]
        d_null = dfp[(dfp.bias_label == blab) & (dfp.effect == "Null")]
        ax = axes[0, col]
        _line_panel(ax, d_pow, "rmse", "RMSE  ($\\Delta = 0.5$)",
                    show_legend=(col == 0))
        ax.set_title(f"RMSE  —  {ttl}")
        ax = axes[1, col]
        _line_panel(ax, d_null, "coverage", "Coverage  ($\\Delta = 0$)",
                    ref=0.95, ref_label="nominal 95%",
                    ylim=(0.55, 1.02), show_legend=False)
        ax.set_title(f"Coverage  —  {ttl}")
    fig.suptitle("Estimation Quality across the Precision Sensitivity", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "figP3_rmse_coverage_vs_sigmaH.pdf")
    print("  [ok] Figure P3: RMSE / coverage vs σ_H")


def plot_precision_testing(dfp, out_dir):
    """Fig P4: Type-I & Power vs σ_H."""
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.4))
    for col, blab in enumerate(["unbiased", "modBias"]):
        ttl = r"$b_\theta = 0$" if blab == "unbiased" else r"$b_\theta = 0.5$"
        d_null = dfp[(dfp.bias_label == blab) & (dfp.effect == "Null")]
        d_pow  = dfp[(dfp.bias_label == blab) & (dfp.effect == "Power")]
        ax = axes[0, col]
        _line_panel(ax, d_null, "rejection",
                    "Type I Error  ($\\Delta = 0$)",
                    ycol_se="rejection_se",
                    ref=0.05, ref_label=r"$\alpha = 0.05$",
                    ylim=(0, max(0.30, d_null.rejection.max()*1.15)),
                    show_legend=(col==0))
        ax.set_title(f"Type I Error  —  {ttl}")
        ax = axes[1, col]
        _line_panel(ax, d_pow, "rejection",
                    "Statistical Power  ($\\Delta = 0.5$)",
                    ycol_se="rejection_se",
                    ylim=(0, 1.05), show_legend=False)
        ax.set_title(f"Power  —  {ttl}")
    fig.suptitle("Hypothesis Testing across the Precision Sensitivity", y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "figP4_typeI_power_vs_sigmaH.pdf")
    print("  [ok] Figure P4: Type-I / Power vs σ_H")


# ═══════════════════════════════════════════════════════════════════
# Tables
# ═══════════════════════════════════════════════════════════════════

def write_factorial_table(df, scen_order, out_dir):
    """Diagnostics-augmented summary table (LaTeX + CSV)."""
    agg = _aggregate(df, scen_order)
    cols = ["Scenario", "Effect", "Method", "n",
            "Bias", "RMSE", "ESS",
            "Rejection", "Rejection_calibrated", "Coverage", "Width",
            "Mean_W", "Mean_W_x1_0", "Mean_W_x1_1",
            "Mean_Rn", "Mean_Dpdc", "Alloc"]
    out = agg[cols].copy()
    out.to_csv(out_dir / "tables" / "metrics_full.csv", index=False)
    fmt = {"Bias":"{:.4f}","RMSE":"{:.4f}","ESS":"{:.3f}",
           "Rejection":"{:.3f}",
           "Rejection_calibrated":"{:.3f}",
           "Coverage":"{:.3f}","Width":"{:.3f}",
           "Mean_W":"{:.3f}","Mean_W_x1_0":"{:.3f}","Mean_W_x1_1":"{:.3f}",
           "Mean_Rn":"{:.3f}","Mean_Dpdc":"{:.3f}",
           "Alloc":"{:.3f}"}
    for c, f in fmt.items():
        out[c] = out[c].apply(lambda v: f.format(v) if pd.notna(v) and v != "" else "—")
    out_tex = out.rename(columns={
        "Rejection_calibrated": "Power (calibrated)",
        "Mean_W":  r"$\overline{W}$",
        "Mean_W_x1_0": r"$\overline{W}_{X_1=0}$",
        "Mean_W_x1_1": r"$\overline{W}_{X_1=1}$",
        "Mean_Rn": r"$\overline{R_n}$",
        "Mean_Dpdc": r"$\overline{D_{\mathrm{PDC}}}$",
        "Scenario": "Scenario", "Alloc": "Alloc.\\,Ratio",
    })
    out_tex["Scenario"] = out_tex["Scenario"].str.replace("_", "\\_", regex=False)
    try:
        body = out_tex.to_latex(index=False, escape=False, longtable=False)
        tex = (
            "\\begin{table}[!ht]\n\\centering\n"
            "\\caption{Operating characteristics across the conflict-by-precision "
            "design, including the relative efficiency "
            "$\\mathrm{ESS} = (\\mathrm{RMSE}_{\\mathrm{KBCD}}/\\mathrm{RMSE})^2$ "
            "and the borrowing diagnostics $\\overline{R_n}$, "
            "$\\overline{W}$, the stratum-wise weights "
            "$\\overline{W}_{X_1=0}$ and $\\overline{W}_{X_1=1}$, and the "
            "conflict index $\\overline{D_{\\mathrm{PDC}}}$. "
            "Each cell is averaged over Monte~Carlo replications.}\n"
            "\\label{tab:metrics_full}\n"
            f"{body}\n\\end{{table}}\n"
        )
        (out_dir / "tables" / "metrics_full.tex").write_text(tex, encoding="utf-8")
    except Exception as e:
        print(f"  [warn] LaTeX export failed: {e}")


def write_precision_table(dfp, out_dir):
    """Precision sensitivity diagnostics table (complementary to figures)."""
    cols = ["bias_label","b_theta","sigma_h","effect","method","n",
            "rmse","rejection","coverage","width",
            "mean_W","mean_Rn"]
    out = dfp[cols].copy()
    out = out.rename(columns={
        "bias_label":"BiasReg","b_theta":"b_theta","sigma_h":"sigma_H",
        "effect":"Effect","method":"Method",
        "rmse":"RMSE","rejection":"Reject","coverage":"Coverage",
        "width":"Width","mean_W":"Mean_W","mean_Rn":"Mean_Rn"})
    out.to_csv(out_dir / "tables" / "metrics_precision.csv", index=False)
    fmt = {"RMSE":"{:.4f}","Reject":"{:.3f}","Coverage":"{:.3f}",
           "Width":"{:.3f}","Mean_W":"{:.3f}","Mean_Rn":"{:.3f}",
           "b_theta":"{:.2f}","sigma_H":"{:.2f}"}
    for c, f in fmt.items():
        out[c] = out[c].apply(lambda v: f.format(v) if pd.notna(v) else "—")
    out_tex = out.rename(columns={
        "b_theta":  r"$b_\theta$",
        "sigma_H":  r"$\sigma_H$",
        "Mean_W":   r"$\overline{W}$",
        "Mean_Rn":  r"$\overline{R_n}$",
        "BiasReg":  "Bias regime",
    })
    try:
        body = out_tex.to_latex(index=False, escape=False)
        tex = (
            "\\begin{table}[!ht]\n\\centering\n"
            "\\caption{Precision sensitivity experiment: per-cell operating "
            "characteristics and borrowing diagnostics across "
            "$\\sigma_H \\in \\{0.25, 0.5, 1.0, 1.5, 2.5, 4.0\\}$ "
            "at fixed bias regimes $b_\\theta \\in \\{0, 0.5\\}$.}\n"
            "\\label{tab:metrics_precision}\n"
            f"{body}\n\\end{{table}}\n"
        )
        (out_dir / "tables" / "metrics_precision.tex").write_text(tex, encoding="utf-8")
    except Exception as e:
        print(f"  [warn] LaTeX export failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# BIAS-GRADIENT FIGURES & THEORY-VERIFICATION (mode="bias")
#
# Direct test of Theorem 1:
#   D_PDC(b)  ≥  c1·N_H·h̄·b² + c2·log N        (lower bound)
#   W(b)      =  O_p(exp{−c1·N_H·h̄·b²})         (exponential collapse)
# We fit log( D̄_PDC − D̄_PDC(0) ) ~ β · log(b) to recover the predicted
# slope of 2.0, and fit log( W ) ~ a − γ·b² to recover the exponential
# decay constant. These linear-regression overlays appear as inset
# diagnostics on the line plots.
# ═══════════════════════════════════════════════════════════════════

def _gather_bias(df, scenarios):
    rows = []
    for sc, meta in scenarios.items():
        for ef in ["Null", "Power"]:
            for mt in METHOD_ORDER:
                s = df[(df.scenario==sc) & (df.effect_type==ef) & (df.method==mt)]
                if s.empty: continue
                td = s.true_delta.iloc[0]
                cov = ((s.ci_low <= td) & (td <= s.ci_high)).astype(int)
                rmse = np.sqrt(((s.estimated_delta - td)**2).mean())
                rej = s.rejected.mean()
                rej_se = np.sqrt(rej*(1-rej)/max(len(s),1))
                W  = s.mean_W if "mean_W" in s else pd.Series([np.nan])
                Rn = s.mean_Rn if "mean_Rn" in s else pd.Series([np.nan])
                D  = s.mean_Dpdc if "mean_Dpdc" in s else pd.Series([np.nan])
                rows.append(dict(
                    b_theta=meta["b_theta"], sigma_h=meta["sigma_h"],
                    xi=meta.get("xi", meta["b_theta"] / (2.0 * config.SIGMA_0)),
                    effect=ef, method=mt, n=len(s),
                    bias=s.estimation_bias.mean(),
                    bias_se=s.estimation_bias.std(ddof=1)/np.sqrt(len(s)),
                    rmse=rmse,
                    rejection=rej, rejection_se=rej_se,
                    coverage=cov.mean(),
                    width=(s.ci_high - s.ci_low).mean(),
                    mean_W=W.mean(skipna=True),
                    mean_W_se=W.std(ddof=1,skipna=True)/np.sqrt(max(W.dropna().shape[0],1)),
                    mean_Rn=Rn.mean(skipna=True),
                    mean_Dpdc=D.mean(skipna=True),
                ))
    return pd.DataFrame(rows)


def _fit_quadratic(b, y):
    """Fit y = α + γ·b²; return (α, γ, R²)."""
    b = np.asarray(b, float); y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(b)
    if m.sum() < 3: return np.nan, np.nan, np.nan
    bb = b[m]**2
    A = np.column_stack([np.ones_like(bb), bb])
    beta, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    yhat = A @ beta
    ss_res = np.sum((y[m] - yhat)**2)
    ss_tot = np.sum((y[m] - y[m].mean())**2)
    r2 = 1 - ss_res / max(ss_tot, 1e-12)
    return float(beta[0]), float(beta[1]), float(r2)


def plot_bias_diagnostics(dfb, out_dir):
    """Theory-verification: D̄_PDC vs b² (linear) and W̄ vs b² (log-linear)."""
    pow_d = dfb[dfb.effect == "Power"].copy()

    # Theorem 1: D_PDC ≥ c1·N_H·h̄·b² + c2·log N → quadratic in b
    rad = pow_d[pow_d.method == "RADISH"].sort_values("b_theta")
    b_arr = rad.b_theta.values
    D_arr = rad.mean_Dpdc.values
    # Drop saturated points (D > 25, since cap ≈ 27.6)
    keep = D_arr < 25
    a_D, c1_emp, R2_D = _fit_quadratic(b_arr[keep], D_arr[keep])

    # Theorem 1(i): W ≤ exp(−c1·N_H·h̄·b² + ...). Fit log W ~ −γ·b² in unsat.
    W_arr = rad.mean_W.values
    keep_W = (W_arr > 1e-4) & (b_arr > 0)
    a_W, c2_emp, R2_W = _fit_quadratic(b_arr[keep_W], np.log(W_arr[keep_W]))

    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 3.0))

    # Panel A: D_PDC vs b on linear scale, with quadratic fit overlay
    ax = axes[0]
    for mt in METHOD_ORDER:
        d = pow_d[pow_d.method == mt].sort_values("b_theta")
        if d.empty: continue
        if mt == "RADISH":
            ax.plot(d.b_theta, d.mean_Dpdc,
                    marker=METHOD_MARKERS[mt], color=METHOD_COLORS[mt],
                    label=METHOD_LABELS[mt], lw=1.4, mec="white", mew=0.6)
    if np.isfinite(c1_emp):
        bgrid = np.linspace(0, b_arr[keep].max(), 100)
        ax.plot(bgrid, a_D + c1_emp * bgrid**2,
                color="0.25", ls="--", lw=1.0,
                label=f"Quadratic fit  $D = {a_D:.2f} + {c1_emp:.2f}\\,b^2$  ($R^2={R2_D:.3f}$)")
    ax.axhline(-np.log(1e-12), color="0.55", ls=":", lw=0.8,
               label=r"$-\log\epsilon_p \approx 27.6$ (cap)")
    ax.set_xlabel(r"Historical bias  $b_\theta$")
    ax.set_ylabel(r"Mean PDC surprisal  $\overline{D_{\mathrm{PDC}}}$")
    ax.set_title("Theorem 1(i): quadratic growth")
    ax.legend(loc="upper left", fontsize=7.2)

    # Panel B: log W̄ vs b² with linear-fit overlay
    ax = axes[1]
    for mt in METHOD_ORDER:
        d = pow_d[pow_d.method == mt].sort_values("b_theta")
        if d.empty: continue
        if mt == "KBCD": continue   # KBCD W ≡ 0; skip log
        b2 = d.b_theta.values**2
        Wv = d.mean_W.values
        ok = Wv > 1e-4
        ax.plot(b2[ok], Wv[ok], marker=METHOD_MARKERS[mt],
                color=METHOD_COLORS[mt], label=METHOD_LABELS[mt],
                lw=1.4, mec="white", mew=0.6)
    if np.isfinite(c2_emp):
        bg = np.linspace(0, (b_arr[keep_W]**2).max(), 100)
        ax.plot(bg, np.exp(a_W + c2_emp * bg),
                color="0.25", ls="--", lw=1.0,
                label=f"Exponential fit  $W \\propto e^{{{c2_emp:.2f}\\,b^2}}$  ($R^2={R2_W:.3f}$)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$b_\theta^2$")
    ax.set_ylabel(r"Mean borrowing weight  $\overline{W}$  (log scale)")
    ax.set_title("Theorem 1(i): exponential collapse")
    ax.legend(loc="upper right", fontsize=7.2)

    fig.suptitle("Bias Sensitivity (Theorem 1) Verification ($\\sigma_H = 0.5$, $\\Delta = 0.5$)",
                 y=1.04)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "figB1_theorem1_verification.pdf")
    print("  [ok] Figure B1: Theorem 1 verification")
    return dict(a_D=a_D, c1_emp=c1_emp, R2_D=R2_D,
                a_W=a_W, c2_emp=c2_emp, R2_W=R2_W)


def plot_bias_operating(dfb, out_dir):
    """Operating characteristics over the focused low-conflict xi grid."""
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.4))
    null_d = dfb[dfb.effect == "Null"]
    pow_d  = dfb[dfb.effect == "Power"]

    def set_xi_axis(ax):
        """Base-2 log spacing for positive xi while retaining xi=0."""
        from matplotlib.ticker import NullFormatter
        ticks = np.asarray(config.BIAS_XI_GRID, dtype=float)
        ax.set_xscale("symlog", base=2, linthresh=0.01, linscale=0.8)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0" if x == 0 else f"{x:g}" for x in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
        ax.set_xlabel(r"Conflict $\xi$ (base-2 scale for $\xi>0$)")

    # Panel A: Type I error vs b
    ax = axes[0, 0]
    for mt in METHOD_ORDER:
        d = null_d[null_d.method == mt].sort_values("xi")
        if d.empty: continue
        ax.errorbar(d.xi, d.rejection, yerr=1.96*d.rejection_se,
                    marker=METHOD_MARKERS[mt], color=METHOD_COLORS[mt],
                    label=METHOD_LABELS[mt], lw=1.3, mec="white", mew=0.6,
                    capsize=2, elinewidth=0.7)
    ax.axhline(0.05, color="0.25", ls="--", lw=1.0, label=r"$\alpha = 0.05$")
    set_xi_axis(ax)
    ax.set_ylabel(r"Type I error  ($\Delta = 0$)")
    ax.set_ylim(0, max(0.20, null_d.rejection.max()*1.15))
    ax.set_title("(a) Type I error")
    ax.legend(loc="upper left", fontsize=7.2)

    # Panel B: Power
    ax = axes[0, 1]
    for mt in METHOD_ORDER:
        d = pow_d[pow_d.method == mt].sort_values("xi")
        if d.empty: continue
        ax.errorbar(d.xi, d.rejection, yerr=1.96*d.rejection_se,
                    marker=METHOD_MARKERS[mt], color=METHOD_COLORS[mt],
                    lw=1.3, mec="white", mew=0.6, capsize=2, elinewidth=0.7)
    set_xi_axis(ax)
    ax.set_ylabel(r"Power  ($\Delta = 0.5$)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("(b) Statistical power")

    # Panel C: RMSE (Power)
    ax = axes[1, 0]
    for mt in METHOD_ORDER:
        d = pow_d[pow_d.method == mt].sort_values("xi")
        if d.empty: continue
        ax.plot(d.xi, d.rmse, marker=METHOD_MARKERS[mt],
                color=METHOD_COLORS[mt], lw=1.3, mec="white", mew=0.6)
    set_xi_axis(ax)
    ax.set_ylabel(r"RMSE  ($\Delta = 0.5$)")
    ax.set_title("(c) Estimation RMSE")

    # Panel D: Coverage (Null)
    ax = axes[1, 1]
    for mt in METHOD_ORDER:
        d = null_d[null_d.method == mt].sort_values("xi")
        if d.empty: continue
        ax.plot(d.xi, d.coverage, marker=METHOD_MARKERS[mt],
                color=METHOD_COLORS[mt], lw=1.3, mec="white", mew=0.6)
    ax.axhline(0.95, color="0.25", ls="--", lw=1.0)
    set_xi_axis(ax)
    ax.set_ylabel(r"Coverage  ($\Delta = 0$)")
    ax.set_ylim(0.55, 1.02)
    ax.set_title("(d) Wald coverage")

    fig.suptitle(r"Operating Characteristics across the Low-Conflict Path "
                 r"($b_\theta=2\xi\sigma_C$)",
                 y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "figB2_bias_operating.pdf")
    print("  [ok] Figure B2: bias operating characteristics")


def write_bias_table(dfb, fit, out_dir):
    """Bias sensitivity diagnostic + theory-fit table."""
    cols = ["xi", "b_theta", "effect", "method", "n",
            "rmse","rejection","coverage","width",
            "mean_W","mean_Rn","mean_Dpdc"]
    out = dfb[cols].copy()
    out = out.rename(columns={
        "b_theta":"b_theta","effect":"Effect","method":"Method",
        "rmse":"RMSE","rejection":"Reject","coverage":"Coverage",
        "width":"Width",
        "mean_W":"Mean_W","mean_Rn":"Mean_Rn","mean_Dpdc":"Mean_Dpdc"})
    out.to_csv(out_dir / "tables" / "metrics_bias.csv", index=False)

    # Format
    fmt = {"RMSE":"{:.4f}","Reject":"{:.3f}","Coverage":"{:.3f}",
           "Width":"{:.3f}","Mean_W":"{:.4f}","Mean_Rn":"{:.3f}",
           "Mean_Dpdc":"{:.2f}","b_theta":"{:.3f}"}
    for c, f in fmt.items():
        out[c] = out[c].apply(lambda v: f.format(v) if pd.notna(v) else "—")
    out_tex = out.rename(columns={
        "b_theta":  r"$b_\theta$",
        "Mean_W":   r"$\overline{W}$",
        "Mean_Rn":  r"$\overline{R_n}$",
        "Mean_Dpdc":r"$\overline{D_{\mathrm{PDC}}}$",
    })
    try:
        body = out_tex.to_latex(index=False, escape=False, longtable=False)
        cap = (
            "Bias sensitivity experiment at $\\sigma_H = 0.5$. "
            f"Quadratic fit $D = {fit['a_D']:.2f} + {fit['c1_emp']:.2f}\\,b^2$ "
            f"($R^2 = {fit['R2_D']:.3f}$); "
            f"exponential collapse $\\overline{{W}} \\propto "
            f"\\exp\\{{{fit['c2_emp']:.2f}\\,b^2\\}}$ "
            f"($R^2 = {fit['R2_W']:.3f}$). "
            "These coefficients corroborate Theorem~1(i)."
        )
        tex = (
            "\\begin{table}[!ht]\n\\centering\n"
            f"\\caption{{{cap}}}\n"
            "\\label{tab:metrics_bias}\n"
            f"{body}\n\\end{{table}}\n"
        )
        (out_dir / "tables" / "metrics_bias.tex").write_text(tex, encoding="utf-8")
    except Exception as e:
        print(f"  [warn] LaTeX export failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════

def run_analysis(df, out_dir, mode="full"):
    out_dir = Path(out_dir)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    print(f"Running analysis on {len(df)} results (mode={mode})...")

    # Save metrics summary table from main.metrics if available
    try:
        from main import metrics as _metrics
        scenarios = config.get_mode_scenarios(mode) if config else None
        tbl = _metrics(df, scenarios)
        tbl.to_csv(out_dir / "tables" / "metrics_summary.csv", index=False)
        print("  [ok] metrics_summary.csv")
    except Exception as e:
        print(f"  [!!] summary table skipped: {e}")

    if mode == "precision":
        scenarios = config.get_mode_scenarios("precision")
        dfp = _gather_precision(df, scenarios)
        plot_precision_W(dfp, out_dir)
        plot_precision_Rn(dfp, out_dir)
        plot_precision_estimation(dfp, out_dir)
        plot_precision_testing(dfp, out_dir)
        write_precision_table(dfp, out_dir)
        print("Precision sensitivity analysis complete.")
        return

    if mode == "bias":
        scenarios = config.get_mode_scenarios("bias")
        dfb = _gather_bias(df, scenarios)
        fit = plot_bias_diagnostics(dfb, out_dir)
        plot_bias_operating(dfb, out_dir)
        write_bias_table(dfb, fit, out_dir)
        print("Bias sensitivity theory-verification analysis complete.")
        return

    if mode == "global_local":
        scenarios = config.get_mode_scenarios(mode)
        plot_global_local_quadrants(df, list(scenarios), out_dir)
        print("Global-by-local diagnostic analysis complete.")
        return

    # Factorial figures for the mode-specific six- or eight-cell grid.
    scen_order = list(config.get_mode_scenarios(mode).keys()) if config else \
                 sorted(df.scenario.unique())
    plot_figure1_allocation(df, scen_order, out_dir)
    plot_figure3_ci(df, scen_order, out_dir)
    plot_figure4_testing(df, scen_order, out_dir)
    plot_figure5_borrowing_diagnostics(df, scen_order, out_dir)
    plot_figure6_precision_map(df, scen_order, out_dir)
    write_factorial_table(df, scen_order, out_dir)
    print("Factorial analysis complete.")
