"""
analysis.py — Publication-quality figures and tables.

Two output modes
----------------
mode="full"       3×2 factorial scenarios:
    Fig 1  Allocation Ratio   (Null | Power)
    Fig 2  Bias + RMSE        (2×2 metric × effect)
    Fig 3  CI Width + Coverage (2×2 metric × effect)
    Fig 4  Type-I + Power
    Fig 5  Borrowing Diagnostics: mean W, mean R_n         ← NEW
    Fig 6  Precision-Awareness Map: W vs (b_θ, σ_H) heat   ← NEW

mode="precision"  σ_H sensitivity sweep at fixed b_θ:
    Fig P1  Borrowing weight W vs σ_H        (showcase)   ← NEW
    Fig P2  Information ratio R_n vs σ_H                  ← NEW
    Fig P3  RMSE & Coverage vs σ_H                        ← NEW
    Fig P4  Type-I error / Power vs σ_H                   ← NEW

All figures saved as vector PDF at publication quality.
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

PUB_RC = {
    # Biometrics / Biostatistics figure standards: 8-9 pt body fonts,
    # sans-serif for axis labels, serif math, vector PDF (Type42).
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.title_fontsize": 8.5,
    "figure.titlesize":   10,
    # Math text in serif (Computer Modern), per journal house style.
    "mathtext.fontset":   "stix",
    "mathtext.rm":        "STIXGeneral",
    # Lines & axes — thin strokes that print well in B/W reproduction.
    "axes.linewidth":     0.7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linestyle":     ":",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.4,
    # Horizontal-grid-only via axes.grid.axis is not valid in matplotlib;
    # we apply ax.yaxis.grid(True); ax.xaxis.grid(False) selectively below.
    "lines.linewidth":    1.4,
    "lines.markersize":   4.5,
    "lines.markeredgewidth": 0.7,
    # Ticks
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    # Vector PDF (TrueType embed, journal-required)
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.04,
    # Legend
    "legend.frameon":     True,
    "legend.framealpha":  0.95,
    "legend.edgecolor":   "0.5",
    "legend.fancybox":    False,
    "legend.borderpad":   0.4,
    "legend.handlelength":1.6,
    "legend.handletextpad":0.5,
}

plt.rcParams.update(PUB_RC)

# Biometrics column widths (inches): 1col = 3.27", 1.5col = 5", 2col = 6.83"
COL1_W = 3.27
COL15_W = 5.00
COL2_W = 6.83

# Color-blind-friendly palette (Okabe–Ito-derived)
METHOD_ORDER  = ["KBCD", "CAHB", "RADISH"]
METHOD_COLORS = {
    "KBCD":   "#0072B2",   # blue
    "CAHB":   "#D55E00",   # vermillion
    "RADISH": "#009E73",   # bluish green
}
METHOD_MARKERS = {"KBCD": "o", "CAHB": "s", "RADISH": "^"}
METHOD_LABELS  = {
    "KBCD":   "KBCD (no borrowing)",
    "CAHB":   "CAHB",
    "RADISH": "RADISH (proposed)",
}

EFFECT_DISPLAY = {"Null": r"$\Delta = 0$ (Null)", "Power": r"$\Delta = 0.5$ (Power)"}


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), format="pdf")
    plt.close(fig)


def _scen_short_factorial(s):
    if config and s in config.SCENARIOS:
        sc = config.SCENARIOS[s]
        idx = s.split("_")[0]
        return f"{idx}\n$b={sc['b_theta']}$\n$\\sigma_H={sc['sigma_h']}$"
    return s


def _aggregate(df, scenarios):
    df = df.copy()
    td = df["true_delta"]
    df["coverage"] = ((df.ci_low <= td) & (td <= df.ci_high)).astype(int)
    df["width"] = df.ci_high - df.ci_low
    df["sq_err"] = (df.estimated_delta - td) ** 2

    rows = []
    for sc in scenarios:
        for ef in ["Null", "Power"]:
            for mt in METHOD_ORDER:
                s = df[(df.scenario == sc) & (df.effect_type == ef) & (df.method == mt)]
                if s.empty: continue
                rows.append(dict(
                    Scenario=sc, Effect=ef, Method=mt, n=len(s),
                    Bias=s.estimation_bias.mean(),
                    Bias_se=s.estimation_bias.std(ddof=1)/np.sqrt(len(s)),
                    RMSE=np.sqrt(s.sq_err.mean()),
                    Rejection=s.rejected.mean(),
                    Rejection_se=np.sqrt(s.rejected.mean()*(1-s.rejected.mean())/len(s)),
                    Coverage=s.coverage.mean(),
                    Width=s.width.mean(),
                    Width_se=s.width.std(ddof=1)/np.sqrt(len(s)),
                    Alloc=s.allocation_ratio.mean(),
                    Mean_W=s.get("mean_W", pd.Series(dtype=float)).mean(skipna=True),
                    Mean_W_se=s.get("mean_W", pd.Series(dtype=float)).std(ddof=1, skipna=True)/np.sqrt(max(len(s.get("mean_W", pd.Series(dtype=float)).dropna()),1)),
                    Mean_Rn=s.get("mean_Rn", pd.Series(dtype=float)).mean(skipna=True),
                    Mean_Dpdc=s.get("mean_Dpdc", pd.Series(dtype=float)).mean(skipna=True),
                ))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Helpers for grouped bar charts
# ═══════════════════════════════════════════════════════════════════

def _grouped_bar(ax, agg, scenarios, metric, ylabel, ylim=None,
                 ref_line=None, ref_label=None,
                 short_fn=_scen_short_factorial,
                 err_metric=None, value_fontsize=None, value_fmt="{:.3f}"):
    """Grouped bar chart. Set value_fontsize to a number (e.g. 7) to overlay
    numeric values on bars; default is None (no overlay) for clean figures."""
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
        if value_fontsize:
            for bar, v in zip(bars, vals):
                if not np.isfinite(v): continue
                yoff = max(abs(v)*0.012, 0.003)
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (yoff if v >= 0 else -yoff),
                        value_fmt.format(v),
                        ha="center", va="bottom" if v >= 0 else "top",
                        fontsize=value_fontsize)
    if ref_line is not None:
        ax.axhline(ref_line, color="0.25", ls="--", lw=1.1,
                   label=ref_label or f"{ref_line}")
    ax.set_xticks(x)
    ax.set_xticklabels([short_fn(s) for s in scenarios], fontsize=8.5)
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
# 3×2 FACTORIAL FIGURES
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
        ax.set_xticks(tick_pos); ax.set_xticklabels(tick_lab, fontsize=8.5)
        ax.set_title(EFFECT_DISPLAY[eff])
        ax.set_ylim(0.42, 0.72)
    axes[0].set_ylabel("Allocation Ratio to Treatment Arm")
    _shared_method_legend(fig, y=1.06)
    fig.suptitle("Allocation Ratio across Scenarios", y=1.13)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "fig1_allocation_ratio.pdf")
    print("  [ok] Figure 1: allocation ratio")


def plot_figure2_bias_rmse(df, scen_order, out_dir):
    agg = _aggregate(df, scen_order)
    if agg.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.4))
    for col, eff in enumerate(["Null", "Power"]):
        sub = agg[agg.Effect == eff]
        scen = [s for s in scen_order if s in sub.Scenario.values]
        ax = axes[0, col]
        _grouped_bar(ax, sub, scen, "Bias", "Estimation Bias",
                     ref_line=0, ref_label="zero bias",
                     err_metric="Bias_se")
        ax.set_title(f"Bias — {EFFECT_DISPLAY[eff]}")
        ax = axes[1, col]
        _grouped_bar(ax, sub, scen, "RMSE", "RMSE")
        ax.set_title(f"RMSE — {EFFECT_DISPLAY[eff]}")
    _shared_method_legend(fig, y=1.02)
    fig.suptitle("Estimation Bias and RMSE", y=1.05)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "fig2_bias_rmse.pdf")
    print("  [ok] Figure 2: bias / RMSE")


def plot_figure3_ci(df, scen_order, out_dir):
    agg = _aggregate(df, scen_order)
    if agg.empty: return
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.4))
    for col, eff in enumerate(["Null", "Power"]):
        sub = agg[agg.Effect == eff]
        scen = [s for s in scen_order if s in sub.Scenario.values]
        ax = axes[0, col]
        _grouped_bar(ax, sub, scen, "Width", "CI Width",
                     err_metric="Width_se")
        ax.set_title(f"CI Width — {EFFECT_DISPLAY[eff]}")
        ax = axes[1, col]
        _grouped_bar(ax, sub, scen, "Coverage", "Coverage Probability",
                     ref_line=0.95, ref_label="nominal 95%",
                     ylim=(0.55, 1.02))
        ax.set_title(f"Coverage — {EFFECT_DISPLAY[eff]}")
    _shared_method_legend(fig, y=1.02)
    fig.suptitle("Confidence Interval Width and Coverage", y=1.05)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "fig3_ci_width_coverage.pdf")
    print("  [ok] Figure 3: CI width / coverage")


def plot_figure4_testing(df, scen_order, out_dir):
    agg = _aggregate(df, scen_order)
    if agg.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 2.9))
    for ax, eff, ttl in zip(axes, ["Null", "Power"],
                            ["Type I Error  ($\\Delta=0$)",
                             "Statistical Power  ($\\Delta=0.5$)"]):
        sub = agg[agg.Effect == eff]
        scen = [s for s in scen_order if s in sub.Scenario.values]
        ref = 0.05 if eff == "Null" else None
        ref_lab = r"$\alpha = 0.05$" if eff == "Null" else None
        ylim = (0, max(0.28, sub.Rejection.max()*1.18) if eff=="Null" else None)
        _grouped_bar(ax, sub, scen, "Rejection",
                     "Rejection Rate" if eff=="Null" else "Power",
                     ref_line=ref, ref_label=ref_lab,
                     err_metric="Rejection_se",
                     ylim=ylim if eff == "Null" else (0, 1.05))
        ax.set_title(ttl)
    _shared_method_legend(fig, y=1.05)
    fig.suptitle("Hypothesis Testing Performance", y=1.10)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "fig4_typeI_power.pdf")
    print("  [ok] Figure 4: type I / power")


def plot_figure5_borrowing_diagnostics(df, scen_order, out_dir):
    """NEW: showcase how each method borrows. Mean W and mean R_n by scenario."""
    agg = _aggregate(df, scen_order)
    if agg.empty: return
    # use Power runs (richer signal); pattern is similar under Null
    sub = agg[agg.Effect == "Power"]
    scen = [s for s in scen_order if s in sub.Scenario.values]

    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 2.9))
    ax = axes[0]
    _grouped_bar(ax, sub, scen, "Mean_W", r"Mean borrowing weight $\overline{W(x)}$",
                 ref_line=0,
                 err_metric="Mean_W_se",
                 ylim=(0, max(0.28, sub.Mean_W.max()*1.20)))
    ax.set_title(r"Borrowing Intensity")
    # add annotation
    ax.text(0.02, 0.97,
            "RADISH down-weights\nlow-precision history\n(B2, B4, B6)",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6))

    ax = axes[1]
    _grouped_bar(ax, sub, scen, "Mean_Rn", r"Mean information ratio $\overline{R_n(x)}$",
                 ref_line=1.0, ref_label=r"$R_n=1$ (no borrowing)",
                 ylim=(0.95, max(1.6, sub.Mean_Rn.max()*1.10)))
    ax.set_title(r"Effective Information Gain")

    _shared_method_legend(fig, y=1.05)
    fig.suptitle("Borrowing Diagnostics by Scenario  ($\\Delta = 0.5$)", y=1.10)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "fig5_borrowing_diagnostics.pdf")
    print("  [ok] Figure 5: borrowing diagnostics")


def plot_figure6_precision_map(df, scen_order, out_dir):
    """NEW: 2-D precision-bias map of borrowing weight."""
    if config is None: return
    rows = []
    for sc in scen_order:
        meta = config.SCENARIOS.get(sc, {})
        b = meta.get("b_theta"); sh = meta.get("sigma_h")
        for mt in ["CAHB", "RADISH"]:
            s = df[(df.scenario==sc)&(df.effect_type=="Power")&(df.method==mt)]
            if s.empty or "mean_W" not in s: continue
            rows.append(dict(b_theta=b, sigma_h=sh, method=mt,
                             W=s.mean_W.mean(skipna=True)))
    pdat = pd.DataFrame(rows).dropna()
    if pdat.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(COL2_W, 3.0), sharey=True)
    # Build coordinate axes
    bvals = sorted(pdat.b_theta.unique())
    svals = sorted(pdat.sigma_h.unique())
    for ax, mt in zip(axes, ["CAHB", "RADISH"]):
        Z = np.full((len(svals), len(bvals)), np.nan)
        for _, r in pdat[pdat.method == mt].iterrows():
            i = svals.index(r.sigma_h); j = bvals.index(r.b_theta)
            Z[i, j] = r.W
        im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=max(0.4, np.nanmax(Z)*1.05))
        ax.set_xticks(range(len(bvals))); ax.set_xticklabels([f"{b:g}" for b in bvals])
        ax.set_yticks(range(len(svals))); ax.set_yticklabels([f"{s:g}" for s in svals])
        ax.set_xlabel(r"Historical bias  $b_\theta$")
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
            "Bias", "RMSE", "Rejection", "Coverage", "Width",
            "Mean_W", "Mean_Rn", "Mean_Dpdc", "Alloc"]
    out = agg[cols].copy()
    out.to_csv(out_dir / "tables" / "metrics_full.csv", index=False)
    fmt = {"Bias":"{:.4f}","RMSE":"{:.4f}","Rejection":"{:.3f}",
           "Coverage":"{:.3f}","Width":"{:.3f}",
           "Mean_W":"{:.3f}","Mean_Rn":"{:.3f}","Mean_Dpdc":"{:.3f}",
           "Alloc":"{:.3f}"}
    for c, f in fmt.items():
        out[c] = out[c].apply(lambda v: f.format(v) if pd.notna(v) and v != "" else "—")
    out_tex = out.rename(columns={
        "Mean_W":  r"$\overline{W}$",
        "Mean_Rn": r"$\overline{R_n}$",
        "Mean_Dpdc": r"$\overline{D_{\mathrm{PDC}}}$",
        "Scenario": "Scenario", "Alloc": "Alloc.\\,Ratio",
    })
    out_tex["Scenario"] = out_tex["Scenario"].str.replace("_", "\\_", regex=False)
    try:
        body = out_tex.to_latex(index=False, escape=False, longtable=False)
        tex = (
            "\\begin{table}[!ht]\n\\centering\n"
            "\\caption{Operating characteristics across the $3\\times 2$ "
            "factorial design, including borrowing diagnostics "
            "($\\overline{W}$, $\\overline{R_n}$, "
            "$\\overline{D_{\\mathrm{PDC}}}$). "
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
    """Operating characteristics across bias sensitivity: Type-I, power, RMSE, coverage."""
    fig, axes = plt.subplots(2, 2, figsize=(COL2_W, 5.4))
    null_d = dfb[dfb.effect == "Null"]
    pow_d  = dfb[dfb.effect == "Power"]

    # Panel A: Type I error vs b
    ax = axes[0, 0]
    for mt in METHOD_ORDER:
        d = null_d[null_d.method == mt].sort_values("b_theta")
        if d.empty: continue
        ax.errorbar(d.b_theta, d.rejection, yerr=1.96*d.rejection_se,
                    marker=METHOD_MARKERS[mt], color=METHOD_COLORS[mt],
                    label=METHOD_LABELS[mt], lw=1.3, mec="white", mew=0.6,
                    capsize=2, elinewidth=0.7)
    ax.axhline(0.05, color="0.25", ls="--", lw=1.0, label=r"$\alpha = 0.05$")
    ax.set_xlabel(r"Historical bias  $b_\theta$")
    ax.set_ylabel(r"Type I error  ($\Delta = 0$)")
    ax.set_ylim(0, max(0.20, null_d.rejection.max()*1.15))
    ax.set_title("(a) Type I error")
    ax.legend(loc="upper left", fontsize=7.2)

    # Panel B: Power
    ax = axes[0, 1]
    for mt in METHOD_ORDER:
        d = pow_d[pow_d.method == mt].sort_values("b_theta")
        if d.empty: continue
        ax.errorbar(d.b_theta, d.rejection, yerr=1.96*d.rejection_se,
                    marker=METHOD_MARKERS[mt], color=METHOD_COLORS[mt],
                    lw=1.3, mec="white", mew=0.6, capsize=2, elinewidth=0.7)
    ax.set_xlabel(r"Historical bias  $b_\theta$")
    ax.set_ylabel(r"Power  ($\Delta = 0.5$)")
    ax.set_ylim(0.4, 1.02)
    ax.set_title("(b) Statistical power")

    # Panel C: RMSE (Power)
    ax = axes[1, 0]
    for mt in METHOD_ORDER:
        d = pow_d[pow_d.method == mt].sort_values("b_theta")
        if d.empty: continue
        ax.plot(d.b_theta, d.rmse, marker=METHOD_MARKERS[mt],
                color=METHOD_COLORS[mt], lw=1.3, mec="white", mew=0.6)
    ax.set_xlabel(r"Historical bias  $b_\theta$")
    ax.set_ylabel(r"RMSE  ($\Delta = 0.5$)")
    ax.set_title("(c) Estimation RMSE")

    # Panel D: Coverage (Null)
    ax = axes[1, 1]
    for mt in METHOD_ORDER:
        d = null_d[null_d.method == mt].sort_values("b_theta")
        if d.empty: continue
        ax.plot(d.b_theta, d.coverage, marker=METHOD_MARKERS[mt],
                color=METHOD_COLORS[mt], lw=1.3, mec="white", mew=0.6)
    ax.axhline(0.95, color="0.25", ls="--", lw=1.0)
    ax.set_xlabel(r"Historical bias  $b_\theta$")
    ax.set_ylabel(r"Coverage  ($\Delta = 0$)")
    ax.set_ylim(0.55, 1.02)
    ax.set_title("(d) Wald coverage")

    fig.suptitle("Operating Characteristics across the Bias Sensitivity",
                 y=1.02)
    plt.tight_layout()
    _save(fig, out_dir / "plots" / "figB2_bias_operating.pdf")
    print("  [ok] Figure B2: bias operating characteristics")


def write_bias_table(dfb, fit, out_dir):
    """Bias sensitivity diagnostic + theory-fit table."""
    cols = ["b_theta", "effect", "method", "n",
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

    # Default 3×2 factorial figures
    scen_order = list(config.SCENARIOS.keys()) if config else \
                 sorted(df.scenario.unique())
    plot_figure1_allocation(df, scen_order, out_dir)
    plot_figure2_bias_rmse(df, scen_order, out_dir)
    plot_figure3_ci(df, scen_order, out_dir)
    plot_figure4_testing(df, scen_order, out_dir)
    plot_figure5_borrowing_diagnostics(df, scen_order, out_dir)
    plot_figure6_precision_map(df, scen_order, out_dir)
    write_factorial_table(df, scen_order, out_dir)
    print("Factorial analysis complete.")
