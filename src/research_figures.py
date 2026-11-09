"""Readable, reusable aggregate figures for the state and validation notebooks."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import PathPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator


INK, MUTED, GRID = "#17324A", "#506779", "#DCE5EB"
BLUE, TEAL, GOLD, PURPLE = "#1D5C90", "#087E79", "#A55C18", "#694789"
UNAVAILABLE = "#E0E7EC"
PRICE_CMAP = LinearSegmentedColormap.from_list("price", ["#F2F7FB", "#7FADCD", "#0F426C"])
GAP_CMAP = LinearSegmentedColormap.from_list("gap", ["#985219", "#F9F7F0", "#06736E"])
WINDOWS = ["Apr 13–18", "Apr 19–24", "Apr 25–30", "May 1–4"]
USD = FuncFormatter(lambda x, _: f"−${abs(x):,.0f}" if x < 0 else f"${x:,.0f}")
PCT = FuncFormatter(lambda x, _: f"{x:+.0f}%" if x else "0%")


def figure_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "figure.dpi": 130,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": "#A9BBC8",
    })


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=170, facecolor="white", bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", facecolor="white", bbox_inches="tight")


def _left_title(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=12, pad=10)


def state_maps(summary, paths):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.55), layout="constrained")
    fig.suptitle("State asking prices and vehicle-adjusted gaps", fontsize=17, fontweight="bold")
    maximum = max(1, float(summary["gap_pct"].abs().max()))
    gap_limit = max(5, 5 * np.ceil(maximum / 5))
    panels = [
        ("median_all", PRICE_CMAP, Normalize(summary["median_all"].min(), summary["median_all"].max()),
         "A  ·  Median asking price", "USD · all positive listings", USD),
        ("gap_pct", GAP_CMAP, TwoSlopeNorm(vmin=-gap_limit, vcenter=0, vmax=gap_limit),
         "B  ·  Gap versus similar listings", "Percent vs state-free model · supported April listings", PCT),
    ]
    for ax, (field, cmap, norm, title, unit, formatter) in zip(axes, panels, strict=True):
        for state, path in paths.items():
            value = summary[field].get(state, np.nan)
            color = cmap(norm(value)) if np.isfinite(value) else UNAVAILABLE
            ax.add_patch(PathPatch(path, facecolor=color, edgecolor="white", linewidth=.64))
        dc_xy = paths["dc"].vertices.mean(axis=0)
        dc_value = summary[field].get("dc", np.nan)
        dc_color = cmap(norm(dc_value)) if np.isfinite(dc_value) else UNAVAILABLE
        ax.scatter(*dc_xy, s=26, c=[dc_color], edgecolors=INK, linewidths=.6, zorder=4)
        ax.annotate("DC", dc_xy, xytext=(10, -10), textcoords="offset points", fontsize=8, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .7})
        ax.autoscale_view()
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        _left_title(ax, title)
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal",
                          fraction=.07, pad=.018, shrink=.82, aspect=32)
        cb.ax.xaxis.set_major_formatter(formatter)
        cb.outline.set_visible(False)
        cb.set_label(unit, labelpad=7)
    return fig


def state_intervals(summary):
    order = summary.sort_values("gap_pct", na_position="first").copy()
    yy = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 14.8), sharey=True, layout="constrained")
    fig.suptitle("State estimates with 95% vehicle-group bootstrap intervals", fontsize=16, fontweight="bold")
    configurations = (
        ("median_all", "median_low", "median_high", BLUE, "Raw median · all positive listings", "Asking price (USD)", USD),
        ("gap_pct", "gap_low", "gap_high", TEAL, "Adjusted gap · supported April listings", "Difference vs state-free benchmark", PCT),
    )
    for ax, (point, lo, hi, color, title, xlabel, formatter) in zip(axes, configurations, strict=True):
        ax.hlines(yy, order[lo], order[hi], color=color, linewidth=1.4, alpha=.78)
        ax.scatter(order[point], yy, color=color, s=18, zorder=3)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis="x")
        _left_title(ax, title)
    labels = [f"{state.upper()}    {n:,} / {m:,}" for state, n, m in zip(order.index, order["positive_n"], order["supported_n"])]
    axes[0].set_yticks(yy, labels=labels, fontsize=8.7)
    axes[0].set_ylabel("State    raw n / adjusted n")
    axes[1].axvline(0, color=MUTED, linewidth=.9, linestyle="--")
    return fig


def state_rank_shifts(summary):
    eligible = summary.dropna(subset=["rank_change"])
    extremes = eligible.loc[eligible["rank_change"].abs().nlargest(12).index].sort_values("rank_change")
    fig, ax = plt.subplots(figsize=(10, 5.1), layout="constrained")
    colors = np.where(extremes["rank_change"].ge(0), TEAL, GOLD)
    yy = np.arange(len(extremes))
    ax.barh(yy, extremes["rank_change"], color=colors, height=.63)
    ax.axvline(0, color=INK, linewidth=.95)
    labels = [f"{state.upper()}    #{int(a)} → #{int(b)}    (n={n:,})"
              for state, a, b, n in zip(extremes.index, extremes["raw_rank"], extremes["gap_rank"], extremes["supported_n"])]
    ax.set_yticks(yy, labels=labels)
    pad = 5
    ax.set_xlim(min(0, extremes["rank_change"].min())-pad, max(0, extremes["rank_change"].max())+pad)
    for y, change in enumerate(extremes["rank_change"]):
        ax.text(change + (.65 if change >= 0 else -.65), y, f"{change:+.0f}", ha="left" if change >= 0 else "right", va="center")
    ax.set_xlabel("Change in rank (positive = moves toward higher relative asking prices)")
    _left_title(ax, "Largest rank shifts on the same supported April listings")
    ax.grid(axis="x")
    return fig


def state_model_sensitivity(summary, blocks):
    eligible = summary.dropna(subset=["gap_pct", "ridge_gap_pct"])
    limit = max(5, 5 * np.ceil(eligible[["gap_pct", "ridge_gap_pct"]].abs().max().max() / 5))
    selected = eligible.loc[eligible["gap_pct"].abs().nlargest(12).index].sort_values("gap_pct", ascending=False)
    matrix = blocks.pivot(index="state", columns="window", values="gap_pct").reindex(index=selected.index, columns=WINDOWS[:3])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.15), layout="constrained", gridspec_kw={"width_ratios": [1, 1.1]})
    fig.suptitle("Do the geographic gaps depend on the model or week?", fontsize=16, fontweight="bold")
    ax, ax2 = axes
    ax.plot([-limit, limit], [-limit, limit], linestyle="--", color=MUTED, linewidth=.9)
    ax.axhline(0, color=GRID, linewidth=.7)
    ax.axvline(0, color=GRID, linewidth=.7)
    ax.scatter(eligible["ridge_gap_pct"], eligible["gap_pct"], color=BLUE, s=29, alpha=.8)
    for state, row in eligible.loc[(eligible["gap_pct"]-eligible["ridge_gap_pct"]).abs().nlargest(3).index].iterrows():
        ax.annotate(state.upper(), (row["ridge_gap_pct"], row["gap_pct"]), xytext=(5, 3), textcoords="offset points", fontsize=9)
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="Ridge benchmark gap", ylabel="Boosting benchmark gap", aspect="equal")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    _left_title(ax, "A  ·  Alternate model (one dot per jurisdiction)")
    max_block = max(limit, np.nanmax(np.abs(matrix.to_numpy())))
    cmap = GAP_CMAP.copy()
    cmap.set_bad(UNAVAILABLE)
    image = ax2.imshow(matrix, aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-max_block, vcenter=0, vmax=max_block))
    ax2.set_xticks(range(3), matrix.columns)
    ax2.set_yticks(range(len(matrix)), matrix.index.str.upper())
    for i, row in enumerate(matrix.to_numpy()):
        for j, v in enumerate(row):
            ax2.text(j, i, f"{v:+.0f}%" if np.isfinite(v) else "<50", ha="center", va="center", fontsize=8.5,
                     color="white" if np.isfinite(v) and abs(v) > max_block*.62 else INK)
    _left_title(ax2, "B  ·  Time blocks for the 12 largest pooled gaps")
    fig.colorbar(image, ax=ax2, fraction=.04, pad=.04, format=PCT)
    return fig


def validation_by_window(metrics):
    fig, ax = plt.subplots(figsize=(10.6, 5.1), layout="constrained")
    styles = zip([MUTED, GOLD, PURPLE, BLUE, TEAL], ["o", "D", "^", "s", "P"], ["--", "-.", ":", "-", "-"], strict=True)
    for (model, group), (color, marker, line) in zip(metrics.groupby("model", sort=False), styles, strict=True):
        ordered = group.set_index("window").loc[WINDOWS]
        ax.plot(range(4), ordered["mae"], marker=marker, linestyle=line, color=color, label=model, linewidth=2, markersize=6)
    ax.axvspan(2.55, 3.3, color="#EEF4F8", zorder=0)
    ax.set_xticks(range(4), [*WINDOWS[:3], "May 1–4\nHistorical comparison"])
    ax.set_xlim(-.14, 3.3)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Mean absolute error (USD)")
    ax.yaxis.set_major_formatter(USD)
    ax.grid(axis="y")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.51, -.13), fontsize=9.4)
    _left_title(ax, "Price prediction across three April validation blocks")
    return fig


def segment_errors(predictions, model, field, bins, labels):
    frame = predictions.loc[predictions["window"].ne("May 1–4")].copy()
    frame["segment"] = pd.cut(frame[field], bins=bins, labels=labels, right=False)
    frame["signed"] = frame[model] - frame["price"]
    frame["absolute"] = frame["signed"].abs()
    return frame.groupby("segment", observed=True).agg(
        n=("signed", "size"), mae=("absolute", "mean"), median_error=("signed", "median")
    )


def error_profile(price, age, model):
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.1), layout="constrained")
    fig.suptitle(f"Where does {model} miss?  ·  April predictions", fontsize=16, fontweight="bold")
    for col, (table, title) in enumerate([(price, "Observed asking price"), (age, "Vehicle age")]):
        x = np.arange(len(table))
        for row, field, ylabel in [(0, "mae", "MAE (USD)"), (1, "median_error", "Median signed error (USD)")]:
            ax = axes[row, col]
            values = table[field]
            colors = BLUE if row == 0 else np.where(values.ge(0), TEAL, GOLD)
            ax.bar(x, values, color=colors, width=.68)
            ax.set_xticks(x, [f"{label}\n{n:,}" for label, n in zip(table.index, table["n"])], fontsize=8.4)
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(USD)
            ax.axhline(0, color=MUTED, linewidth=.7)
            ax.grid(axis="y")
            if row == 0:
                _left_title(ax, title)
    return fig


def feature_ablation(diagnostics):
    base = diagnostics.loc[diagnostics["specification"].eq("Full features"), "mae"].iloc[0]
    table = diagnostics.loc[~diagnostics["specification"].str.contains("VIN-only")].copy()
    table["delta"] = table["mae"] - base
    table = table.sort_values("delta")
    fig, ax = plt.subplots(figsize=(9.2, 3.4), layout="constrained")
    ax.barh(table["specification"], table["delta"], color=TEAL, height=.58)
    for i, delta in enumerate(table["delta"]):
        ax.text(delta+max(15, table["delta"].max()*.018), i, f"{delta:+,.0f}", va="center", color=INK)
    ax.set_xlim(min(0, table["delta"].min())-20, max(40, table["delta"].max()*1.19))
    ax.axvline(0, color=MUTED, linewidth=.8)
    ax.set_xlabel("Change in MAE relative to all features (USD)")
    _left_title(ax, "Drop one feature group at a time · Apr 25–30")
    ax.grid(axis="x")
    return fig
