"""Aggregate diagnostics for the Craigslist asking-price distribution."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

from src.listings import HIGH_PRICE_REVIEW, LOW_PRICE_REVIEW, RAW, clean_listings, finite_numeric, read_source


INK, MUTED, GRID = "#142D43", "#526777", "#DEE7EB"
BLUE, GOLD, LIGHT = "#146C96", "#A8612B", "#DFEBF1"
PAPER = "#FCFDFD"
BAND = f"${LOW_PRICE_REVIEW:,}–${HIGH_PRICE_REVIEW:,}"
SOURCE = "Source  Austin Reese, Craigslist Cars & Trucks Data, Kaggle v10 (April–May 2021)."


def load_price_inputs(root):
    """Return positive listings and a restricted source view for the zero-price audit."""
    source = read_source(root / RAW)
    identifiers = source["id"].str.strip().replace("", pd.NA).dropna()
    if identifiers.duplicated().any():
        raise ValueError("Q1 requires the pinned Kaggle v10 file with unique nonempty listing IDs.")
    clean, audit, _ = clean_listings(source)
    audit_input = source[["type", "state", "manufacturer"]].apply(
        lambda column: column.str.strip().str.lower().replace("", pd.NA)
    )
    audit_input["price"] = finite_numeric(source["price"])
    if int(audit_input["price"].gt(0).sum()) != len(clean) or not clean["price"].equals(audit_input.loc[clean.index, "price"]):
        raise ValueError("The source price audit did not reconcile to the shared cleaned sample.")
    return clean, audit_input, audit


def price_audit(prices):
    counts = {
        "Unusable or missing": int(prices.isna().sum()),
        "Negative": int(prices.lt(0).sum()),
        "Exactly $0": int(prices.eq(0).sum()),
        "Exactly $1": int(prices.eq(1).sum()),
        f"Other positive prices below ${LOW_PRICE_REVIEW:,}": int(
            (prices.gt(0) & prices.lt(LOW_PRICE_REVIEW) & prices.ne(1)).sum()
        ),
        f"Within {BAND}": int(prices.between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW).sum()),
        f"Above ${HIGH_PRICE_REVIEW:,}": int(prices.gt(HIGH_PRICE_REVIEW).sum()),
    }
    assert sum(counts.values()) == len(prices)
    return pd.Series(counts, name="Listings").to_frame().assign(**{
        "% of source listings": lambda table: 100 * table["Listings"] / len(prices),
    })


def cohort_summary(positive):
    in_band = positive[positive.between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW)]

    def describe(prices):
        percentiles = prices.quantile([.01, .05, .25, .50, .75, .95, .99])
        ordered = np.sort(prices.to_numpy())
        trim = int(len(ordered) * .05)
        trimmed = ordered[trim:len(ordered) - trim] if trim else ordered
        return {
            "Listings": len(prices), "P01": percentiles.loc[.01], "P05": percentiles.loc[.05],
            "P25": percentiles.loc[.25], "Median": percentiles.loc[.50], "P75": percentiles.loc[.75],
            "P95": percentiles.loc[.95], "P99": percentiles.loc[.99],
            "IQR": percentiles.loc[.75] - percentiles.loc[.25],
            "Mean": prices.mean(), "5% trimmed mean": trimmed.mean(), "Maximum": prices.max(),
        }

    return pd.DataFrame([describe(positive), describe(in_band)], index=["All positive", BAND])


def probability(prices, thresholds, *, upper=False, inclusive=True):
    """Empirical probability below (or above) fixed reporting thresholds."""
    ordered = np.sort(prices.to_numpy())
    side = "right" if upper or inclusive else "left"
    below = np.searchsorted(ordered, np.asarray(thresholds), side=side) / len(ordered)
    return 1 - below if upper else below


def heaping(prices):
    band = prices[prices.between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW)]
    dollars = band[band.mod(1).eq(0)].astype("int64")
    categories = np.select(
        [dollars.mod(1000).eq(0), dollars.mod(500).eq(0), dollars.mod(100).eq(0),
         dollars.mod(100).eq(99), dollars.mod(100).eq(95)],
        ["Multiple of $1,000", "Other multiple of $500", "Other multiple of $100",
         "Ends in 99", "Ends in 95"],
        default="Other whole-dollar price",
    )
    counts = pd.Series(categories).value_counts()
    assert counts.sum() == len(dollars)
    endings = pd.DataFrame({"Listings": counts, "%": 100 * counts / len(dollars)})
    frequent = dollars.value_counts().rename("Listings").head(8)
    return endings, frequent, len(dollars), len(band)


def group_screening(source, field):
    prices = source["price"]
    flags = pd.DataFrame({
        "Group": source[field].fillna("(missing)"), "Zero": prices.eq(0), "Positive": prices.gt(0),
        "Low": prices.gt(0) & prices.lt(LOW_PRICE_REVIEW), "High": prices.gt(HIGH_PRICE_REVIEW),
        "In band": prices.between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW),
    })
    grouped = flags.groupby("Group", observed=True)
    result = grouped.sum()
    result.insert(0, "Source n", grouped.size())
    result["Zero % of source"] = 100 * result["Zero"] / result["Source n"]
    result["Outside % of positive"] = 100 * (result["Low"] + result["High"]) / result["Positive"]
    result["Share before %"] = 100 * result["Positive"] / result["Positive"].sum()
    result["Share in band %"] = 100 * result["In band"] / result["In band"].sum()
    result["Share change pp"] = result["Share in band %"] - result["Share before %"]
    assert result["Source n"].sum() == len(source)
    assert (result["Positive"] == result["In band"] + result["Low"] + result["High"]).all()
    return result.sort_values("Source n", ascending=False)


def contextual_screening(positive, min_group=200):
    """Screen departures from each make/type/age band's median log asking price."""
    fields = ["price", "manufacturer", "type", "vehicle_age"]
    subset = positive[fields].dropna().copy()
    subset["age band"] = pd.cut(subset["vehicle_age"], bins=[0, 5, 10, 15, 20, 30, 40, np.inf], right=False)
    subset["log price"] = np.log(subset["price"])
    keys = ["manufacturer", "type", "age band"]
    grouped = subset.groupby(keys, observed=True)["log price"]
    subset["delta"] = subset["log price"] - grouped.transform("median")
    subset["deviation"] = subset["delta"].abs()
    mad = subset.groupby(keys, observed=True)["deviation"].transform("median")
    subset["peer n"] = grouped.transform("size")
    peer_scope = subset["peer n"].ge(min_group) & mad.gt(0)
    usable = subset.loc[peer_scope].copy()
    usable["robust score"] = usable["delta"] / (1.4826 * mad.loc[peer_scope])
    usable["Price band"] = np.select(
        [usable["price"].lt(LOW_PRICE_REVIEW), usable["price"].gt(HIGH_PRICE_REVIEW)],
        [f"Below ${LOW_PRICE_REVIEW:,}", f"Above ${HIGH_PRICE_REVIEW:,}"], default=f"Within {BAND}",
    )
    cohort_order = [f"Below ${LOW_PRICE_REVIEW:,}", f"Within {BAND}", f"Above ${HIGH_PRICE_REVIEW:,}"]

    def compare(limit, sample=usable):
        flags = pd.Series(np.select(
            [sample["robust score"].lt(-limit), sample["robust score"].gt(limit)],
            ["Low relative to peers", "High relative to peers"], default="Between limits",
        ), index=sample.index, name="Peer screen")
        return pd.crosstab(sample["Price band"], flags).reindex(
            index=cohort_order, columns=["Low relative to peers", "Between limits", "High relative to peers"],
            fill_value=0,
        )

    overlap = compare(3.5)
    overlap["Peer screen n"] = overlap.sum(axis=1)
    all_band = pd.Series({
        cohort_order[0]: int(positive["price"].lt(LOW_PRICE_REVIEW).sum()),
        cohort_order[1]: int(positive["price"].between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW).sum()),
        cohort_order[2]: int(positive["price"].gt(HIGH_PRICE_REVIEW).sum()),
    })
    overlap["All positive n"] = all_band
    overlap["Peer coverage %"] = 100 * overlap["Peer screen n"] / overlap["All positive n"]
    assert int(overlap["Peer screen n"].sum()) == len(usable)
    sensitivity = pd.DataFrame([
        {"Peer group minimum": minimum, "Absolute cutoff": limit,
         "Screened n": len(sample), "Flagged n": int(sample["robust score"].abs().gt(limit).sum())}
        for minimum in (min_group, 500)
        for sample in [usable.loc[usable["peer n"].ge(minimum)]]
        for limit in (3.0, 3.5, 4.0)
    ])
    sensitivity["Flagged %"] = 100 * sensitivity["Flagged n"] / sensitivity["Screened n"]
    return overlap, sensitivity, int(usable[keys].drop_duplicates().shape[0]), len(subset)


def setup_plots():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "text.parse_math": False,
        "axes.labelcolor": INK, "axes.edgecolor": MUTED, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.axisbelow": True, "axes.facecolor": PAPER, "figure.facecolor": PAPER,
        "grid.color": GRID, "grid.linewidth": .7, "xtick.color": MUTED, "ytick.color": MUTED,
        "savefig.facecolor": PAPER, "figure.dpi": 118,
    })


def figure(title, subtitle, ncols=2, height=4.8):
    fig, axes = plt.subplots(1, ncols, figsize=(10.5, height), squeeze=False)
    fig.subplots_adjust(left=.10, right=.94, top=.72, bottom=.21, wspace=.30)
    fig.text(.10, .94, title, fontsize=20, fontweight="bold", va="top")
    fig.text(.10, .852, subtitle, fontsize=10.0, color=MUTED, va="top")
    fig.text(.10, .049, SOURCE, fontsize=8.5, color=MUTED)
    return fig, axes[0]


def dollars(value, _=None):
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:g}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:g}M"
    if value >= 1_000:
        return f"${value / 1_000:g}k"
    return f"${value:,.0f}"


def price_distribution(positive):
    band = positive[positive.between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW)]
    fig, (whole, center) = figure(
        "Asking prices span several orders of magnitude",
        f"{len(positive):,} positive listings. The main price range overlaps closely across screening choices.",
    )
    edges = np.geomspace(positive.min(), positive.max() * 1.0001, 69)
    frequencies, _ = np.histogram(positive, bins=edges)
    whole.stairs(100 * frequencies / len(positive), edges, fill=True, color=BLUE, alpha=.85)
    whole.axvline(positive.median(), color=INK, linestyle="--", linewidth=1)
    whole.set(xscale="log", ylabel="Share of positive listings per bin", xlabel="Asking price (USD, log scale)")
    whole.set_title("Full range · equal log-width bins", loc="left", pad=10)
    whole.set_xticks([1, 100, 10_000, 1_000_000, 1_000_000_000],
                     ["$1", "$100", "$10k", "$1M", "$1B"])
    whole.annotate(f"Median {dollars(positive.median())}", (positive.median(), .75),
                   xycoords=("data", "axes fraction"), xytext=(12, 0), textcoords="offset points",
                   fontsize=9, color=INK, bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 2})
    whole.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    whole.grid(axis="y")
    cutoffs = np.geomspace(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW, 155)
    center.plot(cutoffs, 100 * probability(positive, cutoffs), color=BLUE, linewidth=2.5, label="All positive")
    center.plot(cutoffs, 100 * probability(band, cutoffs), color=GOLD, linestyle="--", linewidth=1.8, label=BAND)
    center.axhline(50, color=MUTED, alpha=.55, linewidth=.8)
    center.set(xscale="log", ylim=(0, 103), xlabel="Asking price (USD, log scale)", ylabel="Share at or below price")
    center.set_title("Cumulative distribution · central range", loc="left", pad=10)
    center.set_xticks([500, 2_000, 10_000, 50_000, 150_000], ["$500", "$2k", "$10k", "$50k", "$150k"])
    center.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    center.legend(loc="upper left", frameon=False, fontsize=8.8)
    center.grid(axis="y")
    return fig


def tail_figure(positive):
    fig, (left, right) = figure(
        "The review thresholds isolate different tails",
        f"Empirical probabilities among {len(positive):,} positive listings. Both probability axes use a log scale.",
    )
    specifications = (
        (left, np.unique(np.r_[np.geomspace(1, 8_000, 118), LOW_PRICE_REVIEW]), LOW_PRICE_REVIEW, False,
         [1, 10, 100, 500, 8_000], "Lower tail · below each price"),
        (right, np.unique(np.r_[np.geomspace(10_000, 1_000_000_000, 138), HIGH_PRICE_REVIEW]), HIGH_PRICE_REVIEW, True,
         [10_000, 150_000, 1_000_000, 10_000_000, 1_000_000_000], "Upper tail · above each price"),
    )
    for ax, x, cutoff, upper, ticks, title in specifications:
        y = 100 * probability(positive, x, upper=upper, inclusive=False)
        ax.step(x[y > 0], y[y > 0], where="post", color=BLUE, linewidth=2.2)
        selected = 100 * probability(positive, [cutoff], upper=upper, inclusive=False)[0]
        ax.axvline(cutoff, color=GOLD, linestyle="--", linewidth=1.2)
        ax.scatter([cutoff], [selected], color=GOLD, zorder=3, s=38)
        ax.annotate(f"{selected:.2f}%", (cutoff, selected), xytext=(9, 5 if upper else -15), textcoords="offset points",
                    color=GOLD, fontweight="bold")
        ax.set_title(title, loc="left", pad=10)
        ax.set(xscale="log", yscale="log", xlabel="Asking price (USD, log scale)")
        ax.set_xticks(ticks, [dollars(t) for t in ticks])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
        ax.grid(axis="y", which="major")
    left.set_ylabel("Share below")
    right.set_ylabel("Share above")
    return fig


def heaping_figure(endings, frequent, whole_n):
    fig, (ending_ax, amount_ax) = figure(
        "Common prices cluster on round and just-below endings",
        f"{whole_n:,} whole-dollar listings within {BAND}. Ending categories are mutually exclusive.",
        height=5.5,
    )
    fig.subplots_adjust(left=.22, right=.95, wspace=.42)
    for ax, values, names, title, color, xlabel in (
        (ending_ax, endings["%"].to_numpy(), endings.index.tolist(), "Price endings", BLUE, "Share of whole-dollar listings"),
        (amount_ax, frequent.to_numpy(), [f"${value:,}" for value in frequent.index], "Most common exact amounts", GOLD, "Listings"),
    ):
        bars = ax.barh(names, values, color=color, height=.65)
        labels = [f"{value:.1f}%" if ax is ending_ax else f"{value:,}" for value in values]
        ax.bar_label(bars, labels=labels, padding=4, fontsize=8.7, color=INK)
        ax.set_xlim(0, max(values) * 1.24)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x")
    ending_ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    amount_ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    return fig


def screening_figure(types, overall_zero, overall_outside):
    types = types.loc[types["Positive"].ge(1000)].head(8)
    fig, (zero, outside) = figure(
        "Zero prices and cutoff exposure vary by vehicle type",
        "Eight largest eligible groups. Labels show source counts, including the missing-type group.", height=5.8,
    )
    fig.subplots_adjust(left=.21, right=.95, wspace=.34)
    labels = ["SUV" if label == "suv" else label.title() for label in types.index]
    group_names = [f"{label}  ·  n={n:,}" for label, n in zip(labels, types["Source n"], strict=True)]
    limit = max(types["Zero % of source"].max(), types["Outside % of positive"].max()) * 1.22
    for ax, values, reference, title, xlabel, color in (
        (zero, types["Zero % of source"], overall_zero, "Recorded as $0", "% of source listings in type", BLUE),
        (outside, types["Outside % of positive"], overall_outside, f"Outside {BAND}", "% of positive listings in type", GOLD),
    ):
        bars = ax.barh(group_names, values, height=.59, color=color)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=4, fontsize=8.7, color=INK,
                     bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 1}, zorder=5)
        ax.axvline(reference, color=INK, linestyle="--", linewidth=1.15, label=f"Overall {reference:.1f}%", zorder=3)
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.xaxis.set_major_locator(FixedLocator(np.arange(0, limit, 3)))
        ax.set_xlim(0, limit)
        ax.legend(loc="lower right", fontsize=8, frameon=False)
    outside.set_yticklabels([])
    return fig


def peer_figure(overlap):
    fig, (ax,) = figure(
        "Some prices inside the dollar band still flag against peers",
        "Peer groups use make, vehicle type and age band. Minimum 200 listings and a nonzero median absolute deviation.",
        ncols=1, height=4.6,
    )
    fig.subplots_adjust(left=.25, right=.92, top=.70, bottom=.26)
    columns = ["Low relative to peers", "Between limits", "High relative to peers"]
    styles = [(BLUE, "/"), (LIGHT, ""), (GOLD, "\\")]
    left = np.zeros(len(overlap))
    width = 100 * overlap[columns].div(overlap["Peer screen n"], axis=0).fillna(0)
    groups = [f"{name}\nScreened n={n:,}" for name, n in zip(overlap.index, overlap["Peer screen n"], strict=True)]
    for column, (color, hatch) in zip(columns, styles, strict=True):
        values = width[column].to_numpy()
        ax.barh(groups, values, left=left, height=.58, color=color, edgecolor=PAPER, hatch=hatch, label=column)
        for y, (start, value) in enumerate(zip(left, values, strict=True)):
            if value >= 8:
                ax.text(start + value / 2, y, f"{value:.1f}%", ha="center", va="center",
                        color=INK if color == LIGHT else "white", fontweight="bold", fontsize=9,
                        bbox={"facecolor": color, "edgecolor": "none", "pad": 1})
        left += values
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel="Share of screened listings within each price band")
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.legend(loc="upper center", bbox_to_anchor=(.50, -.29), ncol=3, frameon=False, fontsize=8.8)
    return fig


def save_figure(fig, output, name):
    Path(output).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output) / f"{name}.png", dpi=185, bbox_inches="tight", pad_inches=.15)
    fig.savefig(Path(output) / f"{name}.pdf", bbox_inches="tight", pad_inches=.15)
    plt.show()
