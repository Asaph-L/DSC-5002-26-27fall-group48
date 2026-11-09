
"""Distribution summaries and overlap-weighted comparisons for proposal question 3."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd

from src.listings import MILEAGE_REVIEW


MIN_VEHICLES_PER_ARM = 15
BOOTSTRAPS = 399
SEED = 5002
FIELDS = ("condition", "manufacturer", "fuel", "type")
TITLES = ("Seller-reported condition", "Manufacturer", "Fuel type", "Body type")
AGE_EDGES = [0, 5, 10, 15, 20, 30, np.inf]
MILEAGE_EDGES = [0, 50_000, 100_000, 150_000, 250_000, np.inf]
REGIONS = {
    "Northeast": "ct me ma nh ri vt nj ny pa",
    "Midwest": "il in mi oh wi ia ks mn mo ne nd sd",
    "South": "de fl ga md nc sc va dc wv al ky ms tn ar la ok tx",
    "West": "az co id mt nv nm ut wy ak ca hi or wa",
}
STATE_REGION = {state: region for region, codes in REGIONS.items() for state in codes.split()}


@dataclass(frozen=True)
class Comparison:
    field: str
    first: str
    second: str
    label: str
    controls: tuple[str, ...]


COMPARISONS = (
    Comparison("condition", "good", "excellent", "Good − excellent", ("make_stratum", "type")),
    Comparison("manufacturer", "ford", "chevrolet", "Ford − Chevrolet", ("type", "fuel")),
    Comparison("fuel", "diesel", "gas", "Diesel − gas", ("make_stratum", "type")),
    Comparison("type", "pickup", "suv", "Pickup − SUV", ("make_stratum", "fuel")),
)


def prepare_listings(data):
    """Construct bins for comparable listings while retaining every positive price."""
    columns = ["price", "vehicle_age", "odometer", "state", "vehicle_group", *FIELDS]
    frame = data[columns].copy()
    assert frame["price"].gt(0).all() and frame["vehicle_group"].notna().all()
    frame["region"] = frame["state"].map(STATE_REGION)
    common_makes = (
        frame["manufacturer"].value_counts().rename_axis("make").reset_index(name="n")
        .sort_values(["n", "make"], ascending=[False, True]).head(10)["make"]
    )
    frame["make_stratum"] = frame["manufacturer"].where(frame["manufacturer"].isin(common_makes), "Other make")
    frame.loc[frame["manufacturer"].isna(), "make_stratum"] = pd.NA
    # Zero may be valid for some new listings and a placeholder for others.
    known_mileage = frame["odometer"].where(frame["odometer"].between(1, MILEAGE_REVIEW))
    frame["age_band"] = pd.cut(frame["vehicle_age"], AGE_EDGES, right=False, labels=False)
    frame["mileage_band"] = pd.cut(known_mileage, MILEAGE_EDGES, right=False, labels=False)
    frame["age_fine"] = pd.cut(frame["vehicle_age"], [0, 3, 6, 9, 12, 15, 20, 25, 35, np.inf],
                              right=False, labels=False)
    frame["mileage_fine"] = pd.cut(known_mileage, [0, 25_000, 50_000, 75_000, 100_000, 125_000,
                                                   150_000, 200_000, 250_000, np.inf],
                                  right=False, labels=False)
    return frame


def profile_groups(data):
    """Return aggregate coverage and distribution quantiles for every recorded level."""
    counts = {field: data[field].notna().sum() for field in FIELDS}
    coverage = pd.DataFrame({
        "field": FIELDS,
        "known": [counts[field] for field in FIELDS],
        "unknown": [len(data) - counts[field] for field in FIELDS],
        "known_share": [counts[field] / len(data) for field in FIELDS],
    })
    pieces = []
    for field in FIELDS:
        grouped = data.groupby(field, observed=True)["price"]
        percentiles = grouped.quantile([.10, .25, .50, .75, .90]).unstack()
        percentiles.columns = ["p10", "p25", "median", "p75", "p90"]
        table = percentiles.join(grouped.size().rename("n")).rename_axis("label").reset_index()
        table.insert(0, "field", field)
        pieces.append(table)
    distributions = pd.concat(pieces, ignore_index=True)
    assert distributions.groupby("field")["n"].sum().to_dict() == counts
    return coverage, distributions


@dataclass
class Arm:
    """Prices sorted once for efficient empirical quantiles."""

    price: np.ndarray
    vehicle: np.ndarray
    shared: np.ndarray
    support_price: np.ndarray
    support_vehicle: np.ndarray
    support_cell: np.ndarray


def weighted_median(prices, weights):
    cumulative = np.cumsum(weights)
    if not len(cumulative) or cumulative[-1] <= 0:
        return np.nan
    return float(prices[np.searchsorted(cumulative, cumulative[-1] / 2)])


def effective_n(weights):
    return float(weights.sum() ** 2 / np.dot(weights, weights))


def _prepare_pair(data, comparison, fine=False):
    bins = ("age_fine", "mileage_fine") if fine else ("age_band", "mileage_band")
    fields = [*bins, "region", *comparison.controls]
    pair = data.loc[data[comparison.field].isin([comparison.first, comparison.second])].copy()
    pair["_arm"] = pair[comparison.field].eq(comparison.second).astype("int8")
    observed = pair.loc[pair[fields].notna().all(axis=1)].copy()
    observed["_cell"] = observed.groupby(fields, observed=True, sort=True).ngroup()
    vehicle_counts = observed.groupby(["_cell", "_arm"])["vehicle_group"].nunique().unstack(fill_value=0)
    if len(vehicle_counts.columns) != 2:
        raise ValueError(f"{comparison.label} needs both groups among complete listings.")
    support = vehicle_counts.index[vehicle_counts.ge(MIN_VEHICLES_PER_ARM).all(axis=1)]
    if support.empty:
        raise ValueError(f"{comparison.label} has no comparable strata at the chosen minimum.")
    indexer = pd.Series(np.arange(len(support)), index=support)
    pair["_cell"] = observed["_cell"].map(indexer).reindex(pair.index).fillna(-1).astype(int)
    pair["_vehicle"], vehicle_groups = pd.factorize(pair["vehicle_group"], sort=True)
    arms = []
    for code in (0, 1):
        arm = pair.loc[pair["_arm"].eq(code)].sort_values("price", kind="stable")
        prices, vehicles, cells = [arm[key].to_numpy() for key in ("price", "_vehicle", "_cell")]
        shared = cells >= 0
        arms.append(Arm(prices, vehicles, shared, prices[shared], vehicles[shared], cells[shared]))
    counts = [np.bincount(arm.support_cell, minlength=len(support)) for arm in arms]
    mass = (counts[0] * counts[1]) / (counts[0] + counts[1])
    assert np.all(mass > 0)
    return pair, arms, counts, mass, len(vehicle_groups), fields


def _medians(arm, multiplier, mass):
    raw = multiplier[arm.vehicle]
    overlap = multiplier[arm.support_vehicle]
    cell = arm.support_cell
    denominator = np.bincount(cell, weights=overlap, minlength=len(mass))
    standardized = overlap * mass[cell] / denominator[cell]
    return (
        weighted_median(arm.price, raw),
        weighted_median(arm.support_price, overlap),
        weighted_median(arm.support_price, standardized),
    )


def compare_groups(data, comparison, *, draws=0, seed=SEED, fine=False):
    """Compare raw, overlap-only, and standardized median prices for a prespecified pair."""
    pair, arms, counts, mass, vehicle_count, fields = _prepare_pair(data, comparison, fine)
    estimates = np.asarray([_medians(arm, np.ones(vehicle_count), mass) for arm in arms])
    table = pd.DataFrame({
        "comparison": comparison.label,
        "stage": ("Raw", "Overlap only", "Mix adjusted"),
        "median_first": estimates[0],
        "median_second": estimates[1],
        "gap": estimates[0] - estimates[1],
    })
    if draws:
        rng = np.random.default_rng(seed)
        differences = np.empty((draws, 3))
        retries = 0
        for i in range(draws):
            while True:
                multiplier = rng.poisson(1, size=vehicle_count)
                if all(np.bincount(arm.support_cell, weights=multiplier[arm.support_vehicle],
                                   minlength=len(mass)).min() > 0 for arm in arms):
                    break
                retries += 1
                if retries > draws:
                    raise RuntimeError("Too many resamples lost a common-support stratum.")
            differences[i] = np.subtract(*[_medians(arm, multiplier, mass) for arm in arms])
        table["lo"], table["hi"] = np.quantile(differences, [.025, .975], axis=0)
    support_rows = []
    for j, arm in enumerate(arms):
        weights = mass[arm.support_cell] / counts[j][arm.support_cell]
        assert np.allclose(np.bincount(arm.support_cell, weights, minlength=len(mass)), mass)
        cluster_weights = np.bincount(arm.support_vehicle, weights=weights)
        support_rows.append({
            "comparison": comparison.label,
            "group": [comparison.first, comparison.second][j],
            "raw_n": len(arm.price),
            "overlap_n": len(arm.support_price),
            "retained": float(arm.shared.mean()),
            "listing_ess": effective_n(weights),
            "vehicle_ess": effective_n(cluster_weights),
            "strata": len(mass),
        })
    overlap = pair.loc[pair["_cell"].ge(0)].copy()
    cell = overlap["_cell"].to_numpy()
    arm_index = overlap["_arm"].to_numpy()
    overlap["_weight"] = mass[cell] / np.stack(counts)[arm_index, cell]
    balance = []
    for key in ("vehicle_age", "odometer"):
        raw = pair[key].where(pair[key].between(1, MILEAGE_REVIEW)) if key == "odometer" else pair[key]
        valid = raw.notna()
        a = raw.loc[valid & pair["_arm"].eq(0)].astype(float)
        b = raw.loc[valid & pair["_arm"].eq(1)].astype(float)
        denom = np.sqrt((a.var(ddof=0) + b.var(ddof=0)) / 2)
        means = [np.average(group[key].astype(float), weights=group["_weight"])
                 for _, group in overlap.groupby("_arm")]
        balance.append({
            "comparison": comparison.label,
            "feature": key,
            "raw_smd": (a.mean() - b.mean()) / denom if denom else 0,
            "adjusted_smd": (means[0] - means[1]) / denom if denom else 0,
        })
    return {
        "estimates": table,
        "support": pd.DataFrame(support_rows),
        "balance": pd.DataFrame(balance),
        # Internal frame for aggregate ECDF only. Never written or displayed row by row.
        "overlap": overlap[["price", "_arm", "_weight", "age_band"]],
        "fields": fields,
    }


def summarize_condition(data):
    """Characterize condition nonreporting separately from the focal contrast."""
    frame = data.copy()
    frame["status"] = np.where(frame["condition"].isna(), "Condition unreported", "Other reported condition")
    frame.loc[frame["condition"].eq("good").fillna(False), "status"] = "Good"
    frame.loc[frame["condition"].eq("excellent").fillna(False), "status"] = "Excellent"
    frame["usable_mileage"] = frame["odometer"].where(frame["odometer"].between(1, MILEAGE_REVIEW))
    frame["age_group"] = pd.cut(frame["vehicle_age"], AGE_EDGES, right=False,
                                labels=["0–4", "5–9", "10–14", "15–19", "20–29", "30+"])
    frame["age_group"] = frame["age_group"].cat.add_categories("Unknown").fillna("Unknown")
    table = frame.groupby("status").agg(
        n=("price", "size"), median_price=("price", "median"),
        age_n=("vehicle_age", "count"), median_age=("vehicle_age", "median"),
        mileage_n=("usable_mileage", "count"), median_mileage=("usable_mileage", "median"),
    )
    assert table["n"].sum() == len(data)
    by_age = frame.groupby("age_group", observed=True)["condition"].agg(n="size", reported="count")
    by_age["missing_share"] = 1 - by_age["reported"] / by_age["n"]
    return table, by_age


# Chart palette and formatting are shared across the four notebook figures.
BLUE = "#245A78"
TEAL = "#087F79"
OCHRE = "#A35523"
GRAY = "#5D737E"
INK = "#17323F"
GRID = "#DCE5E9"


def set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "xtick.color": GRAY,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "figure.dpi": 108,
        "text.parse_math": False,
    })


def _money(value, _position=None):
    if value == 0:
        return "$0"
    magnitude = abs(value)
    scale = f"{magnitude / 1000:g}k" if magnitude >= 1000 else f"{magnitude:g}"
    return f"{'−' if value < 0 else ''}\u0024{scale}"


def _label(value):
    return {"suv": "SUV", "mini-van": "Minivan", "offroad": "Off-road",
            "gas": "Gasoline", "rv": "RV", "gmc": "GMC"}.get(value, value.capitalize())


def distribution_figure(coverage, distributions):
    """Aligned percentile intervals ordered by median within each panel."""
    fig, axes = plt.subplots(2, 2, figsize=(13.7, 10.2))
    fig.subplots_adjust(left=.14, right=.978, top=.85, bottom=.11, wspace=.44, hspace=.52)
    fig.text(.14, .953, "Asking prices vary widely within vehicle groups", color=INK, size=21, weight="bold")
    fig.text(.14, .915, "Median dot, middle 50% thick line, 10th to 90th percentile thin line", color=GRAY, size=11.5)
    panels = []
    for field in FIELDS:
        table = distributions.loc[distributions["field"].eq(field)].sort_values(["n", "label"], ascending=[False, True])
        table = table.head(8) if field == "manufacturer" else table.loc[table["n"].ge(200)]
        panels.append(table.sort_values("median", ascending=False))
    minimum = min(float(p["p10"].min()) for p in panels)
    maximum = max(float(p["p90"].max()) for p in panels)
    ticks = [t for t in [1, 10, 100, 1000, 10_000, 100_000, 1_000_000] if minimum / 3 <= t <= maximum * 2]
    for ax, field, title, table in zip(axes.flat, FIELDS, TITLES, panels):
        y = np.arange(len(table))
        ax.hlines(y, table["p10"], table["p90"], color="#A8BCC6", linewidth=1.6)
        ax.hlines(y, table["p25"], table["p75"], color=BLUE, linewidth=5.2)
        ax.scatter(table["median"], y, c=TEAL, edgecolors="white", linewidth=.7, s=44, zorder=4)
        ax.set_yticks(y, [f"{_label(a)}  {b:,}" for a, b in zip(table["label"], table["n"])])
        ax.tick_params(axis="y", length=0, labelsize=10)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(min(minimum * .8, ticks[0]), max(maximum * 1.15, ticks[-1]))
        ax.set_xticks(ticks, [_money(x) for x in ticks])
        ax.set_xlabel("Asking price (USD, log scale)")
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        known = coverage.loc[coverage["field"].eq(field), "known_share"].iat[0]
        ax.set_title(f"{title}  ·  {known:.0%} recorded", loc="left", pad=13, color=INK)
    fig.text(.14, .034, "Craigslist v10 positive prices. Labels include listing counts. Manufacturers show the eight most frequent. Other panels show groups with n ≥ 200.",
             color=GRAY, size=9.5)
    return fig


def contrast_figure(estimates):
    """Compare before/after median differences on one USD scale."""
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(left=.19, right=.95, top=.72, bottom=.19)
    fig.text(.19, .925, "Comparable vehicle mix changes the observed gaps", size=20, color=INK, weight="bold")
    fig.text(.19, .863, "Difference in median asking price. Positive values mean the first named group is higher.",
             size=11, color=GRAY)
    styles = [
        ("Raw", -.22, GRAY, "o"),
        ("Overlap only", 0, OCHRE, "s"),
        ("Mix adjusted", .22, TEAL, "D"),
    ]
    labels = list(estimates["comparison"].unique())
    ax.axvline(0, color="#657A85", lw=1.1)
    for i, label in enumerate(labels):
        for stage, shift, color, marker in styles:
            row = estimates.loc[estimates["comparison"].eq(label) & estimates["stage"].eq(stage)].iloc[0]
            ax.hlines(i + shift, row["lo"], row["hi"], color=color, lw=1.7, zorder=3)
            ax.scatter(row["gap"], i + shift, color=color, marker=marker, s=50, zorder=4)
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(axis="y", length=0, pad=11)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_xlabel("First group minus second group (USD)")
    ax.grid(axis="x", color=GRID, linewidth=.75)
    legend = [Line2D([0], [0], color=c, marker=m, linestyle="none", label=s) for s, _, c, m in styles]
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(-.01, 1.06), ncol=3, frameon=False)
    fig.text(.19, .055, "Pointwise 95% vehicle-group bootstrap intervals. Each contrast uses its own overlap population.",
             color=GRAY, size=10)
    return fig


def _cdf_prices(prices, weights, grid):
    order = np.argsort(prices, kind="stable")
    cumulative = np.concatenate(([0.0], np.cumsum(weights[order])))
    location = np.searchsorted(prices[order], grid, side="right")
    return cumulative[location] / cumulative[-1]


def condition_cdf_figure(data, overlap, estimate):
    """Compare good and excellent price distributions before and after balancing."""
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.11, right=.97, top=.73, bottom=.20, wspace=.22)
    fig.text(.11, .93, "The condition comparison depends on vehicle mix", size=20, color=INK, weight="bold")
    fig.text(.11, .865, "Share of listings with an asking price at or below each value", size=11, color=GRAY)
    grid = np.linspace(500, 75_000, 300)
    for i, (ax, title) in enumerate(zip(axes, ["All positive listings", "Comparable mix, weighted"])):
        row = estimate.loc[estimate["stage"].eq(("Raw", "Mix adjusted")[i])].iloc[0]
        for k, (name, color, linestyle) in enumerate([("good", TEAL, "-"), ("excellent", BLUE, "--")]):
            if i == 0:
                price = data.loc[data["condition"].eq(name), "price"].to_numpy()
                weights = np.ones(len(price))
            else:
                subset = overlap.loc[overlap["_arm"].eq(k)]
                price = subset["price"].to_numpy()
                weights = subset["_weight"].to_numpy()
            ax.plot(grid, _cdf_prices(price, weights, grid), label=f"{name.capitalize()}  ·  n={len(price):,}",
                    color=color, ls=linestyle, lw=2.2)
            median = row[["median_first", "median_second"][k]]
            ax.scatter(median, .5, color=color, zorder=4, s=35, edgecolors="white", linewidth=.7)
        ax.axhline(.5, color=GRID, linewidth=1)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.set_title(title, color=INK, loc="left", pad=9)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_ylim(0, 1.01)
        ax.xaxis.set_major_formatter(FuncFormatter(_money))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Asking price (USD)")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(.11, .07, "Dots mark each median. The view focuses on $500–$75,000. Each curve uses its full cohort as the denominator.",
             color=GRAY, size=9.6)
    return fig


def condition_coverage_figure(condition, by_age):
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.5))
    fig.subplots_adjust(left=.225, right=.96, top=.72, bottom=.25, wspace=.43)
    fig.text(.12, .94, "Condition is missing unevenly across the sample", size=20, color=INK, weight="bold")
    fig.text(.12, .875, "The unreported group shows which listings the condition comparison leaves out", size=11, color=GRAY)
    labels = ["Good", "Excellent", "Other reported condition", "Condition unreported"]
    rows = condition.reindex(labels)
    y = np.arange(len(rows))
    axes[0].barh(y, rows["median_price"], color=[BLUE, BLUE, BLUE, OCHRE], height=.49)
    axes[0].set_yticks(y, [f"{label}\n{count:,} listings" for label, count in zip(rows.index, rows["n"])])
    axes[0].invert_yaxis()
    axes[0].tick_params(axis="y", length=0)
    axes[0].set_xlim(0, rows["median_price"].max() * 1.25)
    axes[0].xaxis.set_major_formatter(FuncFormatter(_money))
    for i, val in enumerate(rows["median_price"]):
        axes[0].annotate(f"{chr(36)}{val:,.0f}", (val, i), xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center", color=INK, fontsize=10)
    axes[0].set_title("Median asking price (USD)", loc="left", pad=12, color=INK)
    pos = np.arange(len(by_age))
    axes[1].bar(pos, by_age["missing_share"], color=[TEAL] * (len(pos) - 1) + [GRAY], width=.66)
    axes[1].set_xticks(pos, by_age.index, rotation=34, ha="right")
    axes[1].set_ylim(0, min(1.07, by_age["missing_share"].max() + .15))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[1].set_title("Condition unreported by age", loc="left", pad=12, color=INK)
    axes[1].set_xlabel("Vehicle age (years)")
    for i, value in enumerate(by_age["missing_share"]):
        axes[1].annotate(f"{value:.0%}", (i, value), xytext=(0, 4), textcoords="offset points",
                         ha="center", color=INK, fontsize=9.5)
    for ax in axes:
        ax.set_axisbelow(True)
    axes[1].grid(axis="y", color=GRID, linewidth=.8)
    axes[0].grid(axis="x", color=GRID, linewidth=.8)
    fig.text(.12, .04, "All positive-price listings. The right panel denominator is all listings in each age band, including unknown age.",
             color=GRAY, size=9.8)
    return fig


def condition_contrast_by_age(overlap):
    """Show whether the adjusted aggregate condition gap hides age-specific patterns."""
    labels = ["0–4", "5–9", "10–14", "15–19", "20–29", "30+"]
    records = []
    for band, group in overlap.groupby("age_band", sort=True):
        arms = [arm.sort_values("price") for _, arm in group.groupby("_arm")]
        if len(arms) != 2 or min(map(len, arms)) < 200:
            continue
        medians = [weighted_median(arm["price"].to_numpy(), arm["_weight"].to_numpy()) for arm in arms]
        records.append({
            "age_years": labels[int(band)],
            "good_n": len(arms[0]),
            "excellent_n": len(arms[1]),
            "good_median": medians[0],
            "excellent_median": medians[1],
            "good_minus_excellent": medians[0] - medians[1],
        })
    return pd.DataFrame(records)


def condition_age_figure(condition_by_age):
    """Display the age-specific direction and magnitude within common support."""
    fig, ax = plt.subplots(figsize=(11.4, 5.9))
    fig.subplots_adjust(left=.29, right=.95, top=.72, bottom=.20)
    fig.text(.29, .935, "Condition gaps differ across age bands", size=20, color=INK, weight="bold")
    fig.text(.29, .874, "Weighted medians within comparable strata. Labels show Good / Excellent counts.",
             size=10.5, color=GRAY)
    y = np.arange(len(condition_by_age))
    gap = condition_by_age["good_minus_excellent"].to_numpy()
    colors = [TEAL if value >= 0 else BLUE for value in gap]
    ax.axvline(0, color=GRAY, linewidth=1.1)
    ax.barh(y, gap, color=colors, height=.49)
    ax.set_yticks(y, [
        f"{row.age_years} years  ·  {row.good_n:,} / {row.excellent_n:,}"
        for row in condition_by_age.itertuples(index=False)
    ])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=9)
    ax.set_xlim(min(gap.min(), 0) - 1700, max(gap.max(), 0) + 1600)
    ax.xaxis.set_major_formatter(FuncFormatter(_money))
    ax.set_xlabel("Good median minus excellent median (USD)")
    ax.grid(axis="x", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    for i, value in enumerate(gap):
        ax.annotate(f"{'−' if value < 0 else '+'}\u0024{abs(value):,.0f}",
                    (value, i), xytext=((-7 if value < 0 else 7), 0),
                    textcoords="offset points", ha="right" if value < 0 else "left",
                    va="center", fontsize=9.5, color=INK)
    fig.text(.29, .067, "Each age band describes its own shared vehicle mix. The 30+ band has the smallest counts.",
             color=GRAY, size=9.5)
    return fig

