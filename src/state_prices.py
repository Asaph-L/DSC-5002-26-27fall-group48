"""State summaries, conditional resampling intervals, and projected boundaries."""

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from matplotlib.path import Path as PlotPath


ATLAS_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3.0.1/states-albers-10m.json"
ATLAS_PATH = Path("data/geo/states-albers-10m.json")
FIPS_TO_STATE = dict(zip(
    "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split(),
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".lower().split(),
    strict=True,
))
MIN_STATE_N = 100


def _median_interval(group, field, random, draws):
    """Resample vehicles within state, preserving all listings for each drawn vehicle."""
    ordered = group.sort_values(field)
    values = ordered[field].to_numpy(dtype=float)
    keys, clusters = pd.factorize(ordered["vehicle_group"])
    replicates = []
    for _ in range(draws):
        selected = random.integers(len(clusters), size=len(clusters))
        multiplicities = np.bincount(selected, minlength=len(clusters))
        cumulative = multiplicities[keys].cumsum()
        n = cumulative[-1]
        indices = np.searchsorted(cumulative, [(n - 1) // 2 + 1, n // 2 + 1])
        replicates.append(values[indices].mean())
    lower, upper = np.quantile(replicates, [.025, .975])
    return np.median(values), lower, upper


def summarize_states(raw, predictions, draws=199):
    """Compare raw medians with the supported April benchmark at the state grain."""
    random = np.random.default_rng(5002)
    valid_states = set(FIPS_TO_STATE.values())
    all_positive = raw.loc[raw["state"].isin(valid_states)]
    observed = predictions.loc[predictions["state"].isin(valid_states)]
    supported = observed.loc[observed["supported"]]
    raw_groups = all_positive.groupby("state", observed=True)
    support_groups = supported.groupby("state", observed=True)
    observed_groups = observed.groupby("state", observed=True)
    rows = []
    for state, group in raw_groups:
        median, low, high = _median_interval(group, "price", random, draws)
        pair = supported.iloc[support_groups.indices.get(state, [])]
        base = observed.iloc[observed_groups.indices.get(state, [])]
        row = {"state": state, "positive_n": len(group), "median_all": median, "median_low": low,
               "median_high": high, "april_n": len(base), "supported_n": len(pair),
               "support_share": len(pair) / len(base) if len(base) else np.nan,
               "raw_supported": pair["price"].median() if len(pair) else np.nan}
        if len(pair) >= MIN_STATE_N:
            estimate, lo, hi = _median_interval(pair, "boost_log_gap", random, draws)
            row.update(dict(zip(("gap_pct", "gap_low", "gap_high"), 100 * np.expm1([estimate, lo, hi]))))
            row["ridge_gap_pct"] = 100 * np.expm1(pair["ridge_log_gap"].median())
            row["all_april_gap_pct"] = 100 * np.expm1(base["boost_log_gap"].median())
        rows.append(row)

    result = pd.DataFrame(rows).set_index("state").sort_index()
    eligible = result["gap_pct"].notna()
    result["raw_rank"] = result["raw_supported"].where(eligible).rank(ascending=False, method="min")
    result["gap_rank"] = result["gap_pct"].rank(ascending=False, method="min")
    result["rank_change"] = result["raw_rank"] - result["gap_rank"]
    blocks = supported.groupby(["state", "window"], observed=True)["boost_log_gap"].agg(["median", "size"])
    blocks["gap_pct"] = 100 * np.expm1(blocks["median"].where(blocks["size"].ge(50)))
    return result, blocks.reset_index()


def load_state_paths(root):
    """Read the versioned D3 Albers-USA projection of Census state boundaries."""
    path = Path(root) / ATLAS_PATH
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(ATLAS_URL, timeout=25) as request:
            data = request.read()
        json.loads(data)
        path.write_bytes(data)
    atlas = json.loads(path.read_text(encoding="utf-8"))
    scale = atlas.get("transform")
    arcs = [np.asarray(arc, dtype=float) for arc in atlas["arcs"]]
    if scale is not None:
        arcs = [arc.cumsum(axis=0) * scale["scale"] + scale["translate"] for arc in arcs]

    def ring(references):
        parts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in references]
        coordinates = np.concatenate([parts[0], *(arc[1:] for arc in parts[1:])])
        if not np.array_equal(coordinates[0], coordinates[-1]):
            coordinates = np.vstack([coordinates, coordinates[0]])
        codes = np.full(len(coordinates), PlotPath.LINETO, dtype=np.uint8)
        codes[0], codes[-1] = PlotPath.MOVETO, PlotPath.CLOSEPOLY
        return PlotPath(coordinates, codes)

    paths = {}
    for geometry in atlas["objects"]["states"]["geometries"]:
        state = FIPS_TO_STATE.get(str(geometry["id"]).zfill(2))
        if state is None:
            continue
        polygons = [geometry["arcs"]] if geometry["type"] == "Polygon" else geometry["arcs"]
        paths[state] = PlotPath.make_compound_path(*(ring(references) for p in polygons for references in p))
    assert set(paths) == set(FIPS_TO_STATE.values())
    return paths
