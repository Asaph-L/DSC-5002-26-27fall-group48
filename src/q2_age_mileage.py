"""Listing-level association and prediction diagnostics for research question 2."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import sparse
from scipy.stats import t
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, SplineTransformer
from threadpoolctl import threadpool_limits

from src.listings import HIGH_PRICE_REVIEW, LOW_PRICE_REVIEW, MILEAGE_REVIEW


SEED = 5002
CONTROLS = [
    "manufacturer", "type", "condition", "fuel", "title_status", "transmission", "drive", "state",
]
AGE_EDGES = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 65, 80, 200])
MILE_EDGES = np.array([0, 25_000, 50_000, 75_000, 100_000, 125_000,
                       150_000, 175_000, 225_000, 300_000, 450_000, 1_000_001])
APRIL_WINDOWS = (("2021-04-11", "2021-04-18"), ("2021-04-18", "2021-04-25"),
                 ("2021-04-25", "2021-05-01"))
MODEL_NAMES = ("Linear ridge", "Additive splines", "Interaction splines")
MIN_SUPPORT = 50


def eligible_rows(data):
    """Build complete age/mileage rows and mark the primary positive price band."""
    eligible = data.loc[
        data["price"].gt(0) & data["vehicle_age"].notna()
        & data["odometer"].between(0, MILEAGE_REVIEW)
    ].copy()
    eligible["log_price"] = np.log(eligible["price"])
    eligible["price_band"] = eligible["price"].between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW)
    return eligible


def binned_summary(frame, field, edges, labels):
    """Compute n and three quartiles for prespecified, left-closed display bands."""
    bands = pd.cut(frame[field], edges, labels=labels, right=False)
    price = frame.groupby(bands, observed=False)["price"]
    return pd.DataFrame({"n": price.size(), "p25": price.quantile(.25),
                         "median": price.median(), "p75": price.quantile(.75)})


def _cluster_interval(estimate, contributions, groups):
    """Approximate t interval using a sum of row influence contributions by state."""
    sums = pd.Series(contributions).groupby(pd.Series(groups).fillna("unknown").astype(str)).sum()
    clusters = len(sums)
    critical = t.ppf(.975, clusters - 1)
    standard_error = np.sqrt((sums.pow(2).sum() * clusters) / (clusters - 1))
    return 100 * np.expm1([estimate, estimate - critical * standard_error,
                           estimate + critical * standard_error])


def _rebound_contrast(frame, strata, threshold):
    """Standardize 40+ versus 20–24 to overlapping strata using harmonic n weights."""
    rows = frame.loc[frame.vehicle_age.ge(40) | frame.vehicle_age.between(20, 24)].copy()
    rows["older"] = rows.vehicle_age.ge(40).astype(int)
    rows["mileage_band"] = pd.cut(rows.odometer, [-1, 50_000, 100_000, 150_000, 250_000, 1_000_000])
    if strata:
        rows = rows.dropna(subset=strata)
        rows["stratum"] = rows.groupby(strata, observed=True).ngroup()
    else:
        rows["stratum"] = 0
    groups = rows.groupby(["stratum", "older"])["log_price"].agg(group_n="size", group_mean="mean")
    n = groups["group_n"].unstack(fill_value=0)
    n = n.loc[n.min(axis=1).ge(threshold)]
    weights = (n[0] * n[1]) / (n[0] + n[1])
    weights = weights / weights.sum()
    means = groups["group_mean"].unstack().reindex(weights.index)
    estimate = ((means[1] - means[0]) * weights).sum()
    retained = rows.loc[rows.stratum.isin(weights.index)].copy()
    retained = retained.join(groups, on=["stratum", "older"])
    influence = (
        (2 * retained.older - 1) * retained.stratum.map(weights)
        * (retained.log_price - retained["group_mean"]) / retained["group_n"]
    )
    effect, low, high = _cluster_interval(estimate, influence.to_numpy(), retained.state.to_numpy())
    return {"estimate": effect, "lower": low, "upper": high,
            "strata": len(weights), "n_20_24": int(n[0].sum()), "n_40_plus": int(n[1].sum())}


def rebound_audit(primary, positive):
    """Raw and overlap-adjusted older-age contrasts with two sensitivity checks."""
    detailed = ["manufacturer", "type", "mileage_band"]
    cases = (
        ("Raw · price band", primary, [], 1),
        ("Same type + mileage", primary, ["type", "mileage_band"], 20),
        ("Same make + type + mileage", primary, detailed, 20),
        ("Detailed · positive mileage", primary.loc[primary.odometer.gt(0)], detailed, 20),
        ("Detailed · all positive prices", positive, detailed, 20),
    )
    return pd.DataFrame([{"comparison": label, **_rebound_contrast(rows, strata, threshold)}
                         for label, rows, strata, threshold in cases]).set_index("comparison")


def _spline(knots, degree=3):
    knots = np.asarray(knots).reshape(-1, 1)
    return SplineTransformer(knots=knots, degree=degree, include_bias=False).fit(knots)


class SplineDesign:
    """Use fixed cubic marginal splines and a lower-dimensional tensor interaction."""

    def __init__(self, frame):
        self.categories = OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist",
                                        min_frequency=30)
        self.categories.fit(self._category_data(frame))
        self.age = _spline([0, 5, 12, 22, 40, 125])
        self.miles = _spline(np.log1p([0, 25_000, 75_000, 150_000, 250_000, 1_000_000]))
        self.age_interaction = _spline([0, 12, 40, 125], degree=2)
        self.mile_interaction = _spline(np.log1p([0, 50_000, 125_000, 1_000_000]), degree=2)

    @staticmethod
    def _category_data(frame):
        return frame[CONTROLS].fillna("(unreported)").astype(str)

    def terms(self, age, miles, spec):
        age = np.asarray(age, dtype=float).reshape(-1, 1)
        miles = np.log1p(np.asarray(miles, dtype=float).reshape(-1, 1))
        if spec == "Linear ridge":
            return np.column_stack([age / 10, miles / 2])
        add = np.column_stack([self.age.transform(age), self.miles.transform(miles)])
        if spec == "Additive splines":
            return add
        age_basis = self.age_interaction.transform(age)
        mile_basis = self.mile_interaction.transform(miles)
        tensor = np.einsum("ij,ik->ijk", age_basis, mile_basis).reshape(len(age), -1)
        return np.column_stack([add, tensor])

    def matrix(self, frame, spec):
        numeric = sparse.csr_matrix(self.terms(frame.vehicle_age, frame.odometer, spec))
        levels = self._category_data(frame)
        # Keep the prediction design defined for a level never observed in training.
        for field, seen, dropped in zip(CONTROLS, self.categories.categories_,
                                        self.categories.drop_idx_):
            levels[field] = levels[field].where(levels[field].isin(seen), seen[dropped])
        return sparse.hstack([numeric, self.categories.transform(levels)], format="csr")


def rolling_spline_scores(primary):
    """Compare the same design specifications in three fixed April windows."""
    from src.research_data import purge_matches

    rows = []
    last = None
    for start, stop in APRIL_WINDOWS:
        before = primary.loc[primary.posting_day.lt(start)]
        test = primary.loc[primary.posting_day.ge(start) & primary.posting_day.lt(stop)]
        train = purge_matches(before, test, mode="combined")
        design = SplineDesign(train)
        window = f"Apr {int(start[-2:])}–{(pd.Timestamp(stop) - pd.Timedelta(days=1)).day}"
        for spec in MODEL_NAMES:
            with threadpool_limits(limits=2):
                fit = Ridge(alpha=10, solver="lsqr").fit(design.matrix(train, spec), train.log_price)
                prediction = fit.predict(design.matrix(test, spec))
            rows.append({"window": window, "model": spec, "train_n": len(train),
                         "purged": len(before) - len(train), "validation_n": len(test),
                         "log_mae": mean_absolute_error(test.log_price, prediction),
                         "usd_mae": mean_absolute_error(test.price, np.exp(prediction))})
        last = train, test
    return pd.DataFrame(rows), last


def fit_adjusted(primary, max_rows=80_000, cluster=True):
    """Fit the prespecified tensor spline in a deterministic April sample."""
    sample = primary.loc[primary.posting_day.lt("2021-05-01")]
    sample = sample.sample(n=min(max_rows, len(sample)), random_state=SEED)
    design = SplineDesign(sample)
    spec = MODEL_NAMES[-1]
    groups = sample.state.fillna("(unreported)").astype(str)
    with threadpool_limits(limits=2):
        matrix = sm.add_constant(design.matrix(sample, spec).toarray(), has_constant="add")
        model = sm.OLS(sample.log_price.to_numpy(), matrix)
        fit = (model.fit(cov_type="cluster", cov_kwds={"groups": groups}, use_t=True)
               if cluster else model.fit())
    terms = design.terms([10], [100_000], spec).shape[1]
    return {
        "design": design, "coef": fit.params[1:terms+1],
        "cov": fit.cov_params()[1:terms+1, 1:terms+1],
        "clusters": groups.nunique(), "n": len(sample), "rank": fit.model.rank,
        "columns": matrix.shape[1],
    }


def adjusted_difference(fit, age, miles, ref_age, ref_miles, interaction=False):
    """Calculate fitted percentage differences and pointwise state-clustered intervals."""
    spec = MODEL_NAMES[-1]
    terms = fit["design"].terms
    age, miles, ref_age, ref_miles = np.broadcast_arrays(age, miles, ref_age, ref_miles)
    difference = terms(age, miles, spec) - terms(ref_age, ref_miles, spec)
    if interaction:
        difference = (terms(age, miles, spec) - terms(age, ref_miles, spec)
                      - terms(ref_age, miles, spec) + terms(ref_age, ref_miles, spec))
    effect = difference @ fit["coef"]
    se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", difference, fit["cov"], difference), 0))
    margin = t.ppf(.975, fit["clusters"] - 1) * se
    return 100 * np.expm1(np.column_stack([effect, effect - margin, effect + margin]))


def support_counts(frame, age, miles):
    """Return observed counts of age-by-odometer cells for queried coordinates."""
    counts, *_ = np.histogram2d(frame.vehicle_age, frame.odometer, bins=[AGE_EDGES, MILE_EDGES])
    age_index = np.searchsorted(AGE_EDGES, age, side="right") - 1
    mile_index = np.searchsorted(MILE_EDGES, miles, side="right") - 1
    return counts[age_index, mile_index].astype(int)


def quantile_diagnostic(train, test, train_cap=90_000):
    """Fit fixed 25th/50th/75th quantile trees and assess held-out calibration."""
    train = train.sample(n=min(len(train), train_cap), random_state=SEED)
    encode = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    category = encode.fit_transform(SplineDesign._category_data(train))
    x_train = np.column_stack([train[["vehicle_age", "odometer"]].to_numpy(dtype=float), category])
    x_test = np.column_stack([test[["vehicle_age", "odometer"]].to_numpy(dtype=float),
                              encode.transform(SplineDesign._category_data(test))])
    categorical = [False, False] + [True] * len(CONTROLS)
    rows = []
    results = pd.DataFrame({"age": test.vehicle_age.to_numpy(dtype=float),
                            "price": test.price.to_numpy(dtype=float)})
    with threadpool_limits(limits=2):
        for q in (.25, .50, .75):
            fit = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, categorical_features=categorical, max_iter=100,
                min_samples_leaf=75, l2_regularization=1, early_stopping=False, random_state=SEED,
            ).fit(x_train, train.log_price)
            forecast = np.exp(fit.predict(x_test))
            baseline = np.repeat(train.price.quantile(q), len(test))
            rows.append({"quantile": q, "test_fraction_below": float(np.mean(test.price <= forecast)),
                         "pinball_usd": mean_pinball_loss(test.price, forecast, alpha=q),
                         "baseline_pinball_usd": mean_pinball_loss(test.price, baseline, alpha=q)})
            results[f"p{round(100*q)}"] = forecast
    results["crossed"] = results.p25.gt(results.p50) | results.p50.gt(results.p75)
    bounds = np.sort(results[["p25", "p50", "p75"]].to_numpy(), axis=1)
    results[["p25", "p50", "p75"]] = bounds
    results["covered"] = results.price.between(results.p25, results.p75)
    results["width"] = results.p75 - results.p25
    labels = ["0–4", "5–9", "10–14", "15–19", "20–24", "25–39", "40+"]
    results["age_band"] = pd.cut(results.age, [0, 5, 10, 15, 20, 25, 40, 200],
                                 labels=labels, right=False)
    by_age = results.groupby("age_band", observed=True).agg(
        n=("price", "size"), observed_median=("price", "median"), median_p25=("p25", "median"),
        median_p50=("p50", "median"), median_p75=("p75", "median"),
        coverage=("covered", "mean"), width=("width", "median"),
    )
    return {"scores": pd.DataFrame(rows).set_index("quantile"), "age": by_age,
            "train_n": len(train), "test_n": len(test), "coverage": float(results.covered.mean()),
            "crossing_rate": float(results.crossed.mean())}
