"""Time-based validation and adjusted associations for asking-price models."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.listings import (
    COLUMNS, HIGH_PRICE_REVIEW, LOW_PRICE_REVIEW, MILEAGE_REVIEW,
    check_input, clean_listings,
)


SEED = 5002
NUMERIC = ("vehicle_age", "odometer")
CATEGORICAL = (
    "manufacturer", "condition", "fuel", "type", "title_status",
    "transmission", "drive", "state",
)
FEATURES = [*NUMERIC, *CATEGORICAL]
MISSING = "__missing__"
OTHER = "__rare__"
MAX_OLS_ROWS = 80_000


def chronological_cohort(path):
    """Build the dated price-band cohort and a nonoverlapping held-out time block."""
    check_input(path)
    source = pd.read_csv(path, usecols=[*COLUMNS, "VIN", "model"], dtype="string")
    cleaned, _, _ = clean_listings(source[list(COLUMNS)])
    band = cleaned.loc[cleaned["price"].between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW)].copy()
    # Reuse local posting day (the date as written in the advertisement).
    posting = source.loc[band.index, "posting_date"]
    dates = pd.to_datetime(posting.str[:10], format="%Y-%m-%d", errors="coerce")
    iso_valid = pd.to_datetime(posting, format="ISO8601", errors="coerce", utc=True).notna()
    dates = dates.where(iso_valid).dropna()
    band = band.loc[dates.index]
    if dates.nunique() < 2:
        raise ValueError("At least two valid posting days are needed for a time split.")

    target_position = int(np.floor(0.8 * len(dates)))
    cutoff = dates.sort_values().iloc[target_position]
    if cutoff == dates.min():
        cutoff = dates.sort_values().drop_duplicates().iloc[1]
    held_out = dates.ge(cutoff)
    first_train = dates.index[~held_out]
    test_rows = dates.index[held_out]

    # VINs and a conservative exact match on stable attributes are split keys only.
    vin = source.loc[band.index, "VIN"].str.strip().str.upper()
    vin = vin.where(vin.str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", na=False))
    model = source.loc[band.index, "model"].str.strip().str.lower().replace("", pd.NA)
    keys = band[["state", "manufacturer", "model_year", "odometer", "type", "paint_color"]].copy()
    keys["model"] = model
    fingerprint = pd.util.hash_pandas_object(keys, index=False).astype("UInt64")
    fingerprint = fingerprint.where(keys[["state", "model_year", "odometer", "model"]].notna().all(axis=1))
    overlapping_vin = vin.loc[first_train].isin(vin.loc[test_rows].dropna())
    overlapping_key = fingerprint.loc[first_train].isin(fingerprint.loc[test_rows].dropna())
    train_rows = first_train[~(overlapping_vin | overlapping_key)]
    if not len(train_rows) or not len(test_rows):
        raise ValueError("Time split and duplicate protection left an empty train or test cohort.")
    assert dates.loc[train_rows].max() < dates.loc[test_rows].min()
    assert not vin.loc[train_rows].isin(vin.loc[test_rows].dropna()).any()
    assert not fingerprint.loc[train_rows].isin(fingerprint.loc[test_rows].dropna()).any()

    train = _model_frame(band.loc[train_rows])
    test = _model_frame(band.loc[test_rows])
    counts = {
        "source_positive": len(cleaned),
        "within_price_band": int(cleaned["price"].between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW).sum()),
        "dated_rows": len(band),
        "initial_train_n": len(first_train),
        "train_n": len(train),
        "purge_n": len(first_train) - len(train),
        "test_n": len(test),
        "train_first_day": dates.loc[train_rows].min().date().isoformat(),
        "train_last_day": dates.loc[train_rows].max().date().isoformat(),
        "test_first_day": dates.loc[test_rows].min().date().isoformat(),
        "test_last_day": dates.loc[test_rows].max().date().isoformat(),
    }
    return train, test, counts


def _model_frame(frame):
    prepared = frame[["price", *FEATURES]].copy()
    prepared["odometer"] = prepared["odometer"].where(prepared["odometer"].le(MILEAGE_REVIEW))
    for field in NUMERIC:
        prepared[field] = prepared[field].astype("float64")
    for field in CATEGORICAL:
        prepared[field] = prepared[field].fillna(MISSING).astype(str)
    return prepared


def prediction_scores(train, test):
    """Evaluate fixed model specifications on the same held-out USD targets."""
    linear_inputs = ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), list(NUMERIC)),
        ("categories", OneHotEncoder(handle_unknown="ignore"), list(CATEGORICAL)),
    ])
    tree_inputs = ColumnTransformer([
        ("numeric", "passthrough", list(NUMERIC)),
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), list(CATEGORICAL)),
    ])
    ridge = make_pipeline(linear_inputs, Ridge(alpha=10, solver="lsqr"))
    boosting = make_pipeline(tree_inputs, HistGradientBoostingRegressor(
        categorical_features=[False] * len(NUMERIC) + [True] * len(CATEGORICAL),
        max_iter=180, max_leaf_nodes=31, min_samples_leaf=45,
        learning_rate=0.08, l2_regularization=1.0, early_stopping=False, random_state=SEED,
    ))
    models = {
        "Training median": DummyRegressor(strategy="median"),
        "Log-price ridge": TransformedTargetRegressor(regressor=ridge, func=np.log, inverse_func=np.exp),
        "Gradient boosting": TransformedTargetRegressor(regressor=boosting, func=np.log, inverse_func=np.exp),
    }
    scores = []
    with threadpool_limits(limits=4):
        for label, model in models.items():
            model.fit(train[FEATURES], train["price"])
            estimate = model.predict(test[FEATURES])
            actual = test["price"]
            assert np.isfinite(estimate).all()
            scores.append({
                "model": label,
                "mae": float(mean_absolute_error(actual, estimate)),
                "rmse": float(np.sqrt(mean_squared_error(actual, estimate))),
                "r2": float(r2_score(actual, estimate)),
            })
    return scores


def adjusted_associations(train):
    """Fit log OLS in a bounded training subset and report state-clustered intervals."""
    available = train.dropna(subset=list(NUMERIC)).loc[lambda frame: frame["state"].ne(MISSING)]
    sample = available.sample(n=min(MAX_OLS_ROWS, len(available)), random_state=SEED).copy()
    sample["log_price"] = np.log(sample["price"])
    sample["age_5y"] = sample["vehicle_age"] / 5
    sample["mileage_10k"] = sample["odometer"] / 10_000
    # Pool small non-geographic category levels for a stable design matrix.
    for field in CATEGORICAL:
        if field != "state":
            counts = sample[field].value_counts()
            sample[field] = sample[field].where(sample[field].map(counts).ge(40), OTHER)

    controls = " + ".join(f"C({field})" for field in CATEGORICAL)
    with threadpool_limits(limits=4):
        result = smf.ols(f"log_price ~ age_5y + mileage_10k + {controls}", data=sample).fit(
            cov_type="cluster", cov_kwds={"groups": sample["state"]}, use_t=True,
        )
    intervals = result.conf_int()
    rows = []
    for field, label in [("age_5y", "+5 years of age"), ("mileage_10k", "+10,000 miles")]:
        estimate = float(100 * np.expm1(result.params[field]))
        lower, upper = (float(value) for value in 100 * np.expm1(intervals.loc[field]))
        assert lower <= estimate <= upper
        rows.append({"contrast": label, "estimate_pct": estimate, "low_pct": lower, "high_pct": upper})
    return rows, len(sample), int(sample["state"].nunique())
