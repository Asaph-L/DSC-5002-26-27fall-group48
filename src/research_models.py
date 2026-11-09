"""Fixed time-block experiments for price prediction and state-free benchmarks."""

import hashlib
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from src.listings import HIGH_PRICE_REVIEW, LOW_PRICE_REVIEW, RAW
from src.price_models import CATEGORICAL, FEATURES, NUMERIC, SEED, _model_frame, price_estimators
from src.research_data import purge_matches


APRIL_WINDOWS = (
    ("Apr 13–18", "2021-04-13", "2021-04-19"),
    ("Apr 19–24", "2021-04-19", "2021-04-25"),
    ("Apr 25–30", "2021-04-25", "2021-05-01"),
)
HISTORICAL_WINDOW = ("May 1–4", "2021-05-01", "2021-05-05")
LINKAGE_FIELDS = ["posting_day", "vin_key", "fingerprint_key", "vehicle_group"]
STATE_FREE = tuple(field for field in CATEGORICAL if field != "state")


def model_cohort(data):
    in_scope = data["price"].between(LOW_PRICE_REVIEW, HIGH_PRICE_REVIEW) & data["posting_day"].notna()
    band = data.loc[in_scope]
    return _model_frame(band).join(band[LINKAGE_FIELDS])


def time_split(cohort, window, mode="combined"):
    label, start, end = window
    prior = cohort.loc[cohort["posting_day"].lt(start)].sort_values("posting_day", kind="stable")
    test = cohort.loc[cohort["posting_day"].ge(start) & cohort["posting_day"].lt(end)]
    train = purge_matches(prior, test, mode)
    assert train["posting_day"].max() < test["posting_day"].min()
    assert not train["vin_key"].isin(test["vin_key"].dropna()).any()
    if mode == "combined":
        assert not train["fingerprint_key"].isin(test["fingerprint_key"].dropna()).any()
    return train, test, {
        "window": label, "earlier_n": len(prior), "train_n": len(train), "test_n": len(test),
        "vin_removed": len(prior) - len(purge_matches(prior, test, "vin")),
        "combined_removed": len(prior) - len(purge_matches(prior, test, "combined")),
    }


def grouped_median(train, test):
    """Back off from make/type/age to make/age, then age, then the training median."""
    train = train.assign(age_band=(train["vehicle_age"] // 5).fillna(-1).clip(upper=8))
    test = test.assign(age_band=(test["vehicle_age"] // 5).fillna(-1).clip(upper=8))
    estimate = pd.Series(np.nan, index=test.index)
    for fields in [["manufacturer", "type", "age_band"], ["manufacturer", "age_band"], ["age_band"]]:
        grouped = train.groupby(fields)["price"].agg(["median", "size"])
        supported = grouped.loc[grouped["size"].ge(40), "median"]
        key = pd.MultiIndex.from_frame(test[fields]) if len(fields) > 1 else test[fields[0]]
        values = pd.Series(supported.reindex(key).to_numpy(), index=test.index)
        estimate = estimate.fillna(values)
    return estimate.fillna(train["price"].median()).to_numpy()


def _predict(train, test, model_name, numeric=NUMERIC, categorical=CATEGORICAL):
    fields = [*numeric, *categorical]
    if model_name == "Grouped median":
        return grouped_median(train, test)
    with threadpool_limits(limits=2):
        if model_name == "CatBoost":
            from catboost import CatBoostRegressor

            model = CatBoostRegressor(
                cat_features=list(categorical), iterations=160, depth=7, learning_rate=.12,
                loss_function="RMSE", random_seed=SEED, thread_count=2,
                verbose=False, allow_writing_files=False,
            )
            model.fit(train[fields], np.log(train["price"]))
            return np.exp(model.predict(test[fields]))
        estimator = price_estimators(numeric=numeric, categorical=categorical)[model_name]
        estimator.fit(train[fields], train["price"])
        return estimator.predict(test[fields])


def score(actual, predicted):
    assert np.isfinite(predicted).all() and (predicted > 0).all()
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) > 1 else np.nan,
        "median_error": float(np.median(np.asarray(predicted) - np.asarray(actual))),
    }


MODELS = ("Training median", "Grouped median", "Log-price ridge", "Gradient boosting", "CatBoost")


def benchmark(cohort):
    """Run fixed specifications on April blocks and the already viewed May comparison."""
    metrics, samples, predictions, diagnostics = [], [], [], []
    for window in (*APRIL_WINDOWS, HISTORICAL_WINDOW):
        train, test, counts = time_split(cohort, window)
        label = window[0]
        samples.append(counts)
        frame = test[["price", "vehicle_age", "odometer", "state"]].copy()
        frame["window"] = label
        for name in MODELS:
            predicted = _predict(train, test, name)
            frame[name] = predicted
            metrics.append({"window": label, "model": name, "n": len(test), **score(test["price"], predicted)})

        if window == APRIL_WINDOWS[-1]:
            specifications = [
                ("Full features", NUMERIC, CATEGORICAL),
                ("Without age and mileage", (), CATEGORICAL),
                ("Without vehicle categories", NUMERIC, ("state",)),
                ("Without state", NUMERIC, STATE_FREE),
            ]
            for desc, numeric, categories in specifications:
                result = (frame["Gradient boosting"] if desc == "Full features" else
                          _predict(train, test, "Gradient boosting", numeric, categories))
                diagnostics.append({"specification": desc, **score(test["price"], result)})
            vin_train, _, _ = time_split(cohort, window, mode="vin")
            vin_result = _predict(vin_train, test, "Gradient boosting")
            diagnostics.append({"specification": "Full features, VIN-only exclusion", **score(test["price"], vin_result)})
        predictions.append(frame)
    return pd.DataFrame(metrics), pd.DataFrame(samples), pd.concat(predictions), pd.DataFrame(diagnostics)


def attribute_support(train, test):
    """Screen for established make/type cells with plausible observed age and mileage ranges."""
    keys = ["manufacturer", "type"]
    known = train.loc[train["manufacturer"].ne("__missing__") & train["type"].ne("__missing__")]
    known = known.dropna(subset=list(NUMERIC))
    grouped = known.groupby(keys, observed=True)
    counts = grouped.size().rename("support_n")
    quantiles = grouped[list(NUMERIC)].quantile([.01, .99]).unstack()
    quantiles.columns = [f"{feature}_{q}" for feature, q in quantiles.columns]
    support = counts.to_frame().join(quantiles)
    limits = test[keys].join(support, on=keys)
    keep = limits["support_n"].ge(100)
    for feature in NUMERIC:
        keep &= test[feature].between(limits[f"{feature}_0.01"], limits[f"{feature}_0.99"])
    return keep


def state_free_predictions(cohort):
    """Predict each April block from earlier records with geography omitted."""
    predictions = []
    for window in APRIL_WINDOWS:
        train, test, _ = time_split(cohort, window)
        frame = test[["price", "state", "vehicle_group"]].copy()
        frame["window"] = window[0]
        frame["supported"] = attribute_support(train, test)
        for name, key in [("Gradient boosting", "boost_log_gap"), ("Log-price ridge", "ridge_log_gap")]:
            estimated = _predict(train, test, name, categorical=STATE_FREE)
            frame[key] = np.log(frame["price"]) - np.log(estimated)
        predictions.append(frame)
    return pd.concat(predictions)


def run_experiment(root, cohort, name):
    """Reuse a local result only when source, specifications and library versions agree."""
    if name == "benchmark":
        files, compute = ("metrics", "samples", "predictions", "diagnostics"), benchmark
    elif name == "states":
        files, compute = ("state_predictions",), lambda frame: (state_free_predictions(frame),)
    else:
        raise ValueError(f"Unknown experiment {name!r}")

    root = Path(root)
    source = (root / RAW).stat()
    digest = hashlib.sha256(f"{source.st_size}:{source.st_mtime_ns}".encode())
    for module in ("listings.py", "price_models.py", "research_data.py", "research_models.py"):
        digest.update((Path(__file__).parent / module).read_bytes())
    for package in ("numpy", "pandas", "scikit-learn", "catboost"):
        digest.update(version(package).encode())
    directory = root / "outputs" / "research-cache" / f"{name}-{digest.hexdigest()[:16]}"
    paths = [directory / f"{file}.parquet" for file in files]
    if all(p.is_file() for p in paths):
        return tuple(pd.read_parquet(p) for p in paths)

    data = compute(cohort)
    directory.mkdir(parents=True, exist_ok=True)
    for p, frame in zip(paths, data, strict=True):
        temp = p.with_suffix(f".{uuid4().hex}.parquet")
        frame.to_parquet(temp)
        temp.replace(p)
    return data
