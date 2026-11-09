"""Read, profile, and clean the Craigslist listings without publishing raw rows."""

from pathlib import Path

import numpy as np
import pandas as pd


RAW = Path("data/raw/vehicles.csv")
PROCESSED = Path("data/processed/listings.csv")
COLUMNS = (
    "id", "price", "year", "manufacturer", "condition", "fuel", "odometer",
    "title_status", "transmission", "drive", "size", "type", "paint_color",
    "state", "posting_date",
)
CATEGORIES = (
    "manufacturer", "condition", "fuel", "title_status", "transmission",
    "drive", "size", "type", "paint_color", "state",
)
MIN_MODEL_YEAR = 1886
MILEAGE_REVIEW = 1_000_000
LOW_PRICE_REVIEW = 500
HIGH_PRICE_REVIEW = 150_000


def check_input(path):
    """Return the CSV header; give actionable errors for a missing/wrong file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Follow the download steps in DATA.md.")
    columns = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"Unexpected dataset schema; missing columns: {missing}")
    return columns


def profile_source(path):
    """Count source rows and nulls for all columns in bounded memory."""
    columns = check_input(path)
    missing = pd.Series(0, index=columns, dtype="int64")
    n_rows = 0
    for chunk in pd.read_csv(path, dtype="string", chunksize=10_000):
        n_rows += len(chunk)
        missing += chunk.isna().sum()
    if n_rows == 0:
        raise ValueError("The source CSV has a header but no data rows.")
    profile = pd.DataFrame({"missing": missing, "missing_pct": 100 * missing / n_rows})
    return n_rows, profile.sort_values("missing", ascending=False, kind="stable")


def read_source(path):
    """Read analytical fields and listing ID; skip free text and contact-like fields."""
    check_input(path)
    return pd.read_csv(path, usecols=list(COLUMNS), dtype="string")


def finite_numeric(values):
    parsed = pd.to_numeric(values, errors="coerce").astype("float64")
    return parsed.where(np.isfinite(parsed))


def parse_source(source):
    data = source.copy()
    data["id"] = data["id"].str.strip().replace("", pd.NA)
    for column in ("price", "year", "odometer"):
        data[column] = finite_numeric(data[column])
    for column in CATEGORIES:
        data[column] = data[column].str.strip().str.lower().replace("", pd.NA)
    posted = pd.to_datetime(data["posting_date"], format="ISO8601", errors="coerce", utc=True)
    data["posted_utc"] = posted
    # Use the year as shown in the listing, so offsets near January 1 do not shift it.
    local_year = finite_numeric(data["posting_date"].str.extract(r"^(\d{4})-", expand=False))
    data["listing_year"] = local_year.where(posted.notna())
    return data


def clean_listings(source):
    """Return a public-safe schema (for local use), row audit, and quality counts."""
    parsed = parse_source(source)
    identified = parsed["id"].notna()
    # Prefer the newest timestamp, then the first source occurrence for ties.
    ranked = parsed.sort_values("posted_utc", ascending=False, na_position="last", kind="stable")
    repeated = ranked["id"].notna() & ranked["id"].duplicated(keep="first")
    dedup = ranked.loc[~repeated].sort_index().copy()
    valid_price = dedup["price"].gt(0)
    data = dedup.loc[valid_price].copy()

    # Values above the newest observed posting year + 1 are unusable as model years.
    # The extra year allows new model years to appear before the next calendar year.
    max_posting_year = data["listing_year"].max()
    year_upper = max_posting_year + 1 if pd.notna(max_posting_year) else 2100
    valid_year = data["year"].between(MIN_MODEL_YEAR, year_upper) & data["year"].mod(1).eq(0)
    data["model_year"] = data["year"].where(valid_year).astype("Int64")
    age = data["listing_year"] - data["model_year"]
    data["vehicle_age"] = age.where(age.ge(0)).astype("Int64")
    negative_mileage = data["odometer"].lt(0)
    data["odometer"] = data["odometer"].where(~negative_mileage)
    data["odometer_review"] = data["odometer"].gt(MILEAGE_REVIEW)
    data["low_price_review"] = data["price"].lt(LOW_PRICE_REVIEW)
    data["high_price_review"] = data["price"].gt(HIGH_PRICE_REVIEW)

    stages = {
        "Parsed source": parsed,
        "Unique known listing IDs (missing IDs kept)": dedup,
        "Positive asking price": data,
    }
    audit = []
    prior = len(parsed)
    for stage, frame in stages.items():
        price = frame["price"].dropna()
        quartiles = price.quantile([0.25, 0.5, 0.75])
        audit.append({
            "stage": stage, "listings": len(frame), "removed": prior - len(frame),
            "numeric_prices": price.size, "min_usd": price.min(),
            "p25_usd": quartiles.loc[0.25], "median_usd": quartiles.loc[0.5],
            "p75_usd": quartiles.loc[0.75], "max_usd": price.max(),
        })
        prior = len(frame)

    quality = pd.Series({
        "Source rows with no listing ID": (~identified).sum(),
        "Duplicate listing ID rows removed": repeated.sum(),
        "Missing / unparseable price after deduplication": dedup["price"].isna().sum(),
        "Non-positive price after deduplication": dedup["price"].le(0).sum(),
        "Missing / unparseable posting date among retained rows": data["listing_year"].isna().sum(),
        "Invalid populated model year among retained rows": (data["year"].notna() & ~valid_year).sum(),
        "Negative calculated age among retained rows": age.lt(0).sum(),
        "Negative odometer among retained rows": negative_mileage.sum(),
        "Odometer above review threshold among retained rows": data["odometer_review"].sum(),
        "Price below low review threshold among retained rows": data["low_price_review"].sum(),
        "Price above high review threshold among retained rows": data["high_price_review"].sum(),
    }, name="listings", dtype="int64")

    assert len(data) + audit[1]["removed"] + audit[2]["removed"] == len(source)
    assert data["price"].gt(0).all() and data["vehicle_age"].dropna().ge(0).all()
    assert data["id"].dropna().is_unique
    if data.empty:
        raise ValueError("No valid positive asking prices remain after cleaning.")
    output = [
        "price", "model_year", "listing_year", "vehicle_age", "odometer",
        *CATEGORIES, "odometer_review", "low_price_review", "high_price_review",
    ]
    return data[output], pd.DataFrame(audit).set_index("stage"), quality
