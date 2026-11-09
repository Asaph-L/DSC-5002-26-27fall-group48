"""Shared local cohort for the four follow-up research questions."""

import json
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.listings import COLUMNS, RAW, check_input, clean_listings


CACHE = Path("data/processed/research_listings.parquet")
SCHEMA_VERSION = 1


def _prepare(path):
    check_input(path)
    source = pd.read_csv(path, usecols=[*COLUMNS, "VIN", "model"], dtype="string")
    data, _, _ = clean_listings(source[list(COLUMNS)])
    posting = source.loc[data.index, "posting_date"]
    valid = pd.to_datetime(posting, format="ISO8601", errors="coerce", utc=True).notna()
    data["posting_day"] = pd.to_datetime(
        posting.str[:10], format="%Y-%m-%d", errors="coerce"
    ).where(valid)
    data["model"] = source.loc[data.index, "model"].str.strip().str.lower().replace("", pd.NA)

    # Keep opaque linkage keys only in the ignored local cache. Never use them as features.
    vin = source.loc[data.index, "VIN"].str.strip().str.upper()
    valid_vin = vin.str.fullmatch(r"[A-HJ-NPR-Z0-9]{17}", na=False)
    data["vin_key"] = pd.util.hash_pandas_object(vin, index=False).astype("UInt64").where(valid_vin)
    fields = ["state", "manufacturer", "model_year", "odometer", "type", "paint_color", "model"]
    exact = data[fields]
    known = exact[["state", "model_year", "odometer", "model"]].notna().all(axis=1)
    data["fingerprint_key"] = pd.util.hash_pandas_object(exact, index=False).astype("UInt64").where(known)

    row = pd.Series(data.index, index=data.index, dtype="string")
    data["vehicle_group"] = (
        ("vin:" + data["vin_key"].astype("string"))
        .fillna("fingerprint:" + data["fingerprint_key"].astype("string"))
        .fillna("row:" + row)
    )
    return data.reset_index(drop=True)


def load_research_data(root):
    """Load all positive asking prices and build the local cache on first use."""
    root = Path(root)
    path = root / RAW
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Follow the download steps in DATA.md.")
    stat = path.stat()
    version = {"schema": SCHEMA_VERSION, "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}
    cache = root / CACHE
    metadata = cache.with_suffix(".json")
    if cache.is_file() and metadata.is_file():
        if json.loads(metadata.read_text(encoding="utf-8")) == version:
            return pd.read_parquet(cache)

    data = _prepare(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    parquet_temp = cache.with_name(f"{cache.stem}.{token}.parquet")
    metadata_temp = metadata.with_name(f"{metadata.stem}.{token}.json")
    data.to_parquet(parquet_temp, index=False)
    metadata_temp.write_text(json.dumps(version), encoding="utf-8")
    parquet_temp.replace(cache)
    metadata_temp.replace(metadata)
    return data


def purge_matches(train, test, mode="combined"):
    """Remove earlier listings whose VIN or exact-attribute key occurs in the test block."""
    if mode not in {"vin", "combined"}:
        raise ValueError(f"Unexpected duplicate policy {mode!r}")
    match = train["vin_key"].isin(test["vin_key"].dropna())
    if mode == "combined":
        match |= train["fingerprint_key"].isin(test["fingerprint_key"].dropna())
    return train.loc[~match]
