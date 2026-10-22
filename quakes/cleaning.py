"""Task 3: cleaning."""
import pandas as pd


def epoch_ms_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'time' and 'updated' (epoch milliseconds) to UTC datetimes."""
    df = df.copy()
    for col in ("time", "updated"):
        if col in df and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], unit="ms", utc=True)
    return df


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """One row per 'id', keeping the row with the greatest 'updated'."""
    return (df.sort_values("updated", kind="stable")
              .drop_duplicates("id", keep="last")
              .reset_index(drop=True))


def keep_earthquakes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise 'type' (strip, lowercase) and keep only 'earthquake'."""
    df = df.copy()
    df["type"] = df["type"].astype("string").str.strip().str.lower()
    return df[df["type"] == "earthquake"].reset_index(drop=True)


def extract_region(place: pd.Series) -> pd.Series:
    """Text after the last comma, or the whole string if there is no comma.
    Missing places become 'Unknown'."""
    region = place.astype("string").str.split(",").str[-1].str.strip()
    region = region.mask(region.isna() | (region == ""), "Unknown")
    return region.astype(object)


def iqr_outlier_mask(s: pd.Series, k: float = 1.5) -> pd.Series:
    """True where a value lies outside [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


def drop_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where 'mag' is missing."""
    return df.dropna(subset=["mag"]).reset_index(drop=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Chain the steps above (think about the order) and add a 'region' column."""
    df = epoch_ms_to_datetime(df)   # types first, so 'updated' sorts correctly
    df = dedupe_latest(df)          # latest version of each event wins
    df = keep_earthquakes(df)
    df = drop_missing_target(df)
    df["region"] = extract_region(df["place"])
    return df
