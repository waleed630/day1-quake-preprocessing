"""Task 1: fetch the USGS GeoJSON feed and turn it into a DataFrame."""
import pandas as pd
import requests

BASE = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary"


def fetch_feed(feed: str = "all_week") -> dict:
    """GET {BASE}/{feed}.geojson and return the parsed dict.

    Use a timeout and raise_for_status(). A misspelled feed name returns
    HTTP 200 with a plain-text body, so raise a clear error if the body
    is not valid JSON.
    """
    url = f"{BASE}/{feed}.geojson"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    try:
        payload = resp.json()
    except ValueError as e:
        raise ValueError(
            f"Feed '{feed}' did not return JSON (check the feed name). "
            f"Body starts with: {resp.text[:80]!r}"
        ) from e
    if "features" not in payload:
        raise ValueError(f"Feed '{feed}' returned JSON without a 'features' list")
    return payload


def geojson_to_df(payload: dict) -> pd.DataFrame:
    """One row per event.

    Columns: 'id' (top level of each feature), every key in 'properties',
    plus 'lon', 'lat', 'depth_km' from geometry.coordinates = [lon, lat, depth].
    """
    rows = []
    for feat in payload.get("features", []):
        coords = (feat.get("geometry") or {}).get("coordinates") or []
        coords = list(coords) + [None] * (3 - len(coords))
        row = {"id": feat.get("id")}
        row.update(feat.get("properties") or {})
        row["lon"], row["lat"], row["depth_km"] = coords[0], coords[1], coords[2]
        rows.append(row)
    return pd.DataFrame(rows)
