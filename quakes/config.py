from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FALLBACK_PATH = ROOT / "data" / "fallback" / "all_week.geojson"
STREAM_DIR = ROOT / "data" / "stream"
PROCESSED_DIR = ROOT / "data" / "processed"

SEED = 42
TARGET = "big_quake"

# Numeric feature columns: all known once the event has been located.
NUMERIC: list[str] = [
    "lat", "lon", "abs_lat", "depth_km", "is_shallow",
    "nst", "nst_missing", "gap", "dmin", "rms",
    "hour", "dayofweek", "update_lag_hours", "is_reviewed",
]
# Categorical feature columns (one-hot encoded).
NOMINAL: list[str] = ["region", "net"]
# Columns that encode the magnitude, or only exist because of it: must be dropped.
LEAKY: list[str] = [
    "title",    # literally contains "M 4.6 - ..."
    "sig",      # significance score computed from magnitude and felt reports
    "mmi",      # ShakeMap intensity, derived from magnitude
    "cdi",      # "Did You Feel It?" intensity, collected after the event
    "felt",     # number of felt reports, after the fact and driven by magnitude
    "alert",    # PAGER alert level, computed from estimated shaking
    "tsunami",  # flag set for large oceanic events once the magnitude is known
    "magType",  # method picked by event size (ml/md small, mb/mww large)
    "types",    # product list (shakemap, losspager, dyfi) exists mostly for big events
    "sources", "ids", "code", "url", "detail", "tz",  # identifiers and links
]
