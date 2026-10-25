"""Part 2, Task A: extract, transform, load, with logging."""
import argparse
import json
import logging
import sqlite3

import pandas as pd

from . import config
from .cleaning import clean
from .features import (add_location_features, add_quality_features, add_target,
                       add_time_features, drop_leaky_columns, group_rare)
from .fetch import fetch_feed, geojson_to_df

log = logging.getLogger("quakes.etl")
DB_PATH = config.PROCESSED_DIR / "quakes.db"
LOG_PATH = config.ROOT / "logs" / "etl.log"
TOP_K_REGIONS = 15


def extract(feed: str = "all_week") -> pd.DataFrame:
    """Fetch the feed as a DataFrame. If the network fails, log a warning and
    use config.FALLBACK_PATH instead."""
    try:
        df = geojson_to_df(fetch_feed(feed))
        log.info("extract: %d rows from live feed '%s'", len(df), feed)
    except Exception as e:
        log.warning("extract: fetch of '%s' failed (%s); using %s", feed, e, config.FALLBACK_PATH)
        with open(config.FALLBACK_PATH, encoding="utf-8") as f:
            df = geojson_to_df(json.load(f))
        log.info("extract: %d rows from fallback file", len(df))
    return df


def read_stream_files() -> pd.DataFrame:
    """Concatenate every data/stream/*.jsonl file (empty DataFrame if none)."""
    frames = [pd.read_json(p, lines=True, convert_dates=False)
              for p in sorted(config.STREAM_DIR.glob("*.jsonl")) if p.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def transform(raw: pd.DataFrame) -> pd.DataFrame:
    """Day 1 steps in one call: clean, time/quality/location features, target,
    group_rare on 'region', drop leaky columns."""
    df = clean(raw)
    df = add_time_features(df)
    df = add_quality_features(df)
    df = add_location_features(df)
    df = add_target(df)
    df["region"] = group_rare(df["region"], top_k=TOP_K_REGIONS)
    df = drop_leaky_columns(df)
    keep = ["id"] + config.NUMERIC + config.NOMINAL + ["mag", config.TARGET]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def load_sqlite(df: pd.DataFrame, db_path, table: str = "quakes") -> None:
    """Upsert on 'id': running twice must not duplicate rows, and a row with an
    existing id must be replaced by the new version.
    Hints: sqlite3, df.to_sql, DELETE before append. Datetime columns need to
    be converted to text first."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    with sqlite3.connect(db_path) as con:
        exists = con.execute("select 1 from sqlite_master where type='table' and name=?",
                             (table,)).fetchone()
        if exists:
            # new columns (e.g. after changing config) must exist before appending
            have = {r[1] for r in con.execute(f'pragma table_info("{table}")')}
            for col in df.columns.difference(list(have)):
                con.execute(f'alter table "{table}" add column "{col}"')
            ids = df["id"].astype(str).tolist()
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                con.execute(f'delete from "{table}" where id in ({",".join("?" * len(chunk))})',
                            chunk)
        df.to_sql(table, con, if_exists="append", index=False)


def _table_count(db_path, table: str = "quakes") -> int:
    with sqlite3.connect(db_path) as con:
        return con.execute(f'select count(*) from "{table}"').fetchone()[0]


def run_etl(mode: str = "batch") -> None:
    """extract -> (merge stream files if mode == 'merge') -> transform -> load.
    Log row counts at every stage with the logging module."""
    log.info("ETL start (mode=%s)", mode)
    raw = extract()
    if mode == "merge":
        stream = read_stream_files()
        log.info("merge: %d stream rows from %s", len(stream), config.STREAM_DIR)
        if len(stream):
            raw = pd.concat([raw, stream], ignore_index=True)
    log.info("raw rows before transform: %d", len(raw))

    df = transform(raw)
    log.info("transform: %d rows, %d columns, positive rate %.3f",
             len(df), df.shape[1], df[config.TARGET].mean())

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    before = _table_count(DB_PATH) if DB_PATH.exists() and _has_table(DB_PATH) else 0
    load_sqlite(df, DB_PATH, "quakes")
    after = _table_count(DB_PATH)
    log.info("load: upserted %d rows into %s (table rows %d -> %d)", len(df), DB_PATH, before, after)
    log.info("ETL done")


def _has_table(db_path, table: str = "quakes") -> bool:
    with sqlite3.connect(db_path) as con:
        return con.execute("select 1 from sqlite_master where type='table' and name=?",
                           (table,)).fetchone() is not None


def main() -> None:
    """argparse: --mode {batch, merge}. Configure logging to LOG_PATH and the
    console, then call run_etl."""
    p = argparse.ArgumentParser(description="Extract the USGS feed, transform it, load it into SQLite.")
    p.add_argument("--mode", choices=["batch", "merge"], default="batch")
    args = p.parse_args()

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
    )
    try:
        run_etl(args.mode)
    except Exception:
        log.exception("ETL failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
