"""Task 5: run the whole pipeline end to end.

Steps (see LAB_GUIDE.md):
 1. fetch all_week (fall back to config.FALLBACK_PATH if the network fails)
 2. merge any data/stream/*.jsonl rows, then dedupe_latest
 3. clean, engineer features, drop leaky columns
 4. split, fit_transform on train only, transform test
 5. print the report, save train.csv, test.csv, preprocessor.joblib
"""
import argparse
import json

import joblib
import pandas as pd

from . import config
from .cleaning import clean, dedupe_latest, epoch_ms_to_datetime
from .features import (add_location_features, add_quality_features, add_target,
                       add_time_features, drop_leaky_columns)
from .fetch import fetch_feed, geojson_to_df
from .transform import build_preprocessor, split_data

TOP_K_REGIONS = 15


def load_batch(feed: str) -> tuple[pd.DataFrame, str]:
    try:
        return geojson_to_df(fetch_feed(feed)), f"live feed '{feed}'"
    except Exception as e:
        print(f"Fetch failed ({e}); using {config.FALLBACK_PATH}")
        with open(config.FALLBACK_PATH, encoding="utf-8") as f:
            return geojson_to_df(json.load(f)), "fallback file"


def load_stream() -> pd.DataFrame:
    files = sorted(config.STREAM_DIR.glob("*.jsonl"))
    frames = [pd.read_json(p, lines=True, convert_dates=False)
              for p in files if p.stat().st_size > 0]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(description="Build the model-ready earthquake dataset.")
    p.add_argument("--feed", default="all_week")
    p.add_argument("--scaler", default="robust", choices=["standard", "minmax", "robust"])
    args = p.parse_args()

    # 1. batch
    raw, source = load_batch(args.feed)
    n_raw = len(raw)

    # 2. merge stream, then latest version of each event wins
    stream = load_stream()
    combined = pd.concat([raw, stream], ignore_index=True) if len(stream) else raw
    combined = epoch_ms_to_datetime(combined)
    versions = combined.groupby("id")["updated"].nunique()
    n_changed = int((versions > 1).sum())
    merged = dedupe_latest(combined)

    # 3. clean, features, leakage
    df = clean(merged)
    n_clean = len(df)
    df = add_time_features(df)
    df = add_quality_features(df)
    df = add_location_features(df)
    df = add_target(df)
    df = drop_leaky_columns(df)
    df = df[config.NUMERIC + config.NOMINAL + ["mag", config.TARGET]]

    # 4. split first; everything learned from data is fit on train only
    X_train, X_test, y_train, y_test = split_data(df)
    top_regions = X_train["region"].value_counts().nlargest(TOP_K_REGIONS).index
    for X in (X_train, X_test):
        X["region"] = X["region"].where(X["region"].isin(top_regions), "Other")
    pre = build_preprocessor(args.scaler)
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)

    # 5. report and save
    print("\n=== Day 1 pipeline report ===")
    print(f"source                       : {source}")
    print(f"raw rows (batch)             : {n_raw}")
    print(f"stream rows merged           : {len(stream)}")
    print(f"events whose 'updated' changed: {n_changed}")
    print(f"rows after dedupe            : {len(merged)}")
    print(f"rows after cleaning          : {n_clean}")
    print(f"train shape (raw / encoded)  : {X_train.shape} / {Xtr.shape}")
    print(f"test shape  (raw / encoded)  : {X_test.shape} / {Xte.shape}")
    print(f"positive rate train / test   : {y_train.mean():.3f} / {y_test.mean():.3f}")
    print(f"scaler                       : {args.scaler}")

    # train/test keep unscaled features plus both targets ('mag' for Day 3,
    # 'big_quake' for Day 4); the fitted preprocessor is saved alongside.
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    mag = df["mag"]
    X_train.assign(mag=mag.loc[X_train.index], **{config.TARGET: y_train}) \
        .to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    X_test.assign(mag=mag.loc[X_test.index], **{config.TARGET: y_test}) \
        .to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    joblib.dump(pre, config.PROCESSED_DIR / "preprocessor.joblib")
    print(f"saved train.csv, test.csv, preprocessor.joblib to {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
