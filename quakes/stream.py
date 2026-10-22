"""Task 2: simulate streaming by polling a feed on a timer."""
import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .fetch import fetch_feed, geojson_to_df


def filter_unseen(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Return only rows whose (id, updated) pair is not in `seen`.

    Add the new pairs to `seen` (modify the set in place).
    """
    pairs = list(zip(df["id"], df["updated"]))
    mask = [p not in seen for p in pairs]
    seen.update(p for p, new in zip(pairs, mask) if new)
    return df.loc[mask]


def run_stream(feed: str, interval_s: int, duration_s: int, out_path: str) -> None:
    """Poll `feed` every `interval_s` seconds for `duration_s` seconds.

    Each poll: fetch, parse, keep unseen rows, append them to `out_path`
    as JSON lines, and print e.g. "[14:02:11] 3 new / 12 fetched".
    A failed poll must not stop the loop.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set = set()
    end = time.time() + duration_s
    while True:
        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            df = geojson_to_df(fetch_feed(feed))
            new = filter_unseen(df, seen) if len(df) else df
            if len(new):
                text = new.to_json(orient="records", lines=True)
                with open(out, "a", encoding="utf-8") as f:
                    f.write(text if text.endswith("\n") else text + "\n")
            print(f"[{stamp}] {len(new)} new / {len(df)} fetched", flush=True)
        except Exception as e:  # one failed poll must not kill the run
            print(f"[{stamp}] poll failed: {e}", flush=True)
        if time.time() + interval_s > end:
            break
        time.sleep(interval_s)


def main() -> None:
    """argparse: --feed (default all_hour), --interval (60), --duration (2400),
    --out (data/stream/stream.jsonl), then call run_stream."""
    p = argparse.ArgumentParser(description="Poll a USGS feed and append new rows as JSON lines.")
    p.add_argument("--feed", default="all_hour")
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--duration", type=int, default=2400)
    p.add_argument("--out", default="data/stream/stream.jsonl")
    a = p.parse_args()
    run_stream(a.feed, a.interval, a.duration, a.out)


if __name__ == "__main__":
    main()
