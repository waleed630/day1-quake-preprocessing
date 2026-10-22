# Day 1 Lab: Answers

Name:

## 1. Leakage
Which columns did you drop and why? Is `tsunami` leaky? Is `magType`?

Dropped (`config.LEAKY`): `title` (contains "M 4.6 - …"), `sig` (computed from magnitude + felt reports), `mmi`, `cdi`, `felt`, `alert` (ShakeMap / DYFI / PAGER products that are derived from the magnitude or collected after people feel the quake), plus `types` (the product list only gains `shakemap`/`losspager`/`dyfi` for big events) and pure identifiers (`ids`, `sources`, `code`, `url`, `detail`, `tz`). **`tsunami` is leaky:** the flag is set only after a large oceanic event is measured; in my pull every `tsunami=1` row was a big quake (2/2). **`magType` is leaky too:** the method is chosen by event size. `mww` was always ≥ 4.5 (21/21), `ml`/`md` almost never (1/1725), so it nearly gives away the label.

## 2. Stream vs batch
How many events changed (same `id`, newer `updated`) during your stream window? What does that tell you about "latest version wins"?

The `all_hour` stream wrote 17 rows covering 9 unique events. **1 event came back with the same `id` and a newer `updated`**, and after merging with the `all_week` batch, `run.py` also reports 1 changed event. One change in a short window looks small, but it proves the feed is not append-only: USGS revises magnitude, depth, location and `status` (automatic → reviewed) after first publication, and the same event can arrive several times. So "latest version wins" is required: sort by `updated` and keep the last row per `id`. Otherwise we either double-count an event (duplicates leak between train and test) or train on a stale, unreviewed magnitude, which means a wrong label. Deduplicating on `id` alone isn't enough for the collector, which is why it tracks `(id, updated)` pairs.

## 3. Outliers
Your decision on negative depth and negative magnitude, with reasoning.

I keep both and do **not** delete IQR outliers blindly. **Negative depth** (28 rows, min −3.2 km) comes from local networks (`av`, `hv`, `uu`, `nc`), where depth is measured relative to sea level or a datum, so events under volcanoes and mountains can sit "above" sea level. That's a real measurement, not an error. **Negative magnitude** (83 rows, min −1.26, almost all `ml` from Alaska's `av` volcano network) is real too: the magnitude scale is logarithmic, and sensitive dense networks record micro-quakes below M 0. The IQR mask flags 272 depths (all deep events, 40–625 km, subduction zones) and 170 magnitudes (mostly the M 4.5+ events, i.e. the positive class!), so deleting them would destroy exactly the signal we want. The robust scaler handles the long tails instead.

## 4. Cardinality
You grouped `region` to top-k. Name one alternative encoding and one risk it carries.

`region` has ~77 distinct values in one week, so one-hot without grouping would add ~77 sparse columns, most seen only a handful of times. **Alternative: target (mean) encoding**, which replaces each region with its share of big quakes. **Risk:** it leaks the target. If the means are computed on the same rows the model trains on (or before the split), rare regions memorise their own labels and the model overfits; it needs out-of-fold / smoothed encoding fit on train only. (Frequency encoding is a safer alternative, but different regions with the same count collide.) Note that `run.py` picks the top-15 regions from the **train split only**, so test frequencies don't leak in.
