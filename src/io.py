import numpy as np
import pandas as pd

from .config import (
    AIRINGS_CACHE,
    FIRST_WEEK,
    LAST_WEEK,
    LOG_PATH,
    SCHEDULE_CACHE,
    SCHEDULE_PATHS,
    VIEWS_CACHE,
)


def _to_unix(values, fmt):
    dt = pd.to_datetime(values, format=fmt, errors="coerce")
    valid = dt.notna()
    out = np.full(len(dt), -1, dtype=np.int64)
    out[valid.to_numpy()] = dt[valid].astype("datetime64[s]").astype("int64").to_numpy()
    return out, valid.to_numpy()


def load_views(use_cache=True):
    if use_cache and VIEWS_CACHE.exists():
        return pd.read_parquet(VIEWS_CACHE)

    raw = pd.read_csv(
        LOG_PATH,
        usecols=["user_id", "vsetv_id", "start_time", "stop_time"],
        dtype={"user_id": "uint64", "vsetv_id": "int32"},
    )
    start, start_ok = _to_unix(raw["start_time"], "%Y-%m-%d %H:%M:%S")
    stop, stop_ok = _to_unix(raw["stop_time"], "%Y-%m-%d %H:%M:%S")
    ok = start_ok & stop_ok & (stop > start)

    views = pd.DataFrame(
        {
            "user_id": raw.loc[ok, "user_id"].to_numpy(dtype=np.uint64),
            "channel_id": raw.loc[ok, "vsetv_id"].to_numpy(dtype=np.int32),
            "start": start[ok],
            "stop": stop[ok],
        }
    )
    views.to_parquet(VIEWS_CACHE, index=False)
    return views


def load_schedule(use_cache=True):
    if use_cache and SCHEDULE_CACHE.exists():
        return pd.read_parquet(SCHEDULE_CACHE)

    frames = []
    for path in SCHEDULE_PATHS:
        part = pd.read_csv(
            path,
            usecols=["channel_id", "start_time", "duration", "tv_show_id"],
            dtype={"channel_id": "int32", "duration": "int64", "tv_show_id": "int64"},
        )
        start, ok = _to_unix(part["start_time"], "%d.%m.%Y %H:%M:%S")
        part = part.loc[ok, ["channel_id", "duration", "tv_show_id"]].copy()
        part["start"] = start[ok]
        frames.append(part)

    schedule = pd.concat(frames, ignore_index=True)
    schedule = schedule[["channel_id", "start", "duration", "tv_show_id"]]
    schedule.to_parquet(SCHEDULE_CACHE, index=False)
    return schedule


def build_airings(schedule, use_cache=True):
    if use_cache and AIRINGS_CACHE.exists():
        return pd.read_parquet(AIRINGS_CACHE)

    labelled = schedule[(schedule["tv_show_id"] != 0) & (schedule["duration"] > 0)].copy()
    labelled = labelled.drop_duplicates(["channel_id", "start", "tv_show_id"])
    labelled["end"] = labelled["start"] + labelled["duration"]
    weeks = pd.to_datetime(labelled["start"], unit="s").dt.isocalendar().week
    labelled["week"] = weeks.astype("int16")
    labelled = labelled[(labelled["week"] >= FIRST_WEEK) & (labelled["week"] <= LAST_WEEK)]
    labelled = labelled.sort_values(
        ["channel_id", "start", "tv_show_id"], kind="mergesort"
    ).reset_index(drop=True)
    labelled["airing_id"] = np.arange(len(labelled), dtype=np.int64)
    labelled = labelled[
        ["airing_id", "channel_id", "start", "end", "duration", "tv_show_id", "week"]
    ]
    labelled.to_parquet(AIRINGS_CACHE, index=False)
    return labelled
