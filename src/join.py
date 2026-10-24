import numpy as np
import pandas as pd

from .config import JOINED_CACHE


def join_views_to_airings(views, airings, use_cache=True):
    if use_cache and JOINED_CACHE.exists():
        return pd.read_parquet(JOINED_CACHE)

    a_start = airings["start"].to_numpy()
    a_end = airings["end"].to_numpy()
    a_id = airings["airing_id"].to_numpy()
    duration_by_id = airings["duration"].to_numpy()

    airings_by_channel = {}
    for channel, pos in airings.groupby("channel_id", sort=False).indices.items():
        pos = np.asarray(pos)
        starts = a_start[pos]
        ends = a_end[pos]
        if len(ends) > 1 and not np.all(ends[1:] >= ends[:-1]):
            raise ValueError(f"airing ends are not sorted on channel {channel}")
        airings_by_channel[int(channel)] = (starts, ends, a_id[pos])

    pieces = []
    for channel, group in views.groupby("channel_id", sort=False):
        channel_airings = airings_by_channel.get(int(channel))
        if channel_airings is None:
            continue
        starts, ends, ids = channel_airings

        view_start = group["start"].to_numpy()
        view_stop = group["stop"].to_numpy()
        user_id = group["user_id"].to_numpy()

        first = np.searchsorted(ends, view_start, side="right")
        last = np.searchsorted(starts, view_stop, side="left")
        counts = np.maximum(last - first, 0)
        active = counts > 0
        if not active.any():
            continue

        active_counts = counts[active]
        total = int(active_counts.sum())
        view_idx = np.repeat(np.flatnonzero(active), active_counts)
        airing_base = np.repeat(first[active], active_counts)
        offsets = np.arange(total, dtype=np.int64) - np.repeat(
            np.cumsum(active_counts) - active_counts, active_counts
        )
        airing_idx = airing_base + offsets

        overlap = np.minimum(view_stop[view_idx], ends[airing_idx]) - np.maximum(
            view_start[view_idx], starts[airing_idx]
        )
        keep = overlap > 0
        if not keep.any():
            continue

        pairs = pd.DataFrame(
            {
                "user_id": user_id[view_idx[keep]],
                "airing_id": ids[airing_idx[keep]],
                "overlap": overlap[keep],
            }
        )
        pairs = pairs.groupby(["user_id", "airing_id"], as_index=False, sort=False)[
            "overlap"
        ].sum()
        pairs["st"] = pairs["overlap"].to_numpy() / duration_by_id[
            pairs["airing_id"].to_numpy()
        ]
        pieces.append(pairs)

    if pieces:
        joined = pd.concat(pieces, ignore_index=True)
    else:
        joined = pd.DataFrame(
            {
                "user_id": pd.Series(dtype="uint64"),
                "airing_id": pd.Series(dtype="int64"),
                "overlap": pd.Series(dtype="float64"),
                "st": pd.Series(dtype="float64"),
            }
        )
    if use_cache and pieces:
        joined.to_parquet(JOINED_CACHE, index=False)
    return joined


def attach_airings(joined, airings):
    airing_ids = joined["airing_id"].to_numpy()
    show_by_id = airings["tv_show_id"].to_numpy()
    week_by_id = airings["week"].to_numpy()
    out = joined.copy()
    out["show_id"] = show_by_id[airing_ids]
    out["week"] = week_by_id[airing_ids]
    return out
