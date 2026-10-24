import numpy as np
import pandas as pd


def clock_check(views, schedule):
    view_duration = views["stop"].to_numpy() - views["start"].to_numpy()
    selected = views[(view_duration >= 1200) & (view_duration <= 7200)]

    stops_by_channel = {
        int(ch): g["stop"].to_numpy()
        for ch, g in selected.groupby("channel_id", sort=False)
    }
    ends = schedule.loc[schedule["duration"] > 0, ["channel_id", "start", "duration"]].copy()
    ends["end"] = ends["start"] + ends["duration"]
    ends_by_channel = {
        int(ch): np.sort(g.to_numpy())
        for ch, g in ends.groupby("channel_id")["end"]
    }

    total = len(selected)
    offsets = range(-3 * 3600, 3 * 3600 + 1, 1800)
    rows = []
    for offset in offsets:
        hits = 0
        for channel, stops in stops_by_channel.items():
            channel_ends = ends_by_channel.get(channel)
            if channel_ends is None or len(channel_ends) == 0:
                continue
            shifted = stops + offset
            idx = np.searchsorted(channel_ends, shifted)
            distance = np.full(len(shifted), np.iinfo(np.int64).max, dtype=np.int64)
            right = idx < len(channel_ends)
            distance[right] = np.abs(channel_ends[idx[right]] - shifted[right])
            left = idx > 0
            distance[left] = np.minimum(
                distance[left], np.abs(channel_ends[idx[left] - 1] - shifted[left])
            )
            hits += int((distance <= 180).sum())
        rows.append((offset / 3600.0, hits / total if total else 0.0))

    table = pd.DataFrame(rows, columns=["offset_hours", "share_within_3min"])
    return table, total
