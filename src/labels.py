import numpy as np
import pandas as pd

from .config import FULL_WATCH_THRESHOLD, MIN_SHOWS


def candidate_shows(airings, window):
    first_week, last_week = window
    week = airings["week"].to_numpy()
    mask = (week >= first_week) & (week <= last_week)
    return np.unique(airings.loc[mask, "tv_show_id"].to_numpy())


def labels_for_window(joined, airings, window):
    first_week, last_week = window
    week_by_id = airings["week"].to_numpy()
    show_by_id = airings["tv_show_id"].to_numpy()

    events = joined[joined["st"] >= FULL_WATCH_THRESHOLD]
    airing_ids = events["airing_id"].to_numpy()
    weeks = week_by_id[airing_ids]
    events = events[(weeks >= first_week) & (weeks <= last_week)].copy()
    events["show_id"] = show_by_id[events["airing_id"].to_numpy()]

    counts = events.groupby(["user_id", "show_id"], as_index=False).size()
    counts = counts.sort_values(
        ["user_id", "size", "show_id"], ascending=[True, False, True]
    )
    shows_per_user = counts.groupby("user_id")["show_id"].transform("size")
    counts = counts[shows_per_user >= MIN_SHOWS]
    top = counts.groupby("user_id", sort=False).head(MIN_SHOWS)

    truths = {int(user): group["show_id"].tolist() for user, group in top.groupby("user_id", sort=False)}
    return truths


def training_events(joined, airings, window):
    first_week, last_week = window
    week_by_id = airings["week"].to_numpy()
    show_by_id = airings["tv_show_id"].to_numpy()

    airing_ids = joined["airing_id"].to_numpy()
    weeks = week_by_id[airing_ids]
    mask = (weeks >= first_week) & (weeks <= last_week)
    events = joined.loc[mask].copy()
    events["show_id"] = show_by_id[events["airing_id"].to_numpy()]
    events["week"] = weeks[mask]
    return events
