import numpy as np
import pandas as pd
from scipy import sparse

from .config import (
    CACHE,
    FIRST_WEEK,
    GENRE_FEATURES_CACHE,
    GENRE_SHOWS_CACHE,
    HALF_LIFE_WEEKS,
    SCHEDULE_PATHS,
    SCHEDULE_PROFILE_CACHE,
    SHOW_META_CACHE,
    TRAIN_WINDOW,
)
from .matrices import user_rows


def load_show_metadata():
    if SHOW_META_CACHE.exists():
        return pd.read_parquet(SHOW_META_CACHE)

    columns = [
        "tv_show_id",
        "tv_show_category",
        "tv_show_genre_1",
        "tv_show_genre_2",
        "tv_show_genre_3",
    ]
    dtypes = {
        "tv_show_id": "int64",
        "tv_show_category": "string",
        "tv_show_genre_1": "string",
        "tv_show_genre_2": "string",
        "tv_show_genre_3": "string",
    }
    frames = []
    for path in SCHEDULE_PATHS:
        part = pd.read_csv(path, usecols=columns, dtype=dtypes)
        part = part[part["tv_show_id"] != 0]
        frames.append(part.drop_duplicates("tv_show_id", keep="first"))
    metadata = pd.concat(frames, ignore_index=True).drop_duplicates(
        "tv_show_id", keep="first"
    )
    metadata = metadata.reset_index(drop=True)
    metadata.to_parquet(SHOW_META_CACHE, index=False)
    return metadata


def build_genre_features():
    if GENRE_FEATURES_CACHE.exists() and GENRE_SHOWS_CACHE.exists():
        return sparse.load_npz(GENRE_FEATURES_CACHE), np.load(GENRE_SHOWS_CACHE)

    metadata = load_show_metadata()
    shows = metadata["tv_show_id"].to_numpy(dtype=np.int64)
    token_columns = [
        "tv_show_category",
        "tv_show_genre_1",
        "tv_show_genre_2",
        "tv_show_genre_3",
    ]

    documents = []
    for row in metadata[token_columns].itertuples(index=False, name=None):
        tokens = []
        for value in row:
            if pd.notna(value):
                token = str(value).strip().lower()
                if token:
                    tokens.append(token)
        documents.append(tokens)

    vocabulary = sorted({token for document in documents for token in document})
    token_index = {token: i for i, token in enumerate(vocabulary)}

    row_indices = []
    column_indices = []
    for row_index, document in enumerate(documents):
        for token in document:
            row_indices.append(row_index)
            column_indices.append(token_index[token])

    data = np.ones(len(row_indices), dtype=np.float32)
    counts = sparse.csr_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(metadata), len(vocabulary)),
        dtype=np.float32,
    )
    document_frequency = counts.getnnz(axis=0)
    idf = np.log((1.0 + len(metadata)) / (1.0 + document_frequency)) + 1.0
    counts = counts.multiply(idf.reshape(1, -1)).tocsr()

    row_norms = np.sqrt(np.asarray(counts.multiply(counts).sum(axis=1)).ravel())
    inverse_norms = np.zeros_like(row_norms)
    nonzero = row_norms > 0
    inverse_norms[nonzero] = 1.0 / row_norms[nonzero]
    features = (sparse.diags(inverse_norms) @ counts).tocsr().astype(np.float32)

    sparse.save_npz(GENRE_FEATURES_CACHE, features)
    np.save(GENRE_SHOWS_CACHE, shows)
    return features, shows


def _feature_rows(shows, feature_shows, features):
    n_features = features.shape[1]
    show_index = {int(show): i for i, show in enumerate(feature_shows)}
    positions = np.fromiter(
        (show_index.get(int(show), -1) for show in shows),
        dtype=np.int64,
        count=len(shows),
    )
    rows = sparse.lil_matrix((len(shows), n_features), dtype=np.float32)
    present = positions >= 0
    if present.any():
        rows[np.flatnonzero(present)] = features[positions[present]]
    return rows.tocsr()


def genre_scores(
    matrix,
    matrix_users,
    matrix_user_index,
    matrix_shows,
    valid_users,
    candidates,
):
    features, feature_shows = build_genre_features()
    matrix_features = _feature_rows(matrix_shows, feature_shows, features)
    candidate_features = _feature_rows(candidates, feature_shows, features)

    users, user_positions = user_rows(valid_users, matrix_user_index)
    present_users = user_positions >= 0
    profiles = (matrix[user_positions[present_users]] @ matrix_features).toarray()
    profiles = profiles.astype(np.float32)
    norms = np.sqrt((profiles * profiles).sum(axis=1))
    nonzero = norms > 0
    profiles[nonzero] /= norms[nonzero, None]
    profiles[~nonzero] = 0.0

    scores = np.zeros((len(users), candidate_features.shape[0]), dtype=np.float32)
    scores[present_users] = (candidate_features @ profiles.T).T
    return scores


def channel_code_map(views, airings):
    channels = np.unique(
        np.concatenate(
            [views["channel_id"].unique(), airings["channel_id"].unique()]
        )
    )
    return channels, {int(channel): i for i, channel in enumerate(channels)}


def _profile_cache_paths(window):
    first_week, last_week = window
    if (first_week, last_week) == tuple(TRAIN_WINDOW):
        return (
            SCHEDULE_PROFILE_CACHE,
            CACHE / "schedule_profile_users.npy",
            CACHE / "schedule_profile_channels.npy",
        )
    suffix = f"{first_week}_{last_week}"
    return (
        CACHE / f"schedule_profile_{suffix}.npz",
        CACHE / f"schedule_profile_{suffix}_users.npy",
        CACHE / f"schedule_profile_{suffix}_channels.npy",
    )


def build_schedule_profile(views, channels, channel_codes, window=TRAIN_WINDOW):
    profile_path, users_path, channels_path = _profile_cache_paths(window)
    if profile_path.exists() and users_path.exists() and channels_path.exists():
        cached_channels = np.load(channels_path)
        if np.array_equal(cached_channels, channels):
            return sparse.load_npz(profile_path), np.load(users_path)

    first_week, last_week = window
    base = pd.Timestamp("2020-03-09") + pd.Timedelta(weeks=first_week - 11)
    after_last = base + pd.Timedelta(weeks=last_week - first_week + 1)
    first_timestamp = int(base.value // 1_000_000_000)
    after_last_timestamp = int(after_last.value // 1_000_000_000)

    selected = views[
        (views["stop"] > first_timestamp) & (views["start"] < after_last_timestamp)
    ].reset_index(drop=True)
    start = selected["start"].to_numpy()
    stop = selected["stop"].to_numpy()
    user = selected["user_id"].to_numpy()
    channel = selected["channel_id"].to_numpy()

    first_hour = start // 3600
    last_hour = (stop - 1) // 3600
    counts = last_hour - first_hour + 1
    total = int(counts.sum())
    view_index = np.repeat(np.arange(len(selected), dtype=np.int64), counts)
    base = np.repeat(np.cumsum(counts) - counts, counts)
    hour_index = np.repeat(first_hour, counts) + (
        np.arange(total, dtype=np.int64) - base
    )

    piece_start = np.maximum(start[view_index], hour_index * 3600)
    piece_stop = np.minimum(stop[view_index], (hour_index + 1) * 3600)
    length = piece_stop - piece_start

    week = (
        pd.to_datetime(hour_index * 3600, unit="s")
        .isocalendar()
        .week.to_numpy()
        .astype(np.int16)
    )
    keep = (week >= first_week) & (week <= last_week) & (length > 0)
    week = week[keep]
    hour_index = hour_index[keep]
    length = length[keep]
    view_index = view_index[keep]

    weight = np.exp(-(last_week - week.astype(np.float64)) / HALF_LIFE_WEEKS)
    weighted_length = length.astype(np.float64) * weight
    weekday = ((hour_index // 24) + 3) % 7
    hour_of_day = hour_index % 24
    hour_of_week = (weekday * 24 + hour_of_day).astype(np.int16)

    pieces = pd.DataFrame(
        {
            "user_id": user[view_index],
            "channel_id": channel[view_index],
            "hour_of_week": hour_of_week,
            "weighted_length": weighted_length,
        }
    )
    grouped = pieces.groupby(
        ["user_id", "channel_id", "hour_of_week"], as_index=False
    )["weighted_length"].sum()

    profile_users = np.sort(selected["user_id"].unique())
    user_index = {int(value): i for i, value in enumerate(profile_users)}
    rows = grouped["user_id"].map(user_index).to_numpy(dtype=np.int64)
    cells = (
        grouped["channel_id"].map(channel_codes).to_numpy(dtype=np.int64) * 168
        + grouped["hour_of_week"].to_numpy(dtype=np.int64)
    )
    week_weights = np.exp(
        -(last_week - np.arange(first_week, last_week + 1, dtype=np.float64))
        / HALF_LIFE_WEEKS
    )
    denominator = 3600.0 * week_weights.sum()
    values = (grouped["weighted_length"].to_numpy() / denominator).astype(np.float32)

    profile = sparse.csr_matrix(
        (values, (rows, cells)),
        shape=(len(profile_users), len(channels) * 168),
        dtype=np.float32,
    )

    sparse.save_npz(profile_path, profile)
    np.save(users_path, profile_users)
    np.save(channels_path, channels)
    return profile, profile_users


def schedule_item_features(airings, candidates, channel_codes, window, n_channels):
    first_week, last_week = window
    selected = airings[
        (airings["week"] >= first_week) & (airings["week"] <= last_week)
    ]
    candidate_array = np.asarray(candidates, dtype=np.int64)
    candidate_index = {int(show): i for i, show in enumerate(candidate_array)}

    show_positions = selected["tv_show_id"].map(candidate_index).to_numpy(dtype=np.int64)
    midpoint = selected["start"].to_numpy() + selected["duration"].to_numpy() // 2
    hour_index = midpoint // 3600
    weekday = ((hour_index // 24) + 3) % 7
    hour_of_day = hour_index % 24
    hour_of_week = weekday * 24 + hour_of_day
    channel_codes_selected = (
        selected["channel_id"].map(channel_codes).to_numpy(dtype=np.int64)
    )

    n_cells = n_channels * 168
    cells = channel_codes_selected * 168 + hour_of_week
    keys = show_positions * n_cells + cells
    unique_keys, counts = np.unique(keys, return_counts=True)
    rows = unique_keys // n_cells
    cols = unique_keys % n_cells
    return sparse.csr_matrix(
        (counts.astype(np.float32), (rows, cols)),
        shape=(len(candidate_array), n_cells),
        dtype=np.float32,
    )


def schedule_scores(
    views,
    airings,
    valid_users,
    candidates,
    window,
    channels,
    channel_codes,
    profile_window=TRAIN_WINDOW,
):
    profile, profile_users = build_schedule_profile(
        views, channels, channel_codes, profile_window
    )
    profile_index = {int(user): i for i, user in enumerate(profile_users)}
    positions = np.fromiter(
        (profile_index.get(int(user), -1) for user in valid_users),
        dtype=np.int64,
        count=len(valid_users),
    )
    present = positions >= 0
    items = schedule_item_features(
        airings, candidates, channel_codes, window, len(channels)
    )
    scores = np.zeros((len(valid_users), len(candidates)), dtype=np.float32)
    if present.any():
        scores[present] = (profile[positions[present]] @ items.T).toarray().astype(
            np.float32
        )
    return scores
