import numpy as np
import pandas as pd
from scipy import sparse


def build_user_show_matrix(scores):
    users = np.sort(scores["user_id"].unique())
    shows = np.sort(scores["show_id"].unique())
    user_index = {int(user): i for i, user in enumerate(users)}
    show_index = {int(show): i for i, show in enumerate(shows)}
    rows = scores["user_id"].map(user_index).to_numpy(dtype=np.int64)
    cols = scores["show_id"].map(show_index).to_numpy(dtype=np.int64)
    values = scores["score"].to_numpy(dtype=np.float32)
    matrix = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(users), len(shows)), dtype=np.float32
    )
    return matrix, users, shows


def candidate_show_index(candidates, show_index):
    candidates = np.asarray(candidates, dtype=np.int64)
    positions = np.fromiter(
        (show_index.get(int(show), -1) for show in candidates),
        dtype=np.int64,
        count=len(candidates),
    )
    return candidates, positions


def user_rows(users, user_index):
    users = np.asarray(users, dtype=np.uint64)
    positions = np.fromiter(
        (user_index.get(int(user), -1) for user in users),
        dtype=np.int64,
        count=len(users),
    )
    return users, positions
