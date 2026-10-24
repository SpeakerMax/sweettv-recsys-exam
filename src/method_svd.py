import numpy as np
from scipy.sparse.linalg import svds

from .matrices import candidate_show_index, user_rows


def puresvd_scores(
    matrix,
    matrix_users,
    matrix_user_index,
    valid_users,
    candidates,
    show_index,
    k,
    seed=42,
):
    weighted = matrix.astype(np.float64).copy()
    weighted.data = np.log1p(weighted.data)
    u, s, vt = svds(weighted, k=k, random_state=seed)
    order = np.argsort(-s)
    u = u[:, order]
    s = s[order]
    vt = vt[order, :]

    candidate_array, candidate_positions = candidate_show_index(candidates, show_index)
    candidate_factors = np.zeros((k, len(candidate_array)), dtype=np.float64)
    present_candidates = candidate_positions >= 0
    candidate_factors[:, present_candidates] = vt[:, candidate_positions[present_candidates]]

    all_scores = (u * s) @ candidate_factors

    users, user_positions = user_rows(valid_users, matrix_user_index)
    present_users = user_positions >= 0
    scores = np.zeros((len(users), len(candidate_array)), dtype=np.float32)
    scores[present_users] = all_scores[user_positions[present_users]].astype(np.float32)
    return scores
