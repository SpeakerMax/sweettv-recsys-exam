import numpy as np
from scipy import sparse

from .matrices import candidate_show_index, user_rows


def ppr_scores(
    matrix,
    matrix_users,
    matrix_user_index,
    valid_users,
    candidates,
    show_index,
    alpha,
    iterations=40,
):
    n_users, n_shows = matrix.shape
    user_degree = np.asarray(matrix.sum(axis=1)).ravel()
    show_degree = np.asarray(matrix.sum(axis=0)).ravel()

    p_user_show = sparse.diags(1.0 / user_degree) @ matrix
    p_show_user = sparse.diags(1.0 / show_degree) @ matrix.T

    users, user_positions = user_rows(valid_users, matrix_user_index)
    present_users = user_positions >= 0
    batch_users = user_positions[present_users]
    batch_size = len(batch_users)

    restart = np.zeros((batch_size, n_users), dtype=np.float32)
    restart[np.arange(batch_size), batch_users] = 1.0
    x_user = restart.copy()
    x_show = np.zeros((batch_size, n_shows), dtype=np.float32)
    final_change = None

    for _ in range(iterations):
        previous = x_user
        x_show = np.asarray(alpha * (p_user_show.T @ x_user.T)).T.astype(np.float32)
        x_user = (
            np.asarray(alpha * (p_show_user.T @ x_show.T)).T
            + (1.0 - alpha) * restart
        ).astype(np.float32)
        final_change = np.abs(x_user - previous).sum(axis=1)

    candidate_array, candidate_positions = candidate_show_index(candidates, show_index)
    batch_scores = np.zeros((batch_size, len(candidate_array)), dtype=np.float32)
    present_candidates = candidate_positions >= 0
    batch_scores[:, present_candidates] = x_show[:, candidate_positions[present_candidates]]

    scores = np.zeros((len(users), len(candidate_array)), dtype=np.float32)
    scores[present_users] = batch_scores

    change_summary = {
        "mean_l1": float(final_change.mean()) if final_change is not None else 0.0,
        "max_l1": float(final_change.max()) if final_change is not None else 0.0,
        "total_l1": float(final_change.sum()) if final_change is not None else 0.0,
    }
    return scores, change_summary
