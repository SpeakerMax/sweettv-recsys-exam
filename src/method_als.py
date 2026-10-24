import numpy as np

from .matrices import candidate_show_index, user_rows


def als_scores(
    matrix,
    matrix_users,
    matrix_user_index,
    valid_users,
    candidates,
    show_index,
    factors=64,
    regularization=0.1,
    alpha=10.0,
    iterations=20,
    seed=42,
):
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
        use_gpu=False,
    )
    model.fit(matrix, show_progress=False)

    candidate_array, candidate_positions = candidate_show_index(candidates, show_index)
    item_factors = np.zeros((len(candidate_array), factors), dtype=np.float32)
    present_candidates = candidate_positions >= 0
    item_factors[present_candidates] = model.item_factors[
        candidate_positions[present_candidates]
    ]

    all_scores = model.user_factors @ item_factors.T

    users, user_positions = user_rows(valid_users, matrix_user_index)
    present_users = user_positions >= 0
    scores = np.zeros((len(users), len(candidate_array)), dtype=np.float32)
    scores[present_users] = all_scores[user_positions[present_users]]
    return scores
