import numpy as np

from .config import MIN_SHOWS


def add_popularity_fallback(scores, users, no_history_users, popularity_scores):
    if len(no_history_users) == 0:
        return scores
    no_history = set(int(user) for user in no_history_users)
    mask = np.array([int(user) in no_history for user in users], dtype=bool)
    scores[mask] = popularity_scores
    return scores


def top_k_predictions(scores, users, candidates, fallback_ranking=None, k=MIN_SHOWS):
    scores = np.asarray(scores)
    candidates = np.asarray(candidates, dtype=np.int64)
    predictions = {}
    for row, user in enumerate(users):
        values = scores[row]
        order = np.lexsort((candidates, -values))[:k]
        recommendation = [int(candidates[index]) for index in order]
        if len(recommendation) < k and fallback_ranking is not None:
            seen = set(recommendation)
            for show in fallback_ranking:
                show = int(show)
                if show not in seen:
                    recommendation.append(show)
                    seen.add(show)
                    if len(recommendation) == k:
                        break
        predictions[int(user)] = recommendation
    return predictions
