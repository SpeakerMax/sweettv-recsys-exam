import numpy as np
import pandas as pd

from .config import FULL_WATCH_THRESHOLD, HALF_LIFE_WEEKS, MIN_SHOWS


def popularity_ranking(training, candidates):
    full = training[training["st"] >= FULL_WATCH_THRESHOLD]
    counts = full.groupby("show_id").size()
    candidate_array = np.array(sorted(int(show) for show in candidates), dtype=np.int64)
    candidate_counts = counts.reindex(candidate_array, fill_value=0).to_numpy()
    order = np.lexsort((candidate_array, -candidate_counts))
    return candidate_array[order]


def pad_with_popularity(recommendation, candidate_ranking, k=MIN_SHOWS):
    if len(recommendation) >= k:
        return recommendation[:k]
    result = list(recommendation)
    seen = set(result)
    for show in candidate_ranking:
        show = int(show)
        if show not in seen:
            result.append(show)
            seen.add(show)
            if len(result) == k:
                break
    return result


def popularity_predictions(users, candidate_ranking, k=MIN_SHOWS):
    top = [int(show) for show in candidate_ranking[:k]]
    return {int(user): list(top) for user in users}


def history_scores(training, last_training_week):
    weeks = training["week"].to_numpy()
    weights = np.exp(-(last_training_week - weeks) / HALF_LIFE_WEEKS)
    values = np.minimum(training["st"].to_numpy(), 1.0)
    scored = pd.DataFrame(
        {
            "user_id": training["user_id"].to_numpy(),
            "show_id": training["show_id"].to_numpy(),
            "score": values * weights,
        }
    )
    return scored.groupby(["user_id", "show_id"], as_index=False)["score"].sum()


def history_predictions(users, scores, candidates, candidate_ranking, k=MIN_SHOWS):
    user_array = np.array(users, dtype=np.uint64)
    candidate_array = np.array(sorted(int(show) for show in candidates), dtype=np.int64)
    subset = scores[
        scores["user_id"].isin(user_array) & scores["show_id"].isin(candidate_array)
    ]
    by_user = {int(user): group for user, group in subset.groupby("user_id", sort=False)}

    predictions = {}
    for user in users:
        user = int(user)
        group = by_user.get(user)
        recommendation = []
        if group is not None:
            shows = group["show_id"].to_numpy()
            values = group["score"].to_numpy()
            order = np.lexsort((shows, -values))
            recommendation = [int(show) for show in shows[order][:k]]
            if len(recommendation) < k:
                seen = set(recommendation)
                for show in candidate_array:
                    show = int(show)
                    if show not in seen:
                        recommendation.append(show)
                        seen.add(show)
                        if len(recommendation) == k:
                            break
        predictions[user] = pad_with_popularity(
            recommendation, candidate_ranking, k
        )
    return predictions
