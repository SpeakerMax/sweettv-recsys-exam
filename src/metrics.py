def ap_at_5(prediction, truth, k=5):
    truth = truth if isinstance(truth, set) else set(truth)
    hits = 0
    score = 0.0
    seen = set()
    for rank, show in enumerate(prediction[:k], start=1):
        if show in seen:
            continue
        seen.add(show)
        if show in truth:
            hits += 1
            score += hits / rank
    return score / k


def mean_ap(predictions, truths, users):
    values = [ap_at_5(predictions[user], truths[user]) for user in users]
    return sum(values) / len(values) if values else 0.0
