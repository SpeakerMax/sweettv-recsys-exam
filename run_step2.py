import time

import numpy as np
import pandas as pd
from scipy import sparse

from src.baselines import history_scores
from src.cbf import channel_code_map, genre_scores, schedule_scores
from src.config import (
    CACHE,
    SCORE_DIR,
    SUBMISSION_PATH,
    TRAIN_WINDOW,
    VALID_WINDOW,
)
from src.io import build_airings, load_schedule, load_views
from src.join import join_views_to_airings
from src.labels import candidate_shows, labels_for_window, training_events
from src.matrices import build_user_show_matrix, candidate_show_index, user_rows
from src.method_als import als_scores
from src.method_ppr import ppr_scores
from src.method_svd import puresvd_scores
from src.metrics import mean_ap
from src.ranking import add_popularity_fallback, top_k_predictions


REFERENCE = {
    ("POPULARITY", "-"): 0.2793,
    ("PERSONAL HISTORY", "-"): 0.3602,
    ("PureSVD", "k=32"): 0.3686,
    ("Implicit ALS", "f=64, reg=0.1, alpha=10, it=20"): 0.0950,
    ("PPR", "alpha=0.5"): 0.3774,
    ("CBF genre", "tfidf"): 0.1252,
    ("CBF schedule", "channel x hour"): 0.3739,
}


def select_candidate_scores(matrix, user_positions, present_users, candidates_positions):
    candidate_array = np.flatnonzero(candidates_positions >= 0)
    scores = np.zeros(
        (len(user_positions), len(candidates_positions)), dtype=np.float32
    )
    if present_users.any() and len(candidate_array):
        subset = matrix[user_positions[present_users]][
            :, candidates_positions[candidate_array]
        ].toarray()
        scores[np.ix_(np.flatnonzero(present_users), candidate_array)] = subset
    return scores


def save_scores(name, scores, valid_users, candidates):
    np.save(SCORE_DIR / f"{name}.npy", scores.astype(np.float32))


def main():
    started = time.perf_counter()

    views = load_views()
    schedule = load_schedule()
    airings = build_airings(schedule)
    joined = join_views_to_airings(views, airings)

    train = training_events(joined, airings, TRAIN_WINDOW)
    truths = labels_for_window(joined, airings, VALID_WINDOW)
    candidates = candidate_shows(airings, VALID_WINDOW)
    valid_users = np.array(sorted(truths), dtype=np.uint64)
    truth_sets = {int(user): set(shows) for user, shows in truths.items()}

    submission = pd.read_csv(SUBMISSION_PATH, dtype={"user_id": "uint64"})
    test_users = set(int(user) for user in submission["user_id"])
    validation_test_users = [int(user) for user in valid_users if int(user) in test_users]

    training_matrix, matrix_users, matrix_shows = build_user_show_matrix(
        history_scores(train, TRAIN_WINDOW[1])
    )
    matrix_user_index = {int(user): i for i, user in enumerate(matrix_users)}
    matrix_show_index = {int(show): i for i, show in enumerate(matrix_shows)}

    sparse.save_npz(CACHE / "W.npz", training_matrix)
    np.save(CACHE / "W_users.npy", matrix_users)
    np.save(CACHE / "W_shows.npy", matrix_shows)

    candidate_array = np.array(sorted(int(show) for show in candidates), dtype=np.int64)
    full_training = train[train["st"] >= 0.8]
    candidate_counts = (
        full_training.groupby("show_id")
        .size()
        .reindex(candidate_array, fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    popularity_order = np.lexsort((candidate_array, -candidate_counts))
    candidate_ranking = candidate_array[popularity_order]
    no_history_users = sorted(set(int(user) for user in valid_users) - set(matrix_user_index))

    np.save(SCORE_DIR / "valid_users.npy", valid_users)
    np.save(SCORE_DIR / "candidates.npy", candidate_array)
    np.save(SCORE_DIR / "popularity_scores.npy", candidate_counts)
    np.save(SCORE_DIR / "no_history_users.npy", np.array(no_history_users, dtype=np.uint64))

    rows = []

    def evaluate_and_store(method, setting, scores, runtime):
        scores = add_popularity_fallback(
            scores, valid_users, no_history_users, candidate_counts
        )
        predictions = top_k_predictions(
            scores, valid_users, candidate_array, fallback_ranking=candidate_ranking
        )
        map_all = mean_ap(predictions, truth_sets, [int(user) for user in valid_users])
        map_test = mean_ap(predictions, truth_sets, validation_test_users)
        name = f"{method}__{setting}".replace(" ", "_").replace("=", "").replace(",", "")
        save_scores(name, scores, valid_users, candidate_array)
        rows.append(
            {
                "method": method,
                "setting": setting,
                "map_all": map_all,
                "map_test": map_test,
                "runtime_s": runtime,
                "file": f"{name}.npy",
            }
        )
        print(
            f"{method:20s} {setting:38s} {map_all:.6f} {map_test:.6f} {runtime:7.2f}s"
        )
        return scores

    method_started = time.perf_counter()
    popularity_scores = np.tile(candidate_counts, (len(valid_users), 1))
    evaluate_and_store(
        "POPULARITY", "-", popularity_scores, time.perf_counter() - method_started
    )

    method_started = time.perf_counter()
    _, user_positions = user_rows(valid_users, matrix_user_index)
    present_users = user_positions >= 0
    _, candidate_positions = candidate_show_index(candidate_array, matrix_show_index)
    history_matrix = select_candidate_scores(
        training_matrix, user_positions, present_users, candidate_positions
    )
    evaluate_and_store(
        "PERSONAL HISTORY", "-", history_matrix, time.perf_counter() - method_started
    )

    for k in (16, 32, 64):
        method_started = time.perf_counter()
        scores = puresvd_scores(
            training_matrix,
            matrix_users,
            matrix_user_index,
            valid_users,
            candidate_array,
            matrix_show_index,
            k,
        )
        evaluate_and_store(
            "PureSVD", f"k={k}", scores, time.perf_counter() - method_started
        )

    method_started = time.perf_counter()
    scores = als_scores(
        training_matrix,
        matrix_users,
        matrix_user_index,
        valid_users,
        candidate_array,
        matrix_show_index,
    )
    evaluate_and_store(
        "Implicit ALS",
        "f=64, reg=0.1, alpha=10, it=20",
        scores,
        time.perf_counter() - method_started,
    )

    print("\nPPR final L1 change (last iteration)")
    print("alpha      mean_l1        max_l1       total_l1")
    for alpha in (0.3, 0.5, 0.7, 0.85):
        method_started = time.perf_counter()
        scores, change = ppr_scores(
            training_matrix,
            matrix_users,
            matrix_user_index,
            valid_users,
            candidate_array,
            matrix_show_index,
            alpha,
        )
        runtime = time.perf_counter() - method_started
        evaluate_and_store("PPR", f"alpha={alpha}", scores, runtime)
        print(
            f"{alpha:<6.2f} {change['mean_l1']:.6e} {change['max_l1']:.6e} {change['total_l1']:.6e}"
        )

    method_started = time.perf_counter()
    scores = genre_scores(
        training_matrix,
        matrix_users,
        matrix_user_index,
        matrix_shows,
        valid_users,
        candidate_array,
    )
    evaluate_and_store(
        "CBF genre", "tfidf", scores, time.perf_counter() - method_started
    )

    method_started = time.perf_counter()
    channels, channel_codes = channel_code_map(views, airings)
    scores = schedule_scores(
        views,
        airings,
        valid_users,
        candidate_array,
        VALID_WINDOW,
        channels,
        channel_codes,
    )
    evaluate_and_store(
        "CBF schedule", "channel x hour", scores, time.perf_counter() - method_started
    )

    results = pd.DataFrame(rows)
    results.to_csv(SCORE_DIR / "manifest.csv", index=False)

    print("\nvalidation MAP@5")
    print(results[["method", "setting", "map_all", "map_test", "runtime_s"]].to_string(index=False))

    print("\nreference check (>0.003 flagged)")
    for _, row in results.iterrows():
        key = (row["method"], row["setting"])
        if key in REFERENCE:
            reference = REFERENCE[key]
            difference = row["map_all"] - reference
            flag = "  <-- CHECK" if abs(difference) > 0.003 else ""
            print(
                f"{row['method']:20s} {row['setting']:38s} "
                f"actual={row['map_all']:.4f} expected={reference:.4f} diff={difference:+.4f}{flag}"
            )

    print(f"\ntotal runtime: {time.perf_counter() - started:.1f}s")
    print(f"score matrices saved in {SCORE_DIR}")


if __name__ == "__main__":
    main()
