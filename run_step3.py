import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.baselines import history_scores
from src.cbf import channel_code_map, genre_scores, schedule_scores
from src.config import (
    LOG_PATH,
    ROOT,
    SCORE_DIR,
    SUBMISSION_PATH,
    TEST_WINDOW,
    VALID_WINDOW,
)
from src.io import build_airings, load_schedule, load_views
from src.join import join_views_to_airings
from src.labels import candidate_shows, labels_for_window, training_events
from src.matrices import build_user_show_matrix
from src.method_ppr import ppr_scores
from src.method_svd import puresvd_scores

EXPECTED_ROWS = 1260
FINAL_TRAIN_WINDOW = (11, 30)
GRID_C2 = (0.5, 1.0, 1.5, 2.0, 3.0)
GRID_SVD = (0.0, 0.5, 1.0, 1.5)
GRID_POP = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_C1 = (0.0, 0.25, 0.5, 1.0)


def print_facts(views, joined):
    st = joined["st"].to_numpy(dtype=np.float64)
    edges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.inf])
    counts, _ = np.histogram(st, bins=edges)
    labels = [f"[{edges[i]:.1f},{edges[i + 1]:.1f})" for i in range(len(edges) - 2)]
    labels.append("[1.0,inf)")
    print("screen time st histogram over all joined (user, airing) pairs")
    print("bin              count       share")
    for label, count in zip(labels, counts):
        print(f"{label:14s} {count:10,d} {count / len(st):10.4f}")

    durations = pd.read_csv(LOG_PATH, usecols=["duraton"])["duraton"].to_numpy()
    quantiles = np.quantile(durations, [0.25, 0.50, 0.75, 0.90])
    print("\nview duration duraton quantiles")
    for q, value in zip((25, 50, 75, 90), quantiles):
        print(f"p{q}: {value:.1f} seconds")


def normalize_per_user(scores):
    maxima = scores.max(axis=1, keepdims=True)
    maxima[maxima == 0] = 1.0
    return scores / maxima


def normalize_global(scores):
    maximum = scores.max()
    if maximum == 0:
        return scores
    return scores / maximum


def map_from_indices(pred_indices, candidates, truth_sets, user_keys):
    total = 0.0
    for row, user in enumerate(user_keys):
        truth = truth_sets[user]
        hits = 0
        score = 0.0
        seen = set()
        for rank, column in enumerate(pred_indices[row], start=1):
            show = int(candidates[column])
            if show in seen:
                continue
            seen.add(show)
            if show in truth:
                hits += 1
                score += hits / rank
        total += score / 5.0
    return total / len(user_keys)


def run_grid(valid_users, candidates, truths, test_users):
    names = {
        "ppr": "PPR__alpha0.7.npy",
        "c2": "CBF_schedule__channel_x_hour.npy",
        "svd": "PureSVD__k16.npy",
        "pop": "POPULARITY__-.npy",
        "c1": "CBF_genre__tfidf.npy",
    }
    matrices = {key: np.load(SCORE_DIR / name) for key, name in names.items()}
    normalized = {
        "ppr": normalize_per_user(matrices["ppr"]),
        "c2": normalize_per_user(matrices["c2"]),
        "svd": normalize_per_user(matrices["svd"]),
        "pop": normalize_global(matrices["pop"]),
        "c1": normalize_per_user(matrices["c1"]),
    }
    truth_sets = {int(user): set(shows) for user, shows in truths.items()}
    user_keys = [int(user) for user in valid_users]

    rows = []
    started = time.perf_counter()
    for w_c2 in GRID_C2:
        base_c2 = normalized["ppr"] + w_c2 * normalized["c2"]
        for w_svd in GRID_SVD:
            base_svd = base_c2 + w_svd * normalized["svd"]
            for w_pop in GRID_POP:
                base_pop = base_svd + w_pop * normalized["pop"]
                for w_c1 in GRID_C1:
                    if w_c1:
                        hybrid = base_pop + w_c1 * normalized["c1"]
                    else:
                        hybrid = base_pop
                    pred_indices = np.argsort(-hybrid, axis=1, kind="stable")[:, :5]
                    score = map_from_indices(
                        pred_indices, candidates, truth_sets, user_keys
                    )
                    rows.append(
                        {
                            "w_c2": w_c2,
                            "w_svd": w_svd,
                            "w_pop": w_pop,
                            "w_c1": w_c1,
                            "map_all": score,
                        }
                    )

    results = pd.DataFrame(rows).sort_values(
        ["map_all", "w_c2", "w_svd", "w_pop", "w_c1"],
        ascending=[False, True, True, True, True],
    )
    results.to_csv(SCORE_DIR / "step3_grid.csv", index=False)
    best = results.iloc[0]
    chosen = {
        "w_c2": float(best["w_c2"]),
        "w_svd": float(best["w_svd"]),
        "w_pop": float(best["w_pop"]),
        "w_c1": float(best["w_c1"]),
    }
    hybrid = (
        normalized["ppr"]
        + chosen["w_c2"] * normalized["c2"]
        + chosen["w_svd"] * normalized["svd"]
        + chosen["w_pop"] * normalized["pop"]
        + chosen["w_c1"] * normalized["c1"]
    )
    pred_indices = np.argsort(-hybrid, axis=1, kind="stable")[:, :5]
    test_mask = np.array([int(user) in test_users for user in valid_users], dtype=bool)
    map_test = map_from_indices(
        pred_indices[test_mask],
        candidates,
        truth_sets,
        [int(user) for user in valid_users[test_mask]],
    )
    elapsed = time.perf_counter() - started
    return results, chosen, float(best["map_all"]), map_test, elapsed


def write_submission(path, users, predictions):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("user_id,tv_show_id\n")
        for user, shows in zip(users, predictions):
            values = " ".join(str(int(show)) for show in shows)
            handle.write(f"{int(user)},{values}\n")


def validate_submission(path, template_users):
    frame = pd.read_csv(path, dtype={"user_id": "uint64", "tv_show_id": "string"})
    assert len(frame) == EXPECTED_ROWS, f"{path}: {len(frame)} rows"
    assert np.array_equal(
        frame["user_id"].to_numpy(dtype=np.uint64), template_users
    ), f"{path}: user order mismatch"
    for value in frame["tv_show_id"]:
        parts = str(value).split()
        assert len(parts) == 5, f"{path}: {value}"
        ids = [int(part) for part in parts]
        assert len(set(ids)) == 5, f"{path}: duplicate ids {value}"
    return True


def main():
    started = time.perf_counter()

    views = load_views()
    schedule = load_schedule()
    airings = build_airings(schedule)
    joined = join_views_to_airings(views, airings)

    print_facts(views, joined)

    truths = labels_for_window(joined, airings, VALID_WINDOW)
    valid_users = np.array(sorted(truths), dtype=np.uint64)
    candidates = candidate_shows(airings, VALID_WINDOW)
    candidates = np.array(sorted(int(show) for show in candidates), dtype=np.int64)
    submission = pd.read_csv(SUBMISSION_PATH, dtype={"user_id": "uint64"})
    test_users = set(int(user) for user in submission["user_id"])

    grid_results, chosen, chosen_map_all, chosen_map_test, grid_seconds = run_grid(
        valid_users, candidates, truths, test_users
    )
    print("\nbest 5 hybrid weight rows on validation")
    print(
        grid_results.head(5).to_string(
            index=False,
            formatters={"map_all": "{:.6f}".format},
        )
    )
    print(f"\nchosen weights: {json.dumps(chosen)}")
    print(f"chosen hybrid validation MAP@5 all users: {chosen_map_all:.6f}")
    print(f"chosen hybrid validation MAP@5 test subset: {chosen_map_test:.6f}")
    print(f"grid runtime: {grid_seconds:.1f}s")

    with open(SCORE_DIR / "step3_chosen_weights.json", "w", encoding="utf-8") as handle:
        json.dump(chosen, handle, indent=2)

    train = training_events(joined, airings, FINAL_TRAIN_WINDOW)
    assert int(train["week"].min()) == 11
    assert int(train["week"].max()) <= 30

    final_candidates = np.array(
        sorted(int(show) for show in candidate_shows(airings, TEST_WINDOW)),
        dtype=np.int64,
    )
    submission_users = submission["user_id"].to_numpy(dtype=np.uint64)

    matrix, matrix_users, matrix_shows = build_user_show_matrix(
        history_scores(train, 30)
    )
    matrix_user_index = {int(user): i for i, user in enumerate(matrix_users)}
    matrix_show_index = {int(show): i for i, show in enumerate(matrix_shows)}

    no_history_users = [
        int(user) for user in submission_users if int(user) not in matrix_user_index
    ]
    no_history_mask = np.array(
        [int(user) not in matrix_user_index for user in submission_users], dtype=bool
    )

    full_training = train[train["st"] >= 0.8]
    popularity_counts = (
        full_training.groupby("show_id")
        .size()
        .reindex(final_candidates, fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    popularity_order = np.lexsort((final_candidates, -popularity_counts))
    popularity_ranking = final_candidates[popularity_order]
    popularity_top5 = popularity_ranking[:5].copy()

    print(f"\ntest users with no usable history: {len(no_history_users):,}")
    print(f"popularity list: {' '.join(str(int(show)) for show in popularity_top5)}")

    method_started = time.perf_counter()
    ppr, ppr_change = ppr_scores(
        matrix,
        matrix_users,
        matrix_user_index,
        submission_users,
        final_candidates,
        matrix_show_index,
        0.7,
    )
    print(
        f"final PPR alpha=0.7 final L1 change: "
        f"mean={ppr_change['mean_l1']:.3e} max={ppr_change['max_l1']:.3e}"
    )

    svd = puresvd_scores(
        matrix,
        matrix_users,
        matrix_user_index,
        submission_users,
        final_candidates,
        matrix_show_index,
        16,
    )

    channels, channel_codes = channel_code_map(views, airings)
    c2 = schedule_scores(
        views,
        airings,
        submission_users,
        final_candidates,
        TEST_WINDOW,
        channels,
        channel_codes,
        profile_window=FINAL_TRAIN_WINDOW,
    )

    c1 = genre_scores(
        matrix,
        matrix_users,
        matrix_user_index,
        matrix_shows,
        submission_users,
        final_candidates,
    )

    popularity = np.tile(popularity_counts, (len(submission_users), 1))
    print(f"final model rebuild runtime: {time.perf_counter() - method_started:.1f}s")

    def rank(scores):
        indices = np.argsort(-scores, axis=1, kind="stable")[:, :5]
        return final_candidates[indices]

    predictions = {
        "popularity": rank(popularity),
        "ppr": rank(ppr),
        "puresvd": rank(svd),
        "cbf_schedule": rank(c2),
    }

    normalized_ppr = normalize_per_user(ppr)
    normalized_c2 = normalize_per_user(c2)
    normalized_svd = normalize_per_user(svd)
    normalized_pop = normalize_global(popularity)
    normalized_c1 = normalize_per_user(c1)
    hybrid = (
        normalized_ppr
        + chosen["w_c2"] * normalized_c2
        + chosen["w_svd"] * normalized_svd
        + chosen["w_pop"] * normalized_pop
        + chosen["w_c1"] * normalized_c1
    )
    predictions["hybrid"] = rank(hybrid)

    if no_history_mask.any():
        for name in predictions:
            predictions[name][no_history_mask] = popularity_top5

    submissions_dir = ROOT / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    for name in ("popularity", "ppr", "puresvd", "cbf_schedule", "hybrid"):
        path = submissions_dir / f"{name}.csv"
        write_submission(path, submission_users, predictions[name])
        validate_submission(path, submission_users)
        print(f"validated {path.name}")

    print(f"\ntotal runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
