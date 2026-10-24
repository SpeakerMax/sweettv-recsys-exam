import subprocess
import sys
import time

import numpy as np
import pandas as pd

import pandas
import numpy
import scipy
import pyarrow

from src.baselines import (
    history_predictions,
    history_scores,
    popularity_predictions,
    popularity_ranking,
)
from src.clock import clock_check
from src.config import (
    FIRST_WEEK,
    FULL_WATCH_THRESHOLD,
    LAST_WEEK,
    SUBMISSION_PATH,
    TEST_WINDOW,
    TRAIN_WINDOW,
    VALID_WINDOW,
)
from src.io import build_airings, load_schedule, load_views
from src.join import attach_airings, join_views_to_airings
from src.labels import candidate_shows, labels_for_window, training_events
from src.metrics import mean_ap


def report_environment():
    print(f"python {sys.version.split()[0]}")
    print(f"pandas {pandas.__version__}")
    print(f"numpy {numpy.__version__}")
    print(f"scipy {scipy.__version__}")
    print(f"pyarrow {pyarrow.__version__}")
    for command in (["gh", "--version"], ["gh", "auth", "status"]):
        result = subprocess.run(command, capture_output=True, text=True)
        output = (result.stdout + result.stderr).strip()
        print(f"$ {' '.join(command)}")
        print(output)


def print_window_counts(schedule, airings):
    labelled = schedule[(schedule["tv_show_id"] != 0) & (schedule["duration"] > 0)]
    labelled = labelled.drop_duplicates(["channel_id", "start", "tv_show_id"])
    print(f"labelled airings before week filter: {len(labelled):,}")
    for name, (first, last) in (
        ("train", TRAIN_WINDOW),
        ("validation", VALID_WINDOW),
        ("test", TEST_WINDOW),
    ):
        count = int(((airings["week"] >= first) & (airings["week"] <= last)).sum())
        print(f"labelled airings weeks {first}-{last} ({name}): {count:,}")


def evaluate(predictions, truths, users):
    return mean_ap(predictions, truths, users)


def main():
    started = time.perf_counter()
    report_environment()

    views = load_views()
    schedule = load_schedule()
    print(f"views: {len(views):,}")
    print(f"schedule rows: {len(schedule):,}")

    airings = build_airings(schedule)
    print_window_counts(schedule, airings)

    clock_table, clock_views = clock_check(views, schedule)
    print(f"\nclock check on {clock_views:,} views lasting 20-120 min")
    print(clock_table.to_string(index=False, formatters={"share_within_3min": "{:.4f}".format}))

    joined = join_views_to_airings(views, airings)
    full = joined["st"].to_numpy() >= FULL_WATCH_THRESHOLD
    print(f"\njoined (user, airing) pairs: {len(joined):,}")
    print(f"share st >= {FULL_WATCH_THRESHOLD}: {full.mean():.4f}")
    print(f"full events weeks 11-30: {int(full.sum()):,}")

    train = training_events(joined, airings, TRAIN_WINDOW)
    truths = labels_for_window(joined, airings, VALID_WINDOW)
    candidates = candidate_shows(airings, VALID_WINDOW)
    valid_users = sorted(truths)
    truth_sets = {user: set(shows) for user, shows in truths.items()}

    submission = pd.read_csv(SUBMISSION_PATH, dtype={"user_id": "uint64"})
    test_users = set(int(user) for user in submission["user_id"])
    validation_test_users = [user for user in valid_users if user in test_users]

    print(f"\nvalidation users: {len(valid_users):,}")
    print(f"validation users also in submission: {len(validation_test_users):,}")
    print(f"validation candidates: {len(candidates):,}")

    candidate_ranking = popularity_ranking(train, candidates)
    scores = history_scores(train, TRAIN_WINDOW[1])

    popularity = popularity_predictions(valid_users, candidate_ranking)
    history = history_predictions(
        valid_users, scores, candidates, candidate_ranking
    )

    pop_all = evaluate(popularity, truth_sets, valid_users)
    pop_test = evaluate(popularity, truth_sets, validation_test_users)
    hist_all = evaluate(history, truth_sets, valid_users)
    hist_test = evaluate(history, truth_sets, validation_test_users)

    print("\nvalidation MAP@5")
    print("method                all users   test users")
    print(f"popularity            {pop_all:.6f}    {pop_test:.6f}")
    print(f"personal history      {hist_all:.6f}    {hist_test:.6f}")

    expected = {
        "labelled airings before week filter": 381_892,
        "joined (user, airing) pairs": 3_014_129,
        "full events weeks 11-30": 615_979,
        "validation users": 4_030,
        "validation candidates": 6_376,
    }
    actual = {
        "labelled airings before week filter": len(
            schedule[(schedule["tv_show_id"] != 0) & (schedule["duration"] > 0)]
            .drop_duplicates(["channel_id", "start", "tv_show_id"])
        ),
        "joined (user, airing) pairs": len(joined),
        "full events weeks 11-30": int(full.sum()),
        "validation users": len(valid_users),
        "validation candidates": len(candidates),
    }
    print("\nreference check")
    for key, expected_value in expected.items():
        print(f"{key:42s} actual={actual[key]:>10,}  expected={expected_value:>10,}")

    print(f"\nruntime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
