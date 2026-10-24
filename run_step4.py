import json
import time

import numpy as np
import pandas as pd

from src.baselines import history_scores
from src.cbf import genre_scores
from src.config import (
    CACHE,
    ROOT,
    SCHEDULE_PATHS,
    SCORE_DIR,
    SUBMISSION_PATH,
    TEST_WINDOW,
    VALID_WINDOW,
)
from src.io import build_airings, load_schedule, load_views
from src.join import join_views_to_airings
from src.labels import candidate_shows, labels_for_window, training_events
from src.matrices import build_user_show_matrix
from src.metrics import ap_at_5

FINAL_TRAIN_WINDOW = (11, 30)
RESULTS_DIR = ROOT / "results"
SHOW_INFO_CACHE = CACHE / "show_info.parquet"
METHOD_FILES = {
    "popularity": "popularity.csv",
    "ppr": "ppr.csv",
    "puresvd": "puresvd.csv",
    "cbf_schedule": "cbf_schedule.csv",
    "hybrid": "hybrid.csv",
}
METHOD_LABELS = {
    "popularity": "POPULARITY",
    "ppr": "PPR alpha=0.7",
    "puresvd": "PureSVD k=16",
    "cbf_schedule": "CBF schedule",
    "cbf_genre": "CBF genre",
    "hybrid": "HYBRID",
}


def load_show_info():
    if SHOW_INFO_CACHE.exists():
        return pd.read_parquet(SHOW_INFO_CACHE)
    columns = ["tv_show_id", "tv_show_title", "tv_show_category", "channel_title"]
    frames = []
    for path in SCHEDULE_PATHS:
        part = pd.read_csv(
            path,
            usecols=columns,
            dtype={
                "tv_show_id": "int64",
                "tv_show_title": "string",
                "tv_show_category": "string",
                "channel_title": "string",
            },
        )
        part = part[part["tv_show_id"] != 0]
        frames.append(part.drop_duplicates("tv_show_id", keep="first"))
    info = pd.concat(frames, ignore_index=True).drop_duplicates(
        "tv_show_id", keep="first"
    )
    info.to_parquet(SHOW_INFO_CACHE, index=False)
    return info


def show_info_map(info):
    result = {}
    for row in info.itertuples(index=False):
        result[int(row.tv_show_id)] = {
            "title": "" if pd.isna(row.tv_show_title) else str(row.tv_show_title),
            "category": ""
            if pd.isna(row.tv_show_category)
            else str(row.tv_show_category),
            "channel": "" if pd.isna(row.channel_title) else str(row.channel_title),
        }
    return result


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def read_final_submission(path):
    frame = pd.read_csv(path, dtype={"user_id": "uint64", "tv_show_id": "string"})
    predictions = {}
    for user, value in zip(frame["user_id"], frame["tv_show_id"]):
        predictions[int(user)] = [int(part) for part in str(value).split()]
    return frame["user_id"].to_numpy(dtype=np.uint64), predictions


def normalize_row(scores):
    maximum = scores.max()
    if maximum == 0:
        return scores
    return scores / maximum


def unlabelled_shares(views, schedule, users):
    week11 = pd.Timestamp("2020-03-09")
    week31 = week11 + pd.Timedelta(weeks=20)
    first = int(week11.value // 1_000_000_000)
    after = int(week31.value // 1_000_000_000)

    selected_views = views[
        np.isin(
            views["user_id"].to_numpy(dtype=np.uint64),
            np.array(users, dtype=np.uint64),
        )
    ].copy()
    viewing_seconds = (
        selected_views.assign(seconds=selected_views["stop"] - selected_views["start"])
        .groupby("user_id")["seconds"]
        .sum()
    )
    viewing_seconds = {int(user): float(value) for user, value in viewing_seconds.items()}

    unlabelled = schedule[
        (schedule["tv_show_id"] == 0)
        & (schedule["duration"] > 0)
        & (schedule["start"] >= first)
        & (schedule["start"] < after)
    ]
    unlabelled = unlabelled.drop_duplicates(["channel_id", "start"]).copy()
    unlabelled["end"] = unlabelled["start"] + unlabelled["duration"]
    unlabelled = unlabelled.sort_values(["channel_id", "start"]).reset_index(drop=True)
    unlabelled["airing_id"] = np.arange(len(unlabelled), dtype=np.int64)
    unlabelled = unlabelled[
        ["airing_id", "channel_id", "start", "end", "duration"]
    ]

    overlap = join_views_to_airings(selected_views, unlabelled, use_cache=False)
    overlap_seconds = overlap.groupby("user_id")["overlap"].sum()
    overlap_seconds = {int(user): float(value) for user, value in overlap_seconds.items()}

    return {
        int(user): overlap_seconds.get(int(user), 0.0)
        / viewing_seconds.get(int(user), 1.0)
        for user in users
    }


def main():
    started = time.perf_counter()
    RESULTS_DIR.mkdir(exist_ok=True)

    views = load_views()
    schedule = load_schedule()
    airings = build_airings(schedule)
    joined = join_views_to_airings(views, airings)
    info = load_show_info()
    info_map = show_info_map(info)

    submission = pd.read_csv(SUBMISSION_PATH, dtype={"user_id": "uint64"})
    submission_users = submission["user_id"].to_numpy(dtype=np.uint64)
    test_users = set(int(user) for user in submission_users)

    train = training_events(joined, airings, FINAL_TRAIN_WINDOW)
    full = train[train["st"] >= 0.8]
    truths = labels_for_window(joined, airings, VALID_WINDOW)
    valid_users = np.load(SCORE_DIR / "valid_users.npy")
    validation_users = [int(user) for user in valid_users]
    valid_test_users = sorted(set(validation_users) & test_users)
    valid_test_array = np.array(valid_test_users, dtype=np.uint64)

    full_counts = full[full["user_id"].isin(valid_test_array)].groupby("user_id").size()
    full_counts = full_counts.reindex(valid_test_array, fill_value=0)
    counts_frame = pd.DataFrame(
        {
            "user_id": [int(user) for user in valid_test_array],
            "count": full_counts.to_numpy(dtype=np.int64),
        }
    )
    p10 = float(np.percentile(counts_frame["count"], 10))
    p90 = float(np.percentile(counts_frame["count"], 90))
    counts_frame["distance_10"] = (counts_frame["count"] - p10).abs()
    counts_frame["distance_90"] = (counts_frame["count"] - p90).abs()
    light = counts_frame.sort_values(["distance_10", "user_id"]).iloc[0]
    heavy = counts_frame.sort_values(["distance_90", "user_id"]).iloc[0]
    light_user = int(counts_frame.at[light.name, "user_id"])
    heavy_user = int(counts_frame.at[heavy.name, "user_id"])

    shares = unlabelled_shares(views, schedule, [heavy_user, light_user])

    final_user_predictions = {}
    for method, filename in METHOD_FILES.items():
        _, predictions = read_final_submission(ROOT / "submissions" / filename)
        final_user_predictions[method] = predictions

    matrix, matrix_users, matrix_shows = build_user_show_matrix(
        history_scores(train, 30)
    )
    matrix_user_index = {int(user): i for i, user in enumerate(matrix_users)}
    matrix_show_index = {int(show): i for i, show in enumerate(matrix_shows)}
    final_candidates = np.array(
        sorted(int(show) for show in candidate_shows(airings, TEST_WINDOW)),
        dtype=np.int64,
    )
    genre_final_scores = genre_scores(
        matrix,
        matrix_users,
        matrix_user_index,
        matrix_shows,
        submission_users,
        final_candidates,
    )
    genre_indices = np.argsort(-genre_final_scores, axis=1, kind="stable")[:, :5]
    genre_final = final_candidates[genre_indices]
    pop_counts = (
        full.groupby("show_id")
        .size()
        .reindex(final_candidates, fill_value=0)
        .to_numpy(dtype=np.float32)
    )
    pop_ranking = final_candidates[np.lexsort((final_candidates, -pop_counts))]
    no_history_mask = np.array(
        [int(user) not in matrix_user_index for user in submission_users], dtype=bool
    )
    genre_final[no_history_mask] = pop_ranking[:5]
    genre_predictions = {
        int(user): list(genre_final[row])
        for row, user in enumerate(submission_users)
    }
    final_user_predictions["cbf_genre"] = genre_predictions

    chosen = json.loads((SCORE_DIR / "step3_chosen_weights.json").read_text())
    validation_candidates = np.load(SCORE_DIR / "candidates.npy")
    validation_index = {int(user): row for row, user in enumerate(valid_users)}
    validation_matrices = {
        "popularity": np.load(SCORE_DIR / "POPULARITY__-.npy"),
        "ppr": np.load(SCORE_DIR / "PPR__alpha0.7.npy"),
        "puresvd": np.load(SCORE_DIR / "PureSVD__k16.npy"),
        "cbf_schedule": np.load(SCORE_DIR / "CBF_schedule__channel_x_hour.npy"),
        "cbf_genre": np.load(SCORE_DIR / "CBF_genre__tfidf.npy"),
    }

    def validation_predictions(user):
        row = validation_index[int(user)]
        raw = {
            method: validation_matrices[method][row]
            for method in validation_matrices
        }
        hybrid_row = (
            normalize_row(raw["ppr"])
            + chosen["w_c2"] * normalize_row(raw["cbf_schedule"])
            + chosen["w_svd"] * normalize_row(raw["puresvd"])
            + chosen["w_pop"]
            * (raw["popularity"] / validation_matrices["popularity"].max())
            + chosen["w_c1"] * normalize_row(raw["cbf_genre"])
        )
        output = {
            method: [
                int(validation_candidates[index])
                for index in np.argsort(-scores, kind="stable")[:5]
            ]
            for method, scores in raw.items()
        }
        output["hybrid"] = [
            int(validation_candidates[index])
            for index in np.argsort(-hybrid_row, kind="stable")[:5]
        ]
        return output

    user_sections = []
    for label, user in (("HEAVY", heavy_user), ("LIGHT", light_user)):
        user_full = full[full["user_id"] == user]
        fully_watched_airings = len(user_full)
        distinct_shows = int(user_full["show_id"].nunique())
        fully_watched_shows = set(int(show) for show in user_full["show_id"].unique())

        top_shows = (
            user_full.groupby("show_id")
            .size()
            .reset_index(name="count")
            .sort_values(["count", "show_id"], ascending=[False, True])
            .head(5)
        )
        top_rows = []
        for row in top_shows.itertuples(index=False):
            show = int(row.show_id)
            meta = info_map.get(show, {"title": "", "category": "", "channel": ""})
            top_rows.append(
                [show, meta["title"], meta["category"], meta["channel"], int(row.count)]
            )

        section = [f"## {label} user `{user}`", ""]
        section.append(
            f"- Fully watched airings in weeks 11-30: **{fully_watched_airings:,}**"
        )
        section.append(f"- Distinct fully watched shows in weeks 11-30: **{distinct_shows:,}**")
        section.append(f"- Share of viewing seconds on unlabelled airtime: **{shares[user]:.4f}**")
        section.append("")
        section.append("### Top 5 shows by fully watched count, weeks 11-30")
        section.append("")
        section.append(
            markdown_table(
                ["id", "title", "category", "channel", "fully watched airings"],
                top_rows,
            )
        )
        section.append("")
        section.append("### Final Top-5 for weeks 31-42")
        section.append("")
        final_rows = []
        for method in ("popularity", "ppr", "puresvd", "cbf_schedule", "cbf_genre", "hybrid"):
            for rank, show in enumerate(final_user_predictions[method][user], start=1):
                show = int(show)
                meta = info_map.get(show, {"title": "", "category": "", "channel": ""})
                final_rows.append(
                    [
                        METHOD_LABELS[method],
                        rank,
                        show,
                        meta["title"],
                        meta["category"],
                        "yes" if show in fully_watched_shows else "no",
                    ]
                )
        section.append(
            markdown_table(
                ["method", "rank", "id", "title", "category", "fully watched 11-30"],
                final_rows,
            )
        )
        section.append("")
        section.append("### Validation check, weeks 21-30")
        section.append("")
        truth = truths[user]
        truth_rows = []
        for show in truth:
            show = int(show)
            meta = info_map.get(show, {"title": "", "category": "", "channel": ""})
            truth_rows.append([show, meta["title"]])
        section.append("**Actual validation truth (top 5):**")
        section.append("")
        section.append(markdown_table(["id", "title"], truth_rows))
        section.append("")
        truth_set = set(int(show) for show in truth)
        predictions = validation_predictions(user)
        validation_rows = []
        for method in ("popularity", "ppr", "puresvd", "cbf_schedule", "cbf_genre", "hybrid"):
            marked = []
            for show in predictions[method]:
                marker = "*" if show in truth_set else ""
                marked.append(f"{show}{marker}")
            ap = ap_at_5(predictions[method], truth_set)
            validation_rows.append(
                [METHOD_LABELS[method], ", ".join(marked), f"{ap:.6f}"]
            )
        section.append(
            markdown_table(
                ["method", "validation top-5 ids (hits marked *)", "AP@5"],
                validation_rows,
            )
        )
        section.append("")
        user_sections.append("\n".join(section))

    header = [
        "# Two-user examples",
        "",
        f"HEAVY chosen at the 90th percentile, LIGHT at the 10th percentile of fully watched airings in weeks 11-30 among the 1,150 validation/test users.",
        f"Percentiles: p10 = {p10:.2f}, p90 = {p90:.2f}. Ties go to the smaller user id.",
        "",
    ]
    report = "\n".join(header) + "\n".join(user_sections)
    (RESULTS_DIR / "two_users.md").write_text(report, encoding="utf-8")

    step2 = pd.read_csv(SCORE_DIR / "manifest.csv")
    step2.to_csv(RESULTS_DIR / "step2_validation.csv", index=False)

    grid = pd.read_csv(SCORE_DIR / "step3_grid.csv")
    grid.sort_values(
        ["map_all", "w_c2", "w_svd", "w_pop", "w_c1"],
        ascending=[False, True, True, True, True],
    ).head(5).to_csv(RESULTS_DIR / "hybrid_grid_top5.csv", index=False)

    kaggle = pd.DataFrame(
        [
            {
                "file": "hybrid.csv",
                "ref": 57590784,
                "publicScore": 0.44670,
                "privateScore": 0.44039,
                "status": "COMPLETE",
            },
            {
                "file": "ppr.csv",
                "ref": 57590789,
                "publicScore": 0.43309,
                "privateScore": 0.42321,
                "status": "COMPLETE",
            },
            {
                "file": "puresvd.csv",
                "ref": 57590792,
                "publicScore": 0.43061,
                "privateScore": 0.41875,
                "status": "COMPLETE",
            },
            {
                "file": "cbf_schedule.csv",
                "ref": 57590795,
                "publicScore": 0.41879,
                "privateScore": 0.40339,
                "status": "COMPLETE",
            },
            {
                "file": "popularity.csv",
                "ref": 57590797,
                "publicScore": 0.31942,
                "privateScore": 0.30620,
                "status": "COMPLETE",
            },
        ]
    )
    kaggle.to_csv(RESULTS_DIR / "kaggle_scores.csv", index=False)

    print(f"HEAVY user: {heavy_user} (count={int(heavy['count'])}, distance={heavy['distance_90']:.2f})")
    print(f"LIGHT user: {light_user} (count={int(light['count'])}, distance={light['distance_10']:.2f})")
    print(f"p10={p10:.2f} p90={p90:.2f}")
    print(f"unlabelled share HEAVY={shares[heavy_user]:.4f} LIGHT={shares[light_user]:.4f}")
    print(f"saved {RESULTS_DIR / 'two_users.md'}")
    print(f"saved {RESULTS_DIR / 'step2_validation.csv'}")
    print(f"saved {RESULTS_DIR / 'hybrid_grid_top5.csv'}")
    print(f"saved {RESULTS_DIR / 'kaggle_scores.csv'}")
    print(f"runtime: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
