# SweetTV TV Program Recommender

Solution for the Kaggle competition **`sweettv-tv-program-recommender`**. The task is to recommend the five TV shows a user is most likely to watch in a future four-week period. The competition metric is **MAP@5**.

The final approach is a weighted hybrid of collaborative, content-based, and graph-based rankers, with a popularity fallback for users without usable history. The best public/private Kaggle scores were **0.44670 / 0.44039**.

## Problem setup

- **Train period:** weeks 11-30 (2020-03-09 through 2020-07-26).
- **Validation period:** weeks 21-30.
- **Test period:** weeks 31-42.
- **Candidate pool:** shows with at least one labelled airing in the target window. Repeats are allowed in the submission, but recommendations are deduplicated.
- **Positive label:** a `(user, show)` pair is positive when the user fully watched at least one airing of the show.
  - `st = overlap_seconds / airing_duration`
  - fully watched if `st >= 0.8`
  - `tv_show_id == 0` is unlabelled airtime and is dropped.
- **User labels:** for each user, the top five shows by fully watched airing count, with ties broken by smaller `tv_show_id`.

The validation set has **4,030** users; **1,150** of them also appear in the test submission users.

## Repository layout

```
.
├── run_step1.py          # data checks, labelling, MAP@5, baselines
├── run_step2.py          # SVD, ALS, PPR, CBF, validation table
├── run_step3.py          # hybrid grid, final submissions
├── run_step4.py          # two-user case study and result CSVs
├── src/
│   ├── config.py         # paths and constants
│   ├── io.py             # raw data loading and caching
│   ├── clock.py          # week/time helpers
│   ├── join.py           # view-to-airing interval join
│   ├── labels.py         # candidate pool and target labels
│   ├── metrics.py        # AP@5 / MAP@5
│   ├── baselines.py      # popularity and personal history
│   ├── matrices.py       # user-show matrix utilities
│   ├── method_svd.py     # PureSVD
│   ├── method_als.py     # implicit ALS
│   ├── method_ppr.py     # personalized PageRank
│   ├── cbf.py            # genre TF-IDF and channel x hour CBF
│   └── ranking.py        # top-k and fallback logic
├── submissions/          # final Kaggle submission files
└── results/              # validation, grid, Kaggle scores, case study
```

Raw data and intermediate caches are intentionally not committed (see `.gitignore`). Place the competition files in `data/raw/` before running the pipeline.

## Methods

| Method | Description |
|---|---|
| Popularity | Counts fully watched airings per show in the training window; ties by show id. |
| Personal history | Sums `min(st, 1) * exp(-(last_week - week) / 8)` per user-show. |
| PureSVD | Truncated SVD on `log1p(user-show score)` with `k = 16, 32, 64`. |
| Implicit ALS | 64 factors, regularization `0.1`, confidence `alpha=10`, 20 iterations. |
| PPR | Personalized PageRank over the bipartite user-show graph with restart `alpha`. |
| CBF genre | TF-IDF over category and genre fields, L2-normalized show profiles, cosine scores against user profiles. |
| CBF schedule | User profiles over `(channel, hour_of_week)` with an 8-week exponential time decay; show features from their aired `(channel, hour_of_week)` slots. |
| Hybrid | Weighted sum of per-user max-normalized score matrices. |

The hybrid uses:

```
0.7-PPR
+ 1.00 * CBF schedule
+ 0.50 * PureSVD(k=16)
+ 0.00 * popularity
+ 0.25 * CBF genre
```

Popularity is normalized by its global maximum; the other components are normalized per user. Users with no usable history receive the popularity top five.

## Validation results (weeks 21-30)

MAP@5 on all 4,030 validation users and on the 1,150 validation users who are also test users:

| Method | Setting | MAP@5 all | MAP@5 test subset |
|---|---|---:|---:|
| Popularity | - | 0.279273 | 0.305087 |
| Personal history | - | 0.360413 | 0.398719 |
| PureSVD | k=16 | 0.371119 | 0.403907 |
| PureSVD | k=32 | 0.368623 | 0.400568 |
| PureSVD | k=64 | 0.363184 | 0.396716 |
| Implicit ALS | f=64, reg=0.1, alpha=10, it=20 | 0.095081 | 0.079533 |
| PPR | alpha=0.3 | 0.366347 | 0.404261 |
| PPR | alpha=0.5 | 0.377374 | 0.413759 |
| PPR | alpha=0.7 | 0.380907 | 0.414968 |
| PPR | alpha=0.85 | 0.349722 | 0.377504 |
| CBF genre | tfidf | 0.163265 | 0.178771 |
| CBF schedule | channel x hour | 0.373893 | 0.404687 |
| **Hybrid** | **selected weights** | **0.396572** | **0.428516** |

The hybrid improves over the best single model on validation (PPR `alpha=0.7`, 0.380907) and over the personal-history baseline (0.360413).

## Kaggle results

| Submission | Ref | Public | Private |
|---|---:|---:|---:|
| `submissions/hybrid.csv` | 57590784 | 0.44670 | 0.44039 |
| `submissions/ppr.csv` | 57590789 | 0.43309 | 0.42321 |
| `submissions/puresvd.csv` | 57590792 | 0.43061 | 0.41875 |
| `submissions/cbf_schedule.csv` | 57590795 | 0.41879 | 0.40339 |
| `submissions/popularity.csv` | 57590797 | 0.31942 | 0.30620 |

## Two-user case study

`results/two_users.md` contains detailed examples for a heavy and a light user. The users were chosen at the 90th and 10th percentiles of fully watched airings in weeks 11-30.

| User | Fully watched airings | Distinct shows | Unlabelled share of viewing seconds |
|---|---:|---:|---:|
| HEAVY `3717184880213603078` | 360 | 127 | 0.4854 |
| LIGHT `2860008338744295871` | 38 | 24 | 0.2767 |

The report includes each user's top five shows, the final top five from every method, and a validation check on weeks 21-30 with per-method AP@5. A large unlabelled share indicates why pure content/schedule signals are noisy: much of the viewing time cannot be tied to a known show.

## How to reproduce

Requirements: Python 3.13, `numpy`, `pandas`, `scipy`, `pyarrow`, `implicit`.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Put the competition files in `data/raw/`:

- `dataset11-30.csv`
- `export_arh_11-20-final.csv`
- `export_arh_21-30-final.csv`
- `export_arh_31-42-final.csv`
- `submission.csv`

Then run the pipeline in order:

```bash
.venv/bin/python run_step1.py
.venv/bin/python run_step2.py
.venv/bin/python run_step3.py
.venv/bin/python run_step4.py
```

Step 1 validates the join, labels, and baselines. Step 2 writes the validation score matrices and manifest. Step 3 runs the hybrid grid, rebuilds the final model on weeks 11-30, and writes `submissions/`. Step 4 produces the case study and the CSVs in `results/`.

The cached score matrices in `data/cache/scores/` make Step 3/4 re-runs fast; the first run is dominated by the view-to-airing join and the PPR/SVD fits.

## Notes and limitations

- `st >= 0.8` is a proxy for "fully watched"; it does not capture intentional rewatching or background viewing.
- About two test users have no usable history and fall back to popularity.
- ALS performs poorly because the interaction matrix is extremely sparse and the confidence weighting favors popular shows.
- Genre CBF is weak: many shows share broad categories, so genre alone is not discriminative. Schedule CBF is substantially better because it captures channel and time-slot preferences.
- The final hybrid was selected on weeks 21-30 and then refit on weeks 11-30; the Kaggle scores are the out-of-sample check.
