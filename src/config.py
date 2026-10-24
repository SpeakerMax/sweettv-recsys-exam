from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

LOG_PATH = RAW / "dataset11-30.csv"
SCHEDULE_PATHS = (
    RAW / "export_arh_11-20-final.csv",
    RAW / "export_arh_21-30-final.csv",
    RAW / "export_arh_31-42-final.csv",
)
SUBMISSION_PATH = RAW / "submission.csv"

VIEWS_CACHE = CACHE / "views.parquet"
SCHEDULE_CACHE = CACHE / "schedule.parquet"
AIRINGS_CACHE = CACHE / "airings.parquet"
JOINED_CACHE = CACHE / "joined.parquet"
SHOW_META_CACHE = CACHE / "show_meta.parquet"
GENRE_FEATURES_CACHE = CACHE / "genre_tfidf.npz"
GENRE_SHOWS_CACHE = CACHE / "genre_shows.npy"
SCHEDULE_PROFILE_CACHE = CACHE / "schedule_profile.npz"
SCORE_DIR = CACHE / "scores"
SCORE_DIR.mkdir(parents=True, exist_ok=True)

FIRST_WEEK = 11
LAST_WEEK = 42
TRAIN_WINDOW = (11, 20)
VALID_WINDOW = (21, 30)
TEST_WINDOW = (31, 42)

FULL_WATCH_THRESHOLD = 0.8
MIN_SHOWS = 5
HALF_LIFE_WEEKS = 8.0
