from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MOVIES_PATH = DATA_DIR / "movies.csv"
RATINGS_PATH = DATA_DIR / "ratings.csv"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"

def load_data():
    """Load MovieLens files, downloading and extracting them on first use."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not MOVIES_PATH.exists() or not RATINGS_PATH.exists():
        archive = DATA_DIR / "ml-latest-small.zip"
        try:
            urlretrieve(DATA_URL, archive)
            with ZipFile(archive, "r") as zf:
                # Extract only the two files this project needs.
                for name in ("ml-latest-small/movies.csv", "ml-latest-small/ratings.csv"):
                    target = DATA_DIR / name.split("/")[-1]
                    with zf.open(name) as source, open(target, "wb") as dest:
                        dest.write(source.read())
        except Exception as exc:
            raise RuntimeError(
                "Could not download MovieLens. Check your internet connection, or manually download "
                "ml-latest-small.zip from https://grouplens.org/datasets/movielens/ and put movies.csv "
                "and ratings.csv into the project's data folder."
            ) from exc

    movies = pd.read_csv(MOVIES_PATH)
    ratings = pd.read_csv(RATINGS_PATH)
    movies["genres"] = movies["genres"].fillna("(no genres listed)")
    ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    return {"movies": movies, "ratings": ratings}

def _genre_columns(movies):
    all_genres = sorted({
        genre
        for cell in movies["genres"].fillna("")
        for genre in str(cell).split("|")
        if genre and genre != "(no genres listed)"
    })
    return all_genres

def _build_features(ratings, movies, genre_columns, profile_source=None):
    """Build one feature row per rating using only the supplied profile source."""
    source = ratings if profile_source is None else profile_source
    movie_genres = movies[["movieId", "genres"]].copy()
    for genre in genre_columns:
        movie_genres[f"movie_genre_{genre}"] = movie_genres["genres"].apply(
            lambda value: int(genre in str(value).split("|"))
        )
    genre_feature_cols = [f"movie_genre_{g}" for g in genre_columns]
    base = ratings[["userId", "movieId"]].merge(
        movie_genres[["movieId"] + genre_feature_cols], on="movieId", how="left"
    )

    # A user's genre profile is computed from their historical ratings in profile_source.
    source_join = source[["userId", "movieId", "rating"]].merge(
        movie_genres[["movieId"] + genre_feature_cols], on="movieId", how="left"
    )
    if len(source_join):
        source_join[genre_feature_cols] = source_join[genre_feature_cols].fillna(0)
        source_join["positive_weight"] = source_join["rating"] - 2.5
        for col in genre_feature_cols:
            source_join[f"profile_{col}"] = source_join[col] * source_join["positive_weight"]
        profile_cols = [f"profile_{c}" for c in genre_feature_cols]
        profiles = source_join.groupby("userId")[profile_cols].mean()
        profiles.columns = [f"user_{c.replace('profile_movie_genre_', '')}" for c in profile_cols]
        user_mean = source.groupby("userId")["rating"].mean().rename("user_mean_rating")
        user_count = source.groupby("userId")["rating"].count().rename("user_rating_count")
        profiles = profiles.join(user_mean).join(user_count)
    else:
        profile_cols = [f"user_{g}" for g in genre_columns]
        profiles = pd.DataFrame(columns=profile_cols + ["user_mean_rating", "user_rating_count"])

    movie_stats = source.groupby("movieId")["rating"].agg(["mean", "count"]).rename(
        columns={"mean": "movie_mean_rating", "count": "movie_rating_count"}
    )
    base = base.join(profiles, on="userId")
    base = base.join(movie_stats, on="movieId")
    user_feature_cols = [c for c in base.columns if c.startswith("user_")]
    base[user_feature_cols] = base[user_feature_cols].fillna(0)
    base[["movie_mean_rating", "movie_rating_count"]] = base[
        ["movie_mean_rating", "movie_rating_count"]
    ].fillna(0)
    feature_cols = genre_feature_cols + user_feature_cols + ["movie_mean_rating", "movie_rating_count"]
    # Log transform count features to reduce scale differences.
    base["movie_rating_count"] = np.log1p(base["movie_rating_count"])
    if "user_rating_count" in base:
        base["user_rating_count"] = np.log1p(base["user_rating_count"])
    return base[feature_cols].astype(float), feature_cols

def train_model(data):
    """Train an MLP on a held-out random split of ratings."""
    ratings = data["ratings"].copy()
    movies = data["movies"].copy()
    ratings["liked"] = (ratings["rating"] >= 4.0).astype(int)
    train_idx, test_idx = train_test_split(
        np.arange(len(ratings)), test_size=0.2, random_state=42,
        stratify=ratings["liked"] if ratings["liked"].nunique() > 1 else None
    )
    train_ratings = ratings.iloc[train_idx].copy()
    test_ratings = ratings.iloc[test_idx].copy()
    genre_columns = _genre_columns(movies)

    X_train, feature_cols = _build_features(
        train_ratings, movies, genre_columns, profile_source=train_ratings
    )
    # For test features, user/movie summaries are based only on training ratings.
    X_test, _ = _build_features(
        test_ratings, movies, genre_columns, profile_source=train_ratings
    )
    y_train = train_ratings["liked"].to_numpy()
    y_test = test_ratings["liked"].to_numpy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=42
        ))
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }
    bundle = {
        "genre_columns": genre_columns,
        "feature_cols": feature_cols,
        "train_ratings": train_ratings.drop(columns=["liked"]),
        "movies": movies,
        "ratings": ratings.drop(columns=["liked"]),
    }
    return model, bundle, metrics

def _feature_for_pair(data, bundle, user_id, movie_id):
    movies = bundle["movies"]
    train_ratings = bundle["train_ratings"]
    movie_row = movies.loc[movies["movieId"] == int(movie_id)]
    if movie_row.empty:
        return None
    user_history = train_ratings.loc[train_ratings["userId"] == int(user_id)]
    # For an existing user, use their training history; if no history, use a global profile.
    if user_history.empty:
        profile_source = train_ratings
        synthetic_user_id = -999999
        pair = pd.DataFrame({"userId": [synthetic_user_id], "movieId": [int(movie_id)]})
    else:
        profile_source = train_ratings
        pair = pd.DataFrame({"userId": [int(user_id)], "movieId": [int(movie_id)]})
    X, _ = _build_features(pair, movies, bundle["genre_columns"], profile_source=profile_source)
    # For new/synthetic user, use average training user profile by assigning the synthetic ID's profile
    # via a simple fallback profile. Rebuild pair features with a known user where needed.
    expected = bundle["feature_cols"]
    X = X.reindex(columns=expected, fill_value=0)
    return X

def predict_movie_for_user(data, model, bundle, user_id, movie_id):
    X = _feature_for_pair(data, bundle, user_id, movie_id)
    if X is None:
        return None
    probability = float(model.predict_proba(X)[0][1])
    return {"probability": probability}

def recommend_for_user(data, model, bundle, user_id, n=10):
    movies = bundle["movies"].copy()
    all_ratings = bundle["ratings"]
    watched = set(all_ratings.loc[all_ratings["userId"] == int(user_id), "movieId"].astype(int))
    candidates = movies.loc[~movies["movieId"].isin(watched)].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["movieId", "title", "genres", "predicted_like_probability", "average_rating"])
    # Score candidates in one batch for speed.
    pairs = pd.DataFrame({"userId": int(user_id), "movieId": candidates["movieId"].astype(int).to_numpy()})
    X, _ = _build_features(pairs, movies, bundle["genre_columns"], profile_source=bundle["train_ratings"])
    X = X.reindex(columns=bundle["feature_cols"], fill_value=0)
    candidates["predicted_like_probability"] = model.predict_proba(X)[:, 1]
    stats = all_ratings.groupby("movieId")["rating"].mean().rename("average_rating")
    candidates = candidates.join(stats, on="movieId")
    candidates["average_rating"] = candidates["average_rating"].fillna(0)
    return candidates.sort_values("predicted_like_probability", ascending=False).head(n)[
        ["movieId", "title", "genres", "predicted_like_probability", "average_rating"]
    ].reset_index(drop=True)
