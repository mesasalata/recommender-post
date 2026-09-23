"""
Prepare MovieLens 33M data for the recommender-system experiment.

Default experimental design
---------------------------
- Use the complete ratings.csv file.
- Restrict the common movie catalog to movies with Tag Genome features.
- Require at least five interactions per retained user.
- Split chronologically within each user:
    latest interaction       -> test
    second-latest interaction -> validation
    all earlier interactions -> training
- Build warm-start sampled ranking candidates:
    one relevant held-out item and 999 uniformly sampled unobserved
    training-catalog items.
 - Precompute sparse top-k movie-movie adjusted-cosine similarities.

Parallel processing
-------------------
The three raw-file loads run concurrently. Movie-similarity blocks, the two
content profiles, and the two ranking candidate sets are computed in forked
worker processes that inherit input arrays through copy-on-write memory.
Forked results are gathered in deterministic order, so artifacts are
identical to the serial pipeline. Set PARALLEL_WORKERS=1 to disable
process-level parallelism.

The common validation set is used for:
- content-score scaling;
- hybrid-weight selection;
- MF hyperparameter selection;
- BPR hyperparameter selection;
- the optional post-hoc BPR affine rating mapping.

References
----------
Sarwar, B., Karypis, G., Konstan, J., and Riedl, J. (2001).
"Item-Based Collaborative Filtering Recommendation Algorithms."
WWW 2001, 285-295.

Vig, J., Sen, S., and Riedl, J. (2012).
"The Tag Genome: Encoding Community Knowledge to Support Novel
Interaction." ACM TiiS, 2(3), Article 13.

Cremonesi, P., Koren, Y., and Turrin, R. (2010).
"Performance of Recommender Algorithms on Top-N Recommendation Tasks."
RecSys 2010, 39-46.

Krichene, W., and Rendle, S. (2020).
"On Sampled Metrics for Item Recommendation."
KDD 2020, 1748-1757.
"""

from __future__ import annotations

import csv
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

# Default to single-threaded BLAS so forked workers do not oversubscribe
# the machine. Pre-existing environment values take precedence.
for _blas_variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ.setdefault(_blas_variable, "1")

import numpy as np
from scipy import sparse


NUM_TAGS = 1128
MIN_USER_INTERACTIONS = 5
RAW_DATA_DIR = Path("RawData/33M-grouplens")
OUTPUT_DIR = Path("ProcessedData")
PREDICTIONS_DIR = Path("Predictions")

RELEVANCE_THRESHOLD = 3.5
RANKING_USERS = 500
RANKING_CANDIDATES = 1000

MOVIE_NEIGHBORS = 40
MIN_COMMON_RATERS = 2
SIMILARITY_SHRINKAGE = 10.0
ITEM_SIMILARITY_BLOCK_SIZE = 64

PARALLEL_WORKERS: int | None = None

MOVIELENS_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)",
]


def raw_path(filename: str) -> Path:
    return RAW_DATA_DIR / filename


def output_path(filename: str) -> Path:
    return OUTPUT_DIR / filename


def load_raw_ratings(
    sample_size: int | None = None,
    offset: int = 0,
) -> np.ndarray:
    """Load [userId, movieId, rating, timestamp].

    The default uses the complete file. Nonzero offsets or finite sample
    sizes are supported only for explicitly documented development runs;
    they do not constitute a random MovieLens sample.
    """
    if sample_size is not None and sample_size <= 0:
        raise ValueError("sample_size must be positive or None.")
    if offset < 0:
        raise ValueError("offset must be nonnegative.")

    values = np.loadtxt(
        raw_path("ratings.csv"),
        delimiter=",",
        skiprows=1 + offset,
        max_rows=sample_size,
        dtype=np.float64,
        ndmin=2,
    )
    validate_raw_ratings(values)
    return values


def validate_raw_ratings(ratings: np.ndarray) -> None:
    """Validate the MovieLens rating-table fields used by this pipeline."""
    if ratings.ndim != 2 or ratings.shape[1] != 4:
        raise ValueError("Ratings must have shape (n, 4).")
    if len(ratings) == 0:
        raise ValueError("No ratings were loaded.")
    if not np.all(np.isfinite(ratings)):
        raise ValueError("Ratings contain non-finite values.")

    for column, name in (
        (0, "user IDs"),
        (1, "movie IDs"),
        (3, "timestamps"),
    ):
        values = ratings[:, column]
        if np.any(values < 0) or np.any(values != np.floor(values)):
            raise ValueError(
                f"MovieLens {name} must be nonnegative integers."
            )

    if np.any((ratings[:, 2] < 0.5) | (ratings[:, 2] > 5.0)):
        raise ValueError("Ratings must lie in the interval [0.5, 5.0].")


def build_genre_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Return validated genre vectors and external movie IDs."""
    records: list[tuple[int, list[str]]] = []
    seen_movies: set[int] = set()
    known_genres = set(MOVIELENS_GENRES)

    with raw_path("movies.csv").open(encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header != ["movieId", "title", "genres"]:
            raise ValueError("movies.csv has an unexpected header.")

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise ValueError(
                    f"movies.csv line {line_number} has {len(row)} fields; "
                    "expected 3."
                )

            movie_id = int(row[0])
            if movie_id in seen_movies:
                raise ValueError(
                    f"movies.csv contains duplicate movie ID {movie_id}."
                )
            seen_movies.add(movie_id)

            genres = row[2].split("|")
            unknown = set(genres) - known_genres
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(
                    f"Movie {movie_id} has unknown genres: {names}."
                )

            records.append((movie_id, genres))

    if not records:
        raise ValueError("movies.csv contains no movie records.")

    records.sort(key=lambda value: value[0])
    genre_index = {
        genre: index for index, genre in enumerate(MOVIELENS_GENRES)
    }
    movie_ids = np.asarray(
        [movie_id for movie_id, _ in records],
        dtype=np.int64,
    )
    vectors = np.zeros(
        (len(records), len(MOVIELENS_GENRES)),
        dtype=np.float32,
    )

    for row, (_, genres) in enumerate(records):
        for genre in genres:
            vectors[row, genre_index[genre]] = 1.0

    return vectors, movie_ids


def build_genome_scores() -> tuple[np.ndarray, np.ndarray]:
    """Return complete Tag Genome vectors and external movie IDs.

    The score file is processed as contiguous movie groups. MovieLens
    distributes genome-scores.csv ordered by movie ID and tag ID. Failing
    on a repeated, noncontiguous movie prevents silent corruption if that
    source invariant does not hold.
    """
    tag_ids: list[int] = []

    with raw_path("genome-tags.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header != ["tagId", "tag"]:
            raise ValueError("genome-tags.csv has an unexpected header.")

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 2:
                raise ValueError(
                    f"genome-tags.csv line {line_number} is malformed."
                )
            tag_ids.append(int(row[0]))

    if len(tag_ids) != NUM_TAGS:
        raise ValueError(
            f"Expected {NUM_TAGS} genome tags, found {len(tag_ids)}."
        )
    if len(set(tag_ids)) != len(tag_ids):
        raise ValueError("genome-tags.csv contains duplicate tag IDs.")

    tag_index = {tag_id: index for index, tag_id in enumerate(tag_ids)}
    movie_ids: list[int] = []
    score_rows: list[np.ndarray] = []
    completed_movies: set[int] = set()

    current_movie: int | None = None
    current_scores = np.zeros(NUM_TAGS, dtype=np.float32)
    current_seen = np.zeros(NUM_TAGS, dtype=bool)

    def finish_current_movie() -> None:
        nonlocal current_movie, current_scores, current_seen
        if current_movie is None:
            return

        missing = int(np.sum(~current_seen))
        if missing:
            raise ValueError(
                f"Movie {current_movie} is missing {missing} Tag Genome "
                "scores."
            )

        movie_ids.append(current_movie)
        score_rows.append(current_scores)
        completed_movies.add(current_movie)
        current_movie = None
        current_scores = np.zeros(NUM_TAGS, dtype=np.float32)
        current_seen = np.zeros(NUM_TAGS, dtype=bool)

    with raw_path("genome-scores.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header != ["movieId", "tagId", "relevance"]:
            raise ValueError("genome-scores.csv has an unexpected header.")

        for line_number, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise ValueError(
                    f"genome-scores.csv line {line_number} is malformed."
                )

            movie_id = int(row[0])
            tag_id = int(row[1])
            relevance = float(row[2])

            if tag_id not in tag_index:
                raise ValueError(
                    f"Unknown tag ID {tag_id} at line {line_number}."
                )
            if not np.isfinite(relevance):
                raise ValueError(
                    f"Non-finite relevance at line {line_number}."
                )
            if relevance < 0.0 or relevance > 1.0:
                raise ValueError(
                    f"Relevance outside [0, 1] at line {line_number}."
                )

            if current_movie is None:
                if movie_id in completed_movies:
                    raise ValueError(
                        f"Movie {movie_id} occurs in noncontiguous groups."
                    )
                current_movie = movie_id
            elif movie_id != current_movie:
                finish_current_movie()
                if movie_id in completed_movies:
                    raise ValueError(
                        f"Movie {movie_id} occurs in noncontiguous groups."
                    )
                current_movie = movie_id

            position = tag_index[tag_id]
            if current_seen[position]:
                raise ValueError(
                    f"Duplicate score for movie {movie_id}, tag {tag_id}."
                )
            current_seen[position] = True
            current_scores[position] = relevance

    finish_current_movie()

    if not movie_ids:
        raise ValueError("genome-scores.csv contains no scores.")

    ids = np.asarray(movie_ids, dtype=np.int64)
    scores = np.stack(score_rows)
    order = np.argsort(ids, kind="stable")
    return scores[order], ids[order]


def filter_and_remap(
    ratings: np.ndarray,
    genome_movie_ids: np.ndarray,
    min_user_interactions: int = MIN_USER_INTERACTIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filter to the common Tag Genome catalog and densely remap IDs."""
    if ratings.ndim != 2 or ratings.shape[1] != 4:
        raise ValueError("Ratings must have shape (n, 4).")
    if len(ratings) == 0:
        raise ValueError("No ratings were loaded.")

    external_movies = ratings[:, 1].astype(np.int64)

    keep = np.isin(external_movies, genome_movie_ids)
    ratings = ratings[keep]
    if len(ratings) == 0:
        raise ValueError("No ratings remain after Tag Genome filtering.")

    users = ratings[:, 0].astype(np.int64)
    unique_users, counts = np.unique(users, return_counts=True)
    retained_users = unique_users[counts >= min_user_interactions]

    ratings = ratings[np.isin(users, retained_users)]
    if len(ratings) == 0:
        raise ValueError("No users satisfy the minimum-history requirement.")

    user_ids = np.unique(ratings[:, 0].astype(np.int64))
    movie_ids = np.unique(ratings[:, 1].astype(np.int64))

    remapped = np.empty_like(ratings, dtype=np.float64)
    remapped[:, 0] = np.searchsorted(
        user_ids, ratings[:, 0].astype(np.int64)
    )
    remapped[:, 1] = np.searchsorted(
        movie_ids, ratings[:, 1].astype(np.int64)
    )
    remapped[:, 2:] = ratings[:, 2:]

    pairs = remapped[:, :2].astype(np.int64)
    if len(np.unique(pairs, axis=0)) != len(pairs):
        raise ValueError(
            "Duplicate user-item interactions were found. The algorithms "
            "assume at most one rating per user-item pair."
        )

    return remapped, user_ids, movie_ids


def chronological_split(
    interactions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each user chronologically.

    Equal timestamps are resolved by original ratings.csv row order. This
    is a deterministic tie-break and does not imply that the corresponding
    ratings occurred in that order.
    """
    users = interactions[:, 0].astype(np.int64)
    source_order = np.arange(len(interactions), dtype=np.int64)
    order = np.lexsort((source_order, interactions[:, 3], users))
    sorted_rows = interactions[order]

    _, starts, counts = np.unique(
        sorted_rows[:, 0].astype(np.int64),
        return_index=True,
        return_counts=True,
    )
    if np.any(counts < MIN_USER_INTERACTIONS):
        raise ValueError(
            "Every retained user must have at least "
            f"{MIN_USER_INTERACTIONS} ratings."
        )

    validation_indices = starts + counts - 2
    test_indices = starts + counts - 1

    validation_mask = np.zeros(len(sorted_rows), dtype=bool)
    test_mask = np.zeros(len(sorted_rows), dtype=bool)
    validation_mask[validation_indices] = True
    test_mask[test_indices] = True
    train_mask = ~(validation_mask | test_mask)

    train = sorted_rows[train_mask, :3].copy()
    validation = sorted_rows[validation_mask, :3].copy()
    test = sorted_rows[test_mask, :3].copy()
    return train, validation, test


def align_features(
    retained_movie_ids: np.ndarray,
    source_movie_ids: np.ndarray,
    source_vectors: np.ndarray,
) -> np.ndarray:
    """Align source feature rows to dense experiment movie indices."""
    if len(source_movie_ids) == 0:
        raise ValueError("The source feature table is empty.")
    if source_vectors.ndim != 2:
        raise ValueError("Source vectors must be two-dimensional.")
    if len(source_movie_ids) != len(source_vectors):
        raise ValueError("Source movie IDs and feature rows do not align.")
    if len(np.unique(source_movie_ids)) != len(source_movie_ids):
        raise ValueError("Source feature movie IDs are not unique.")

    source_order = np.argsort(source_movie_ids, kind="stable")
    sorted_ids = source_movie_ids[source_order]
    positions = np.searchsorted(sorted_ids, retained_movie_ids)
    in_bounds = positions < len(sorted_ids)

    valid = np.zeros(len(retained_movie_ids), dtype=bool)
    valid[in_bounds] = (
        sorted_ids[positions[in_bounds]]
        == retained_movie_ids[in_bounds]
    )
    if not np.all(valid):
        missing = retained_movie_ids[~valid]
        preview = ", ".join(str(value) for value in missing[:10])
        raise ValueError(
            f"Required features are missing for movie IDs: {preview}."
        )

    return source_vectors[source_order[positions]]


def l2_normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float32),
        where=norms > 0,
    )


def compute_means(
    train: np.ndarray,
    n_users: int,
    n_movies: int,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute global, user, and movie means plus train-presence masks."""
    if len(train) == 0:
        raise ValueError("Training data is empty.")

    global_mean = float(np.mean(train[:, 2]))
    users = train[:, 0].astype(np.int64)
    movies = train[:, 1].astype(np.int64)
    ratings = train[:, 2]

    user_sums = np.bincount(users, weights=ratings, minlength=n_users)
    user_counts = np.bincount(users, minlength=n_users)
    movie_sums = np.bincount(movies, weights=ratings, minlength=n_movies)
    movie_counts = np.bincount(movies, minlength=n_movies)

    user_means = np.full(n_users, global_mean, dtype=np.float64)
    movie_means = np.full(n_movies, global_mean, dtype=np.float64)

    user_seen = user_counts > 0
    movie_seen = movie_counts > 0
    user_means[user_seen] = user_sums[user_seen] / user_counts[user_seen]
    movie_means[movie_seen] = movie_sums[movie_seen] / movie_counts[movie_seen]

    return global_mean, user_means, movie_means, user_seen, movie_seen


def compute_user_profiles(
    train: np.ndarray,
    normalized_features: np.ndarray,
    user_means: np.ndarray,
    n_users: int,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Build residual-weighted content centroids and L2-normalize them.

    The profile is a Rocchio-like signed centroid:

        p_u = normalize(sum_i (r_ui - mean_u) * normalize(x_i))

    This is a documented content-recommender variant, not a formula from
    Salton et al. alone.
    """
    profiles = np.zeros(
        (n_users, normalized_features.shape[1]),
        dtype=np.float32,
    )

    for start in range(0, len(train), chunk_size):
        rows = train[start:start + chunk_size]
        users = rows[:, 0].astype(np.int64)
        movies = rows[:, 1].astype(np.int64)
        residuals = rows[:, 2] - user_means[users]
        contributions = (
            normalized_features[movies]
            * residuals[:, np.newaxis]
        )
        np.add.at(profiles, users, contributions)

    return l2_normalize_rows(profiles)


def build_interaction_indices(
    train: np.ndarray,
    n_users: int,
    n_movies: int,
) -> tuple[sparse.csr_matrix, sparse.csc_matrix]:
    """Build CSR and CSC training-rating matrices."""
    users = train[:, 0].astype(np.int64)
    movies = train[:, 1].astype(np.int64)
    ratings = train[:, 2].astype(np.float64)

    matrix = sparse.csr_matrix(
        (ratings, (users, movies)),
        shape=(n_users, n_movies),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix, matrix.tocsc()


def _movie_similarity_block(
    centered_matrix: sparse.csr_matrix,
    squared_matrix: sparse.csr_matrix,
    observed: sparse.csr_matrix,
    start: int,
    end: int,
    top_k: int,
    min_common: int,
    shrinkage: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sparse similarities for one movie row block."""
    left = centered_matrix[:, start:end]
    left_squared = squared_matrix[:, start:end]
    left_observed = observed[:, start:end]

    numerator = (left.T @ centered_matrix).toarray()
    first_norm = (left_squared.T @ observed).toarray()
    second_norm = (left_observed.T @ squared_matrix).toarray()
    support = (left_observed.T @ observed).toarray()

    denominator = np.sqrt(first_norm * second_norm)
    similarity = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    similarity[support < min_common] = 0.0
    similarity *= support / (support + shrinkage)

    local_rows = np.arange(end - start)
    similarity[local_rows, start + local_rows] = 0.0

    row_part_list: list[np.ndarray] = []
    col_part_list: list[np.ndarray] = []
    value_part_list: list[np.ndarray] = []

    for local_row in range(end - start):
        values = similarity[local_row]
        nonzero = np.flatnonzero(values != 0.0)
        if len(nonzero) == 0:
            continue

        # Primary key: descending absolute similarity.
        # Secondary key: ascending dense movie index.
        order = np.lexsort((nonzero, -np.abs(values[nonzero])))
        selected = nonzero[order[:top_k]]

        row_part_list.append(
            np.full(
                len(selected),
                start + local_row,
                dtype=np.int32,
            )
        )
        col_part_list.append(selected.astype(np.int32))
        value_part_list.append(values[selected].astype(np.float32))

    if not row_part_list:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, np.empty(0, dtype=np.float32)

    return (
        np.concatenate(row_part_list),
        np.concatenate(col_part_list),
        np.concatenate(value_part_list),
    )


def compute_topk_movie_similarity(
    train: np.ndarray,
    n_users: int,
    n_movies: int,
    top_k: int = MOVIE_NEIGHBORS,
    min_common: int = MIN_COMMON_RATERS,
    shrinkage: float = SIMILARITY_SHRINKAGE,
    block_size: int = ITEM_SIMILARITY_BLOCK_SIZE,
    workers: int = 1,
) -> sparse.csr_matrix:
    """Compute sparse top-k adjusted-cosine movie similarities.

    Similarity is calculated over pair-specific co-raters. It is then
    multiplied by n_common / (n_common + shrinkage), and similarities with
    fewer than min_common co-raters are removed.

    Processing movie rows in blocks avoids materializing a full dense
    movie-movie matrix. Blocks can be processed concurrently in forked
    workers when workers is greater than one.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if min_common <= 0:
        raise ValueError("min_common must be positive.")
    if shrinkage < 0:
        raise ValueError("shrinkage must be nonnegative.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    global_mean, user_means, _, _, _ = compute_means(
        train, n_users, n_movies
    )
    del global_mean

    users = train[:, 0].astype(np.int64)
    movies = train[:, 1].astype(np.int64)
    centered = train[:, 2] - user_means[users]

    shape = (n_users, n_movies)
    centered_matrix = sparse.csr_matrix(
        (centered, (users, movies)),
        shape=shape,
    )
    squared_matrix = sparse.csr_matrix(
        (centered ** 2, (users, movies)),
        shape=shape,
    )
    observed = sparse.csr_matrix(
        (np.ones(len(train)), (users, movies)),
        shape=shape,
    )

    _PARALLEL_STATE.update(
        centered_matrix=centered_matrix,
        squared_matrix=squared_matrix,
        observed=observed,
    )
    jobs = [
        (
            _movie_similarity_worker,
            (
                start,
                min(start + block_size, n_movies),
                top_k,
                min_common,
                shrinkage,
            ),
        )
        for start in range(0, n_movies, block_size)
    ]
    # Forked workers inherit the sparse matrices through _PARALLEL_STATE via
    # copy-on-write memory, so the matrices themselves are never pickled.

    block_results = run_serial_or_forked(workers, jobs)
    row_parts = [part for part, _, _ in block_results]
    col_parts = [part for _, part, _ in block_results]
    value_parts = [part for _, _, part in block_results]

    if all(len(part) == 0 for part in row_parts):
        return sparse.csr_matrix((n_movies, n_movies), dtype=np.float32)

    rows = np.concatenate(row_parts)
    columns = np.concatenate(col_parts)
    values = np.concatenate(value_parts)

    result = sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(n_movies, n_movies),
        dtype=np.float32,
    )
    result.sort_indices()
    return result


def sample_unobserved(
    rng: np.random.Generator,
    catalog: np.ndarray,
    excluded: set[int],
    count: int,
) -> np.ndarray:
    """Uniformly sample distinct catalog items outside an exclusion set."""
    if catalog.ndim != 1:
        raise ValueError("The candidate catalog must be one-dimensional.")
    if count < 0:
        raise ValueError("The requested sample count cannot be negative.")
    if len(np.unique(catalog)) != len(catalog):
        raise ValueError("The candidate catalog contains duplicates.")

    excluded_array = np.fromiter(excluded, dtype=np.int64)
    eligible = catalog[
        ~np.isin(catalog, excluded_array, assume_unique=False)
    ]
    if len(eligible) < count:
        raise ValueError(
            f"Requested {count} negatives but only {len(eligible)} "
            "eligible catalog items exist."
        )

    return rng.choice(eligible, size=count, replace=False).astype(
        np.int32,
        copy=False,
    )


def build_ranking_candidates(
    target: np.ndarray,
    all_known: np.ndarray,
    train_movie_seen: np.ndarray,
    n_users: int,
    n_eval_users: int,
    n_candidates: int,
    relevance_threshold: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build fixed warm-start one-positive candidate sets.

    Negatives are absent from every known interaction, including the other
    held-out split. This all-known-item exclusion uses future information
    for candidate construction and is an explicit evaluation design choice.

    Eligible rows are randomly ordered before viability checks. The function
    continues until n_eval_users viable rows have been collected or all
    eligible rows have been considered.
    """
    if n_eval_users <= 0:
        raise ValueError("n_eval_users must be positive.")
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least two.")
    if relevance_threshold < 0.5 or relevance_threshold > 5.0:
        raise ValueError("The relevance threshold must lie in [0.5, 5.0].")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("Target interactions must have shape (n, 3).")
    if all_known.ndim != 2 or all_known.shape[1] != 3:
        raise ValueError("all_known must have shape (n, 3).")
    if len(train_movie_seen) == 0:
        raise ValueError("The training-presence mask is empty.")

    rng = np.random.default_rng(seed)
    catalog = np.flatnonzero(train_movie_seen).astype(np.int32)
    if len(catalog) < n_candidates:
        raise ValueError(
            f"The training catalog has {len(catalog)} items, fewer than "
            f"the requested {n_candidates} candidates."
        )

    known: list[set[int]] = [set() for _ in range(n_users)]
    for user, movie in all_known[:, :2].astype(np.int64):
        if user < 0 or user >= n_users:
            raise ValueError("all_known contains an invalid user index.")
        known[int(user)].add(int(movie))

    target_movies = target[:, 1].astype(np.int64)
    eligible = target[
        (target[:, 2] >= relevance_threshold)
        & train_movie_seen[target_movies]
    ]
    order = rng.permutation(len(eligible))

    users: list[int] = []
    candidate_rows: list[np.ndarray] = []
    positive_indices: list[int] = []

    for eligible_index in order:
        if len(users) >= n_eval_users:
            break

        row = eligible[eligible_index]
        user = int(row[0])
        positive = int(row[1])

        try:
            negatives = sample_unobserved(
                rng,
                catalog,
                known[user],
                n_candidates - 1,
            )
        except ValueError:
            continue

        candidates = np.concatenate(
            (
                np.asarray([positive], dtype=np.int32),
                negatives,
            )
        )
        rng.shuffle(candidates)

        matches = np.flatnonzero(candidates == positive)
        if len(matches) != 1:
            raise RuntimeError(
                "Candidate construction did not preserve one positive."
            )

        users.append(user)
        candidate_rows.append(candidates)
        positive_indices.append(int(matches[0]))

    if not candidate_rows:
        raise ValueError(
            "No viable ranking rows could be constructed. Reduce the "
            "candidate count or inspect catalog eligibility."
        )

    return {
        "users": np.asarray(users, dtype=np.int32),
        "candidates": np.stack(candidate_rows),
        "positive_indices": np.asarray(
            positive_indices,
            dtype=np.int32,
        ),
        "n_candidates": np.asarray(n_candidates, dtype=np.int32),
        "seed": np.asarray(seed, dtype=np.int64),
        "relevance_threshold": np.asarray(
            relevance_threshold,
            dtype=np.float64,
        ),
    }


def save_ranking_candidates(filename: str, values: dict[str, np.ndarray]) -> None:
    np.savez(output_path(filename), **values)


def resolve_worker_count() -> int:
    """Return the number of process workers for parallel stages.

    Process-level work uses forked workers that inherit input arrays
    through copy-on-write memory. Set PARALLEL_WORKERS=1 to force the
    serial code path.
    """
    if PARALLEL_WORKERS is not None:
        if PARALLEL_WORKERS < 1:
            raise ValueError("PARALLEL_WORKERS must be positive.")
        return PARALLEL_WORKERS
    return max(1, (multiprocessing.cpu_count() or 1) - 2)


def run_serial_or_forked(
    workers: int,
    jobs: list[tuple],
    stage_name: str | None = None,
) -> list:
    """Run (function, args) jobs serially or in forked workers.

    Forked workers inherit large input arrays through copy-on-write
    memory, so submitted jobs do not need to pickle those inputs.
    Results are returned in submission order, so serial and parallel
    execution produce identical results.

    When stage_name is provided, prints a start/finish banner and
    per-chunk progress from workers so the user can see which stage
    is running.
    """
    if stage_name is not None:
        print(f"[parallel] {stage_name}: {len(jobs)} chunks, {workers} workers")

    if workers <= 1 or len(jobs) <= 1:
        return [function(*arguments) for function, arguments in jobs]

    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=multiprocessing.get_context("fork"),
    ) as pool:
        futures = [
            pool.submit(function, *arguments)
            for function, arguments in jobs
        ]
        results = []
        for i, future in enumerate(futures):
            results.append(future.result())
            if stage_name is not None:
                pct = int((i + 1) / len(futures) * 100)
                print(
                    f"\r[parallel] {stage_name}: "
                    f"{i + 1}/{len(futures)} chunks ({pct}%)",
                    end="",
                    flush=True,
                )
        if stage_name is not None:
            print()  # newline after the in-place progress line
            print(f"[parallel] {stage_name}: done")
        return results


def run_overlapped(threads: int, jobs: list[tuple]) -> list:
    """Run (function, args) jobs concurrently in threads.

    Intended for I/O-bound or GIL-releasing work such as the raw-file
    loading phase.
    """
    if threads <= 1 or len(jobs) <= 1:
        return [function(*arguments) for function, arguments in jobs]

    with ThreadPoolExecutor(max_workers=min(threads, len(jobs))) as pool:
        futures = [
            pool.submit(function, *arguments)
            for function, arguments in jobs
        ]
        return [future.result() for future in futures]


_PARALLEL_STATE: dict[str, Any] = {}


def _movie_similarity_worker(
    start: int,
    end: int,
    top_k: int,
    min_common: int,
    shrinkage: float,
) -> tuple:
    return _movie_similarity_block(
        _PARALLEL_STATE["centered_matrix"],
        _PARALLEL_STATE["squared_matrix"],
        _PARALLEL_STATE["observed"],
        start,
        end,
        top_k,
        min_common,
        shrinkage,
    )


def _accumulate_profile_heap(
    rows: np.ndarray,
    normalized_features: np.ndarray,
    user_means: np.ndarray,
    n_local_users: int,
    user_offset: int,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """Accumulate residual-weighted feature sums for one contiguous block.

    Rows hold dense global user IDs and must already be ordered by user.
    The heap is indexed by user ID minus user_offset, so each user's
    contributions are added in exactly the row order the serial pipeline
    uses, keeping the float32 accumulation bit-identical.
    """
    heap = np.zeros(
        (n_local_users, normalized_features.shape[1]),
        dtype=np.float32,
    )

    for start in range(0, len(rows), chunk_size):
        chunk = rows[start:start + chunk_size]
        users = chunk[:, 0].astype(np.int64) - user_offset
        movies = chunk[:, 1].astype(np.int64)
        residuals = chunk[:, 2] - user_means[chunk[:, 0].astype(np.int64)]
        contributions = (
            normalized_features[movies]
            * residuals[:, np.newaxis]
        )
        np.add.at(heap, users, contributions)

    return heap


def _profile_block_worker(
    row_start: int,
    row_end: int,
    user_offset: int,
    n_local_users: int,
    features_key: str,
) -> np.ndarray:
    state = _PARALLEL_STATE
    return _accumulate_profile_heap(
        state["train_by_user"][row_start:row_end],
        state[features_key],
        state["user_means"],
        n_local_users,
        user_offset,
    )


def profile_block_bounds(
    train: np.ndarray,
    workers: int,
) -> list[tuple[int, int, int, int]]:
    """Split user-grouped training rows into balanced contiguous blocks.

    Returns (row_start, row_end, user_offset, n_local_users) tuples whose
    user ranges partition all users in order. The chronological split
    emits rows grouped by user; if that invariant ever breaks, the rows
    are stably re-sorted first so per-user row order is preserved.
    """
    users = train[:, 0].astype(np.int64)
    if np.any(np.diff(users) < 0):
        order = np.argsort(users, kind="stable")
        train = train[order]
        users = users[order]
    _PARALLEL_STATE["train_by_user"] = train

    _, group_starts, group_counts = np.unique(
        users, return_index=True, return_counts=True
    )
    n_groups = len(group_starts)
    if n_groups == 0:
        raise ValueError("The training split is empty.")

    blocks = max(1, min(workers, n_groups))
    if blocks == 1:
        return [(0, len(train), int(users[0]), int(n_groups))]

    cumulative = np.cumsum(group_counts)
    total = int(cumulative[-1])
    targets = np.arange(1, blocks) * (total / blocks)
    split_groups = np.searchsorted(cumulative, targets, side="left") + 1
    split_groups = np.unique(np.clip(split_groups, 1, n_groups - 1))

    bounds: list[tuple[int, int, int, int]] = []
    previous_group = 0
    for boundary in list(split_groups) + [n_groups]:
        row_start = int(group_starts[previous_group])
        row_end = (
            int(group_starts[boundary])
            if boundary < n_groups
            else len(train)
        )
        user_offset = int(users[row_start])
        bounds.append(
            (row_start, row_end, user_offset, boundary - previous_group)
        )
        previous_group = boundary
    return bounds


def compute_content_profiles_parallel(
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate both content-profile families across forked workers.

    Training rows are partitioned into contiguous user blocks and each
    block accumulates a private float32 heap per feature family. Every
    user belongs to exactly one block, so the returned heaps are placed
    into the full matrices rather than summed, keeping the results
    bit-identical to the serial pipeline. Genre and genome block jobs are
    interleaved in submission order so each worker receives one cheap and
    one expensive job.
    """
    n_users = _PARALLEL_STATE["n_users"]
    genre_features = _PARALLEL_STATE["genre_normalized"]
    genome_features = _PARALLEL_STATE["genome_normalized"]
    bounds = profile_block_bounds(_PARALLEL_STATE["train"], workers)

    jobs = []
    for row_start, row_end, user_offset, n_local_users in bounds:
        jobs.append(
            (
                _profile_block_worker,
                (
                    row_start,
                    row_end,
                    user_offset,
                    n_local_users,
                    "genre_normalized",
                ),
            )
        )
        jobs.append(
            (
                _profile_block_worker,
                (
                    row_start,
                    row_end,
                    user_offset,
                    n_local_users,
                    "genome_normalized",
                ),
            )
        )

    heaps = run_serial_or_forked(workers, jobs, stage_name="content profiles")

    genre_profiles = np.zeros(
        (n_users, genre_features.shape[1]), dtype=np.float32
    )
    genome_profiles = np.zeros(
        (n_users, genome_features.shape[1]), dtype=np.float32
    )
    for index, (_, _, user_offset, n_local_users) in enumerate(bounds):
        genre_profiles[user_offset:user_offset + n_local_users] = (
            heaps[2 * index]
        )
        genome_profiles[user_offset:user_offset + n_local_users] = (
            heaps[2 * index + 1]
        )

    return l2_normalize_rows(genre_profiles), l2_normalize_rows(genome_profiles)


def _validation_ranking_worker() -> dict[str, np.ndarray]:
    return build_ranking_candidates(
        _PARALLEL_STATE["validation_target"],
        _PARALLEL_STATE["all_known"],
        _PARALLEL_STATE["movie_seen"],
        _PARALLEL_STATE["n_users"],
        RANKING_USERS,
        RANKING_CANDIDATES,
        RELEVANCE_THRESHOLD,
        seed=2025,
    )


def _test_ranking_worker() -> dict[str, np.ndarray]:
    return build_ranking_candidates(
        _PARALLEL_STATE["test_target"],
        _PARALLEL_STATE["all_known"],
        _PARALLEL_STATE["movie_seen"],
        _PARALLEL_STATE["n_users"],
        RANKING_USERS,
        RANKING_CANDIDATES,
        RELEVANCE_THRESHOLD,
        seed=2026,
    )


def build_dataset(
    sample_size: int | None = None,
    offset: int = 0,
) -> None:
    """Build all processed artifacts.

    Leave sample_size=None and offset=0 for the stated full-data study.
    """
    workers = resolve_worker_count()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading ratings, genres, and Tag Genome ({workers} workers)...")
    loaded_ratings, loaded_genre, loaded_genome = run_overlapped(
        min(3, workers),
        [
            (load_raw_ratings, (sample_size, offset)),
            (build_genre_vectors, ()),
            (build_genome_scores, ()),
        ],
    )
    raw_ratings = loaded_ratings
    genre_source, genre_movie_ids = loaded_genre
    genome_source, genome_movie_ids = loaded_genome

    print("Filtering and remapping IDs...")
    interactions, external_users, external_movies = filter_and_remap(
        raw_ratings,
        genome_movie_ids,
    )
    del raw_ratings

    n_users = len(external_users)
    n_movies = len(external_movies)

    print("Chronological train/validation/test split...")
    train, validation, test = chronological_split(interactions)

    genre_vectors = align_features(
        external_movies,
        genre_movie_ids,
        genre_source,
    )
    genome_vectors = align_features(
        external_movies,
        genome_movie_ids,
        genome_source,
    )
    genre_normalized = l2_normalize_rows(genre_vectors)
    genome_normalized = l2_normalize_rows(genome_vectors)

    print("Computing means and train-presence masks...")
    (
        global_mean,
        user_means,
        movie_means,
        user_seen,
        movie_seen,
    ) = compute_means(train, n_users, n_movies)

    print(f"Computing residual-weighted content profiles "
          f"({workers} workers)...")
    _PARALLEL_STATE.update(
        train=train,
        genre_normalized=genre_normalized,
        genome_normalized=genome_normalized,
        user_means=user_means,
        n_users=n_users,
    )
    genre_profiles, genome_profiles = compute_content_profiles_parallel(
        workers
    )

    print("Building sparse interaction matrices...")
    ratings_csr, ratings_csc = build_interaction_indices(
        train,
        n_users,
        n_movies,
    )

    print(f"Computing sparse top-k adjusted-cosine movie similarities "
          f"({workers} workers)...")
    item_similarity = compute_topk_movie_similarity(
        train,
        n_users,
        n_movies,
        workers=workers,
    )

    all_known = np.vstack((train, validation, test))

    print(f"Building validation and test ranking candidates "
          f"({workers} workers)...")
    _PARALLEL_STATE.update(
        validation_target=validation,
        test_target=test,
        all_known=all_known,
        movie_seen=movie_seen,
    )
    validation_ranking, test_ranking = run_serial_or_forked(
        workers,
        [
            (_validation_ranking_worker, ()),
            (_test_ranking_worker, ()),
        ],
    )

    np.save(output_path("interactions_train.npy"), train)
    np.save(output_path("interactions_validation.npy"), validation)
    np.save(output_path("interactions_test.npy"), test)

    np.save(output_path("external_user_ids.npy"), external_users)
    np.save(output_path("external_movie_ids.npy"), external_movies)

    np.save(output_path("genre_vectors_norm.npy"), genre_normalized)
    np.save(output_path("genome_scores_norm.npy"), genome_normalized)
    np.save(output_path("user_genre_profiles.npy"), genre_profiles)
    np.save(output_path("user_genome_profiles.npy"), genome_profiles)

    np.save(output_path("user_mean_ratings.npy"), user_means)
    np.save(output_path("movie_mean_ratings.npy"), movie_means)
    np.save(output_path("user_seen_train.npy"), user_seen)
    np.save(output_path("movie_seen_train.npy"), movie_seen)

    sparse.save_npz(output_path("ratings_train_csr.npz"), ratings_csr)
    sparse.save_npz(output_path("ratings_train_csc.npz"), ratings_csc)
    sparse.save_npz(
        output_path("item_similarity_top40.npz"),
        item_similarity,
    )

    save_ranking_candidates(
        "ranking_validation_1000.npz",
        validation_ranking,
    )
    save_ranking_candidates(
        "ranking_test_1000.npz",
        test_ranking,
    )

    np.savez(
        output_path("metadata.npz"),
        n_users=n_users,
        n_movies=n_movies,
        global_mean_rating=global_mean,
        train_size=len(train),
        validation_size=len(validation),
        test_size=len(test),
        minimum_user_interactions=MIN_USER_INTERACTIONS,
        minimum_training_interactions=MIN_USER_INTERACTIONS - 2,
        chronological_tie_break="source row order",
        full_dataset=np.asarray(
            sample_size is None and offset == 0,
            dtype=bool,
        ),
        sample_size=-1 if sample_size is None else sample_size,
        offset=offset,
        relevance_threshold=RELEVANCE_THRESHOLD,
        item_neighbors=MOVIE_NEIGHBORS,
        minimum_common_raters=MIN_COMMON_RATERS,
        similarity_shrinkage=SIMILARITY_SHRINKAGE,
    )

    print(
        f"Saved {n_users:,} users, {n_movies:,} movies, "
        f"{len(train):,} training interactions."
    )


def load_arrays(path: str | Path = OUTPUT_DIR) -> dict[str, object]:
    base = Path(path)
    with np.load(base / "metadata.npz", allow_pickle=False) as source:
        metadata = {key: source[key].item() for key in source.files}

    return {
        "train": np.load(
            base / "interactions_train.npy",
            allow_pickle=False,
        ),
        "validation": np.load(
            base / "interactions_validation.npy",
            allow_pickle=False,
        ),
        "test": np.load(
            base / "interactions_test.npy",
            allow_pickle=False,
        ),
        "genre_vectors": np.load(
            base / "genre_vectors_norm.npy",
            allow_pickle=False,
        ),
        "genome_vectors": np.load(
            base / "genome_scores_norm.npy",
            allow_pickle=False,
        ),
        "genre_profiles": np.load(
            base / "user_genre_profiles.npy",
            allow_pickle=False,
        ),
        "genome_profiles": np.load(
            base / "user_genome_profiles.npy",
            allow_pickle=False,
        ),
        "user_means": np.load(
            base / "user_mean_ratings.npy",
            allow_pickle=False,
        ),
        "movie_means": np.load(
            base / "movie_mean_ratings.npy",
            allow_pickle=False,
        ),
        "user_seen": np.load(
            base / "user_seen_train.npy",
            allow_pickle=False,
        ),
        "movie_seen": np.load(
            base / "movie_seen_train.npy",
            allow_pickle=False,
        ),
        "ratings_csr": sparse.load_npz(
            base / "ratings_train_csr.npz"
        ).tocsr(),
        "ratings_csc": sparse.load_npz(
            base / "ratings_train_csc.npz"
        ).tocsc(),
        "movie_similarity": sparse.load_npz(
            base / "item_similarity_top40.npz"
        ).tocsr(),
        "metadata": metadata,
    }


def load_ranking(
    split: str,
    path: str | Path = OUTPUT_DIR,
) -> dict[str, np.ndarray]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")

    filename = Path(path) / f"ranking_{split}_1000.npz"
    with np.load(filename, allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def main() -> None:
    build_dataset()


if __name__ == "__main__":
    main()
