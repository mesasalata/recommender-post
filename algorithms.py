"""
Corrected recommender algorithms for the MovieLens 33M experiment.

Algorithms
----------
1. Global-mean rating baseline.
2. Genre residual-weighted content profile.
3. Tag Genome residual-weighted content profile.
4. Tag-genome weighted hybrid: a confidence-weighted ensemble of the
   Tag Genome content profile and the User-based k-NN (algorithm 5).
5. User-based k-nearest-neighbor collaborative filtering.
6. Movie-based k-nearest-neighbor collaborative filtering.
7. Biased matrix factorization trained with SGD.
8. BPR-MF with thresholded explicit-feedback positives.
9. MF+BPR hybrid: a dual-headed blend of Biased MF (7) and BPR-MF (8).
   The rating head blends the two algorithms' rating predictions
   (lambda * MF + (1 - lambda) * BPR-mapped) with lambda chosen on
   validation RMSE. The ranking head blends the two algorithms' raw latent
   scores after per-row z-score standardisation (lambda * z_MF +
   (1 - lambda) * z_BPR) with a second lambda chosen on validation
   NDCG@10. Two independent lambdas = "dual-headed".

Important specifications
------------------------
- Content rating predictions use:

      prediction = user_mean + beta * cosine(profile, movie)

  beta is fitted on the shared validation set with beta >= 0.

- The hybrid blends the Tag Genome content prediction and the User k-NN
  prediction. Where k-NN had usable neighbors the row is
  alpha * content + (1 - alpha) * user_knn; where it had none the row
  uses content alone. The blend weight alpha is selected on the shared
  validation set.

- User k-NN uses:
    top-k = 40,
    minimum common movies = 2,
    similarity shrinkage = 10,
    signed similarities,
    absolute-similarity normalization.

- Movie k-NN uses sparse top-40 adjusted-cosine similarities prepared by
  prepare_data.py with the same overlap and shrinkage settings.

- BPR positives are training ratings >= 3.5. Unobserved movies are sampled
  dynamically as negatives. This is a documented explicit-rating
  adaptation of implicit-feedback BPR.

- BPR raw scores are primary for ranking. Its rating metrics use a
  validation-fitted nonnegative affine mapping, which is not part of the
  original BPR algorithm. The affine map (slope, intercept) is persisted
  inside bpr_model.npz so post-hoc combinations (the MF+BPR hybrid's rating
  head) reuse the exact mapping the BPR stage fitted, without re-deriving it.

Parallelization
---------------
The per-row prediction loops (k-NN, hybrid, ranking evaluation) and the two
regularization grids run in forked worker processes via
prepare_data.run_serial_or_forked. Forking shares the loaded arrays with the
workers through copy-on-write memory, so nothing large is copied or pickled,
and results return in submission order, keeping the serial and parallel code
paths bit-identical. Set PARALLEL_WORKERS=1 to force the serial path. The
SGD/BPR inner loops themselves are not parallelized: each training run has
shared mutable state, while the independent grid settings parallelize
without changing the algorithms.

References
----------
Salton, G., Wong, A., and Yang, C. S. (1975).
"A Vector Space Model for Automatic Indexing." CACM, 18(11), 613-620.

Rocchio, J. J. (1971). "Relevance Feedback in Information Retrieval."
In The SMART Retrieval System, 313-323.

Pazzani, M. J., and Billsus, D. (2007).
"Content-Based Recommendation Systems." In The Adaptive Web, 325-341.

Burke, R. (2002). "Hybrid Recommender Systems: Survey and Experiments."
UMUAI, 12(4), 331-370.

Resnick, P. et al. (1994). "GroupLens: An Open Architecture for
Collaborative Filtering of Netnews." CSCW 1994, 175-186.

Herlocker, J. L. et al. (1999). "An Algorithmic Framework for Performing
Collaborative Filtering." SIGIR 1999, 230-237.

Sarwar, B. et al. (2001). "Item-Based Collaborative Filtering
Recommendation Algorithms." WWW 2001, 285-295.

Koren, Y., Bell, R., and Volinsky, C. (2009).
"Matrix Factorization Techniques for Recommender Systems."
Computer, 42(8), 30-37.

Rendle, S. et al. (2009). "BPR: Bayesian Personalized Ranking from
Implicit Feedback." UAI 2009, 452-461.

Järvelin, K., and Kekäläinen, J. (2002).
"Cumulated Gain-Based Evaluation of IR Techniques."
ACM TOIS, 20(4), 422-446.

Krichene, W., and Rendle, S. (2020).
"On Sampled Metrics for Item Recommendation." KDD 2020, 1748-1757.
"""

from __future__ import annotations

import os

# The parallel stages below fork worker processes. BLAS thread pools do not
# fork cleanly, and N workers each running a full BLAS thread pool would
# oversubscribe the cores: the parallelism comes from the processes, not
# from threads. Restrict BLAS to one thread per process before any array
# library is imported. Launch with ALGORITHMS_LIMIT_BLAS_THREADS=0 to keep
# the library defaults.
if os.environ.get("ALGORITHMS_LIMIT_BLAS_THREADS", "1") != "0":
    for _variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(_variable, "1")

import json
import time
import argparse
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import sparse

import prepare_data as data


K = 10
RATING_BOUNDS = (0.5, 5.0)
USER_NEIGHBORS = 40
MIN_COMMON_ITEMS = 2
USER_SIMILARITY_SHRINKAGE = 10.0
BPR_RELEVANCE_THRESHOLD = 3.5
BPR_NEGATIVES_PER_POSITIVE = 4
BPR_SELECTION_TOLERANCE = 1e-12

MF_REGULARIZATION_GRID = (0.02, 0.05)
BPR_REGULARIZATION_GRID = (0.0025, 0.01)
HYBRID_ALPHA_GRID = np.linspace(0.0, 1.0, 11)
# The MF+BPR hybrid uses a dual-headed design: two independent blend weights,
# one per head, each selected from this grid. A higher value of lambda
# weights Biased MF more heavily; (1 - lambda) weights BPR-MF.
LAMBDA_GRID = np.linspace(0.0, 1.0, 11)
# NDCG/RMSE tolerance for MF+BPR lambda tie-breaking. On ties within this
# value the larger lambda (the MF-leaning choice) is kept, mirroring the BPR
# regularisation tie policy.
MF_BPR_SELECTION_TOLERANCE = 1e-12

# Number of forked worker processes for prediction stages. The default
# follows prepare_data.py's rule. Set PARALLEL_WORKERS=1 to force the
# serial code path, which reproduces the original results bit-for-bit.
PARALLEL_WORKERS: int | None = None

# Module state inherited by forked workers through copy-on-write memory.
# The fork copies the parent's address space, so workers read these objects
# without copying or pickling them. Workers never replace the dicts, and the
# algorithm objects they read are not mutated during a stage, so forked
# copies stay invisible to the parent.
_STAGE_STATE: dict[str, Any] = {}
_STAGE_MODEL: Any = None
_GRID_STATE: dict[str, Any] = {}


def resolve_prediction_workers() -> int:
    """Return the number of worker processes for prediction stages."""
    if PARALLEL_WORKERS is not None:
        if PARALLEL_WORKERS < 1:
            raise ValueError("PARALLEL_WORKERS must be positive.")
        return PARALLEL_WORKERS
    return max(1, (os.cpu_count() or 1) - 2)


def chunk_bounds(count: int, workers: int) -> list[tuple[int, int]]:
    """Split range(count) into contiguous (start, end) row ranges.

    Chunks are several times more numerous than workers so the pool's task
    queue balances uneven per-row costs (rows for popular movies are far
    more expensive). Contiguous chunks also keep one user's rows together
    inside a chunk, so per-process memoized similarity rows stay warm
    while that user's candidate movies are scored.
    """
    if count <= 0 or workers <= 0:
        raise ValueError("count and workers must be positive.")
    segments = min(4 * workers, count)
    base, remainder = divmod(count, segments)

    bounds: list[tuple[int, int]] = []
    start = 0
    for segment in range(segments):
        size = base + (1 if segment < remainder else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def _stage_chunk_worker(start: int, end: int) -> np.ndarray:
    """Compute one [start:end) chunk of the current prediction stage.

    Runs in a forked worker. The stage model and its input arrays are
    inherited through copy-on-write memory; only this chunk's numbers are
    pickled back to the parent.
    """
    return _STAGE_MODEL.chunk(_STAGE_STATE, start, end)


def run_stage(model: Any, arrays: dict[str, Any], stage_name: str | None = None) -> np.ndarray:
    """Evaluate one prediction stage over forked workers, in row order.

    The stage model implements chunk(state, start, end) returning the rows
    for [start:end). Results are returned in submission order and
    concatenated, so PARALLEL_WORKERS=1 and parallel runs produce identical
    arrays. A fresh worker pool is used per stage: the memoized state
    workers build (for example UserKNN similarity rows) is only reusable
    within the stage, and discarding the workers bounds memory.
    """
    global _STAGE_MODEL
    workers = resolve_prediction_workers()
    count = len(next(iter(arrays.values())))
    if count == 0:
        raise ValueError("Cannot evaluate an empty stage.")
    jobs = [
        (_stage_chunk_worker, (start, end))
        for start, end in chunk_bounds(count, workers)
    ]

    _STAGE_MODEL = model
    _STAGE_STATE.clear()
    _STAGE_STATE.update(arrays)

    parts = data.run_serial_or_forked(workers, jobs, stage_name)
    return np.concatenate(parts)


class RowMapStage:
    """Stage applying a per-row fn(user, movie) to interactions in order."""

    def __init__(self, fn: Callable[[int, int], float]) -> None:
        self.fn = fn

    def chunk(self, state: dict[str, Any], start: int, end: int) -> np.ndarray:
        interactions = state["interactions"]
        return np.asarray(
            [
                self.fn(int(interactions[row][0]), int(interactions[row][1]))
                for row in range(start, end)
            ],
            dtype=np.float64,
        )


class RowCountStage:
    """Stage applying a per-row fn(user, movie) -> int to interactions in order.

    Used by the user-kNN stage to emit the number of usable neighbors each
    prediction actually used, which the genome hybrid consumes as a confidence
    signal. The per-row function is the same memoizing similarity-row accessor
    as the float prediction, so the forked workers parallelize it identically
    to the prediction stage.
    """

    def __init__(self, fn: Callable[[int, int], int]) -> None:
        self.fn = fn

    def chunk(self, state: dict[str, Any], start: int, end: int) -> np.ndarray:
        interactions = state["interactions"]
        return np.asarray(
            [
                int(self.fn(int(interactions[row][0]), int(interactions[row][1])))
                for row in range(start, end)
            ],
            dtype=np.int32,
        )


class RankingStage:
    """Stage scoring fixed candidate rows and deriving rank metrics.

    The row-loop body of ranking_metrics runs unchanged, but over one
    [start:end) chunk of rows per worker. Concatenation restores the
    original per-user metric order, so means are identical.
    """

    def __init__(self, name: str, score: Callable[[int, np.ndarray], np.ndarray], cutoff: int) -> None:
        self.name = name
        self.score = score
        self.cutoff = cutoff

    def chunk(self, state: dict[str, Any], start: int, end: int) -> np.ndarray:
        users = state["users"]
        candidates = state["candidates"]
        positive_indices = state["positive_indices"]

        hits = np.zeros(end - start, dtype=np.float64)
        ndcgs = np.zeros(end - start, dtype=np.float64)
        reciprocal_ranks = np.zeros(end - start, dtype=np.float64)

        for offset, row in enumerate(range(start, end)):
            user = int(users[row])
            movie_ids = candidates[row]
            positive = int(positive_indices[row])

            scores = require_finite(
                self.score(user, movie_ids),
                f"{self.name} ranking scores for row {row}",
            )
            if scores.shape != (len(movie_ids),):
                raise ValueError(
                    f"{self.name} returned {scores.shape}; expected "
                    f"{(len(movie_ids),)}."
                )

            order = np.argsort(-scores, kind="stable")
            positions = np.flatnonzero(order == positive)
            if len(positions) != 1:
                raise RuntimeError(
                    "The positive candidate did not have one rank."
                )

            rank = int(positions[0]) + 1
            if rank <= self.cutoff:
                hits[offset] = 1.0
                ndcgs[offset] = 1.0 / np.log2(rank + 1.0)
                reciprocal_ranks[offset] = 1.0 / rank

        return np.column_stack((hits, ndcgs, reciprocal_ranks))


class PrecomputedRankingStage:
    """Ranking-stage variant that consumes precomputed candidate scores.

    Identical metric computation to RankingStage, but instead of calling a
    scorer per row it reads the row's scores from a 2-D array. This is how
    a stage evaluates ranking without redoing the (expensive) per-candidate
    scoring pass. The per-row rank/hit/ndcg/mrr logic is unchanged.
    """

    def __init__(
        self,
        name: str,
        scores: np.ndarray,
        cutoff: int,
    ) -> None:
        self.name = name
        self.scores = np.asarray(scores, dtype=np.float64)
        self.cutoff = cutoff

    def chunk(self, state: dict[str, Any], start: int, end: int) -> np.ndarray:
        users = state["users"]
        positive_indices = state["positive_indices"]

        hits = np.zeros(end - start, dtype=np.float64)
        ndcgs = np.zeros(end - start, dtype=np.float64)
        reciprocal_ranks = np.zeros(end - start, dtype=np.float64)

        for offset, row in enumerate(range(start, end)):
            scores = require_finite(
                self.scores[row],
                f"{self.name} precomputed ranking scores for row {row}",
            )
            positive = int(positive_indices[row])

            order = np.argsort(-scores, kind="stable")
            positions = np.flatnonzero(order == positive)
            if len(positions) != 1:
                raise RuntimeError(
                    "The positive candidate did not have one rank."
                )

            rank = int(positions[0]) + 1
            if rank <= self.cutoff:
                hits[offset] = 1.0
                ndcgs[offset] = 1.0 / np.log2(rank + 1.0)
                reciprocal_ranks[offset] = 1.0 / rank

        return np.column_stack((hits, ndcgs, reciprocal_ranks))


def require_finite(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        bad = int(np.sum(~np.isfinite(values)))
        raise ValueError(f"{name} contains {bad} non-finite values.")
    return values


def rating_metrics(
    name: str,
    predictions: np.ndarray,
    actual: np.ndarray,
    bounds: tuple[float, float] = RATING_BOUNDS,
) -> dict[str, float | int | str]:
    """Report raw metrics and separately named bounded metrics.

    Raw metrics are primary. Bounded metrics clip finite predictions to the
    valid rating interval. Non-finite predictions are rejected rather than
    silently omitted or propagated.
    """
    predictions = require_finite(predictions, f"{name} predictions")
    actual = require_finite(actual, "actual ratings")

    if len(bounds) != 2 or not np.all(np.isfinite(bounds)):
        raise ValueError("Rating bounds must contain two finite values.")
    if bounds[0] >= bounds[1]:
        raise ValueError("The lower rating bound must be below the upper.")
    if predictions.shape != actual.shape:
        raise ValueError("Predictions and actual ratings must have one shape.")
    if len(predictions) == 0:
        raise ValueError("Cannot evaluate an empty rating set.")

    errors = predictions - actual
    bounded = np.clip(predictions, bounds[0], bounds[1])
    bounded_errors = bounded - actual
    out_of_range = (
        (predictions < bounds[0]) | (predictions > bounds[1])
    )

    return {
        "name": name,
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "bounded_rmse": float(
            np.sqrt(np.mean(bounded_errors ** 2))
        ),
        "bounded_mae": float(np.mean(np.abs(bounded_errors))),
        "bounded_bias": float(np.mean(bounded_errors)),
        "n_out_of_range": int(np.sum(out_of_range)),
        "out_of_range_pct": float(100 * np.mean(out_of_range)),
        "n_non_finite": 0,
        "n_total": len(predictions),
    }


def validate_ranking_inputs(
    users: np.ndarray,
    candidates: np.ndarray,
    positive_indices: np.ndarray,
    k: int,
) -> None:
    """Validate the fixed one-positive candidate representation."""
    users = np.asarray(users)
    candidates = np.asarray(candidates)
    positive_indices = np.asarray(positive_indices)

    if candidates.ndim != 2:
        raise ValueError("Candidates must be a two-dimensional array.")
    if users.ndim != 1 or positive_indices.ndim != 1:
        raise ValueError(
            "Ranking users and positive indices must be one-dimensional."
        )
    if len(users) != len(candidates):
        raise ValueError("Ranking users and candidate rows do not align.")
    if len(positive_indices) != len(candidates):
        raise ValueError(
            "Positive indices and candidate rows do not align."
        )
    if len(candidates) == 0 or candidates.shape[1] == 0:
        raise ValueError("Ranking evaluation cannot use an empty set.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if not np.issubdtype(candidates.dtype, np.integer):
        raise ValueError("Candidate movie indices must be integers.")
    if not np.issubdtype(users.dtype, np.integer):
        raise ValueError("Ranking user indices must be integers.")
    if not np.issubdtype(positive_indices.dtype, np.integer):
        raise ValueError("Positive positions must be integers.")

    width = candidates.shape[1]
    if np.any((positive_indices < 0) | (positive_indices >= width)):
        raise ValueError("A positive index lies outside its candidate row.")

    for row, values in enumerate(candidates):
        if len(np.unique(values)) != width:
            raise ValueError(
                f"Ranking candidate row {row} contains duplicates."
            )


def ranking_metrics(
    name: str,
    users: np.ndarray,
    candidates: np.ndarray,
    positive_indices: np.ndarray,
    scorer: Callable[[int, np.ndarray], np.ndarray],
    k: int = K,
    scores: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    """Evaluate fixed sampled sets containing exactly one relevant movie.

    These are protocol-specific sampled metrics, not estimates that can be
    assumed to preserve full-catalog model comparisons. See Krichene and
    Rendle (2020).

    If scores (2-D, shape (len(users), candidates.shape[1])) is provided it
    is used directly and the scorer is skipped entirely; this is how a stage
    reuses its own saved ranking scores or the hybrid reuses its parents'.
    When scores is None the scorer runs exactly as before. The metric
    computation (per-row hit/ndcg/mrr, stable descending sort) is unchanged
    in both cases.
    """
    validate_ranking_inputs(users, candidates, positive_indices, k)

    cutoff = min(k, candidates.shape[1])
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(users), candidates.shape[1]):
            raise ValueError(
                f"Precomputed ranking scores have shape {scores.shape}; "
                f"expected {(len(users), candidates.shape[1])}."
            )
        stage = PrecomputedRankingStage(name, scores, cutoff)
        metrics = run_stage(
            stage,
            {
                "users": users,
                "candidates": candidates,
                "positive_indices": positive_indices,
            },
            stage_name=f"{name} ranking (precomputed)",
        )
        hits, ndcgs, reciprocal_ranks = metrics.T
    else:
        stage = RankingStage(name, scorer, cutoff)
        metrics = run_stage(
            stage,
            {
                "users": users,
                "candidates": candidates,
                "positive_indices": positive_indices,
            },
            stage_name=f"{name} ranking",
        )
        hits, ndcgs, reciprocal_ranks = metrics.T

    return {
        "name": name,
        "hit_rate@k": float(np.mean(hits)),
        "ndcg@k": float(np.mean(ndcgs)),
        "mrr@k": float(np.mean(reciprocal_ranks)),
        "n_evaluated": len(users),
        "n_candidates": int(candidates.shape[1]),
        "k": k,
    }


def predict_global_mean(
    interactions: np.ndarray,
    global_mean: float,
) -> np.ndarray:
    """Predict the training-set global mean for every user-movie pair."""
    return np.full(len(interactions), global_mean, dtype=np.float64)


def fit_nonnegative_content_scale(
    interactions: np.ndarray,
    profiles: np.ndarray,
    movie_features: np.ndarray,
    user_means: np.ndarray,
) -> float:
    """Fit beta >= 0 for user_mean + beta * cosine similarity.

    Symbols: beta is the non-negative scale that maps content cosine
    similarity to a rating residual. The closed-form least-squares solution
    is beta = dot(sim, targets) / dot(sim, sim), clamped to >= 0 so the
    content component never pushes predictions in the wrong direction.
    """
    users = interactions[:, 0].astype(np.int64)
    movies = interactions[:, 1].astype(np.int64)
    similarities = np.sum(
        profiles[users] * movie_features[movies],
        axis=1,
    )
    targets = interactions[:, 2] - user_means[users]
    denominator = float(np.dot(similarities, similarities))

    if denominator <= np.finfo(float).eps:
        return 0.0

    return max(0.0, float(np.dot(similarities, targets) / denominator))


def predict_content(
    interactions: np.ndarray,
    profiles: np.ndarray,
    movie_features: np.ndarray,
    user_means: np.ndarray,
    beta: float,
) -> np.ndarray:
    """Predict user mean plus a validation-scaled content similarity.

    Symbols: beta is the non-negative scale fitted on validation data;
    the prediction is r_bar_u + beta * cosine(p_u, x_i), where p_u is the
    user's residual-weighted profile and x_i is the movie feature vector.
    """
    users = interactions[:, 0].astype(np.int64)
    movies = interactions[:, 1].astype(np.int64)
    similarities = np.sum(
        profiles[users] * movie_features[movies],
        axis=1,
    )
    return user_means[users] + beta * similarities


class UserKNN:
    """User-based k-NN with Pearson-like mean-centered cosine.

    Similarity uses common rated movies, requires a minimum overlap, and is
    shrunk by n_common / (n_common + shrinkage). For one target user,
    similarities to every other user that shares at least one stored movie
    are computed vectorized through sparse inner products and memoized:
    both test prediction and ranking score the same user against many
    movies, and computing one row of similarities replaces the original
    per-co-rater row intersections with one set of matrix products. A
    movie with tens of thousands of co-raters therefore no longer costs
    tens of thousands of Python-level intersection calls, and no growing
    per-pair cache is needed.

    Symbols: s(u,v) is mean-centered cosine similarity, s'(u,v) is the
    shrinkage-adjusted similarity, n_uv is the co-rated movie count,
    r_ui is the observed rating, r_bar_u is the user mean, N_k(u) is
    the set of k nearest neighbours, r_vi is the neighbour's rating.
    """

    def __init__(
        self,
        ratings_csr: sparse.csr_matrix,
        ratings_csc: sparse.csc_matrix,
        user_means: np.ndarray,
        movie_means: np.ndarray,
        top_k: int = USER_NEIGHBORS,
        min_common: int = MIN_COMMON_ITEMS,
        shrinkage: float = USER_SIMILARITY_SHRINKAGE,
    ) -> None:
        self.csr = ratings_csr
        self.csc = ratings_csc
        self.user_means = user_means
        self.movie_means = movie_means
        self.top_k = top_k
        self.min_common = min_common
        self.shrinkage = shrinkage

        # Precomputed matrix views for vectorized similarity computation.
        # All share the stored-entry pattern of ratings_csr, so overlap
        # counts match an explicit index intersection of two rating rows,
        # including stored entries whose centered value is zero.
        centered = ratings_csr.astype(np.float64, copy=True)
        centered.sort_indices()
        rows = np.repeat(
            np.arange(centered.shape[0]), np.diff(centered.indptr)
        )
        centered.data -= user_means[rows]
        del rows
        self.centered = centered
        self.centered_t = centered.T.tocsr()

        support = ratings_csr.copy()
        support.sort_indices()
        support.data = np.ones(support.nnz, dtype=np.float64)
        self.support = support
        self.support_t = support.T.tocsr()

        self.centered_sq = centered.multiply(centered).tocsr()
        self.centered_sq.sort_indices()
        self.centered_sq_t = self.centered_sq.T.tocsr()

        self._memo_user: int = -1
        self._memo: tuple[np.ndarray, np.ndarray] | None = None

    def _similarity_row(self, user: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (users, similarities) for users sharing a stored movie.

        The similarities use the same formula as the original per-pair
        computation: mean-centered cosine over the common stored movies,
        shrunk by n_common / (n_common + shrinkage), and zero when the
        overlap is below the minimum or the denominator vanishes. The
        user's own entry is zero, matching the original self-similarity
        special case. The result is memoized because both test prediction
        and ranking score the same user against many movies.
        """
        if user == self._memo_user and self._memo is not None:
            return self._memo

        overlaps = (
            self.support.getrow(user) @ self.support_t
        ).tocsr()
        overlaps.sort_indices()
        users = overlaps.indices.copy()
        counts = overlaps.data

        numerator = (
            self.centered.getrow(user) @ self.centered_t
        ).tocsr()
        numerator.sort_indices()
        num = self._gather(numerator, users)

        self_sq = (
            self.centered_sq.getrow(user) @ self.support_t
        ).tocsr()
        self_sq.sort_indices()
        sq_first = self._gather(self_sq, users)

        other_sq = (
            self.support.getrow(user) @ self.centered_sq_t
        ).tocsr()
        other_sq.sort_indices()
        sq_second = self._gather(other_sq, users)

        denominator = np.sqrt(sq_first) * np.sqrt(sq_second)
        similarities = np.zeros(len(users), dtype=np.float64)
        eligible = (counts >= self.min_common) & (denominator > 0.0)
        # Same rounding order as the original per-pair formula: divide
        # the overlap-weighted cosine numerator by the denominator, then
        # multiply by the pre-rounded shrinkage ratio count/(count+s).
        similarities[eligible] = (num[eligible] / denominator[eligible]) * (
            counts[eligible] / (counts[eligible] + self.shrinkage)
        )
        similarities[users == user] = 0.0

        self._memo_user = user
        self._memo = (users, similarities)
        return self._memo

    @staticmethod
    def _gather(
        row: sparse.csr_matrix, keys: np.ndarray
    ) -> np.ndarray:
        """Take a sparse row's values at sorted keys; 0.0 where absent."""
        pos = np.searchsorted(row.indices, keys)
        hit = pos < len(row.indices)
        values = np.zeros(len(keys), dtype=np.float64)
        take = np.zeros(len(keys), dtype=bool)
        take[hit] = row.indices[pos[hit]] == keys[hit]
        values[take] = row.data[pos[take]]
        return values

    def similarity(self, first: int, second: int) -> float:
        if first == second:
            return 0.0

        users, similarities = self._similarity_row(first)
        pos = np.searchsorted(users, second)
        if pos < len(users) and users[pos] == second:
            return float(similarities[pos])
        return 0.0

    def predict_one_with_count(
        self, user: int, movie: int
    ) -> tuple[float, int]:
        """Return (prediction, n_usable_neighbors) for one user-movie pair.

        n_usable_neighbors is the count of co-raters retained after the
        top-k |similarity| selection, i.e. the number of neighbors that
        actually contributed to the prediction. It is 0 for both fallback
        paths (no usable neighbor, or a vanishing denominator), in which
        case the prediction is the movie mean. The genome hybrid consumes
        this count as a confidence signal so it can route rows where k-NN
        had no evidence to the content component instead.
        """
        column = self.csc.getcol(movie)
        neighbors = column.indices
        ratings = column.data

        row_users, row_similarities = self._similarity_row(user)
        pos = np.searchsorted(row_users, neighbors)
        hit = pos < len(row_users)
        match = np.zeros(len(neighbors), dtype=bool)
        match[hit] = row_users[pos[hit]] == neighbors[hit]
        similarities = np.zeros(len(neighbors), dtype=np.float64)
        similarities[match] = row_similarities[pos[match]]

        usable = np.flatnonzero(similarities != 0.0)
        if len(usable) == 0:
            return float(self.movie_means[movie]), 0

        if len(usable) > self.top_k:
            order = np.lexsort(
                (
                    neighbors[usable],
                    -np.abs(similarities[usable]),
                )
            )
            usable = usable[order[: self.top_k]]

        selected_users = neighbors[usable]
        selected_similarities = similarities[usable]
        residuals = (
            ratings[usable] - self.user_means[selected_users]
        )
        denominator = np.sum(np.abs(selected_similarities))

        if denominator <= 0:
            return float(self.movie_means[movie]), 0

        return (
            float(
                self.user_means[user]
                + np.dot(selected_similarities, residuals) / denominator
            ),
            int(len(usable)),
        )

    def predict_one(self, user: int, movie: int) -> float:
        prediction, _ = self.predict_one_with_count(user, movie)
        return prediction

    def neighbor_count(self, user: int, movie: int) -> int:
        _, count = self.predict_one_with_count(user, movie)
        return count

    def predict(self, interactions: np.ndarray) -> np.ndarray:
        return run_stage(
            RowMapStage(self.predict_one),
            {"interactions": interactions},
            stage_name="UserKNN prediction",
        )

    def predict_counts(self, interactions: np.ndarray) -> np.ndarray:
        return run_stage(
            RowCountStage(self.neighbor_count),
            {"interactions": interactions},
            stage_name="UserKNN neighbor counts",
        )

    def score_user(self, user: int, movies: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.predict_one(user, int(movie)) for movie in movies],
            dtype=np.float64,
        )


class MovieKNN:
    """Item-based k-NN with sparse adjusted-cosine similarities.

    The adjusted cosine similarities are computed on user-mean-centered
    ratings, so the prediction weights the user's mean-centered ratings
    (residuals) for the neighbouring movies and re-adds the user mean:

        r_hat_ui = r_bar_u + sum_j s(i,j) * (r_uj - r_bar_u) / sum_j |s(i,j)|

    Without the centering, predictions collapse toward the population
    rating level and ignore the target user's rating scale.

    Symbols: s(i,j) is adjusted cosine similarity, s'(i,j) is the
    shrinkage-adjusted similarity, n_ij is the co-rater count, r_ui is
    the observed rating, r_bar_u is the user mean, N_k(i) is the set of
    k nearest movies, r_uj is the neighbour's rating.
    """

    def __init__(
        self,
        ratings_csr: sparse.csr_matrix,
        movie_similarity: sparse.csr_matrix,
        user_means: np.ndarray,
        movie_means: np.ndarray,
    ) -> None:
        self.ratings = ratings_csr
        self.similarity = movie_similarity
        self.user_means = user_means
        self.movie_means = movie_means

    def predict_one(self, user: int, movie: int) -> float:
        history = self.ratings.getrow(user)
        similarity_row = self.similarity.getrow(movie)

        common, history_pos, similarity_pos = np.intersect1d(
            history.indices,
            similarity_row.indices,
            assume_unique=True,
            return_indices=True,
        )

        # Defensively exclude the target movie if predicting an already-rated
        # pair. Test and ranking targets are normally absent from training.
        keep = common != movie
        history_pos = history_pos[keep]
        similarity_pos = similarity_pos[keep]

        if len(history_pos) == 0:
            return float(self.movie_means[movie])

        similarities = similarity_row.data[similarity_pos]
        denominator = np.sum(np.abs(similarities))
        if denominator <= 0:
            return float(self.movie_means[movie])

        residuals = history.data[history_pos] - self.user_means[user]
        return float(
            self.user_means[user]
            + np.dot(similarities, residuals) / denominator
        )

    def predict(self, interactions: np.ndarray) -> np.ndarray:
        return run_stage(
            RowMapStage(self.predict_one),
            {"interactions": interactions},
            stage_name="ItemKNN prediction",
        )

    def score_user(self, user: int, movies: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.predict_one(user, int(movie)) for movie in movies],
            dtype=np.float64,
        )


class GenomeHybrid:
    """Confidence-weighted ensemble of Tag Genome content and User k-NN.

    The hybrid blends the two already-evaluated methods -- the Tag Genome
    content profile and the User-based k-NN -- instead of inventing its own
    collaborative signal. For each row (user u, movie i):

      - c(u,i) is the Tag Genome content prediction
        r_bar_u + beta * p_u^T x_i.
      - k(u,i) is the User-based k-NN prediction.
      - n(u,i) is the number of usable k-NN neighbors actually used,
        i.e. the k-NN confidence. n is 0 exactly when k-NN fell back to
        the movie mean because it had no usable evidence.

    The prediction is the convex blend when k-NN has evidence, and the
    content prediction alone when it does not:

        r_hat = alpha * c + (1 - alpha) * k     if n > 0
        r_hat = c                               if n == 0

    The content-only fallback is what lets the ensemble weakly dominate
    the plain k-NN parent: on rows where k-NN degrades to a crude movie
    mean the content signal carries the prediction instead. alpha is
    selected on the shared validation set by minimizing validation RMSE
    over HYBRID_ALPHA_GRID (first grid point wins ties).

    Symbols: alpha is the blend weight, beta is the content scale, p_u is
    the user genome profile, x_i is the movie genome vector, r_bar_u is
    the user mean, and n(u,i) is the k-NN usable-neighbor count.
    """

    def __init__(
        self,
        profiles: np.ndarray,
        movie_features: np.ndarray,
        user_means: np.ndarray,
        movie_means: np.ndarray,
        beta: float,
        content_test: np.ndarray,
        userknn_test: np.ndarray,
        counts_test: np.ndarray,
        user_knn: "UserKNN",
    ) -> None:
        # Content component: fast per-row dot product.
        self.profiles = profiles
        self.movie_features = movie_features
        self.user_means = user_means
        self.movie_means = movie_means
        self.beta = beta
        # Saved test-row outputs from the genome_content and user_knn
        # stages; the rating prediction blends these directly.
        self.content_test = np.asarray(content_test, dtype=np.float64)
        self.userknn_test = np.asarray(userknn_test, dtype=np.float64)
        self.counts_test = np.asarray(counts_test, dtype=np.int32)
        # Live k-NN used for validation alpha selection and ranking
        # candidates (candidates are warm-start training-catalog movies,
        # so the k-NN count is always > 0 there).
        self.user_knn = user_knn

    def content_for(self, user: int, movie: int) -> float:
        return float(
            self.user_means[user]
            + self.beta
            * np.dot(self.profiles[user], self.movie_features[movie])
        )

    def blend_row(
        self,
        content: float,
        userknn: float,
        count: int,
        alpha: float,
    ) -> float:
        """Confidence-aware blend of one content/k-NN/count row."""
        if count > 0:
            return float(alpha * content + (1.0 - alpha) * userknn)
        return float(content)

    def prediction_from_components(
        self,
        content: np.ndarray,
        userknn: np.ndarray,
        counts: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Vectorized confidence-aware blend over matched component rows."""
        content = np.asarray(content, dtype=np.float64)
        userknn = np.asarray(userknn, dtype=np.float64)
        counts = np.asarray(counts)
        blended = alpha * content + (1.0 - alpha) * userknn
        blended[counts == 0] = content[counts == 0]
        return blended

    def predict(
        self,
        interactions: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Blend the saved test-row content and k-NN outputs.

        interactions is the test matrix; its row order must match the
        saved outputs, which it does because both the genome_content and
        user_knn stages save one value per test row in test-row order.
        """
        del interactions  # row count is asserted by the caller
        return self.prediction_from_components(
            self.content_test,
            self.userknn_test,
            self.counts_test,
            alpha,
        )

    def validation_components(
        self,
        interactions: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute (content, userknn, counts) on arbitrary rows.

        Used to select alpha on the shared validation set, whose outputs
        are not saved by the genome_content / user_knn stages.
        """
        content = np.asarray(
            [
                self.content_for(int(row[0]), int(row[1]))
                for row in interactions
            ],
            dtype=np.float64,
        )
        userknn = self.user_knn.predict(interactions)
        counts = self.user_knn.predict_counts(interactions)
        return content, userknn, counts

    def score_user(
        self,
        user: int,
        movies: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Blend content and k-NN per candidate movie for ranking.

        Ranking candidates are warm-start training-catalog movies, so the
        k-NN count is always > 0 and every candidate uses the convex
        blend; the content-only fallback applies only to rating rows.
        """
        movies = np.asarray(movies)
        content = np.asarray(
            [self.content_for(user, int(movie)) for movie in movies],
            dtype=np.float64,
        )
        userknn = self.user_knn.score_user(user, movies)
        counts = self.user_knn.predict_counts(
            np.column_stack(
                (
                    np.full(len(movies), user, dtype=np.int64),
                    movies,
                )
            )
        )
        return self.prediction_from_components(
            content, userknn, counts, alpha
        )


def choose_hybrid_alpha(
    actual: np.ndarray,
    content: np.ndarray,
    userknn: np.ndarray,
    counts: np.ndarray,
) -> float:
    """Select the blend weight by minimizing validation RMSE.

    The confidence-aware rule is applied inside the grid search so the
    chosen alpha optimizes the same function the hybrid predicts: rows
    with no k-NN neighbor (counts == 0) contribute the content prediction
    for every alpha and therefore do not bias the selection.
    """
    best_alpha = 0.0
    best_rmse = np.inf

    for alpha in HYBRID_ALPHA_GRID:
        predictions = alpha * content + (1.0 - alpha) * userknn
        predictions[counts == 0] = content[counts == 0]
        rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)

    return best_alpha


def zscore_rows(
    rows: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-row z-score standardisation of a 2-D score matrix.

    Each row (one user's candidate score vector) is centred by its own mean
    and scaled by its own population standard deviation (ddof=0):
    z_c = (s_c - mean(s)) / std(s). A row with (near-)zero variance is
    treated as constant and yields all-zero z-values (matching the
    reference implementation), so that row fuses purely on the other model's
    standardised scores. Standardisation is order-preserving within a row,
    so blending z-rows and re-sorting recovers the blended ranking. The
    summation order (np.mean / np.std over the row) is what the C port must
    reproduce for bit-for-bit parity.
    """
    rows = np.asarray(rows, dtype=np.float64)
    out = np.empty_like(rows)
    for r in range(rows.shape[0]):
        std = float(np.std(rows[r]))
        if std < eps:
            out[r] = 0.0
        else:
            out[r] = (rows[r] - float(np.mean(rows[r]))) / std
    return out


def choose_rating_lambda(
    actual: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """Select the rating-head blend weight by minimising validation RMSE.

    The rating head predicts  lambda * A + (1 - lambda) * B, where A is the
    MF rating prediction and B is the BPR affine-mapped rating. lambda is
    chosen from LAMBDA_GRID to minimise validation RMSE. On ties within
    MF_BPR_SELECTION_TOLERANCE the LARGER lambda is kept (the MF-leaning
    choice), a prespecified deterministic rule mirroring the BPR
    regularisation tie policy.
    """
    actual = np.asarray(actual, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    best_lambda = 0.0
    best_rmse = np.inf
    for lam in LAMBDA_GRID:
        predictions = lam * a + (1.0 - lam) * b
        rmse = float(np.sqrt(np.mean((predictions - actual) ** 2)))
        improves = rmse < best_rmse - MF_BPR_SELECTION_TOLERANCE
        ties = abs(rmse - best_rmse) <= MF_BPR_SELECTION_TOLERANCE
        if improves or (ties and float(lam) > best_lambda):
            best_rmse = rmse
            best_lambda = float(lam)
    return best_lambda


def choose_ranking_lambda(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    users: np.ndarray,
    candidates: np.ndarray,
    positive_indices: np.ndarray,
) -> float:
    """Select the ranking-head blend weight by maximising validation NDCG@K.

    Both input score matrices are per-row z-standardised (they come from
    algorithms with incompatible raw scales), then blended as
    lambda * zA + (1 - lambda) * zB for each candidate row. lambda is chosen
    from LAMBDA_GRID to maximise validation NDCG@K on the shared validation
    candidate rows. On ties within MF_BPR_SELECTION_TOLERANCE the LARGER
    lambda is kept (the MF-leaning choice), mirroring the BPR regularisation
    tie policy.
    """
    z_a = zscore_rows(scores_a)
    z_b = zscore_rows(scores_b)
    best_lambda = 0.0
    best_ndcg = -np.inf
    for lam in LAMBDA_GRID:
        blended = lam * z_a + (1.0 - lam) * z_b
        metrics = ranking_metrics(
            "MF+BPR ranking-head validation",
            users,
            candidates,
            positive_indices,
            lambda user, movies: np.zeros(len(movies)),
            k=K,
            scores=blended,
        )
        ndcg = float(metrics["ndcg@k"])
        improves = ndcg > best_ndcg + MF_BPR_SELECTION_TOLERANCE
        ties = abs(ndcg - best_ndcg) <= MF_BPR_SELECTION_TOLERANCE
        if improves or (ties and float(lam) > best_lambda):
            best_ndcg = ndcg
            best_lambda = float(lam)
    return best_lambda


def train_biased_mf(
    train: np.ndarray,
    n_users: int,
    n_movies: int,
    global_mean: float,
    n_factors: int = 50,
    learning_rate: float = 0.005,
    regularization: float = 0.02,
    epochs: int = 20,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Train standard bias-aware matrix factorization by SGD.

    Model: r_hat_ui = mu + b_u + b_i + p_u . q_i.
    Symbols: mu is global_mean, b_u/b_i are user/movie biases, p_u/q_i are
    latent factor vectors, gamma is learning_rate, lambda is regularization.
    """
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0, 0.1, (n_users, n_factors))
    movie_factors = rng.normal(0, 0.1, (n_movies, n_factors))
    user_bias = np.zeros(n_users)
    movie_bias = np.zeros(n_movies)

    users = train[:, 0].astype(np.int64)
    movies = train[:, 1].astype(np.int64)
    ratings = train[:, 2]

    user_seen = np.bincount(users, minlength=n_users) > 0
    movie_seen = np.bincount(movies, minlength=n_movies) > 0
    indices = np.arange(len(train))

    for epoch in range(epochs):
        rng.shuffle(indices)

        for index in indices:
            user = users[index]
            movie = movies[index]
            rating = ratings[index]

            old_user = user_factors[user].copy()
            old_movie = movie_factors[movie].copy()
            prediction = (
                global_mean
                + user_bias[user]
                + movie_bias[movie]
                + np.dot(old_user, old_movie)
            )
            error = rating - prediction

            user_bias[user] += learning_rate * (
                error - regularization * user_bias[user]
            )
            movie_bias[movie] += learning_rate * (
                error - regularization * movie_bias[movie]
            )
            user_factors[user] += learning_rate * (
                error * old_movie - regularization * old_user
            )
            movie_factors[movie] += learning_rate * (
                error * old_user - regularization * old_movie
            )

        epoch_predictions = (
            global_mean
            + user_bias[users]
            + movie_bias[movies]
            + np.sum(
                user_factors[users] * movie_factors[movies],
                axis=1,
            )
        )
        epoch_rmse = np.sqrt(
            np.mean((ratings - epoch_predictions) ** 2)
        )
        print(
            f"    MF epoch {epoch + 1}/{epochs}: "
            f"final-model train RMSE={epoch_rmse:.4f}"
        )

    # Random factors are invalid cold-start predictions. Zeroing unseen
    # factors makes the model fall back to the identifiable bias terms.
    user_factors[~user_seen] = 0.0
    movie_factors[~movie_seen] = 0.0

    return {
        "global_mean": global_mean,
        "user_bias": user_bias,
        "movie_bias": movie_bias,
        "user_factors": user_factors,
        "movie_factors": movie_factors,
        "user_seen": user_seen,
        "movie_seen": movie_seen,
    }


def predict_biased_mf(
    interactions: np.ndarray,
    model: dict[str, np.ndarray | float],
) -> np.ndarray:
    users = interactions[:, 0].astype(np.int64)
    movies = interactions[:, 1].astype(np.int64)

    user_seen = model["user_seen"][users]
    movie_seen = model["movie_seen"][movies]

    user_bias = model["user_bias"][users] * user_seen
    movie_bias = model["movie_bias"][movies] * movie_seen
    interactions_term = np.sum(
        model["user_factors"][users]
        * model["movie_factors"][movies],
        axis=1,
    )

    return (
        float(model["global_mean"])
        + user_bias
        + movie_bias
        + interactions_term
    )


def _mf_grid_worker(regularization: float) -> dict[str, np.ndarray | float]:
    """Train one biased-MF configuration in a forked worker."""
    print(f"  Training MF with regularization={regularization}...")
    return train_biased_mf(
        _GRID_STATE["train"],
        _GRID_STATE["n_users"],
        _GRID_STATE["n_movies"],
        _GRID_STATE["global_mean"],
        regularization=regularization,
    )


def _bpr_grid_worker(regularization: float) -> dict[str, np.ndarray]:
    """Train one BPR-MF configuration in a forked worker."""
    print(f"  Training BPR with regularization={regularization}...")
    return train_bpr(
        _GRID_STATE["train"],
        _GRID_STATE["n_users"],
        _GRID_STATE["n_movies"],
        regularization=regularization,
    )


def run_grid(jobs: list[tuple], stage_name: str | None = None) -> list:
    """Run independent grid-training jobs in forked workers.

    Each regularization setting trains a separate model, so the grid is
    embarrassingly parallel and the models are unchanged by concurrent
    training. Job arguments stay scalar: workers read the training arrays
    from _GRID_STATE through copy-on-write forked memory, so the large
    training matrix is never pickled. Results keep submission order.
    """
    return data.run_serial_or_forked(
        resolve_prediction_workers(), jobs, stage_name
    )


def select_mf_model(
    train: np.ndarray,
    validation: np.ndarray,
    n_users: int,
    n_movies: int,
    global_mean: float,
) -> tuple[dict[str, np.ndarray | float], float]:
    best_model = None
    best_regularization = 0.0
    best_rmse = np.inf

    # The grid settings are independent training runs; train them
    # concurrently and compare the returned models in grid order below.
    _GRID_STATE.clear()
    _GRID_STATE.update(
        train=train,
        n_users=n_users,
        n_movies=n_movies,
        global_mean=global_mean,
    )
    models = run_grid(
        [(_mf_grid_worker, (regularization,))
         for regularization in MF_REGULARIZATION_GRID],
        stage_name="MF grid search",
    )

    for regularization, model in zip(MF_REGULARIZATION_GRID, models):
        predictions = predict_biased_mf(validation, model)
        rmse = float(
            np.sqrt(np.mean((predictions - validation[:, 2]) ** 2))
        )
        if rmse < best_rmse:
            best_model = model
            best_regularization = regularization
            best_rmse = rmse

    assert best_model is not None
    return best_model, best_regularization


# Model persistence so a re-run does not retrain MF / BPR.
#
# Both models are dicts of arrays plus a scalar; np.savez round-trips them
# exactly (float64 / bool). We store n_users / n_movies so a loader can
# reject a file that belongs to a different dataset shape before it is used.


def _save_mf_model(
    model: dict[str, np.ndarray | float],
    regularization: float,
    n_users: int,
    n_movies: int,
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    """Persist a selected biased-MF model to Predictions/mf_model.npz."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination / "mf_model.npz",
        n_users=np.int64(n_users),
        n_movies=np.int64(n_movies),
        regularization=np.float64(regularization),
        global_mean=np.float64(model["global_mean"]),
        user_bias=model["user_bias"],
        movie_bias=model["movie_bias"],
        user_factors=model["user_factors"],
        movie_factors=model["movie_factors"],
        user_seen=model["user_seen"],
        movie_seen=model["movie_seen"],
    )
    print(f"Saved MF model to {destination / 'mf_model.npz'}")


def _load_mf_model(
    n_users: int,
    n_movies: int,
    path: str | Path = data.PREDICTIONS_DIR,
) -> tuple[dict[str, np.ndarray | float], float] | None:
    """Load a persisted biased-MF model, or None if absent/inconsistent.

    Returns (model, regularization). The model dict is in the same shape as
    train_biased_mf's output, so predict_biased_mf works unchanged.
    """
    filename = Path(path) / "mf_model.npz"
    if not filename.exists():
        return None
    # .item() reads a scalar that may be stored 0-D (Python writer) or 1-D
    # length-1 (C writer), so the two implementations' model files are
    # mutually readable.
    with np.load(filename) as saved:
        if int(saved["n_users"].item()) != n_users or int(saved["n_movies"].item()) != n_movies:
            return None
        model: dict[str, np.ndarray | float] = {
            "global_mean": float(saved["global_mean"].item()),
            "user_bias": saved["user_bias"].copy(),
            "movie_bias": saved["movie_bias"].copy(),
            "user_factors": saved["user_factors"].copy(),
            "movie_factors": saved["movie_factors"].copy(),
            "user_seen": saved["user_seen"].copy(),
            "movie_seen": saved["movie_seen"].copy(),
        }
        regularization = float(saved["regularization"].item())
    return model, regularization


def _save_bpr_model(
    model: dict[str, np.ndarray],
    regularization: float,
    n_users: int,
    n_movies: int,
    affine_mapping: dict[str, float] | None = None,
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    """Persist a selected BPR-MF model to Predictions/bpr_model.npz.

    The validation-fitted nonnegative affine rating map (slope, intercept)
    is stored alongside the latent factors so post-hoc combinations (the
    MF+BPR hybrid's rating head) can reuse the exact mapping the BPR stage
    used, without re-fitting it. When affine_mapping is None the two
    members are omitted and _load_bpr_model reports an absent mapping.
    """
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    members = dict(
        n_users=np.int64(n_users),
        n_movies=np.int64(n_movies),
        regularization=np.float64(regularization),
        user_factors=model["user_factors"],
        movie_factors=model["movie_factors"],
        user_trained=model["user_trained"],
        movie_trained=model["movie_trained"],
    )
    if affine_mapping is not None:
        members["slope"] = np.asarray(float(affine_mapping["slope"]))
        members["intercept"] = np.asarray(float(affine_mapping["intercept"]))
    np.savez(destination / "bpr_model.npz", **members)
    print(f"Saved BPR model to {destination / 'bpr_model.npz'}")


def _load_bpr_model(
    n_users: int,
    n_movies: int,
    path: str | Path = data.PREDICTIONS_DIR,
) -> tuple[
    dict[str, np.ndarray], float, dict[str, float] | None
] | None:
    """Load a persisted BPR-MF model, or None if absent/inconsistent.

    Returns (model, regularization, affine_mapping). The model dict matches
    train_bpr's output, so predict_bpr_scores / bpr_score_user work
    unchanged. affine_mapping is {"slope", "intercept"} when the file stores
    it (newer runs), else None (older files pre-dating the affine
    persistence) — callers then re-fit the mapping from validation scores.
    """
    filename = Path(path) / "bpr_model.npz"
    if not filename.exists():
        return None
    # .item() reads a scalar stored 0-D (Python) or 1-D length-1 (C) so both
    # implementations' model files are mutually readable.
    with np.load(filename) as saved:
        # Older files have no n_users/n_movies shape guards; tolerate their
        # absence but still validate when present.
        has_shape = "n_users" in saved and "n_movies" in saved
        if has_shape and (
            int(saved["n_users"].item()) != n_users
            or int(saved["n_movies"].item()) != n_movies
        ):
            return None
        model: dict[str, np.ndarray] = {
            "user_factors": saved["user_factors"].copy(),
            "movie_factors": saved["movie_factors"].copy(),
            "user_trained": saved["user_trained"].copy(),
            "movie_trained": saved["movie_trained"].copy(),
        }
        regularization = (
            float(saved["regularization"].item())
            if "regularization" in saved
            else 0.0
        )
        if "slope" in saved and "intercept" in saved:
            affine_mapping = {
                "slope": float(saved["slope"].item()),
                "intercept": float(saved["intercept"].item()),
            }
        else:
            affine_mapping = None
    return model, regularization, affine_mapping


def build_bpr_training_data(
    train: np.ndarray,
    n_users: int,
    threshold: float = BPR_RELEVANCE_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[set[int]]]:
    """Return threshold-positive pairs and all observed-movie exclusions.

    Symbols: threshold is the minimum rating to count as a BPR positive
    (3.5); observed is a per-user set of all training-catalog movie indices
    seen by that user, used to exclude observed movies from negative
    sampling; positives are the (user, movie) pairs with rating >= threshold.
    """
    observed: list[set[int]] = [set() for _ in range(n_users)]

    for row in train:
        observed[int(row[0])].add(int(row[1]))

    positives = train[train[:, 2] >= threshold]
    return (
        positives[:, 0].astype(np.int64),
        positives[:, 1].astype(np.int64),
        observed,
    )


def draw_unobserved_movie(
    rng: np.random.Generator,
    catalog: np.ndarray,
    excluded: set[int],
    rejection_attempts: int = 64,
) -> int | None:
    """Draw uniformly from catalog movies absent from excluded.

    Rejection sampling is used for the normal sparse-history case. A direct
    eligible-set fallback guarantees termination for dense histories and
    correctly handles exclusions containing movies outside the catalog.

    Symbols: catalog is the set of all training-catalog movie indices;
    excluded is the set of movies already observed by the user (positives
    and other observed ratings); rejection_attempts bounds how many random
    draws are tried before falling back to the explicit eligible-set scan.
    """
    if catalog.ndim != 1 or len(catalog) == 0:
        raise ValueError("The BPR catalog must be a nonempty vector.")
    if rejection_attempts < 0:
        raise ValueError("rejection_attempts must be nonnegative.")

    for _ in range(rejection_attempts):
        movie = int(catalog[rng.integers(0, len(catalog))])
        if movie not in excluded:
            return movie

    eligible = np.asarray(
        [movie for movie in catalog if int(movie) not in excluded],
        dtype=catalog.dtype,
    )
    if len(eligible) == 0:
        return None
    return int(eligible[rng.integers(0, len(eligible))])


def train_bpr(
    train: np.ndarray,
    n_users: int,
    n_movies: int,
    n_factors: int = 50,
    learning_rate: float = 0.01,
    regularization: float = 0.01,
    epochs: int = 20,
    negatives_per_positive: int = BPR_NEGATIVES_PER_POSITIVE,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Train BPR-MF with dynamic unobserved-negative sampling.

    This is a thresholded explicit-feedback adaptation: ratings >= 3.5 are
    positive, other observed ratings are excluded from negative sampling,
    and unobserved training-catalog movies are potential negatives.

    Symbols: x_ui = p_u . q_i is the raw preference score, x_uij is the
    score difference p_u . (q_i - q_j), sigma is the logistic sigmoid,
    lambda is regularization, gamma is learning_rate.
    """
    rng = np.random.default_rng(seed)
    user_factors = rng.normal(0, 0.01, (n_users, n_factors))
    movie_factors = rng.normal(0, 0.01, (n_movies, n_factors))

    positive_users, positive_movies, observed = build_bpr_training_data(
        train,
        n_users,
    )
    if len(positive_users) == 0:
        raise ValueError("No BPR-positive training interactions exist.")

    train_movies = train[:, 1].astype(np.int64)
    catalog = np.unique(train_movies)
    user_trained = np.zeros(n_users, dtype=bool)
    movie_trained = np.zeros(n_movies, dtype=bool)
    order = np.arange(len(positive_users))

    for epoch in range(epochs):
        rng.shuffle(order)

        for index in order:
            user = int(positive_users[index])
            positive = int(positive_movies[index])

            for _ in range(negatives_per_positive):
                negative = draw_unobserved_movie(
                    rng,
                    catalog,
                    observed[user],
                )
                if negative is None:
                    continue

                old_user = user_factors[user].copy()
                old_positive = movie_factors[positive].copy()
                old_negative = movie_factors[negative].copy()

                difference = np.dot(
                    old_user,
                    old_positive - old_negative,
                )
                gradient = 1.0 / (
                    1.0 + np.exp(np.clip(difference, -500, 500))
                )

                user_factors[user] += learning_rate * (
                    gradient * (old_positive - old_negative)
                    - regularization * old_user
                )
                movie_factors[positive] += learning_rate * (
                    gradient * old_user
                    - regularization * old_positive
                )
                movie_factors[negative] += learning_rate * (
                    -gradient * old_user
                    - regularization * old_negative
                )

                user_trained[user] = True
                movie_trained[positive] = True
                movie_trained[negative] = True

        sample_size = min(2000, len(positive_users))
        sample = rng.choice(
            len(positive_users),
            sample_size,
            replace=False,
        )
        valid = 0
        correct = 0

        for index in sample:
            user = int(positive_users[index])
            negative = draw_unobserved_movie(
                rng,
                catalog,
                observed[user],
            )
            if negative is None:
                continue
            positive = int(positive_movies[index])
            positive_score = np.dot(
                user_factors[user],
                movie_factors[positive],
            )
            negative_score = np.dot(
                user_factors[user],
                movie_factors[negative],
            )
            correct += int(positive_score > negative_score)
            valid += 1

        accuracy = correct / valid if valid else 0.0
        print(
            f"    BPR epoch {epoch + 1}/{epochs}: "
            f"sampled pair accuracy={accuracy:.4f}"
        )

    # Untrained random factors must not create arbitrary recommendations.
    user_factors[~user_trained] = 0.0
    movie_factors[~movie_trained] = 0.0

    return {
        "user_factors": user_factors,
        "movie_factors": movie_factors,
        "user_trained": user_trained,
        "movie_trained": movie_trained,
    }


def predict_bpr_scores(
    interactions: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    users = interactions[:, 0].astype(np.int64)
    movies = interactions[:, 1].astype(np.int64)
    return np.sum(
        model["user_factors"][users]
        * model["movie_factors"][movies],
        axis=1,
    )


def bpr_score_user(
    model: dict[str, np.ndarray],
    user: int,
    movies: np.ndarray,
) -> np.ndarray:
    """Return raw BPR-MF preference scores for candidate movies."""
    return (
        model["movie_factors"][movies]
        @ model["user_factors"][user]
    )


def validation_ndcg(
    model: dict[str, np.ndarray],
    validation_ranking: dict[str, np.ndarray],
) -> float:
    """Evaluate BPR on the fixed validation candidate rows."""
    metrics = ranking_metrics(
        "BPR validation candidate",
        validation_ranking["users"],
        validation_ranking["candidates"],
        validation_ranking["positive_indices"],
        lambda user, movies: bpr_score_user(model, user, movies),
        k=K,
    )
    return float(metrics["ndcg@k"])


def select_bpr_model(
    train: np.ndarray,
    validation_ranking: dict[str, np.ndarray],
    n_users: int,
    n_movies: int,
) -> tuple[dict[str, np.ndarray], float, float]:
    """Select BPR regularization by fixed-validation NDCG@10.

    If NDCG values tie within BPR_SELECTION_TOLERANCE, stronger
    regularization is preferred as a prespecified deterministic rule.
    """
    best_model: dict[str, np.ndarray] | None = None
    best_regularization = -np.inf
    best_ndcg = -np.inf

    # The grid settings are independent training runs; train them
    # concurrently, then run the selection logic serially below so the
    # deterministic tie policy and validation NDCG prints stay ordered.
    _GRID_STATE.clear()
    _GRID_STATE.update(
        train=train,
        n_users=n_users,
        n_movies=n_movies,
    )
    models = run_grid(
        [(_bpr_grid_worker, (regularization,))
         for regularization in BPR_REGULARIZATION_GRID],
        stage_name="BPR grid search",
    )

    for regularization, model in zip(BPR_REGULARIZATION_GRID, models):
        ndcg = validation_ndcg(model, validation_ranking)
        print(f"    Fixed validation NDCG@{K}={ndcg:.6f}")

        improves_ndcg = ndcg > best_ndcg + BPR_SELECTION_TOLERANCE
        ties_ndcg = (
            abs(ndcg - best_ndcg) <= BPR_SELECTION_TOLERANCE
        )
        prefers_regularization = regularization > best_regularization

        if improves_ndcg or (ties_ndcg and prefers_regularization):
            best_model = model
            best_regularization = float(regularization)
            best_ndcg = ndcg

    if best_model is None:
        raise RuntimeError("BPR model selection produced no model.")

    return best_model, best_regularization, best_ndcg


def validate_processed_configuration(
    values: dict[str, object],
    validation_ranking: dict[str, np.ndarray],
    test_ranking: dict[str, np.ndarray],
) -> None:
    """Reject incompatible or stale processed artifacts.

    Symbols: metadata keys checked here are produced by prepare_data.py:
    minimum_user_interactions (min ratings per user after filtering),
    item_neighbors (top-k for movie similarity), minimum_common_raters
    (min co-raters for movie similarity), similarity_shrinkage (shrinkage
    parameter for movie-movie adjusted cosine), chronological_tie_break
    (timestamp tie-breaking policy), RANKING_CANDIDATES (candidate count
    per user), RELEVANCE_THRESHOLD (rating >= threshold for relevance).
    """
    metadata = values["metadata"]

    expected = {
        "minimum_user_interactions": data.MIN_USER_INTERACTIONS,
        "item_neighbors": data.MOVIE_NEIGHBORS,
        "minimum_common_raters": data.MIN_COMMON_RATERS,
    }
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise ValueError(
                f"Incompatible metadata {key}: found {actual!r}, "
                f"expected {expected_value!r}. Rerun prepare_data.py."
            )

    actual_shrinkage = float(metadata.get("similarity_shrinkage", np.nan))
    if not np.isclose(
        actual_shrinkage,
        data.SIMILARITY_SHRINKAGE,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "Incompatible movie-similarity shrinkage. "
            "Rerun prepare_data.py."
        )

    if metadata.get("chronological_tie_break") != "source row order":
        raise ValueError(
            "Processed data does not contain the selected timestamp "
            "tie policy. Rerun prepare_data.py."
        )

    for split, ranking in (
        ("validation", validation_ranking),
        ("test", test_ranking),
    ):
        validate_ranking_inputs(
            ranking["users"],
            ranking["candidates"],
            ranking["positive_indices"],
            K,
        )
        stored_count = int(ranking["n_candidates"])
        if stored_count != ranking["candidates"].shape[1]:
            raise ValueError(
                f"{split} ranking candidate metadata is inconsistent."
            )
        if stored_count != data.RANKING_CANDIDATES:
            raise ValueError(
                f"{split} candidates were prepared with {stored_count} "
                "movies; expected "
                f"{data.RANKING_CANDIDATES}. Rerun prepare_data.py."
            )

        threshold = float(ranking["relevance_threshold"])
        if not np.isclose(
            threshold,
            data.RELEVANCE_THRESHOLD,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                f"{split} ranking relevance threshold is incompatible."
            )


def fit_nonnegative_affine_mapping(
    scores: np.ndarray,
    ratings: np.ndarray,
) -> dict[str, float]:
    """Fit rating = slope * score + intercept with slope >= 0.

    Symbols: slope is 'a' in the essay notation (hat_r = a * x + b),
    intercept is 'b'. The nonnegative constraint prevents the mapping
    from reversing the learned ranking.
    """
    scores = require_finite(scores, "BPR validation scores")
    ratings = require_finite(ratings, "BPR validation ratings")

    score_mean = float(np.mean(scores))
    rating_mean = float(np.mean(ratings))
    centered_scores = scores - score_mean
    denominator = float(np.dot(centered_scores, centered_scores))

    if denominator <= np.finfo(float).eps:
        slope = 0.0
    else:
        slope = max(
            0.0,
            float(
                np.dot(centered_scores, ratings - rating_mean)
                / denominator
            ),
        )

    return {
        "slope": slope,
        "intercept": rating_mean - slope * score_mean,
    }


def map_bpr_to_ratings(
    interactions: np.ndarray,
    model: dict[str, np.ndarray],
    mapping: dict[str, float],
) -> np.ndarray:
    scores = predict_bpr_scores(interactions, model)
    return mapping["slope"] * scores + mapping["intercept"]


# The hybrid is listed LAST in execution order (not in this tuple) because it
# reuses the genome_content and user_knn outputs. STAGE_NAMES is only used for
# the --stage / --stages CLI choices and defaults to all stages; the actual
# execution order is fixed in run_all (hybrid runs after every parent stage).
STAGE_NAMES = (
    "global_mean",
    "genre_content",
    "genome_content",
    "user_knn",
    "movie_knn",
    "mf",
    "bpr",
    "hybrid",
    "mf_bpr",
)

def run_all(
    path: str | Path = data.OUTPUT_DIR,
    stages: tuple[str, ...] = STAGE_NAMES,
) -> dict[str, dict[str, object]]:
    values = data.load_arrays(path)
    validation_ranking = data.load_ranking("validation", path)
    test_ranking = data.load_ranking("test", path)
    validate_processed_configuration(
        values,
        validation_ranking,
        test_ranking,
    )

    train = values["train"]
    validation = values["validation"]
    test = values["test"]
    metadata = values["metadata"]

    n_users = int(metadata["n_users"])
    n_movies = int(metadata["n_movies"])
    global_mean = float(metadata["global_mean_rating"])

    results: dict[str, dict[str, object]] = {}

    _stage_display = {
        "global_mean": "Global mean",
        "genre_content": "Genre content profile",
        "genome_content": "Tag Genome content profile",
        "user_knn": "User-based k-NN",
        "movie_knn": "Movie-based k-NN",
        "mf": "Biased MF",
        "bpr": "BPR-MF thresholded positives",
        "hybrid": "Tag Genome weighted hybrid",
        "mf_bpr": "MF+BPR hybrid",
    }
    _stage_index = {s: i + 1 for i, s in enumerate(STAGE_NAMES)}

    def stage_banner(stage_key: str, status: str) -> None:
        """Top-level progress marker so a long run shows which stage it is in.

        One banner per stage (not per sub-step); the detailed file I/O lines
        (Saved/Loaded ...) already print the individual artifact names.
        """
        print(
            f"\n=== Stage {_stage_index.get(stage_key, '?')}/{len(STAGE_NAMES)}: "
            f"{_stage_display.get(stage_key, stage_key)} — {status} ==="
        )

    def evaluate(
        name: str,
        predictions: np.ndarray,
        scorer: Callable[[int, np.ndarray], np.ndarray],
        training_seconds: float = 0.0,
        prediction_seconds: float = 0.0,
        choices: dict[str, object] | None = None,
        ranking_scores: np.ndarray | None = None,
    ) -> None:
        rating = rating_metrics(name, predictions, test[:, 2])
        # When ranking_scores is supplied the metrics consume it directly and
        # the scorer is not invoked (precomputed / reused scores). Otherwise
        # the scorer runs, as for the global-mean random baseline fallback.
        ranking = ranking_metrics(
            name,
            test_ranking["users"],
            test_ranking["candidates"],
            test_ranking["positive_indices"],
            scorer,
            scores=ranking_scores,
        )
        results[name] = {
            "rating": rating,
            "ranking": ranking,
            "timing": {
                "training_seconds": training_seconds,
                "prediction_seconds": prediction_seconds,
            },
            "selected_choices": choices or {},
        }
        print(
            f"{name}: RMSE={rating['rmse']:.4f}, "
            f"HR@{K}={ranking['hit_rate@k']:.4f}, "
            f"NDCG@{K}={ranking['ndcg@k']:.4f}"
        )

    def random_baseline_scorer(
        user: int,
        movies: np.ndarray,
    ) -> np.ndarray:
        # The rating baseline is constant. Ranking therefore requires an
        # explicit reproducible random tie policy. seed = 42 ^ user keeps the
        # scores deterministic and identical to the original per-row scorer,
        # so the global-mean metrics do not change.
        seed = np.uint64(42) ^ np.uint64(user)
        rng = np.random.default_rng(seed)
        return rng.random(len(movies))

    if "global_mean" in stages:
        # Reuse: if the global-mean predictions and ranking scores were
        # already saved, read them and skip the (trivial) recompute.
        loaded_pred = load_predictions("global_mean")
        loaded_scores = load_ranking_scores("global_mean")
        stage_banner(
            "global_mean",
            "reusing saved outputs" if (loaded_pred is not None and loaded_scores is not None) else "computing fresh",
        )
        if loaded_pred is not None and loaded_scores is not None:
            wall_load = time.perf_counter()
            average_predictions = loaded_pred
            ranking_scores = loaded_scores
            wall_load = time.perf_counter() - wall_load
            meta = load_meta("global_mean") or {}
            evaluate(
                "Global mean",
                average_predictions,
                random_baseline_scorer,
                training_seconds=0.0,
                prediction_seconds=wall_load,
                choices={"ranking_tie_policy": "seeded random ordering"},
                ranking_scores=ranking_scores,
            )
            save_meta("global_mean", {
                "training_seconds": 0.0,
                "prediction_seconds": wall_load,
                "wall_predict": 0.0,
                "wall_rank": 0.0,
                "reused": True,
                "reused_model": False,
                "loaded_from": [
                    "global_mean_predictions.npy",
                    "global_mean_ranking_scores.npy",
                ],
                "computed_fresh": False,
            })
        else:
            # Fresh: predict the constant and score all candidates (the
            # seeded random vectors). Timing includes both passes.
            wall_predict = time.perf_counter()
            average_predictions = predict_global_mean(test, global_mean)
            wall_predict = time.perf_counter() - wall_predict

            wall_rank = time.perf_counter()
            ranking_scores = score_rows(
                random_baseline_scorer,
                test_ranking["users"],
                test_ranking["candidates"],
                "Global mean",
            )
            wall_rank = time.perf_counter() - wall_rank

            save_predictions(average_predictions, "global_mean")
            save_ranking_scores(ranking_scores, "global_mean")
            save_meta("global_mean", {
                "training_seconds": 0.0,
                "prediction_seconds": wall_predict + wall_rank,
                "wall_predict": wall_predict,
                "wall_rank": wall_rank,
                "reused": False,
                "reused_model": False,
                "loaded_from": [],
                "computed_fresh": True,
            })
            evaluate(
                "Global mean",
                average_predictions,
                random_baseline_scorer,
                training_seconds=0.0,
                prediction_seconds=wall_predict + wall_rank,
                choices={"ranking_tie_policy": "seeded random ordering"},
                ranking_scores=ranking_scores,
            )

    def make_content_scorer(
        selected_profiles: np.ndarray,
        selected_features: np.ndarray,
        selected_beta: float,
    ) -> Callable[[int, np.ndarray], np.ndarray]:
        def scorer(user: int, movies: np.ndarray) -> np.ndarray:
            return (
                values["user_means"][user]
                + selected_beta
                * (
                    selected_features[movies]
                    @ selected_profiles[user]
                )
            )
        return scorer

    for name, profiles, features, stage_key in (
        (
            "Genre content profile",
            values["genre_profiles"],
            values["genre_vectors"],
            "genre_content",
        ),
        (
            "Tag Genome content profile",
            values["genome_profiles"],
            values["genome_vectors"],
            "genome_content",
        ),
    ):
        if stage_key not in stages:
            continue

        loaded_pred = load_predictions(stage_key)
        loaded_scores = load_ranking_scores(stage_key)
        stage_banner(
            stage_key,
            "reusing saved outputs" if (loaded_pred is not None and loaded_scores is not None) else "computing fresh",
        )
        if loaded_pred is not None and loaded_scores is not None:
            # Reuse: read saved test predictions and ranking scores.
            wall_load = time.perf_counter()
            predictions = loaded_pred
            ranking_scores = loaded_scores
            wall_load = time.perf_counter() - wall_load
            meta = load_meta(stage_key) or {}
            beta = float(meta.get("validation_fitted_beta", np.nan))
            scorer = make_content_scorer(profiles, features, beta)
            evaluate(
                name,
                predictions,
                scorer,
                training_seconds=0.0,
                prediction_seconds=wall_load,
                choices={"validation_fitted_beta": beta},
                ranking_scores=ranking_scores,
            )
            save_meta(stage_key, {
                "training_seconds": 0.0,
                "prediction_seconds": wall_load,
                "wall_predict": 0.0,
                "wall_rank": 0.0,
                "reused": True,
                "reused_model": False,
                "loaded_from": [
                    f"{stage_key}_predictions.npy",
                    f"{stage_key}_ranking_scores.npy",
                ],
                "computed_fresh": False,
                "validation_fitted_beta": beta,
            })
        else:
            # Fresh: fit beta on validation, predict test, score candidates.
            wall_train = time.perf_counter()
            beta = fit_nonnegative_content_scale(
                validation,
                profiles,
                features,
                values["user_means"],
            )
            wall_train = time.perf_counter() - wall_train

            wall_predict = time.perf_counter()
            predictions = predict_content(
                test,
                profiles,
                features,
                values["user_means"],
                beta,
            )
            wall_predict = time.perf_counter() - wall_predict

            scorer = make_content_scorer(profiles, features, beta)
            wall_rank = time.perf_counter()
            ranking_scores = score_rows(
                scorer,
                test_ranking["users"],
                test_ranking["candidates"],
                name,
            )
            wall_rank = time.perf_counter() - wall_rank

            save_predictions(predictions, stage_key)
            save_ranking_scores(ranking_scores, stage_key)
            save_meta(stage_key, {
                "training_seconds": wall_train,
                "prediction_seconds": wall_predict + wall_rank,
                "wall_predict": wall_predict,
                "wall_rank": wall_rank,
                "reused": False,
                "reused_model": False,
                "loaded_from": [],
                "computed_fresh": True,
                "validation_fitted_beta": beta,
            })
            evaluate(
                name,
                predictions,
                scorer,
                training_seconds=wall_train,
                prediction_seconds=wall_predict + wall_rank,
                choices={"validation_fitted_beta": beta},
                ranking_scores=ranking_scores,
            )

    if "user_knn" in stages:
        loaded_pred = load_predictions("user_knn")
        loaded_scores = load_ranking_scores("user_knn")
        _uk_counts_present = (
            Path(data.PREDICTIONS_DIR) / "user_knn_counts.npy"
        ).exists()
        _uk_reuse = (
            loaded_pred is not None
            and loaded_scores is not None
            and _uk_counts_present
        )
        stage_banner(
            "user_knn",
            "reusing saved outputs"
            if _uk_reuse
            else "computing fresh (predictions + ranking scores + neighbor counts)"
            if (loaded_pred is None or loaded_scores is None)
            else "reusing predictions/scores; regenerating missing neighbor counts",
        )
        if loaded_pred is not None and loaded_scores is not None:
            # Reuse: read saved predictions and ranking scores.
            wall_load = time.perf_counter()
            user_predictions = loaded_pred
            ranking_scores = loaded_scores
            wall_load = time.perf_counter() - wall_load
            # The neighbor-count array (needed by the hybrid) must also be
            # present; if it was lost, regenerate it from the model.
            user_counts_file = (
                Path(data.PREDICTIONS_DIR) / "user_knn_counts.npy"
            )
            counts_knn: UserKNN | None = None
            if not user_counts_file.exists():
                counts_knn = UserKNN(
                    values["ratings_csr"],
                    values["ratings_csc"],
                    values["user_means"],
                    values["movie_means"],
                )
                user_counts = counts_knn.predict_counts(test)
                save_counts(user_counts, "user_knn_counts")
            scorer = (
                counts_knn.score_user
                if counts_knn is not None
                else (lambda user, movies: np.zeros(len(movies)))
            )
            evaluate(
                "User-based k-NN",
                user_predictions,
                scorer,
                training_seconds=0.0,
                prediction_seconds=wall_load,
                choices={
                    "top_k": USER_NEIGHBORS,
                    "minimum_common_movies": MIN_COMMON_ITEMS,
                    "shrinkage": USER_SIMILARITY_SHRINKAGE,
                },
                ranking_scores=ranking_scores,
            )
            save_meta("user_knn", {
                "training_seconds": 0.0,
                "prediction_seconds": wall_load,
                "wall_predict": 0.0,
                "wall_rank": 0.0,
                "reused": True,
                "reused_model": False,
                "loaded_from": [
                    "user_knn_predictions.npy",
                    "user_knn_ranking_scores.npy",
                ],
                "computed_fresh": False,
            })
        else:
            # Fresh: predict test rows + counts, score all candidates.
            user_knn = UserKNN(
                values["ratings_csr"],
                values["ratings_csc"],
                values["user_means"],
                values["movie_means"],
            )
            wall_predict = time.perf_counter()
            user_predictions = user_knn.predict(test)
            user_counts = user_knn.predict_counts(test)
            wall_predict = time.perf_counter() - wall_predict

            wall_rank = time.perf_counter()
            ranking_scores = score_rows(
                user_knn.score_user,
                test_ranking["users"],
                test_ranking["candidates"],
                "User-based k-NN",
            )
            wall_rank = time.perf_counter() - wall_rank

            save_predictions(user_predictions, "user_knn")
            save_counts(user_counts, "user_knn_counts")
            save_ranking_scores(ranking_scores, "user_knn")
            save_meta("user_knn", {
                "training_seconds": 0.0,
                "prediction_seconds": wall_predict + wall_rank,
                "wall_predict": wall_predict,
                "wall_rank": wall_rank,
                "reused": False,
                "reused_model": False,
                "loaded_from": [],
                "computed_fresh": True,
            })
            evaluate(
                "User-based k-NN",
                user_predictions,
                user_knn.score_user,
                training_seconds=0.0,
                prediction_seconds=wall_predict + wall_rank,
                choices={
                    "top_k": USER_NEIGHBORS,
                    "minimum_common_movies": MIN_COMMON_ITEMS,
                    "shrinkage": USER_SIMILARITY_SHRINKAGE,
                },
                ranking_scores=ranking_scores,
            )

    if "movie_knn" in stages:
        loaded_pred = load_predictions("movie_knn")
        loaded_scores = load_ranking_scores("movie_knn")
        stage_banner(
            "movie_knn",
            "reusing saved outputs" if (loaded_pred is not None and loaded_scores is not None) else "computing fresh",
        )
        if loaded_pred is not None and loaded_scores is not None:
            # Reuse: read saved predictions and ranking scores.
            wall_load = time.perf_counter()
            movie_predictions = loaded_pred
            ranking_scores = loaded_scores
            wall_load = time.perf_counter() - wall_load
            evaluate(
                "Movie-based k-NN",
                movie_predictions,
                lambda user, movies: np.zeros(len(movies)),
                training_seconds=0.0,
                prediction_seconds=wall_load,
                choices={
                    "top_k": int(metadata["item_neighbors"]),
                    "minimum_common_raters": int(
                        metadata["minimum_common_raters"]
                    ),
                    "shrinkage": float(metadata["similarity_shrinkage"]),
                },
                ranking_scores=ranking_scores,
            )
            save_meta("movie_knn", {
                "training_seconds": 0.0,
                "prediction_seconds": wall_load,
                "wall_predict": 0.0,
                "wall_rank": 0.0,
                "reused": True,
                "reused_model": False,
                "loaded_from": [
                    "movie_knn_predictions.npy",
                    "movie_knn_ranking_scores.npy",
                ],
                "computed_fresh": False,
            })
        else:
            # Fresh: predict test rows, score all candidates.
            movie_knn = MovieKNN(
                values["ratings_csr"],
                values["movie_similarity"],
                values["user_means"],
                values["movie_means"],
            )
            wall_predict = time.perf_counter()
            movie_predictions = movie_knn.predict(test)
            wall_predict = time.perf_counter() - wall_predict

            wall_rank = time.perf_counter()
            ranking_scores = score_rows(
                movie_knn.score_user,
                test_ranking["users"],
                test_ranking["candidates"],
                "Movie-based k-NN",
            )
            wall_rank = time.perf_counter() - wall_rank

            save_predictions(movie_predictions, "movie_knn")
            save_ranking_scores(ranking_scores, "movie_knn")
            save_meta("movie_knn", {
                "training_seconds": 0.0,
                "prediction_seconds": wall_predict + wall_rank,
                "wall_predict": wall_predict,
                "wall_rank": wall_rank,
                "reused": False,
                "reused_model": False,
                "loaded_from": [],
                "computed_fresh": True,
            })
            evaluate(
                "Movie-based k-NN",
                movie_predictions,
                movie_knn.score_user,
                training_seconds=0.0,
                prediction_seconds=wall_predict + wall_rank,
                choices={
                    "top_k": int(metadata["item_neighbors"]),
                    "minimum_common_raters": int(
                        metadata["minimum_common_raters"]
                    ),
                    "shrinkage": float(metadata["similarity_shrinkage"]),
                },
                ranking_scores=ranking_scores,
            )

    if "mf" in stages:
        loaded_pred = load_predictions("mf")
        loaded_scores = load_ranking_scores("mf")
        _mf_reused_model = _load_mf_model(n_users, n_movies) is not None
        _mf_status = (
            "reusing saved outputs"
            if (loaded_pred is not None and loaded_scores is not None)
            else ("model on disk — reusing model, re-predicting" if _mf_reused_model else "training fresh (SGD grid)")
        )
        stage_banner("mf", _mf_status)

        # mf_scorer_for is a module-level helper (shared with the MF+BPR
        # stage); it is not redefined here.
        if loaded_pred is not None and loaded_scores is not None:
            # Sub-case (1): fully reused. Read predictions + scores; credit
            # the source stage's original training time (not 0) so the MF
            # row keeps its provenance.
            wall_load = time.perf_counter()
            mf_predictions = loaded_pred
            ranking_scores = loaded_scores
            wall_load = time.perf_counter() - wall_load
            meta = load_meta("mf") or {}
            mf_training = float(meta.get("training_seconds", 0.0))
            mf_regularization = float(meta.get("selected_regularization", np.nan))
            evaluate(
                "Biased MF",
                mf_predictions,
                mf_scorer_for({}),
                training_seconds=mf_training,
                prediction_seconds=wall_load,
                choices={"selected_regularization": mf_regularization},
                ranking_scores=ranking_scores,
            )
            save_meta("mf", {
                "training_seconds": mf_training,
                "prediction_seconds": wall_load,
                "wall_predict": 0.0,
                "wall_rank": 0.0,
                "reused": True,
                "reused_model": True,
                "loaded_from": [
                    "mf_predictions.npy",
                    "mf_ranking_scores.npy",
                ],
                "computed_fresh": False,
                "selected_regularization": mf_regularization,
            })
        else:
            # Sub-cases (2) and (3): train (or load) the model, then predict.
            reused_model = False
            wall_train = time.perf_counter()
            loaded_mf = _load_mf_model(n_users, n_movies)
            if loaded_mf is not None:
                # Sub-case (2): model was persisted from a prior run; skip
                # SGD. Credit the recorded training time as provenance.
                mf_model, mf_regularization = loaded_mf
                reused_model = True
                meta = load_meta("mf") or {}
                mf_training = float(meta.get("training_seconds", 0.0))
            else:
                # Sub-case (3): fully fresh; train and select on validation.
                mf_model, mf_regularization = select_mf_model(
                    train,
                    validation,
                    n_users,
                    n_movies,
                    global_mean,
                )
                mf_training = time.perf_counter() - wall_train
                _save_mf_model(
                    mf_model,
                    mf_regularization,
                    n_users,
                    n_movies,
                )
            # Predict test rows and score all candidates from the model.
            wall_predict = time.perf_counter()
            mf_predictions = predict_biased_mf(test, mf_model)
            wall_predict = time.perf_counter() - wall_predict

            wall_rank = time.perf_counter()
            ranking_scores = score_rows(
                mf_scorer_for(mf_model),
                test_ranking["users"],
                test_ranking["candidates"],
                "Biased MF",
            )
            wall_rank = time.perf_counter() - wall_rank

            save_predictions(mf_predictions, "mf")
            save_ranking_scores(ranking_scores, "mf")
            save_meta("mf", {
                "training_seconds": mf_training,
                "prediction_seconds": wall_predict + wall_rank,
                "wall_predict": wall_predict,
                "wall_rank": wall_rank,
                "reused": (loaded_pred is not None),
                "reused_model": reused_model,
                "loaded_from": [],
                "computed_fresh": True,
                "selected_regularization": mf_regularization,
            })
            evaluate(
                "Biased MF",
                mf_predictions,
                mf_scorer_for(mf_model),
                training_seconds=mf_training,
                prediction_seconds=wall_predict + wall_rank,
                choices={"selected_regularization": mf_regularization},
                ranking_scores=ranking_scores,
            )

    if "bpr" in stages:
        loaded_pred = load_predictions("bpr")
        loaded_scores = load_ranking_scores("bpr")
        _bpr_reused_model = _load_bpr_model(n_users, n_movies) is not None  # type: ignore[misc]
        _bpr_status = (
            "reusing saved outputs"
            if (loaded_pred is not None and loaded_scores is not None)
            else ("model on disk — reusing model, re-predicting" if _bpr_reused_model else "training fresh (SGD, longest stage)")
        )
        stage_banner("bpr", _bpr_status)

        def bpr_choices(
            regularization: float,
            validation_ndcg: float,
            mapping: dict[str, float],
        ) -> dict[str, object]:
            return {
                "positive_threshold": BPR_RELEVANCE_THRESHOLD,
                "dynamic_negatives_per_positive":
                    BPR_NEGATIVES_PER_POSITIVE,
                "selected_regularization": regularization,
                "selection_metric": f"fixed_validation_ndcg@{K}",
                "selected_validation_ndcg": validation_ndcg,
                "regularization_tie_tolerance":
                    BPR_SELECTION_TOLERANCE,
                "regularization_tie_policy":
                    "prefer stronger regularization",
                "negative_draws_are_independent": True,
                "negative_draws_may_repeat": True,
                "affine_mapping": mapping,
                "rating_metrics_are_post_hoc": True,
            }

        if loaded_pred is not None and loaded_scores is not None:
            # Sub-case (1): fully reused. Read predictions + scores; credit
            # the source stage's original training time (not 0) so the BPR
            # row keeps its ~10.8 h provenance.
            wall_load = time.perf_counter()
            bpr_predictions = loaded_pred
            ranking_scores = loaded_scores
            wall_load = time.perf_counter() - wall_load
            meta = load_meta("bpr") or {}
            bpr_training = float(meta.get("training_seconds", 0.0))
            bpr_regularization = float(meta.get("selected_regularization", np.nan))
            bpr_validation_ndcg = float(meta.get("selected_validation_ndcg", np.nan))
            bpr_affine = meta.get("affine_mapping", {"slope": np.nan, "intercept": np.nan})
            evaluate(
                "BPR-MF thresholded positives",
                bpr_predictions,
                lambda user, movies: np.zeros(len(movies)),
                training_seconds=bpr_training,
                prediction_seconds=wall_load,
                choices=bpr_choices(
                    bpr_regularization, bpr_validation_ndcg, bpr_affine
                ),
                ranking_scores=ranking_scores,
            )
            save_meta("bpr", {
                "training_seconds": bpr_training,
                "prediction_seconds": wall_load,
                "wall_predict": 0.0,
                "wall_rank": 0.0,
                "reused": True,
                "reused_model": True,
                "loaded_from": [
                    "bpr_predictions.npy",
                    "bpr_ranking_scores.npy",
                ],
                "computed_fresh": False,
                "selected_regularization": bpr_regularization,
                "selected_validation_ndcg": bpr_validation_ndcg,
                "affine_mapping": bpr_affine,
            })
        else:
            # Sub-cases (2) and (3): load or train the model, then predict.
            reused_model = False
            bpr_meta = load_meta("bpr") or {}
            stored_affine = bpr_meta.get("affine_mapping")
            stored_ndcg = bpr_meta.get("selected_validation_ndcg")

            wall_train = time.perf_counter()
            loaded_bpr = _load_bpr_model(n_users, n_movies)
            if loaded_bpr is not None:
                # Sub-case (2): model persisted from a prior run; skip the
                # per-triplet SGD. Prefer the affine map stored in the model
                # file (newer runs); fall back to meta, then to a fresh fit,
                # so an older model file without the map still reproduces
                # predict exactly.
                bpr_model, bpr_regularization, loaded_affine = loaded_bpr
                reused_model = True
                bpr_training = float(bpr_meta.get("training_seconds", 0.0))
                bpr_validation_ndcg = float(stored_ndcg)
                if loaded_affine is not None:
                    bpr_affine = loaded_affine
                elif stored_affine is not None:
                    bpr_affine = {
                        "slope": float(stored_affine["slope"]),
                        "intercept": float(stored_affine["intercept"]),
                    }
                else:
                    # Neither the model file nor meta carries the mapping:
                    # re-fit it from the validation scores (deterministic for
                    # this model).
                    _val_scores = predict_bpr_scores(validation, bpr_model)
                    bpr_affine = fit_nonnegative_affine_mapping(
                        _val_scores, validation[:, 2]
                    )
            else:
                # Sub-case (3): fully fresh; train, select, fit mapping.
                (
                    bpr_model,
                    bpr_regularization,
                    bpr_validation_ndcg,
                ) = select_bpr_model(
                    train,
                    validation_ranking,
                    n_users,
                    n_movies,
                )
                validation_scores = predict_bpr_scores(validation, bpr_model)
                bpr_affine = fit_nonnegative_affine_mapping(
                    validation_scores,
                    validation[:, 2],
                )
                bpr_training = time.perf_counter() - wall_train
                _save_bpr_model(
                    bpr_model,
                    bpr_regularization,
                    n_users,
                    n_movies,
                    affine_mapping=bpr_affine,
                )

            # Predict test ratings and score all candidates from the model.
            wall_predict = time.perf_counter()
            bpr_predictions = map_bpr_to_ratings(
                test,
                bpr_model,
                bpr_affine,
            )
            wall_predict = time.perf_counter() - wall_predict

            wall_rank = time.perf_counter()
            ranking_scores = score_rows(
                lambda user, movies: bpr_score_user(bpr_model, user, movies),
                test_ranking["users"],
                test_ranking["candidates"],
                "BPR-MF thresholded positives",
            )
            wall_rank = time.perf_counter() - wall_rank

            save_predictions(bpr_predictions, "bpr")
            save_ranking_scores(ranking_scores, "bpr")
            save_meta("bpr", {
                "training_seconds": bpr_training,
                "prediction_seconds": wall_predict + wall_rank,
                "wall_predict": wall_predict,
                "wall_rank": wall_rank,
                "reused": (loaded_pred is not None),
                "reused_model": reused_model,
                "loaded_from": [],
                "computed_fresh": True,
                "selected_regularization": bpr_regularization,
                "selected_validation_ndcg": bpr_validation_ndcg,
                "affine_mapping": bpr_affine,
            })
            evaluate(
                "BPR-MF thresholded positives",
                bpr_predictions,
                lambda user, movies: bpr_score_user(bpr_model, user, movies),
                training_seconds=bpr_training,
                prediction_seconds=wall_predict + wall_rank,
                choices=bpr_choices(
                    bpr_regularization, bpr_validation_ndcg, bpr_affine
                ),
                ranking_scores=ranking_scores,
            )

    # ------------------------------------------------------------------
    # Hybrid: runs LAST so it can reuse the genome_content and user_knn
    # outputs (predictions, counts, and ranking scores) saved above.
    # ------------------------------------------------------------------
    if "hybrid" in stages:
        pred_dir = Path(data.PREDICTIONS_DIR)
        # Test-row inputs: genome content, user-knn predictions, user-knn
        # counts. The hybrid blends these; if any is absent, compute and
        # save it so the next run reuses it.
        genome_content_pred = load_predictions("genome_content")
        user_knn_pred = load_predictions("user_knn")
        user_knn_counts = None
        user_knn_counts_file = pred_dir / "user_knn_counts.npy"
        if user_knn_counts_file.exists():
            user_knn_counts = np.load(user_knn_counts_file)

        loaded_from: list[str] = []
        wall_load = time.perf_counter()
        if genome_content_pred is not None:
            loaded_from.append("genome_content_predictions.npy")
        if user_knn_pred is not None:
            loaded_from.append("user_knn_predictions.npy")
        if user_knn_counts is not None:
            loaded_from.append("user_knn_counts.npy")

        user_knn_model: UserKNN | None = None

        if genome_content_pred is None:
            # Compute genome content test predictions and save them.
            beta_tmp = fit_nonnegative_content_scale(
                validation,
                values["genome_profiles"],
                values["genome_vectors"],
                values["user_means"],
            )
            genome_content_pred = predict_content(
                test,
                values["genome_profiles"],
                values["genome_vectors"],
                values["user_means"],
                beta_tmp,
            )
            save_predictions(genome_content_pred, "genome_content")
            loaded_from.append("genome_content_predictions.npy (computed)")

        if user_knn_pred is None:
            # Compute user-knn test predictions + counts and save them.
            user_knn_model = UserKNN(
                values["ratings_csr"],
                values["ratings_csc"],
                values["user_means"],
                values["movie_means"],
            )
            user_knn_pred = user_knn_model.predict(test)
            user_knn_counts = user_knn_model.predict_counts(test)
            save_predictions(user_knn_pred, "user_knn")
            save_counts(user_knn_counts, "user_knn_counts")
            loaded_from.append("user_knn_predictions.npy (computed)")
            loaded_from.append("user_knn_counts.npy (computed)")

        if user_knn_counts is None:
            # Counts were lost but predictions exist: regenerate them.
            if user_knn_model is None:
                user_knn_model = UserKNN(
                    values["ratings_csr"],
                    values["ratings_csc"],
                    values["user_means"],
                    values["movie_means"],
                )
            user_knn_counts = user_knn_model.predict_counts(test)
            save_counts(user_knn_counts, "user_knn_counts")
            loaded_from.append("user_knn_counts.npy (computed)")

        # Ranking-score inputs: genome content ranking scores + user-knn
        # ranking scores (both 2-D). If absent, compute and save them.
        genome_content_scores = load_ranking_scores("genome_content")
        user_knn_scores = load_ranking_scores("user_knn")
        if genome_content_scores is not None:
            loaded_from.append("genome_content_ranking_scores.npy")
        if user_knn_scores is not None:
            loaded_from.append("user_knn_ranking_scores.npy")

        stage_banner(
            "hybrid",
            "blending saved parent outputs"
            if (genome_content_scores is not None and user_knn_scores is not None)
            else "recomputing missing parent ranking scores",
        )

        if genome_content_scores is None:
            beta_for_rank = fit_nonnegative_content_scale(
                validation,
                values["genome_profiles"],
                values["genome_vectors"],
                values["user_means"],
            )
            scorer = make_content_scorer(
                values["genome_profiles"],
                values["genome_vectors"],
                beta_for_rank,
            )
            genome_content_scores = score_rows(
                scorer,
                test_ranking["users"],
                test_ranking["candidates"],
                "Tag Genome content (for hybrid)",
            )
            save_ranking_scores(genome_content_scores, "genome_content")
            loaded_from.append("genome_content_ranking_scores.npy (computed)")

        if user_knn_scores is None:
            if user_knn_model is None:
                user_knn_model = UserKNN(
                    values["ratings_csr"],
                    values["ratings_csc"],
                    values["user_means"],
                    values["movie_means"],
                )
            user_knn_scores = score_rows(
                user_knn_model.score_user,
                test_ranking["users"],
                test_ranking["candidates"],
                "User k-NN (for hybrid)",
            )
            save_ranking_scores(user_knn_scores, "user_knn")
            loaded_from.append("user_knn_ranking_scores.npy (computed)")
        wall_load = time.perf_counter() - wall_load

        # Validation components for alpha selection. Reuse saved val arrays
        # if present; otherwise compute and save them.
        val_content_file = pred_dir / "hybrid_val_content.npy"
        val_userknn_file = pred_dir / "hybrid_val_userknn.npy"
        val_counts_file = pred_dir / "hybrid_val_counts.npy"
        wall_val_load = time.perf_counter()
        if (
            val_content_file.exists()
            and val_userknn_file.exists()
            and val_counts_file.exists()
        ):
            val_content = np.load(val_content_file)
            val_userknn = np.load(val_userknn_file)
            val_counts = np.load(val_counts_file)
            loaded_from.extend([
                "hybrid_val_content.npy",
                "hybrid_val_userknn.npy",
                "hybrid_val_counts.npy",
            ])
            val_computed = False
        else:
            val_content = None
            val_computed = True
        wall_val_load = time.perf_counter() - wall_val_load

        # Fit the content scale (needed for validation content and for the
        # content ranking scorer if it had to be computed above).
        wall_beta = time.perf_counter()
        genome_beta = fit_nonnegative_content_scale(
            validation,
            values["genome_profiles"],
            values["genome_vectors"],
            values["user_means"],
        )
        wall_beta = time.perf_counter() - wall_beta

        # If validation components were not reused, compute and save them.
        if val_computed:
            wall_val_compute = time.perf_counter()
            val_users = validation[:, 0].astype(np.int64)
            val_movies = validation[:, 1].astype(np.int64)
            val_content = (
                values["user_means"][val_users]
                + genome_beta
                * np.sum(
                    values["genome_profiles"][val_users]
                    * values["genome_vectors"][val_movies],
                    axis=1,
                )
            )
            if user_knn_model is None:
                user_knn_model = UserKNN(
                    values["ratings_csr"],
                    values["ratings_csc"],
                    values["user_means"],
                    values["movie_means"],
                )
            val_userknn = user_knn_model.predict(validation)
            val_counts = user_knn_model.predict_counts(validation)
            np.save(val_content_file, val_content)
            np.save(val_userknn_file, val_userknn)
            np.save(val_counts_file, val_counts)
            wall_val_compute = time.perf_counter() - wall_val_compute
            loaded_from.extend([
                "hybrid_val_content.npy (computed)",
                "hybrid_val_userknn.npy (computed)",
                "hybrid_val_counts.npy (computed)",
            ])
        else:
            wall_val_compute = 0.0

        # Select alpha on validation (confidence rule applied inside grid).
        wall_alpha = time.perf_counter()
        alpha = choose_hybrid_alpha(
            validation[:, 2],
            val_content,
            val_userknn,
            val_counts,
        )
        wall_alpha = time.perf_counter() - wall_alpha

        # Test rating predictions: confidence blend of content & userknn.
        # Where the k-NN had usable neighbours (count > 0) the row is the
        # convex blend; where it had none (count == 0) the row uses content
        # alone. This is the same rule GenomeHybrid.predict applies.
        wall_blend = time.perf_counter()
        hybrid_predictions = (
            alpha * genome_content_pred
            + (1.0 - alpha) * user_knn_pred
        )
        hybrid_predictions[user_knn_counts == 0] = (
            genome_content_pred[user_knn_counts == 0]
        )
        wall_blend = time.perf_counter() - wall_blend

        # Test ranking scores: blend of the two parents' ranking scores.
        # Candidates are warm-start (in-training) movies so the k-NN count
        # is always > 0 and the plain convex blend applies.
        wall_rank_blend = time.perf_counter()
        ranking_scores = (
            alpha * genome_content_scores
            + (1.0 - alpha) * user_knn_scores
        )
        wall_rank_blend = time.perf_counter() - wall_rank_blend

        # Total prediction wall: load + blend + rank-blend.
        prediction_time = wall_load + wall_blend + wall_rank_blend
        # Total training wall: beta fit + val load + val compute + alpha sel.
        training_time = wall_beta + wall_val_load + wall_val_compute + wall_alpha

        save_predictions(hybrid_predictions, "hybrid")
        save_ranking_scores(ranking_scores, "hybrid")
        save_meta("hybrid", {
            "training_seconds": training_time,
            "prediction_seconds": prediction_time,
            "wall_predict": wall_blend,
            "wall_rank": wall_rank_blend,
            "reused": len(loaded_from) > 0,
            "reused_model": False,
            "loaded_from": loaded_from,
            "computed_fresh": len(loaded_from) == 0,
            "validation_fitted_beta": genome_beta,
            "validation_selected_alpha": alpha,
        })
        evaluate(
            "Tag Genome weighted hybrid",
            hybrid_predictions,
            lambda user, movies: np.zeros(len(movies)),
            training_seconds=training_time,
            prediction_seconds=prediction_time,
            choices={
                "validation_fitted_beta": genome_beta,
                "validation_selected_alpha": alpha,
                "alpha_grid": HYBRID_ALPHA_GRID.tolist(),
                "knn_confidence_fallback": "content when 0 usable neighbors",
            },
            ranking_scores=ranking_scores,
        )

    # ------------------------------------------------------------------
    # MF+BPR hybrid: runs LAST so it can reuse the mf and bpr outputs
    # (test predictions, test ranking scores, and the persisted models)
    # saved above. Dual-headed: one lambda for the rating blend (chosen on
    # validation RMSE) and one for the per-row z-scored ranking blend
    # (chosen on validation NDCG@10). See the essay "Hybridization" section.
    # ------------------------------------------------------------------
    if "mf_bpr" in stages:
        pred_dir = Path(data.PREDICTIONS_DIR)
        # Test-row inputs: MF rating predictions + BPR (affine-mapped) rating
        # predictions. Both are saved 1-D arrays in test-row order.
        mf_pred = load_predictions("mf")
        bpr_pred = load_predictions("bpr")
        # Test ranking inputs: MF raw latent scores + BPR raw latent scores.
        # Both are saved 2-D arrays (n_rank_users, n_rank_candidates).
        mf_scores = load_ranking_scores("mf")
        bpr_scores = load_ranking_scores("bpr")

        loaded_from: list[str] = []
        wall_load = time.perf_counter()
        if mf_pred is not None:
            loaded_from.append("mf_predictions.npy")
        if bpr_pred is not None:
            loaded_from.append("bpr_predictions.npy")
        if mf_scores is not None:
            loaded_from.append("mf_ranking_scores.npy")
        if bpr_scores is not None:
            loaded_from.append("bpr_ranking_scores.npy")
        wall_load = time.perf_counter() - wall_load

        # The two latent-factor models must be available to score the
        # validation rows (for lambda selection). If either is missing,
        # this stage cannot run: it is a cross-stage blend and trains
        # nothing of its own.
        mf_model_loaded = _load_mf_model(n_users, n_movies)
        bpr_model_loaded = _load_bpr_model(n_users, n_movies)
        mf_model, mf_regularization = mf_model_loaded or (None, float("nan"))
        bpr_model, bpr_regularization, bpr_affine_loaded = (
            bpr_model_loaded or (None, float("nan"), None)
        )

        if mf_model is None or bpr_model is None:
            stage_banner(
                "mf_bpr",
                "SKIPPED: a parent model (mf_model.npz / bpr_model.npz) is missing",
            )
            print(
                "  The MF+BPR stage needs both persisted latent-factor "
                "models to score the validation rows. Run the 'mf' and "
                "'bpr' stages first."
            )
            # Record a results block so downstream consumers (visualizer)
            # can detect the stage as absent rather than as zero.
            results["MF+BPR hybrid"] = {
                "rating": None,
                "ranking": None,
                "timing": {
                    "training_seconds": 0.0,
                    "prediction_seconds": 0.0,
                },
                "selected_choices": {
                    "skipped_reason": "missing parent model",
                },
            }
            print(
                f"\n=== All {len(stages)} requested stage(s) complete ==="
            )
            return results

        # Affine mapping for the BPR rating term: prefer the value stored in
        # the model file (authoritative, Y's spec); fall back to the BPR
        # stage's meta; else re-fit from validation scores so an older model
        # file without the map still yields the correct (non-degenerate)
        # slope instead of collapsing the rating head to pure MF.
        bpr_meta = load_meta("bpr") or {}
        if bpr_affine_loaded is not None:
            bpr_affine = bpr_affine_loaded
        else:
            stored = bpr_meta.get("affine_mapping")
            if stored is not None and "slope" in stored:
                bpr_affine = {
                    "slope": float(stored["slope"]),
                    "intercept": float(stored["intercept"]),
                }
            else:
                _val_scores = predict_bpr_scores(validation, bpr_model)
                bpr_affine = fit_nonnegative_affine_mapping(
                    _val_scores, validation[:, 2]
                )

        stage_banner(
            "mf_bpr",
            "dual-headed blend of saved MF/BPR outputs",
        )

        # ---- Validation components for both lambdas (computed once) ----
        # Rating head validation: MF rating predictions + BPR affine-mapped
        # rating predictions on the shared validation rows.
        # Ranking head validation: MF raw latent scores + BPR raw latent
        # scores on the fixed validation candidate rows.
        wall_val = time.perf_counter()
        val_mf_rating = predict_biased_mf(validation, mf_model)
        val_bpr_rating = map_bpr_to_ratings(validation, bpr_model, bpr_affine)

        val_users = validation_ranking["users"]
        val_candidates = validation_ranking["candidates"]
        val_pos = validation_ranking["positive_indices"]
        val_mf_scores = score_rows(
            mf_scorer_for(mf_model), val_users, val_candidates,
            "MF (validation, for MF+BPR)",
        )
        val_bpr_scores = score_rows(
            lambda u, m: bpr_score_user(bpr_model, u, m),
            val_users, val_candidates,
            "BPR (validation, for MF+BPR)",
        )
        wall_val = time.perf_counter() - wall_val

        # ---- Select the two lambdas ----
        # Save the validation components so the λ-sweep figure (visualizer
        # fig 08) can redraw both heads' curves without re-scoring.
        np.save(pred_dir / "mf_bpr_val_mf_rating.npy", val_mf_rating)
        np.save(pred_dir / "mf_bpr_val_bpr_rating.npy", val_bpr_rating)
        np.save(pred_dir / "mf_bpr_val_actual.npy", validation[:, 2])

        wall_alpha_r = time.perf_counter()
        lambda_rating = choose_rating_lambda(
            validation[:, 2], val_mf_rating, val_bpr_rating,
        )
        wall_alpha_r = time.perf_counter() - wall_alpha_r

        wall_alpha_k = time.perf_counter()
        lambda_ranking = choose_ranking_lambda(
            val_mf_scores, val_bpr_scores,
            val_users, val_candidates, val_pos,
        )
        wall_alpha_k = time.perf_counter() - wall_alpha_k

        # ---- Test rating predictions: rating-head blend ----
        wall_blend = time.perf_counter()
        if mf_pred is None:
            mf_pred = predict_biased_mf(test, mf_model)
            save_predictions(mf_pred, "mf")
            loaded_from.append("mf_predictions.npy (computed)")
        if bpr_pred is None:
            bpr_pred = map_bpr_to_ratings(test, bpr_model, bpr_affine)
            save_predictions(bpr_pred, "bpr")
            loaded_from.append("bpr_predictions.npy (computed)")
        hybrid_predictions = (
            lambda_rating * mf_pred
            + (1.0 - lambda_rating) * bpr_pred
        )
        wall_blend = time.perf_counter() - wall_blend

        # ---- Test ranking scores: per-row z-scored ranking-head blend ----
        wall_rank_blend = time.perf_counter()
        if mf_scores is None:
            mf_scores = score_rows(
                mf_scorer_for(mf_model),
                test_ranking["users"],
                test_ranking["candidates"],
                "MF (test, for MF+BPR)",
            )
            save_ranking_scores(mf_scores, "mf")
            loaded_from.append("mf_ranking_scores.npy (computed)")
        if bpr_scores is None:
            bpr_scores = score_rows(
                lambda u, m: bpr_score_user(bpr_model, u, m),
                test_ranking["users"],
                test_ranking["candidates"],
                "BPR (test, for MF+BPR)",
            )
            save_ranking_scores(bpr_scores, "bpr")
            loaded_from.append("bpr_ranking_scores.npy (computed)")
        z_mf = zscore_rows(mf_scores)
        z_bpr = zscore_rows(bpr_scores)
        ranking_scores = (
            lambda_ranking * z_mf + (1.0 - lambda_ranking) * z_bpr
        )
        wall_rank_blend = time.perf_counter() - wall_rank_blend

        training_time = wall_val + wall_alpha_r + wall_alpha_k
        prediction_time = wall_load + wall_blend + wall_rank_blend

        save_predictions(hybrid_predictions, "mf_bpr")
        save_ranking_scores(ranking_scores, "mf_bpr")
        save_meta("mf_bpr", {
            "training_seconds": training_time,
            "prediction_seconds": prediction_time,
            "wall_predict": wall_blend,
            "wall_rank": wall_rank_blend,
            "reused": len(loaded_from) > 0,
            "reused_model": True,
            "loaded_from": loaded_from,
            "computed_fresh": len(loaded_from) == 0,
            "lambda_rating": lambda_rating,
            "lambda_ranking": lambda_ranking,
            "lambda_grid": LAMBDA_GRID.tolist(),
            "rating_head_parent_rmse_mf":
                float(np.sqrt(np.mean((val_mf_rating - validation[:, 2]) ** 2))),
            "rating_head_parent_rmse_bpr_mapped":
                float(np.sqrt(np.mean((val_bpr_rating - validation[:, 2]) ** 2))),
            "ranking_head_parent_ndcg_mf":
                float(ranking_metrics(
                    "mf val", val_users, val_candidates, val_pos,
                    lambda u, m: np.zeros(len(m)), scores=val_mf_scores,
                )["ndcg@k"]),
            "ranking_head_parent_ndcg_bpr":
                float(ranking_metrics(
                    "bpr val", val_users, val_candidates, val_pos,
                    lambda u, m: np.zeros(len(m)), scores=val_bpr_scores,
                )["ndcg@k"]),
            "bpr_affine": bpr_affine,
            "mf_regularization": mf_regularization,
            "bpr_regularization": bpr_regularization,
        })
        evaluate(
            "MF+BPR hybrid",
            hybrid_predictions,
            lambda user, movies: np.zeros(len(movies)),
            training_seconds=training_time,
            prediction_seconds=prediction_time,
            choices={
                "dual_headed": True,
                "lambda_rating": lambda_rating,
                "lambda_ranking": lambda_ranking,
                "lambda_grid": LAMBDA_GRID.tolist(),
                "rating_head_selection": "validation RMSE",
                "ranking_head_selection": "validation NDCG@10",
                "ranking_standardisation": "per-row z-score",
                "bpr_affine": bpr_affine,
            },
            ranking_scores=ranking_scores,
        )

    print(f"\n=== All {len(stages)} requested stage(s) complete ===")
    return results


def save_predictions(
    predictions: np.ndarray,
    stage: str,
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    """Save a 1D prediction array as a .npy file in the predictions directory."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    filename = destination / f"{stage}_predictions.npy"
    np.save(filename, predictions)
    print(f"Saved predictions to {filename}")


def save_counts(
    counts: np.ndarray,
    stage: str,
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    """Save a 1D integer array as a plain .npy file (no _predictions suffix).

    The user-kNN neighbor-count array is saved as {stage}.npy (e.g.
    user_knn_counts.npy) to match both the Python read path and the C
    implementation, which expects that exact name. save_predictions is not
    used here because it always appends _predictions.
    """
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    filename = destination / f"{stage}.npy"
    np.save(filename, np.asarray(counts, dtype=np.int32))
    print(f"Saved counts to {filename}")


# ---------------------------------------------------------------------------
# Cross-stage output reuse
#
# Each stage, when it computes fresh, saves its test-row predictions, its
# ranking-candidate scores, and a small sidecar meta file. On a later run a
# stage that finds those artifacts present re-reads them instead of
# re-fitting / re-predicting / re-scoring, which is what makes the (otherwise
# expensive) hybrid stage cheap: it blends already-computed arrays.
#
# Timing is "credit-the-consumer": a stage that reuses saved work is charged
# the wall time of loading that work (prediction_seconds = load time,
# training_seconds = 0 for stateless stages, or the source stage's recorded
# training time for MF/BPR so a reused BPR row keeps its provenance). The
# results JSON schema is unchanged (timing still has training_seconds +
# prediction_seconds only); the audit trail lives in {stage}_meta.json.
# ---------------------------------------------------------------------------


def load_predictions(
    stage: str,
    path: str | Path = data.PREDICTIONS_DIR,
) -> np.ndarray | None:
    """Load a stage's saved test-row predictions, or None if absent.

    Matching loader for save_predictions: reads {stage}_predictions.npy.
    """
    filename = Path(path) / f"{stage}_predictions.npy"
    if not filename.exists():
        return None
    return np.load(filename)


def save_ranking_scores(
    scores: np.ndarray,
    stage: str,
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    """Save a stage's 2-D ranking-candidate scores as a .npy file.

    Shape is (n_rank_users, n_rank_candidates), one row per evaluated
    ranking user. This is the per-candidate score vector the ranking
    metrics consume; saving it lets the hybrid (and re-runs) reuse the
    expensive scoring pass instead of redoing it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    filename = destination / f"{stage}_ranking_scores.npy"
    np.save(filename, scores)
    print(f"Saved ranking scores to {filename}")


def load_ranking_scores(
    stage: str,
    path: str | Path = data.PREDICTIONS_DIR,
) -> np.ndarray | None:
    """Load a stage's saved ranking-candidate scores, or None if absent.

    Matching loader for save_ranking_scores: reads {stage}_ranking_scores.npy.
    """
    filename = Path(path) / f"{stage}_ranking_scores.npy"
    if not filename.exists():
        return None
    return np.load(filename)


def save_meta(
    stage: str,
    meta: dict[str, object],
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    """Write a stage's timing / provenance sidecar {stage}_meta.json."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    filename = destination / f"{stage}_meta.json"
    with filename.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)
    print(f"Saved meta to {filename}")


def load_meta(
    stage: str,
    path: str | Path = data.PREDICTIONS_DIR,
) -> dict[str, object] | None:
    """Read a stage's {stage}_meta.json, or None if absent."""
    filename = Path(path) / f"{stage}_meta.json"
    if not filename.exists():
        return None
    with filename.open("r", encoding="utf-8") as file:
        return json.load(file)


def mf_scorer_for(
    model: dict[str, np.ndarray | float],
) -> Callable[[int, np.ndarray], np.ndarray]:
    """Build a per-user MF candidate scorer from a loaded model dict.

    Module-level (not a closure inside the 'mf' stage) so cross-stage
    consumers such as the MF+BPR hybrid can score arbitrary candidate rows
    with the same MF model, regardless of which stages are requested.
    """
    def scorer(user: int, movies: np.ndarray) -> np.ndarray:
        interactions = np.column_stack(
            (
                np.full(len(movies), user, dtype=np.int64),
                movies,
                np.zeros(len(movies)),
            )
        )
        return predict_biased_mf(interactions, model)
    return scorer


def score_rows(
    scorer: Callable[[int, np.ndarray], np.ndarray],
    users: np.ndarray,
    candidates: np.ndarray,
    stage_name: str,
) -> np.ndarray:
    """Score every ranking-candidate row with scorer and stack to (rows, cols).

    This is the expensive per-candidate pass (the whole reason k-NN / MF /
    BPR ranking is slow). Computing it once, outside the metrics, means it is
    timed honestly and reusable. scorer(user, candidate_row) -> 1-D scores.
    """
    return np.asarray(
        [scorer(int(user), candidates[row]) for row, user in enumerate(users)],
        dtype=np.float64,
    )


def save_results(
    results: dict[str, dict[str, object]],
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    filename = destination / "results.json"
    with filename.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved results to {filename}")


def save_stage_results(
    results: dict[str, dict[str, object]],
    stage: str,
    path: str | Path = data.PREDICTIONS_DIR,
) -> None:
    """Save results for a single stage to a separate file."""
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    filename = destination / f"results_{stage}.json"
    with filename.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Saved stage results to {filename}")


def merge_stage_results(
    path: str | Path = data.PREDICTIONS_DIR,
    stages: tuple[str, ...] = STAGE_NAMES,
) -> dict[str, dict[str, object]]:
    """Merge individual stage result files into a single results dict."""
    destination = Path(path)
    merged: dict[str, dict[str, object]] = {}
    for stage in stages:
        filename = destination / f"results_{stage}.json"
        if filename.exists():
            with filename.open("r", encoding="utf-8") as file:
                stage_results = json.load(file)
            merged.update(stage_results)
            print(f"Loaded {stage} results from {filename}")
        else:
            print(f"Warning: {filename} not found, skipping {stage}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MovieLens recommender algorithms."
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_NAMES,
        help="Run only a single algorithm stage.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_NAMES,
        default=list(STAGE_NAMES),
        help="Run a subset of algorithm stages (default: all).",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge individual stage result files into results.json.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default=data.OUTPUT_DIR,
        help="Path to the ProcessedData directory.",
    )
    args = parser.parse_args()

    if args.merge:
        stages_to_merge = tuple(args.stages)
        merged = merge_stage_results(data.PREDICTIONS_DIR, stages_to_merge)
        save_results(merged, data.PREDICTIONS_DIR)
        return

    if args.stage:
        stages = (args.stage,)
    else:
        stages = tuple(args.stages)

    results = run_all(args.path, stages)
    if len(stages) == 1:
        save_stage_results(results, stages[0], data.PREDICTIONS_DIR)
    else:
        save_results(results, data.PREDICTIONS_DIR)


if __name__ == "__main__":
    main()
