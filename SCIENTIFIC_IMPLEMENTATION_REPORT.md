## Scientific implementation report

This report documents the recommender-system algorithms actually implemented
in `algorithms.py` and `prepare_data.py`. It describes the closest scientific
design for each implemented algorithm, distinguishes standard algorithms from
adaptations, regularized variants, hybrids, and post-hoc extensions, and
records deviations, optional implementation choices, and citations.

The implementation contains nine evaluated methods, but not all nine are
standalone algorithms copied directly from one publication.

This report subsumes the earlier `recommender_citations.md` reference
document.

## Implementation status summary

| # | Result name | Classification | Canonical or modified? |
|---|---|---|---|
| 1 | Global mean | Rating baseline | Standard basic baseline |
| 2 | Genre content profile | Content-based | Custom residual-weighted Rocchio-like variation |
| 3 | Tag Genome content profile | Content-based | Custom variation using Tag Genome features |
| 4 | Tag Genome weighted hybrid | Weighted ensemble | Content x User-k-NN ensemble with a confidence fallback |
| 5 | User-based k-NN | Neighborhood collaborative filtering | Regularized implementation variation |
| 6 | Item-based k-NN | Neighborhood collaborative filtering | Sparse, shrunk variation of adjusted-cosine item CF |
| 7 | Biased MF | Latent-factor rating model | Standard model with implementation and selection choices |
| 8 | BPR-MF thresholded positives | Pairwise latent-factor ranker | Explicit-feedback adaptation plus post-hoc rating mapping |
| 9 | MF+BPR hybrid | Dual-headed rank/rating ensemble | MF x BPR blend: rating head on raw ratings, ranking head on per-row z-scores |

The methods requiring the clearest qualification are:

- `Genre content profile` is not a direct implementation of Salton et al.
  Its residual-weighted user profile and validation-fitted rating scale are
  recommender-specific implementation choices.
- `Tag Genome content profile` is the same custom profile method with Tag
  Genome feature vectors.
- `Tag Genome weighted hybrid` is a weighted ensemble of the Tag Genome
  content profile and the User-based k-NN, with a confidence fallback to the
  content prediction where the k-NN had no usable neighbours. It is not an
  implementation of Fab or another single published hybrid.
- `User-based k-NN` and `Item-based k-NN` include overlap thresholds,
  similarity shrinkage, signed similarities, absolute normalization, and
  fallback rules.
- `BPR-MF thresholded positives` adapts explicit ratings to implicit BPR.
  Its nonnegative affine rating mapping is a post-hoc extension and is not
  part of the BPR algorithm.
- `MF+BPR hybrid` is a dual-headed blend of Biased MF and BPR-MF. It trains
  nothing; it reuses the two parents' saved predictions, ranking scores, and
  persisted models. The rating head blends raw rating predictions with a
  single $\lambda$ selected on validation RMSE; the ranking head blends
  per-row z-standardised candidate scores with a second $\lambda$ selected on
  validation NDCG@10. The per-row z-scoring is a standardisation choice
  (motivated by the two algorithms' incompatible raw scales) and is not part
  of either parent algorithm.

## Common experimental design

The default preparation uses the complete MovieLens 33M `ratings.csv` file.
Movies are restricted to the catalog for which MovieLens Tag Genome features
are available. This common-catalog restriction ensures that every algorithm
is evaluated on the same movie domain, but it means the experiment is not a
full-catalog MovieLens experiment.

Users must have at least five retained interactions, yielding at least
three training interactions per user. Interactions are ordered
by timestamp within each user:

- The latest interaction is test data.
- The second-latest interaction is validation data.
- Earlier interactions are training data.

Equal timestamps are ordered by original `ratings.csv` row order. This
deterministic tie-break does not imply that the corresponding ratings
occurred in that order.

### Validation-set usage

The validation set is shared across all algorithms. Validation interactions
are never used as base model-fitting interactions: they do not contribute to
profile construction, neighborhood similarities, or stochastic gradient
updates.

However, the validation set is used for model selection and for fitting
predictive calibration parameters:

- Nonnegative rating-scale fitting for the content models.
- Hybrid blend-weight selection.
- Matrix-factorization regularization selection.
- BPR regularization selection through fixed validation NDCG@10.
- The BPR post-hoc affine rating mapping.

The validation set therefore participates in model selection and parameter
calibration, while base interaction-level model fitting uses only training
interactions.

### Chronological holdout

Every retained user contributes:

- All but the latest two interactions to training.
- The second-latest interaction to validation.
- The latest interaction to testing.

Validation and test are distinct one-interaction-per-user splits.

### Run-time output reuse and timing

The algorithms are described by their mathematical behaviour; this section
records an implementation-level property that does not change any formula or
metric. To avoid recomputing expensive work, each stage persists the
intermediate quantities it produces — its test-row rating predictions, its
full ranking-candidate score matrix, and (for the matrix-factorization and
BPR stages) its trained model. On a later run, a stage that finds these
artifacts present reads them back instead of re-fitting, re-predicting, or
re-scoring. The Tag Genome hybrid, which runs last, consumes the genome
content and user-kNN outputs in this way, so it never re-runs the k-NN over
the ranking candidates.

This reuse does not alter any result: the metrics are computed from the same
scores, merely loaded rather than recomputed, and the persisted MF/BPR models
are bit-for-bit the trained parameters. Reported timing is *credit-the-consumer*:
a stage that reuses saved work is charged the wall time of loading it, and the
MF/BPR stages retain their original training-time provenance when their saved
model is reused. The full file list, ordering, and timing rule are documented
in `ARCHITECTURE.md` ("Cross-stage output reuse" and "Timing interpretation").

### Rating metrics

Raw rating metrics are primary. Predictions are also clipped to the valid
rating range for separately named bounded metrics. Non-finite predictions
cause evaluation to fail rather than being silently ignored.

## 1. Global-mean rating baseline

### Implemented model

The prediction for every user-item pair is:

$$
\hat r_{ui}=\mu,
$$

where $\mu$ is the mean of the training ratings.

For ranking, all rating scores would be tied. The implementation therefore
uses a reproducible seeded random ordering. Ranking results for this method
represent a random-ranking baseline rather than ranking by the global mean.

### Classification

This is a standard null model. It is not the stronger user-item bias
baseline:

$$
\hat r_{ui}=\mu+b_u+b_i.
$$

Closest source:

- Koren, Y., Bell, R., and Volinsky, C. (2009). “Matrix Factorization
  Techniques for Recommender Systems.” *Computer*, 42(8), 30–37.
  [DOI](https://doi.org/10.1109/MC.2009.263)

Deviations and choices:

- This is a basic rating baseline rather than the stronger regularized
  baseline $\mu+b_u+b_i$.
- All ranking scores are tied. Ranking therefore uses a reproducible seeded
  random ordering. The ranking result should be interpreted as a random
  ranking baseline, not as meaningful ranking by the global mean.

## 2. Genre residual-weighted content profile

### Implemented model

Each movie has an L2-normalized multi-hot genre vector $x_i$. The genre
vectors are 20-dimensional and cover the MovieLens genres, including
`(no genres listed)`. The user profile is:

$$
p_u=
\operatorname{normalize}
\left(
\sum_{i\in I_u}
(r_{ui}-\bar r_u)x_i
\right).
$$

The rating prediction is:

$$
\hat r_{ui}
=
\bar r_u+\beta p_u^\mathsf{T}x_i,
\qquad \beta\geq0.
$$

The scale $\beta$ is fitted on the shared validation set by nonnegative least
squares.

### Classification

This is a custom content-based variation.

The vector-space cosine foundation comes from Salton et al., while the signed
residual-weighted profile is closer to a Rocchio-style relevance-feedback
adaptation. Salton et al. should not be cited as the source of the complete
rating-prediction formula.

Implementation-specific choices:

- Movie vectors are L2-normalized multi-hot genre vectors.
- Rating residuals provide signed positive and negative profile weights.
- The final profile is L2-normalized.
- The validation-fitted $\beta$ maps cosine similarity to a rating residual.
- A zero-norm profile predicts the user's training mean.
- Predictions are not internally clipped.

Citations:

- Salton, G., Wong, A., and Yang, C. S. (1975). “A Vector Space Model for
  Automatic Indexing.” *Communications of the ACM*, 18(11), 613–620.
  [DOI](https://doi.org/10.1145/361219.361220)
- Rocchio, J. J. (1971). “Relevance Feedback in Information Retrieval.”
  In G. Salton, editor, *The SMART Retrieval System: Experiments in
  Automatic Document Processing*, 313–323. Prentice-Hall.
- Pazzani, M. J., and Billsus, D. (2007). “Content-Based Recommendation
  Systems.” In *The Adaptive Web*, 325–341. Springer.
  [DOI](https://doi.org/10.1007/978-3-540-72079-9_10)
- Lops, P., de Gemmis, M., and Semeraro, G. (2011). “Content-based
  Recommender Systems: State of the Art and Trends.” In *Recommender
  Systems Handbook*, 73–105. Springer.
  [DOI](https://doi.org/10.1007/978-0-387-85820-3_3)

## 3. Tag Genome residual-weighted content profile

### Implemented model

This method uses the same profile and rating formulas as the genre content
model. Each movie is instead represented by its normalized 1128-dimensional
MovieLens Tag Genome relevance vector.

### Classification

This is a custom content-based variation combining:

- Vector-space content similarity.
- Rocchio-like signed residual weighting.
- MovieLens Tag Genome features.
- Validation-based rating-scale fitting.

The Tag Genome paper supplies the feature representation, not the complete
recommendation or rating-prediction formula.

Implementation-specific choices:

- The experiment is restricted to movies with Tag Genome data.
- Tag relevance values are used as content-feature magnitudes.
- Movie feature rows are L2-normalized.
- User profiles use signed rating residuals.
- The rating scale is fitted on validation data.
- Predictions are not internally clipped.

Citations:

- Vig, J., Sen, S., and Riedl, J. (2012). “The Tag Genome: Encoding
  Community Knowledge to Support Novel Interaction.” *ACM Transactions on
  Interactive Intelligent Systems*, 2(3), Article 13.
  [DOI](https://doi.org/10.1145/2362394.2362395)
- Salton, G., Wong, A., and Yang, C. S. (1975). “A Vector Space Model for
  Automatic Indexing.” *Communications of the ACM*, 18(11), 613–620.
  [DOI](https://doi.org/10.1145/361219.361220)
- Rocchio, J. J. (1971). “Relevance Feedback in Information Retrieval.”
  In *The SMART Retrieval System: Experiments in Automatic Document
  Processing*, 313–323.
- Pazzani, M. J., and Billsus, D. (2007). “Content-Based Recommendation
  Systems.” In *The Adaptive Web*, 325–341.
  [DOI](https://doi.org/10.1007/978-3-540-72079-9_10)
- Lops, P., de Gemmis, M., and Semeraro, G. (2011). “Content-based
  Recommender Systems: State of the Art and Trends.” In *Recommender
  Systems Handbook*, 73–105.
  [DOI](https://doi.org/10.1007/978-0-387-85820-3_3)

## 4. Tag Genome weighted hybrid

### Implemented model

The method is a confidence-weighted ensemble of two already-evaluated
methods in this study: the Tag Genome content profile (method 3) and the
User-based k-NN (method 5). For each row $(u,i)$ it uses:

- $\hat r_{ui}^{\mathrm{content}} = \bar r_u + \beta\, p_u^\top x_i$, the
  Tag Genome content prediction;
- $\hat r_{ui}^{\mathrm{knn}}$, the User-based k-NN prediction;
- $n_{ui}$, the number of usable k-NN neighbours actually used for $(u,i)$
  after the top-$k$ selection. $n_{ui}=0$ exactly when the k-NN fell back to
  the movie mean $\bar r_i$ because it had no usable evidence.

The prediction is a convex blend where the k-NN has evidence, and content
alone where it does not:

$$
\hat r_{ui}^{\mathrm{hyb}}
=
\begin{cases}
\alpha\,\hat r_{ui}^{\mathrm{content}} + (1-\alpha)\,\hat r_{ui}^{\mathrm{knn}}
& \text{if } n_{ui} > 0, \\
\hat r_{ui}^{\mathrm{content}}
& \text{if } n_{ui} = 0.
\end{cases}
$$

The global blend weight $\alpha$ is selected from the validation grid
$0.0, 0.1, \dots, 1.0$ by minimising validation RMSE (first grid point wins
ties). The confidence rule is applied inside the grid search, so
zero-neighbour rows (which contribute the content prediction for every
$\alpha$) do not bias the selection.

The ensemble reuses the saved outputs of its two parents: the content
prediction, the k-NN prediction, and the k-NN neighbour count. The k-NN
stage therefore emits one additional per-row count array, and the hybrid
must run after both parent stages.

### Why the fallback gives research value

The plain k-NN parent is $\hat r_{ui}^{\mathrm{knn}}$ when $n_{ui}>0$ and the
crude movie mean $\bar r_i$ when $n_{ui}=0$. The ensemble is
$\alpha\,\hat r_{ui}^{\mathrm{content}} + (1-\alpha)\,\hat r_{ui}^{\mathrm{knn}}$
when $n_{ui}>0$ and $\hat r_{ui}^{\mathrm{content}}$ when $n_{ui}=0$. On every
row the ensemble therefore weakly dominates the k-NN parent: where the k-NN
is strong it contributes through the blend, and where the k-NN degrades to a
movie mean the content signal replaces it. Combined with the content parent
contributing its own strength, this is the mechanism by which the ensemble is
intended to surpass both of its parents.

### Classification

This is a weighted ensemble under Burke's hybrid taxonomy: it combines a
content-based and a collaborative-filtering recommender by a convex blend,
with a confidence-based routing rule that swaps in the content prediction
when the collaborative component has no usable neighbourhood.

It is not a direct implementation of Fab. Fab is relevant as an early system
combining content-based and collaborative signals, but its exact architecture
is not reproduced here. The collaborative component is the conventional
User-based k-NN of method 5 (overlap-based rating similarity), not a
content-profile similarity.

Implementation-specific choices:

- The k-NN parent uses its own documented settings (top-40, minimum two
  common movies, shrinkage 10, signed similarities, absolute-similarity
  normalization).
- The neighbour count $n_{ui}$ is the number of co-raters retained after the
  top-$k$ selection; it is $0$ on both k-NN fallback paths.
- The convex blend weight $\alpha$ is global across rows, selected on
  validation.
- The content-only fallback applies only to rating rows; ranking candidates
  are warm-start training-catalog movies and always have $n_{ui}>0$.
- The content scale $\beta$ and the blend weight $\alpha$ are both selected
  on validation data, and this selection is counted in the hybrid's reported
  training time.

Citations:

- Burke, R. (2002). “Hybrid Recommender Systems: Survey and Experiments.”
  *User Modeling and User-Adapted Interaction*, 12(4), 331–370.
  [DOI](https://doi.org/10.1023/A:1021240730564)
- Balabanović, M., and Shoham, Y. (1997). “Fab: Content-Based,
  Collaborative Recommendation.” *Communications of the ACM*, 40(3),
  66–72. [DOI](https://doi.org/10.1145/245108.245124)
- Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., and Riedl, J.
  (1994). “GroupLens: An Open Architecture for Collaborative Filtering of
  Netnews.” In *Proceedings of CSCW 1994*, 175–186.
  [DOI](https://doi.org/10.1145/192844.192905)
- Pazzani, M. J., and Billsus, D. (2007). “Content-Based Recommendation
  Systems.” In *The Adaptive Web*, 325–341.
  [DOI](https://doi.org/10.1007/978-3-540-72079-9_10)
- Herlocker, J. L., Konstan, J. A., Borchers, A., and Riedl, J. (1999).
  “An Algorithmic Framework for Performing Collaborative Filtering.” In
  *Proceedings of SIGIR 1999*, 230–237.
  [DOI](https://doi.org/10.1145/312629.312688)

## 5. User-based k-nearest-neighbor collaborative filtering

### Implemented model

Similarity is mean-centered cosine, equivalent to a Pearson-like similarity
using full-training user means:

$$
s(u,v)=
\frac{
\sum_{i\in I_u\cap I_v}
(r_{ui}-\bar r_u)(r_{vi}-\bar r_v)
}{
\sqrt{
\sum_{i\in I_u\cap I_v}(r_{ui}-\bar r_u)^2
}
\sqrt{
\sum_{i\in I_u\cap I_v}(r_{vi}-\bar r_v)^2
}
}.
$$

The implementation applies support shrinkage:

$$
s'(u,v)=
\frac{n_{uv}}{n_{uv}+10}s(u,v).
$$

Pairs with fewer than two common movies are discarded. Prediction uses up to
40 neighbors:

$$
\hat r_{ui}
=
\bar r_u+
\frac{
\sum_{v\in N_i(u)}
s'(u,v)(r_{vi}-\bar r_v)
}{
\sum_{v\in N_i(u)}|s'(u,v)|
}.
$$

### Classification

This is a regularized user-based k-NN variation, not an exact reproduction of
one historical GroupLens implementation.

Implementation-specific choices:

- Similarity requires at least two common movies.
- Full-training user means are used rather than means recomputed over only
  the common-item subset.
- Similarity shrinkage is fixed at 10.
- Negative similarities are retained.
- Neighbor selection uses absolute similarity.
- The denominator uses absolute similarities.
- At most 40 neighbors are used.
- Similarities are calculated on demand and cached.
- Missing neighborhoods fall back to the target movie's training mean.

Citations:

- Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., and Riedl, J. (1994).
  “GroupLens: An Open Architecture for Collaborative Filtering of Netnews.”
  In *Proceedings of CSCW 1994*, 175–186.
  [DOI](https://doi.org/10.1145/192844.192905)
- Herlocker, J. L., Konstan, J. A., Borchers, A., and Riedl, J. (1999).
  “An Algorithmic Framework for Performing Collaborative Filtering.”
  In *Proceedings of SIGIR 1999*, 230–237.
  [DOI](https://doi.org/10.1145/312624.312682)
- Herlocker, J. L., Konstan, J. A., Terveen, L. G., and Riedl, J. T. (2004).
  “Evaluating Collaborative Filtering Recommender Systems.”
  *ACM Transactions on Information Systems*, 22(1), 5–53.
  [DOI](https://doi.org/10.1145/963770.963772)

## 6. Item-based k-nearest-neighbor collaborative filtering

### Implemented model

Adjusted cosine similarity is:

$$
s(i,j)=
\frac{
\sum_{u\in U_{ij}}
(r_{ui}-\bar r_u)(r_{uj}-\bar r_u)
}{
\sqrt{
\sum_{u\in U_{ij}}(r_{ui}-\bar r_u)^2
}
\sqrt{
\sum_{u\in U_{ij}}(r_{uj}-\bar r_u)^2
}
}.
$$

Pair-specific co-raters are used in both denominator terms. The implementation
then applies support shrinkage:

$$
s'(i,j)=
\frac{n_{ij}}{n_{ij}+10}s(i,j).
$$

Only similarities supported by at least two co-raters are retained, and only
the top 40 absolute similarities per target item are stored.

Prediction is the Sarwar weighted-sum form:

$$
\hat r_{ui}
=
\frac{
\sum_{j\in I_u}s'(i,j)r_{uj}
}{
\sum_{j\in I_u}|s'(i,j)|
}.
$$

### Classification

This is a sparse, support-shrunk variation of Sarwar et al.'s item-based
adjusted-cosine method.

The code and result name use `Item-based k-NN`. This is the appropriate
general term even though the application domain consists of movies.

Implementation-specific choices:

- At least two common raters are required.
- Similarity shrinkage is fixed at 10.
- Only the 40 largest absolute similarities per target movie are retained.
- Negative similarities are retained.
- The prediction denominator uses absolute similarities.
- The target movie is excluded defensively from the user's history.
- Similarities are computed in blocks and stored as a sparse matrix.

Citation:

- Sarwar, B., Karypis, G., Konstan, J., and Riedl, J. (2001). “Item-Based
  Collaborative Filtering Recommendation Algorithms.” In *Proceedings of the
  10th International Conference on World Wide Web*, 285–295.
  [DOI](https://doi.org/10.1145/371920.372071)

## 7. Biased matrix factorization

### Implemented model

The model is:

$$
\hat r_{ui}
=
\mu+b_u+b_i+p_u^\mathsf{T}q_i.
$$

The implementation minimizes per-interaction SGD regularization of the
squared error:

$$
\sum_{(u,i)\in K}
\left[
\left(r_{ui}-\mu-b_u-b_i-p_u^\mathsf{T}q_i\right)^2
+
\lambda
\left(
b_u^2+b_i^2+\lVert p_u\rVert^2+\lVert q_i\rVert^2
\right)
\right].
$$

This makes clear that frequently observed parameter rows receive
regularization more frequently under the implemented SGD schedule.

For prediction error $e_{ui}=r_{ui}-\hat r_{ui}$, the updates are:

$$
b_u
\leftarrow
b_u+\gamma(e_{ui}-\lambda b_u),
$$

$$
b_i
\leftarrow
b_i+\gamma(e_{ui}-\lambda b_i),
$$

$$
p_u
\leftarrow
p_u+\gamma(e_{ui}q_i-\lambda p_u),
$$

$$
q_i
\leftarrow
q_i+\gamma(e_{ui}p_u-\lambda q_i).
$$

The implementation copies the previous user and item vectors before updating
either vector, so both factor updates use the same pre-update state.

### Classification

The prediction model and SGD objective are standard biased matrix
factorization. The fixed factor count, learning rate, common regularization
coefficient, validation grid, and cold-start handling are implementation
choices.

Implementation-specific choices:

- One regularization coefficient is shared by user biases, item biases, user
  factors, and item factors.
- The latent dimension is 50.
- The learning rate is 0.005.
- Training runs for 20 epochs.
- Regularization is selected from a small validation grid.
- The displayed epoch RMSE is recomputed after each epoch with the resulting
  model; it is not an online-update error.
- Factors for users or items receiving no training interaction are zeroed.
- Cold-start predictions therefore use only identifiable bias terms:
  $\mu+b_i$, $\mu+b_u$, or $\mu$.
- Predictions are not internally clipped.

Citations:

- Koren, Y. (2008). “Factorization Meets the Neighborhood: A Multifaceted
  Collaborative Filtering Model.” In *Proceedings of KDD 2008*, 426–434.
  [DOI](https://doi.org/10.1145/1401890.1401944)
- Koren, Y., Bell, R., and Volinsky, C. (2009). “Matrix Factorization
  Techniques for Recommender Systems.” *Computer*, 42(8), 30–37.
  [DOI](https://doi.org/10.1109/MC.2009.263)

## 8. BPR-MF with thresholded positives

### Base model

The raw preference score is:

$$
x_{ui}=p_u^\mathsf{T}q_i.
$$

For an observed positive item $i$ and unobserved item $j$, BPR optimizes a
pairwise objective based on:

$$
\log\sigma(x_{ui}-x_{uj})
-
\lambda\lVert\Theta\rVert^2.
$$

The score difference is:

$$
x_{uij}
=
p_u^\mathsf{T}(q_i-q_j).
$$

The implementation uses:

$$
\frac{\partial\log\sigma(x_{uij})}{\partial x_{uij}}
=
\sigma(-x_{uij}).
$$

### Explicit-feedback adaptation

The MovieLens input contains explicit ratings, whereas the original BPR
method assumes implicit positive feedback.

The implementation adapts the data as follows:

- MovieLens explicit ratings are adapted to BPR feedback.
- Training ratings of at least 3.5 are positives.
- Lower observed ratings are not positives, but are excluded from negative
  sampling.
- Negatives are uniformly sampled from unobserved training-catalog items.
- Four fresh negatives are attempted per positive in every epoch. These
  draws are independent and may repeat across different positives within
  an epoch.
- Negatives are dynamically resampled rather than permanently precomputed.
- A single regularization coefficient is used for all factors.
- The base model has no item-bias term.
- Users and items receiving no BPR update have zero factor vectors rather
  than arbitrary random scores.
- BPR regularization is selected by NDCG@10 on the fixed validation
  candidate file. Each row contains one relevant validation item and 999
  sampled negatives. If two regularization candidates have NDCG values
  equal within $10^{-12}$, the larger regularization value is selected.
  This top-N selection rule is an experimental design choice, not part of
  the original BPR algorithm.
- Raw BPR scores are used for ranking.

### Post-hoc rating mapping

Raw BPR scores are ranked directly. For secondary rating diagnostics, the
implementation fits:

$$
\hat r_{ui}=a x_{ui}+b,
\qquad a\geq0,
$$

on the shared validation ratings.

The nonnegative constraint prevents the mapping from reversing the learned
ranking. The fitted coefficients $(a, b)$ are persisted inside
`Predictions/bpr_model.npz` (members `slope` / `intercept`) so that
post-hoc combinations — in particular the MF+BPR hybrid's rating head,
section 9 — reuse the exact mapping the BPR stage fitted rather than
re-deriving it. Older model files that pre-date this persistence carry no
affine members; the loader then reports an absent mapping and the
consumer re-fits it from the model's validation scores (deterministic for
that model), so a model-reuse run still reproduces the BPR rating
predictions exactly.

### Classification

This method must be described as:

- BPR-MF adapted from explicit ratings through thresholded positives.
- Dynamic uniform unobserved-negative sampling.
- A validation-fitted post-hoc nonnegative affine rating mapping, persisted
  with the model for reuse by downstream hybrid stages.

The affine mapping is not part of Rendle et al.'s BPR algorithm. It should
not be attributed to that paper, and its rating metrics do not turn BPR into
a pointwise rating-prediction model. BPR rating metrics are secondary
diagnostics.

### C/Python bit-parity requirements

The parallel C implementation (`c_impl/`) reproduces the Python BPR training
bit-for-bit only if three details are matched. These are the properties that
make the negative-sampling stream identical across the two implementations,
and they are worth recording because they are easy to break:

- **Sorted training catalog.** The catalog of candidate negative movies must
  be emitted in ascending movie-ID order (matching Python's
  `np.unique`). `draw_unobserved_movie` indexes the catalog with
  `rng.integers(0, catalog_size)`, so any other ordering (e.g. insertion
  order) changes which movie each random index selects and desynchronises the
  whole training trajectory.
- **Single dot product for the BPR difference.** The gradient uses
  `p_u · (q_i − q_j)` as one dot product (Python's
  `np.dot(old_user, old_positive − old_negative)`), not
  `p_u·q_i − p_u·q_j`. The two are algebraically equal but reduce the sum in
  a different order, diverging in the last bits and compounding over SGD.
- **Monitor consumes RNG draws.** The per-epoch sampled-pair-accuracy monitor
  draws its sample with `rng.choice(n, size, replace=False)`, which consumes
  RNG state. The C monitor must do the same (via `rng_choice_noreplace`); a
  monitor that reuses the already-shuffled order would leave the RNG at a
  different position for the next epoch's negative draws.

The C RNG itself is a bit-exact reproduction of NumPy's `default_rng` (PCG64),
verified against NumPy ground truth by `c_impl/tests/rng_parity.c` (run via
`make parity` in `c_impl/`).

Citation:

- Rendle, S., Freudenthaler, C., Gantner, Z., and Schmidt-Thieme, L. (2009).
  “BPR: Bayesian Personalized Ranking from Implicit Feedback.” In
  *Proceedings of the Twenty-Fifth Conference on Uncertainty in Artificial
  Intelligence*, 452–461. AUAI Press.
  [arXiv version](https://arxiv.org/abs/1205.2618)

## 9. MF+BPR hybrid (dual-headed)

### Implemented model

The MF+BPR hybrid is a dual-headed ensemble of Biased MF (section 7) and
BPR-MF (section 8). It is trained by no gradient descent; it reuses the two
parents' already-saved test-row rating predictions, test ranking scores, and
persisted models, and it selects two blend weights on the shared validation
set. The "dual-headed" name refers to the two independent $\lambda$ values —
one per head — rather than to a single fused output.

**Rating head.** The prediction is the convex blend of the two parents'
rating predictions:

$$
\hat r^{\text{MF+BPR}}_{ui}
  = \lambda_r\,\hat r^{\text{MF}}_{ui}
  + (1-\lambda_r)\,\hat r^{\text{BPR}}_{ui},
$$

where $\hat r^{\text{BPR}}_{ui}$ is BPR-MF's post-hoc affine-mapped rating
(section 8). $\lambda_r$ is chosen from $\{0,0.1,\dots,1.0\}$ to minimise
validation-set RMSE (first grid point wins ties). In the current dataset BPR's
affine mapping is degenerate (fitted slope $0$, because the validation BPR
scores carry no linear rating signal), so the BPR-mapped rating is a constant
and $\lambda_r$ selects $1.0$: the rating head reduces to pure Biased MF.
This is an honest consequence of the data, not a tuning choice.

**Ranking head.** The two parents' raw candidate-score rows are incompatible
in scale (Biased MF scores sit near the rating mean $\approx3.5$; BPR raw
scores are centred near $0$ with larger variance — see figure 09). They are
therefore per-row z-standardised before blending:

$$
z^{(A)}_c = \frac{s^{(A)}_c - \overline{s^{(A)}}}{\sigma^{(A)}},
\qquad
\hat z_c = \lambda_k\,z^{(\text{MF})}_c + (1-\lambda_k)\,z^{(\text{BPR})}_c,
$$

where the mean and standard deviation are computed over the candidate row
$\sigma$ is floored at $10^{-12}$ to guard against zero-variance rows.
$\lambda_k$ is chosen from the same grid to maximise validation NDCG@10.
Per-row z-scoring is order-preserving within a row, so ranking the blended
$\hat z$ recovers the intended candidate order. The standardisation is a
hybrid-design choice (motivated by Cormack et al.'s reciprocal-rank fusion
discussion of score comparability) and is not part of either parent.

### Classification

- Dual-headed weighted ensemble under Burke's hybrid taxonomy.
- No new model parameters are learned; the only fitted quantities are the two
  validation-selected scalars $\lambda_r$ and $\lambda_k$.
- The rating head is a plain convex blend (standard weighted hybrid).
- The ranking head is a convex blend *after* per-row z-standardisation (a
  rank-space fusion).

### Deviations and implementation choices

- Both $\lambda$ values are global across rows and selected on the shared
  validation set; the ranking head's validation rows are the fixed validation
  candidate file (seed 2025).
- The stage runs last (after `mf` and `bpr`) and is skipped with a warning if
  either parent model is absent.
- The rating head's RMSE and the ranking head's NDCG@10 are reported
  separately in `selected_choices`; the rating metric is dominated by the MF
  parent (per above) and the ranking metric is the hybrid's contribution.
- The validation components ($\hat r^{\text{MF}}_{\text{val}}$,
  $\hat r^{\text{BPR}}_{\text{val}}$, $y_{\text{val}}$) are saved as
  `mf_bpr_val_{mf,bpr}_rating.npy` / `mf_bpr_val_actual.npy` so the λ-sweep
  figure (08) can be regenerated without re-scoring.

Citations:

- Burke, R. (2002). “Hybrid Recommender Systems: Survey and Experiments.”
  In *Hybrid Information Retrieval Systems*, 23–242. (weighted ensemble)
- Cormack, G. V., Clarke, C. L. A., and Buettcher, S. (2009). “Reciprocal
  Rank Fusion outperforms Condorcet and individual Rank Learning Methods.”
  In *Proceedings of the 32nd Annual International ACM SIGIR Conference*,
  75–84. (score standardisation for rank fusion)

## Ranking protocol and metrics

### Sampled-candidate ranking

The protocol follows the general one-plus-sampled-negatives evaluation
pattern associated with Cremonesi et al. but is not an exact reproduction.
ReRecom uses one positive plus 999 negatives, chronological validation and
test interactions, a training-observed candidate catalog, and exclusion of
every known interaction.

Each evaluated candidate row contains:

- One relevant held-out movie with rating at least 3.5.
- 999 distinct uniformly sampled negatives drawn from movies observed in
  training.
- At most 500 eligible users are retained for each split.
- Fixed candidate seeds: validation 2025, test 2026.

The same fixed candidate sets are used by every algorithm.

### Candidate-construction caveat: held-out-period information

Candidate negatives exclude all known interactions for the user across
training, validation, and test. This all-known-item candidate-construction
rule uses held-out-period information and is an explicit evaluation design
choice with concrete consequences:

- When validation candidates are constructed, the user's chronologically
  later test interaction is already known and excluded from the validation
  negatives.
- The protocol is therefore a fixed sampled-candidate offline evaluation,
  not a strictly prospective chronological simulation.
- The leakage is limited to candidate-set construction. No test labels
  enter model fitting, model selection, or score calibration.
- Results are valid only for these fixed candidate sets.

### Ranking measures

The reported measures are:

- Hit Rate at 10.
- NDCG at 10.
- Truncated MRR at 10.

There is exactly one relevant movie in each candidate set. Consequently:

$$
\operatorname{Recall@K}=\operatorname{HitRate@K},
$$

$$
\operatorname{Precision@K}
=
\frac{\operatorname{HitRate@K}}{K},
$$

and:

$$
\operatorname{AP@K}
=
\operatorname{truncatedRR@K}.
$$

Precision, recall, and MAP are therefore not separately reported.

Exact tied ranking scores use stable descending sorting, so equal scores are
broken deterministically by their stored candidate order. The global-mean
baseline is an exception: it uses explicit seeded random scores because all
of its rating scores are equal.

### Protocol choices and limitations

- A held-out rating of at least 3.5 is relevant.
- Each evaluated user has one relevant candidate.
- Candidate sets and seeds are fixed across algorithms.
- Unobserved movies are treated as nonrelevant even though some may be
  unknown positives.
- Sampled metrics are not unbiased estimates of full-catalog metrics.
  Following the concerns established by Krichene and Rendle, the resulting
  ranking values are protocol-specific sampled metrics, not estimates that
  can be assumed to preserve full-catalog model comparisons.
- Cremonesi et al. provides methodological precedent for sampled top-N
  evaluation but is not an exact implemented protocol.

### Metric citations

- Cremonesi, P., Koren, Y., and Turrin, R. (2010). “Performance of
  Recommender Algorithms on Top-N Recommendation Tasks.” In *Proceedings of
  RecSys 2010*, 39–46. [DOI](https://doi.org/10.1145/1864708.1864721)
- Krichene, W., and Rendle, S. (2020). “On Sampled Metrics for Item
  Recommendation.” In *Proceedings of KDD 2020*, 1748–1757.
  [DOI](https://doi.org/10.1145/3394486.3403226)
- Järvelin, K., and Kekäläinen, J. (2002). “Cumulated Gain-Based Evaluation
  of IR Techniques.” *ACM Transactions on Information Systems*, 20(4),
  422–446. [DOI](https://doi.org/10.1145/582415.582418)

## Recommended result names

The current result names are scientifically defensible if their variations
are described. More explicit alternatives are:

| Current name | More explicit publication name |
|---|---|
| Global mean | Training global-mean baseline |
| Genre content profile | Residual-weighted genre content profile |
| Tag Genome content profile | Residual-weighted Tag Genome content profile |
| Tag Genome weighted hybrid | Tag Genome content x User-k-NN weighted ensemble |
| User-based k-NN | Shrunk mean-centered user k-NN |
| Item-based k-NN | Sparse shrunk adjusted-cosine item k-NN |
| Biased MF | Validation-selected biased MF with SGD |
| BPR-MF thresholded positives | Thresholded explicit-feedback BPR-MF |

The fourth method should always be identified as a weighted ensemble of the
Tag Genome content profile and the User-based k-NN, with a confidence
fallback.
The second, third, fifth, sixth, and eighth methods should be identified as
variations or adaptations when methodological precision is important.

## Corrected bibliographic details

The following details replace inaccurate entries in the earlier files:

- Herlocker et al. (1999) is cited as SIGIR 1999, pages 230–237.
- Cremonesi et al. (2010) is cited as RecSys 2010, pages 39–46.
- The BPR title uses the source spelling “Personalized.”
- Koren (2008) is explicitly cited when the implementation text refers to
  the 2008 factorization work.
- Salton et al. are cited for vector-space cosine similarity, not for the
  complete rating-prediction formulas.
- Burke is cited for the weighted-ensemble (hybrid) taxonomy.
- The affine BPR rating mapping is not attributed to the BPR source.
