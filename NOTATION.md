# Notation Glossary

This document defines every mathematical symbol used in the algorithm
descriptions across `SCIENTIFIC_IMPLEMENTATION_REPORT.md` and `essay.qmd`.
Cross-reference specific symbols from each algorithm section.

## Dataset and split notation

| Symbol | Meaning |
|---|---|
| $\mu$ | Global mean of the training ratings. |
| $\bar{r}_u$ | Mean rating of user $u$ over the training interactions. |
| $\bar{r}_i$ | Mean rating of movie $i$ over the training interactions. |
| $r_{ui}$ | Observed rating given by user $u$ to movie $i$. |
| $I_u$ | Set of movies rated by user $u$ in the training set. |
| $U_i$ | Set of users who rated movie $i$ in the training set. |
| $U_{ij}$ | Set of users who rated both movies $i$ and $j$ in training. |
| $n_{uv}$ | Number of movies co-rated by users $u$ and $v$. |
| $n_{ij}$ | Number of users who rated both movies $i$ and $j$. |
| $N_k(i)$ | Set of $k$ nearest neighbours of movie $i$. |
| $N_k(u)$ | Set of $k$ nearest neighbours of user $u$. |

## Content-based notation

| Symbol | Meaning |
|---|---|
| $x_i$ | L2-normalised feature vector for movie $i$ (genre or Tag Genome). |
| $p_u$ | L2-normalised residual-weighted user profile vector. |
| $\beta$ | Non-negative content rating-scale parameter, fitted on validation. |

## Collaborative filtering notation

| Symbol | Meaning |
|---|---|
| $s(u,v)$ | Mean-centred cosine similarity between users $u$ and $v$. |
| $s'(u,v)$ | Shrinkage-adjusted user similarity: $\frac{n_{uv}}{n_{uv}+10}\,s(u,v)$. |
| $s(i,j)$ | Adjusted cosine similarity between movies $i$ and $j$. |
| $s'(i,j)$ | Shrinkage-adjusted movie similarity: $\frac{n_{ij}}{n_{ij}+10}\,s(i,j)$. |

## Matrix factorisation notation

| Symbol | Meaning |
|---|---|
| $p_u$ | Latent user factor vector (50-dimensional). |
| $q_i$ | Latent movie factor vector (50-dimensional). |
| $b_u$ | Learned user bias term. |
| $b_i$ | Learned movie bias term. |
| $\gamma$ | Learning rate for SGD updates. |
| $\lambda$ | Regularization coefficient. |
| $\Theta$ | All model parameters collectively. |

## BPR notation

| Symbol | Meaning |
|---|---|
| $x_{ui}$ | Raw BPR preference score: $p_u^\top q_i$. |
| $x_{uij}$ | Score difference: $p_u^\top(q_i - q_j)$. |
| $\sigma$ | Logistic sigmoid function: $\sigma(z) = 1/(1+e^{-z})$. |
| $a, b$ | Post-hoc nonnegative affine mapping coefficients: $\hat{r}_{ui} = a\,x_{ui} + b$. Persisted in `bpr_model.npz` (members `slope`, `intercept`) for reuse by the MF+BPR hybrid's rating head. |

## Hybrid notation

The Tag Genome weighted hybrid is a confidence-weighted ensemble of the Tag
Genome content profile and the User-based k-NN.

| Symbol | Meaning |
|---|---|
| $\alpha$ | Global hybrid blend weight (content weight); selected on validation. |
| $\hat{r}^{\text{content}}_{ui}$ | Tag Genome content prediction: $\bar r_u + \beta\, p_u^\top x_i$. |
| $\hat{r}^{\text{knn}}_{ui}$ | User-based k-NN prediction. |
| $n_{ui}$ | Number of usable k-NN neighbours used for $(u,i)$ after top-$k$ selection; $0$ when k-NN falls back to $\bar r_i$. |
| $\hat{r}^{\text{hyb}}_{ui}$ | Blended hybrid prediction. |

The hybrid prediction rule is:

$$
\hat r^{\text{hyb}}_{ui}
=
\begin{cases}
\alpha\,\hat r^{\text{content}}_{ui} + (1-\alpha)\,\hat r^{\text{knn}}_{ui}
& \text{if } n_{ui} > 0, \\
\hat r^{\text{content}}_{ui}
& \text{if } n_{ui} = 0.
\end{cases}
$$

## MF+BPR hybrid notation

The MF+BPR hybrid is a dual-headed blend of Biased MF and BPR-MF with two
independent blend weights (one per head), each selected on validation.

| Symbol | Meaning |
|---|---|
| $\lambda_r$ | Rating-head blend weight (fraction of Biased MF); selected on validation RMSE. |
| $\lambda_k$ | Ranking-head blend weight (fraction of Biased MF); selected on validation NDCG@10. |
| $s^{(A)}_c$ | Algorithm $A$'s raw score for candidate $c$ in a user's row ($A\in\{\text{MF},\text{BPR}\}$). |
| $\overline{s^{(A)}}$, $\sigma^{(A)}$ | Per-row mean and standard deviation of the raw scores, computed over the candidate row for algorithm $A$. |
| $z^{(A)}_c$ | Per-row z-standardised score: $(s^{(A)}_c - \overline{s^{(A)}})/\sigma^{(A)}$, with $\sigma$ floored at $10^{-12}$. |
| $\hat r^{\text{MF}}_{ui}$, $\hat r^{\text{BPR}}_{ui}$ | Biased MF rating prediction; BPR-MF post-hoc affine-mapped rating. |

Rating head:

$$
\hat r^{\text{MF+BPR}}_{ui}
= \lambda_r\,\hat r^{\text{MF}}_{ui} + (1-\lambda_r)\,\hat r^{\text{BPR}}_{ui}.
$$

Ranking head (per candidate row, after z-standardisation):

$$
\hat z_{c} = \lambda_k\,z^{\text{MF}}_{c} + (1-\lambda_k)\,z^{\text{BPR}}_{c}.
$$

Candidates are ranked by descending $\hat z_c$.

## Evaluation notation

| Symbol | Meaning |
|---|---|
| $K$ | Ranking cutoff (10). |
| $k$ | Rank position of the relevant movie in a candidate list. |
| $\text{NDCG@K}$ | Normalised discounted cumulative gain at cutoff $K$. |
| $\text{HR@K}$ | Hit rate at cutoff $K$. |
| $\text{MRR@K}$ | Truncated mean reciprocal rank at cutoff $K$. |
