"""Edge k-means (EKM): prototypes and their connectivity graph from one objective function.

Reference implementation of the method of the paper

    S. Ubukata, T. Maekawa, A. Notsu, K. Honda,
    "Edge k-means: Learning Prototypes and Their Connectivity Graph by Minimum-Cost Edge Assignment".

EKM assigns each data point x_i to an ordered pair (j, k) of prototypes at the cost

    ||x_i - b_j||^2 + lam * ||x_i - b_k||^2        (0 < lam < 1)

and alternates this assignment with the update of the prototypes.  The optimal pair is the nearest and the
second-nearest prototype of the data point, so the number w_jk of data points assigned to the pair {j, k} is the
weight of the edge {j, k} of the prototype graph (the connections of competitive Hebbian learning with the weights
of CONN).  Clusters of arbitrary shape are the groups of prototypes that remain connected when weak edges are cut.

    import ekm
    model = ekm.fit(X)                              # prototypes model.B, edge weights model.W
    labels = ekm.clusters(model, n_clusters=2)      # cluster of each data point

Requires NumPy and SciPy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

CHUNK = 4096            # data points per block of distances (bounds the memory at CHUNK x K numbers)


@dataclass
class Model:
    B: np.ndarray           # prototypes, K x d
    W: np.ndarray           # edge weights (support counts), K x K, symmetric: W[j, k] = number of data points
                            # whose nearest and second-nearest prototypes are j and k, in either order
    pairs: np.ndarray       # nearest and second-nearest prototype of each data point, n x 2
    lam: float              # weight of the second prototype
    iterations: int
    converged: bool


def _distance_blocks(X: np.ndarray, B: np.ndarray):
    """Blocks of squared distances between data points and prototypes: yields (rows, D)."""
    xx, bb = np.einsum("ij,ij->i", X, X), np.einsum("ij,ij->i", B, B)
    for a in range(0, len(X), CHUNK):
        rows = slice(a, min(a + CHUNK, len(X)))
        D = X[rows] @ (-2.0 * B).T
        D += xx[rows, None]
        D += bb
        yield rows, np.maximum(D, 0.0, out=D)


def two_nearest(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Nearest and second-nearest prototype of each data point (n x 2); ties go to the smaller index."""
    pairs = np.empty((len(X), 2), dtype=np.int64)
    for rows, D in _distance_blocks(X, B):
        index = np.arange(len(D))
        pairs[rows, 0] = D.argmin(axis=1)
        D[index, pairs[rows, 0]] = np.inf
        pairs[rows, 1] = D.argmin(axis=1)
    return pairs


def kmeans_pp(X: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding (Arthur and Vassilvitskii, 2007)."""
    chosen = [int(rng.integers(len(X)))]
    d = ((X - X[chosen[0]]) ** 2).sum(axis=1)
    for _ in range(1, K):
        total = d.sum()
        if total > 0:
            cdf = (d / total).cumsum()
            chosen.append(min(int((cdf / cdf[-1]).searchsorted(rng.random(), side="right")), len(X) - 1))
        else:
            chosen.append(int(rng.integers(len(X))))
        d = np.minimum(d, ((X - X[chosen[-1]]) ** 2).sum(axis=1))
    return X[chosen].copy()


def _update(X: np.ndarray, pairs: np.ndarray, lam: float, B: np.ndarray) -> np.ndarray:
    """Means with weight 1 for the data points whose nearest prototype is j and lam for those whose second-nearest
    prototype is j.  A prototype with zero total weight keeps its position."""
    n, K = len(X), len(B)
    A = csr_matrix((np.tile([1.0, lam], n), pairs.ravel(), np.arange(0, 2 * n + 1, 2)), shape=(n, K))
    total = np.asarray(A.sum(axis=0)).ravel()
    out = B.copy()
    out[total > 0] = (A.T @ X)[total > 0] / total[total > 0, None]
    return out


def _anneal(X: np.ndarray, B: np.ndarray, epochs: int, final_range: float) -> np.ndarray:
    """Annealed initialization: `epochs` updates of batch neural gas (Cottrell et al., 2006), in which every
    prototype is the mean of the data weighted by exp(-rank / range) and the neighborhood range decreases
    geometrically from K / 2 to `final_range`."""
    K = len(B)
    for t in range(epochs):
        weights = np.exp(-np.arange(K) / ((K / 2.0) * (final_range / (K / 2.0)) ** (t / max(epochs - 1, 1))))
        numerator, denominator = np.zeros_like(B), np.zeros(K)
        for rows, D in _distance_blocks(X, B):
            H = np.empty_like(D)
            np.put_along_axis(H, np.argsort(D, axis=1, kind="stable"), np.broadcast_to(weights, D.shape), axis=1)
            numerator += H.T @ X[rows]
            denominator += H.sum(axis=0)
        B = B.copy()
        B[denominator > 0] = numerator[denominator > 0] / denominator[denominator > 0, None]
    return B


def fit(X, n_prototypes: int | None = None, lam: float = 0.3, annealing_epochs: int = 0, max_iter: int = 500,
        seed: int | None = 0, init: np.ndarray | None = None, final_range: float = 0.001) -> Model:
    """Edge k-means.

    n_prototypes       K; by default the rule ceil(sqrt(n)).
    lam                weight of the second prototype, 0 < lam < 1 (0.3 in the paper).  lam = 0 gives k-means
                       followed by the counting of the edges, and lam = 1 gives (k,2)-means.
    annealing_epochs   T > 0 precedes EKM by T annealing updates (annealed EKM; T = 20 in the paper).
    init               initial prototypes (K x d); by default k-means++ seeding with `seed`.
    The iteration stops when no data point changes its pair (for lam = 0, its nearest prototype) or after
    max_iter iterations.
    """
    X = np.ascontiguousarray(X, dtype=float)
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lam must lie in [0, 1]")
    K = int(n_prototypes) if n_prototypes else math.ceil(math.sqrt(len(X)))
    B = np.array(init, dtype=float) if init is not None else kmeans_pp(X, K, np.random.default_rng(seed))
    if len(B) < 2:
        raise ValueError("at least two prototypes are needed")
    if annealing_epochs > 0:
        B = _anneal(X, B, annealing_epochs, final_range)
    used = slice(None) if lam > 0 else 0            # what must stop changing: the pairs, or the nearest prototypes
    previous, converged = None, False
    for iterations in range(1, max_iter + 1):
        pairs = two_nearest(X, B)
        if previous is not None and np.array_equal(pairs[:, used], previous[:, used]):
            converged = True
            break
        previous = pairs
        B = _update(X, pairs, lam, B)
    else:
        pairs = two_nearest(X, B)
    directed = np.bincount(pairs[:, 0] * len(B) + pairs[:, 1], minlength=len(B) ** 2).reshape(len(B), len(B))
    return Model(B=B, W=directed + directed.T, pairs=pairs, lam=lam, iterations=iterations, converged=converged)


# ------------------------------------------------------------------------------------------- clusters from the graph
def _counted(labels: np.ndarray, sizes: np.ndarray, n_clusters: int, min_fraction: float) -> int:
    """Number of clusters with at least min_fraction * n / n_clusters data points (and at least one)."""
    points = np.bincount(labels, weights=sizes)
    return int(np.count_nonzero(points >= max(1.0, min_fraction * sizes.sum() / n_clusters)))


def cut(W: np.ndarray, theta: float) -> np.ndarray:
    """Group of each prototype: connected components of the graph without the edges of weight below theta."""
    return connected_components(csr_matrix(W >= max(theta, 1)), directed=False)[1]


def threshold_cut(W: np.ndarray, n_clusters: int, sizes: np.ndarray, min_fraction: float = 0.1) -> np.ndarray:
    """Group of each prototype when the edges are removed in increasing order of their weights until n_clusters
    clusters are counted.  A cluster is counted if it holds at least min_fraction * n / n_clusters data points;
    smaller ones, made of outlying points, keep their own labels.  If no threshold gives n_clusters counted
    clusters, the first threshold with the largest number of counted clusters is used."""
    best = cut(W, 1)
    best_count = _counted(best, sizes, n_clusters, min_fraction)
    for w in np.unique(W[W > 0]):
        if best_count >= n_clusters:
            break
        labels = cut(W, w + 1)                      # removes every edge of weight <= w
        count = _counted(labels, sizes, n_clusters, min_fraction)
        if count > best_count:
            best, best_count = labels, count
    return best


def density_linkage(W: np.ndarray, B: np.ndarray, n_clusters: int, sizes: np.ndarray, method: str = "average",
                    min_fraction: float = 0.1) -> np.ndarray:
    """Group of each prototype by the segmentation of skeleton clustering (Wei and Chen, 2024): hierarchical
    clustering with the similarity w_jk / ||b_j - b_k|| (the Voronoi density), cut by the same counting rule."""
    bb = np.einsum("ij,ij->i", B, B)
    distance = B @ B.T
    distance *= -2.0
    distance += bb[:, None]
    distance += bb[None, :]
    distance = np.sqrt(np.maximum(distance, 0.0))
    np.fill_diagonal(distance, 1.0)
    similarity = W / np.maximum(distance, 1e-12)
    D = similarity.max() - similarity
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method=method)
    best, best_count = None, -1
    for groups in range(min(n_clusters, len(B)), len(B) + 1):
        labels = fcluster(Z, groups, criterion="maxclust") - 1
        count = _counted(labels, sizes, n_clusters, min_fraction)
        if count > best_count:
            best, best_count = labels, count
        if count >= n_clusters:
            break
    return best


def clusters(model: Model, n_clusters: int | None = None, theta: float | None = None,
             method: str = "density") -> np.ndarray:
    """Cluster of each data point: the group of its nearest prototype.

    theta        remove the edges with weight below theta; the number of clusters is then an outcome.
    n_clusters   number of clusters; method "density" (density linkage, the default) or "threshold" (threshold cut).
    """
    nearest = model.pairs[:, 0]
    sizes = np.bincount(nearest, minlength=len(model.B)).astype(float)
    if theta is not None:
        groups = cut(model.W, theta)
    elif n_clusters is None:
        raise ValueError("give n_clusters or theta")
    elif method == "density":
        groups = density_linkage(model.W, model.B, n_clusters, sizes)
    elif method == "threshold":
        groups = threshold_cut(model.W, n_clusters, sizes)
    else:
        raise ValueError("method must be 'density' or 'threshold'")
    return groups[nearest]


# ------------------------------------------------------------------------------------------- the objective function
def objective(X, model: Model) -> dict:
    """The objective function J = sum_i ||x_i - b_j||^2 + lam ||x_i - b_k||^2 at the pairs of the model, and the two
    terms of its decomposition J = error + energy:

        error  = (1 + lam) sum_i ||x_i - b_jk||^2,   b_jk = (b_j + lam b_k) / (1 + lam)   (a point on the edge),
        energy = lam / (1 + lam) * sum_{j<k} w_jk ||b_j - b_k||^2                          (graph Laplacian energy).
    """
    X = np.asarray(X, dtype=float)
    B, lam = model.B, model.lam
    first, second = B[model.pairs[:, 0]], B[model.pairs[:, 1]]
    J = ((X - first) ** 2).sum() + lam * ((X - second) ** 2).sum()
    error = (1 + lam) * ((X - (first + lam * second) / (1 + lam)) ** 2).sum()
    laplacian = np.diag(model.W.sum(axis=1)) - model.W
    energy = lam / (1 + lam) * np.trace(B.T @ laplacian @ B)
    return {"J": float(J), "error": float(error), "energy": float(energy)}
