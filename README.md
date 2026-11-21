# edge-kmeans

Reference implementation of **edge k-means (EKM)**, a prototype-based clustering method that learns the prototypes
and the graph connecting them from one objective function.

EKM assigns each data point *x* to an ordered pair (*j*, *k*) of prototypes at the cost
‖*x* − *b<sub>j</sub>*‖² + λ‖*x* − *b<sub>k</sub>*‖² (0 < λ < 1) and alternates this assignment with the update of
the prototypes. The optimal pair is the nearest and the second-nearest prototype of the data point, so the number of
data points assigned to a pair is the weight of the edge between the two prototypes. Cutting the weak edges leaves
clusters of arbitrary shape. One iteration costs about as much as one iteration of *k*-means, and no annealing is
needed.

The method is described in the paper

> S. Ubukata, T. Maekawa, A. Notsu, K. Honda, "Edge *k*-means: Learning Prototypes and Their Connectivity Graph by
> Minimum-Cost Edge Assignment" (submitted for publication).

## Usage

```python
import ekm

model = ekm.fit(X)                               # X: array of n data points in d dimensions
labels = ekm.clusters(model, n_clusters=2)       # cluster of each data point
```

- `ekm.fit(X, n_prototypes=None, lam=0.3)` returns the prototypes `model.B`, the edge weights `model.W`, and the
  nearest and second-nearest prototype of each data point `model.pairs`. By default, the number of prototypes is
  ⌈√n⌉ and the initial prototypes are chosen by k-means++ seeding (`seed`).
- `ekm.clusters(model, n_clusters=S)` extracts `S` clusters by density linkage; `method="threshold"` uses the
  threshold cut. `ekm.clusters(model, theta=t)` removes the edges with weights below `t` and returns the connected
  components, so that the number of clusters is an outcome.
- `ekm.fit(X, annealing_epochs=20)` precedes EKM by a short annealing (annealed EKM), which helps when only a few
  prototypes represent a cluster.
- `ekm.objective(X, model)` returns the objective function and its two terms, the quantization error measured from
  points on the edges and the graph Laplacian energy weighted by `model.W`.

`python example.py` runs EKM on two half circles with background noise, the controlled data of the paper, and prints
the adjusted Rand index of the extracted clusters.

The experiments reported in the paper used a faster implementation of the same algorithm (compiled inner loops),
which returns the same results.

## Requirements

Python 3 with NumPy and SciPy (tested with Python 3.14, NumPy 2.5, and SciPy 1.18). Matplotlib is used only for the
figure that the example writes.

## License

MIT License; see [LICENSE](LICENSE).
