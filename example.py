"""Edge k-means on two half circles with background noise (the controlled data of the paper).

    python example.py            prints the results; with matplotlib installed, also writes example.png

Of n two-dimensional points, a fraction eta is uniform on a rectangle around two half circles of radius 1, and the
others lie on the half circles, displaced by a vector uniform on a disk of radius r.  The two half circles are the
clusters to recover; they are not convex, so k-means with two prototypes cannot separate them.
"""
import numpy as np
from scipy.special import comb

import ekm


def two_half_circles(n=2000, r=0.1, eta=0.02, seed=0):
    rng = np.random.default_rng(seed)
    n_noise = round(eta * n)
    half = (n - n_noise) // 2
    parts, labels = [], []
    for which, m in enumerate((half, n - n_noise - half)):
        t = rng.uniform(0.0, np.pi, m)
        base = np.c_[np.cos(t), np.sin(t)] if which == 0 else np.c_[1.0 - np.cos(t), 0.5 - np.sin(t)]
        radius, angle = r * np.sqrt(rng.uniform(0.0, 1.0, m)), rng.uniform(0.0, 2.0 * np.pi, m)
        parts.append(base + np.c_[radius * np.cos(angle), radius * np.sin(angle)])
        labels.append(np.full(m, which))
    parts.append(np.c_[rng.uniform(-1.0 - r, 2.0 + r, n_noise), rng.uniform(-0.5 - r, 1.0 + r, n_noise)])
    labels.append(np.full(n_noise, -1))                       # background points
    return np.vstack(parts), np.concatenate(labels)


def adjusted_rand_index(a, b):
    table = np.unique(np.c_[a, b], axis=0, return_counts=True)[1]
    pairs = comb(table, 2).sum()
    rows, cols = comb(np.unique(a, return_counts=True)[1], 2).sum(), comb(np.unique(b, return_counts=True)[1], 2).sum()
    expected = rows * cols / comb(len(a), 2)
    return (pairs - expected) / ((rows + cols) / 2 - expected)


def main():
    X, y = two_half_circles()
    model = ekm.fit(X, lam=0.3)                                # K = ceil(sqrt(n)) = 45 prototypes
    print(f"{len(X)} data points, {len(model.B)} prototypes, {model.iterations} iterations, converged: {model.converged}")

    # every data point supports one edge, and the objective is a quantization error plus a Laplacian energy
    terms = ekm.objective(X, model)
    print(f"edge weights: {np.count_nonzero(np.triu(model.W))} edges, total weight {np.triu(model.W).sum()}")
    print(f"objective {terms['J']:.6f} = error {terms['error']:.6f} + Laplacian energy {terms['energy']:.6f}"
          f"  (difference {terms['J'] - terms['error'] - terms['energy']:.1e})")

    on_circles = y >= 0
    for method in ("threshold", "density"):
        labels = ekm.clusters(model, n_clusters=2, method=method)
        print(f"{method:9s}: ARI {adjusted_rand_index(y[on_circles], labels[on_circles]):.3f} on the points of the half circles")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    nearest = model.pairs[:, 0]
    sizes = np.bincount(nearest, minlength=len(model.B)).astype(float)
    groups = ekm.density_linkage(model.W, model.B, 2, sizes)   # group of each prototype
    labels = groups[nearest]
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.scatter(X[:, 0], X[:, 1], s=3, c=labels, cmap="coolwarm", alpha=0.5, linewidths=0)
    for j, k in zip(*np.nonzero(np.triu(model.W))):
        same = groups[j] == groups[k]
        ax.plot(*model.B[[j, k]].T, color="black" if same else "gray", linestyle="-" if same else ":",
                linewidth=0.3 + 2.2 * model.W[j, k] / model.W.max())
    ax.scatter(model.B[:, 0], model.B[:, 1], s=28, facecolor="white", edgecolor="black", linewidths=1.0, zorder=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig("example.png", dpi=200, bbox_inches="tight")
    print("written example.png (line width: edge weight; dotted: edges between the two clusters)")


if __name__ == "__main__":
    main()
