import numpy as np
from kmean import KMeans
from distances import manhattan_distance, euclidean_distance


def potential(
        samples: np.ndarray,
        centroids: np.ndarray,
        assignments: np.ndarray,
        distance=euclidean_distance
) -> float:
    assigned = centroids[assignments]

    if distance == manhattan_distance:
        return float(np.sum(np.abs(samples - assigned)))

    return float(np.sum((samples - assigned) ** 2))


def potential_curve(samples: np.ndarray, k_values, n_restarts: int = 5, distance=euclidean_distance) -> np.ndarray:
    """For each K, the best (lowest) potential over n_restarts (seed r per restart)."""
    out = []
    for K in k_values:
        best = np.inf
        for r in range(n_restarts):
            np.random.seed(r)
            km = KMeans(K, distance=distance).fit(samples)
            best = min(best, potential(samples, km.centroids, km.labels, distance))
        out.append(best)
    return np.array(out)


def silhouette_samples(
        samples: np.ndarray,
        assignments: np.ndarray,
        distance=euclidean_distance
) -> np.ndarray:
    r"""Per-point silhouette $s_m \in [-1, 1]$ (0 for singleton clusters)."""
    M = len(samples)
    s = np.zeros(M)
    D = distance(samples, samples)
    clusters = np.unique(assignments)
    if len(clusters) < 2:
        return s

    for m in range(M):
        own = assignments == assignments[m]
        size = own.sum()
        if size == 1:
            continue

        a = D[m, own].sum() / (size - 1)
        b = min(D[m, assignments == k].mean() for k in clusters if k != assignments[m])

        denominator = max(a, b)
        s[m] = (b - a) / denominator if denominator > 0 else 0.0
    return s


def silhouette_score(
        samples: np.ndarray,
        assignments: np.ndarray,
        distance=euclidean_distance
) -> float:
    return float(np.mean(silhouette_samples(samples, assignments, distance)))


def silhouette_curve(
        samples: np.ndarray,
        k_values,
        n_restarts: int = 5,
        distance=euclidean_distance
) -> np.ndarray:
    out = []
    for K in k_values:
        best_pot, best_labels = np.inf, None

        for r in range(n_restarts):
            np.random.seed(r)

            km = KMeans(K, distance=distance).fit(samples)
            pot = potential(samples, km.centroids, km.labels, distance)

            if pot < best_pot:
                best_pot, best_labels = pot, km.labels

        out.append(silhouette_score(samples, best_labels, distance))
    return np.array(out)
