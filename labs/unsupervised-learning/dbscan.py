import numpy as np
from distances import euclidean_distance

class DBSCAN:
    """Density-based clustering. Hyperparameters set on construction; after `fit`,
    `self.labels` holds the result (-1 = noise, clusters = 0, 1, 2, ...)."""

    def __init__(self, eps, min_pts, distance=None):
        self.eps = eps
        self.min_pts = min_pts
        self.distance = distance if distance is not None else euclidean_distance
        self.labels = None

    # Task 3a: components
    def region_query(self, samples: np.ndarray, m: int) -> np.ndarray:
        """Indices of samples which are closer than `self.eps` to sample `m`
        with respect to `self.distance`."""
        distances = self.distance(samples[m:m+1], samples)[0]   # shape (M,)
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors

    def find_core_points(self, samples: np.ndarray) -> np.ndarray:
        r"""Boolean mask of core points (neighborhood size >= `self.min_pts`)."""
        is_core = np.zeros(len(samples), dtype=bool)
        for m in range(len(samples)):
            is_core[m] = len(self.region_query(samples, m)) >= self.min_pts
        return is_core

    # Task 3b: the full algorithm
    def fit(self, samples: np.ndarray) -> "DBSCAN":
        r"""Run DBSCAN, set `self.labels` (-1 = noise), return `self`."""
        UNVISITED, NOISE = -2, -1
        labels = np.full(len(samples), UNVISITED, dtype=int)
        cluster = 0

        for m in range(len(samples)):
            if labels[m] != UNVISITED:
                continue

            neighbors = self.region_query(samples, m)
            if len(neighbors) < self.min_pts:
                labels[m] = NOISE
                continue

            labels[m] = cluster
            queue = list(neighbors)
            while queue:
                j = queue.pop()
                if labels[j] == NOISE:
                    labels[j] = cluster
                if labels[j] != UNVISITED:
                    continue
                labels[j] = cluster
                j_neighbors = self.region_query(samples, j)
                if len(j_neighbors) >= self.min_pts:
                    queue.extend(j_neighbors)

            cluster += 1

        self.labels = labels
        return self