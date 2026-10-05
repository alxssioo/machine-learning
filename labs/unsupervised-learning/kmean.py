import numpy as np
from distances import manhattan_distance, euclidean_distance

class KMeans:
    """
    K-Means clustering algorithm.

    Attributes:
        n_clusters: Number of clusters.
        distance: Distance function used to compare samples.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        centroids: Current cluster centroids.
        centroid_history: Centroid positions over all iterations.
        labels: Final cluster assignment for each sample.
    """

    def __init__(self, n_clusters, distance=None, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.distance = distance if distance is not None else euclidean_distance
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.centroid_history = None
        self.labels = None

    def initialize_centroids(self, samples: np.ndarray) -> np.ndarray:
        """
        Randomly choose the initial cluster centroids from the dataset.

        K-Means needs a starting position for each centroid before the algorithm
        can begin. This function uses Forgy initialization, which means that
        existing samples are randomly selected and used as the initial centroids.

        Args:
            samples (np.ndarray):
                Dataset with shape:
                    (n_samples, n_features)

                Example:
                    If we have 100 two-dimensional points:
                    samples.shape == (100, 2)
                Then:
                    samples.shape[0] == 100   # number of samples
                    samples.shape[1] == 2     # number of features/dimensions

        Returns:
            np.ndarray:
                The randomly selected initial centroids.
                Shape:
                    (n_clusters, n_features)

                Example:
                    If n_clusters = 3 and every sample has 2 features,
                    the returned array has shape:
                        (3, 2)

        How it works:
            1. samples.shape[0] gives the number of samples.
            2. np.random.choice(...) randomly chooses sample indices.
               For example:
                   np.random.choice(
                       100,
                       size=self.n_clusters,
                       replace=False
                   )
               could return:
                   [83, 53, 70]

               size=self.n_clusters:
                   Select exactly one sample for each centroid.

               replace=False:
                   The same sample cannot be selected more than once.

            3. samples[indices] uses NumPy indexing to select those rows.
               For example:
                   samples[[83, 53, 70]]

               returns the samples at indices 83, 53, and 70.
               These samples become the initial centroid positions.
        """

        indices = np.random.choice(
            samples.shape[0],
            size=self.n_clusters,
            replace=False
        )
        centroids = samples[indices]
        return centroids

    def assign(self, samples: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each sample to the nearest centroid.

        Args:
            samples (np.ndarray):
                Dataset with shape:
                    (n_samples, n_features)

                Example:
                    100 two-dimensional points:
                    samples.shape == (100, 2)

            centroids (np.ndarray):
                Current centroid positions with shape:
                    (n_clusters, n_features)

                Example:
                    3 centroids in 2D:
                    centroids.shape == (3, 2)

        Returns:
            np.ndarray:
                One cluster index for every sample.
                Shape:
                    (n_samples)

                Example:
                    [0, 2, 1, 0, 0, 2, ...]

                means:
                    sample 0 belongs to centroid 0
                    sample 1 belongs to centroid 2
                    sample 2 belongs to centroid 1
                    ...

        How it works:
            1. self.distance(samples, centroids) calculates the distance
               from every sample to every centroid.

               If we have:
                   100 samples
                   3 centroids

               then the resulting distance matrix has shape:
                   (100, 3)

               Example:
                   distances = [
                       [1.2, 5.3, 8.1],
                       [4.5, 2.0, 3.7],
                       [6.1, 7.2, 0.8]
                   ]

               Each row belongs to one sample.
               Each column belongs to one centroid.

            2. np.argmin(distances, axis=1) finds the index of the
               smallest distance in each row.

               Example:
                   [1.2, 5.3, 8.1] -> index 0
                   [4.5, 2.0, 3.7] -> index 1
                   [6.1, 7.2, 0.8] -> index 2

               Therefore:
                   assignments = [0, 1, 2]

            3. The returned index is the cluster number assigned to
               each sample.
        """
        distances = self.distance(samples, centroids)
        assignments = np.argmin(distances, axis=1)
        return assignments

    def update_centroids(self,
                         samples: np.ndarray,
                         assignments: np.ndarray,
                         centroids: np.ndarray
                         ) -> np.ndarray:
        """
        Recalculate the position of each centroid based on its assigned samples.

        Args:
            samples (np.ndarray):
                Dataset with shape (n_samples, n_features).

            assignments (np.ndarray):
                Cluster index for each sample.
                Example:
                    [0, 0, 2, 1, 2]

            centroids (np.ndarray):
                Current centroid positions with shape
                (n_clusters, n_features).

        Returns:
            np.ndarray:
                Updated centroid positions.

        How it works:
            1. A copy of the current centroids is created.
               This also ensures the values are floating point.

            2. For each cluster k, we select all samples whose
               assignment is equal to k.

               Example:
                   assignments == k

               creates a boolean mask such as:

                   [True, False, True, False]

               samples[assignments == k]

               then returns only the samples belonging to cluster k.

            3. If the cluster is not empty:
               - Euclidean distance uses the mean of the members.
               - Manhattan distance uses the median of the members.

            4. Empty clusters keep their previous centroid position.
        """

        new_centroids = centroids.copy().astype(float)

        for k in range(len(centroids)):
            members = samples[assignments == k]

            if len(members) > 0:
                if self.distance == euclidean_distance:
                    new_centroids[k] = members.mean(axis=0)

                elif self.distance == manhattan_distance:
                    new_centroids[k] = np.median(members, axis=0)

        return new_centroids

    def has_converged(self, old_centroids: np.ndarray, new_centroids: np.ndarray) -> bool:
        """
        Check whether the centroids have stopped moving significantly.

        self.distance(old_centroids, new_centroids) creates a distance matrix
        comparing every old centroid with every new centroid.

        np.diag(...) keeps only the distances between matching centroids:
            old centroid 0 -> new centroid 0
            old centroid 1 -> new centroid 1
            ...

        np.max(shift) finds the centroid that moved the most.

        If even the largest movement is smaller than self.tol,
        the algorithm is considered converged.
        """

        shift = self.distance(old_centroids, new_centroids)
        shift = np.diag(shift)
        return bool(np.max(shift) < self.tol)

    def fit(self, samples: np.ndarray) -> "KMeans":
        """
        Run the complete K-Means algorithm on the dataset.

        Steps:
            1. Randomly initialize the centroids.
            2. Save the initial centroid positions for visualization.
            3. Repeat until convergence or max_iter is reached:
                - Assign every sample to its nearest centroid.
                - Recalculate the centroid positions.
                - Store the new centroid positions.
                - Check whether the centroids have stopped moving.
            4. Calculate the final cluster assignments.
            5. Return the fitted KMeans object.

        Args:
            samples (np.ndarray):
                Dataset with shape (n_samples, n_features).

        Returns:
            KMeans:
                The fitted KMeans object.
        """

        self.centroids = self.initialize_centroids(samples)
        self.centroid_history = [self.centroids.copy()]

        for _ in range(self.max_iter):
            # Assign each sample to its currently nearest centroid
            labels = self.assign(samples, self.centroids)

            # Move each centroid to the center of its assigned samples
            new_centroids = self.update_centroids(
                samples,
                labels,
                self.centroids,
            )

            # Save the new centroid positions for visualization
            self.centroid_history.append(new_centroids.copy())

            # Stop if the centroids moved less than the tolerance
            if self.has_converged(self.centroids, new_centroids):
                self.centroids = new_centroids
                break

            # Otherwise use the new centroids in the next iteration
            self.centroids = new_centroids

        # One final assignment using the final centroid positions
        self.labels = self.assign(samples, self.centroids)
        return self

