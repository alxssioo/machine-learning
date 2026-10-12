import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from kmean import KMeans
from dbscan import DBSCAN
from distances import manhattan_distance, euclidean_distance
from evaluation import potential_curve, silhouette_curve


def plotKmean():
    # Print the final centroid coordinates
    print("Centroids:")

    print(kmeans.centroids)

    # Print the cluster assignment for every sample
    print("Labels:")
    print(kmeans.labels)

    # Plot all samples.
    # samples[:, 0] selects all x-coordinates.
    # samples[:, 1] selects all y-coordinates.
    # c=kmeans.labels colors each point according to its assigned cluster.
    # alpha=0.5 makes the points slightly transparent.
    plt.scatter(
        samples[:, 0],
        samples[:, 1],
        c=kmeans.labels,
        alpha=0.5
    )

    # Convert the saved centroid positions from a Python list to a NumPy array.
    #
    # Shape:
    #   (n_iterations, n_clusters, n_features)
    #
    # For example, with 3 clusters in 2D:
    #   history.shape -> (number_of_iterations, 3, 2)
    history = np.array(kmeans.centroid_history)

    # Plot the path taken by every centroid.
    for k in range(kmeans.n_clusters):
        plt.plot(
            history[:, k, 0],
            history[:, k, 1],
            marker="o"
        )

    # Plot the final centroid positions.
    plt.scatter(
        kmeans.centroids[:, 0],
        kmeans.centroids[:, 1],
        marker="x",
        s=200,
        linewidths=3
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plotDBSCAN(dbscan: DBSCAN, samples: np.ndarray):
    labels = dbscan.labels

    for lab in np.unique(labels):
        mask = labels == lab

        color = "black" if lab == -1 else mpl.cm.tab10(lab % 10)

        plt.scatter(
            *samples[mask].T,
            s=15,
            color=color,
            label="noise" if lab == -1 else f"cluster {lab}"
        )

    plt.legend()
    plt.title("DBSCAN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


chosenLearning = input("Chose learning algorithm: DBSCAN or Kmeans ").lower()
if chosenLearning not in ("kmeans", "dbscan"):
    raise ValueError("Choose either kmeans or dbscan")
epsilon = 0.1
distance = euclidean_distance

if chosenLearning == "kmeans":
    chosenDistance = input("Choose distance (1 for euclidean, 2 for manhattan): ").strip()
    if chosenDistance == "1":
        distance = euclidean_distance
    elif chosenDistance == "2":
        distance = manhattan_distance
    else:
        raise ValueError("Unknown distance")

# Generate 100 random samples with 0 <= x < 10 and 0 <= y < 10
samples = np.random.uniform(
    low=0,
    high=10,
    size=(100, 2)
)

if chosenLearning == "dbscan":
    rng = np.random.default_rng(0)

    cluster1 = rng.normal(loc=[2, 2], scale=0.5, size=(40, 2))
    cluster2 = rng.normal(loc=[6, 7], scale=0.6, size=(40, 2))
    cluster3 = rng.normal(loc=[8, 3], scale=0.4, size=(40, 2))

    samples = np.vstack([cluster1, cluster2, cluster3])

    epsilon = float(input("Choose epsilon: "))
    min_pts = int(input("Choose minPts: "))

    dbscan = DBSCAN(
        eps=epsilon,
        min_pts=min_pts
    ).fit(samples)

    plotDBSCAN(dbscan, samples)
else:
    # Evaluate different values of K
    ks = np.arange(2, min(11, len(samples)))

    scores = silhouette_curve(
        samples,
        ks,
        distance=distance
    )

    # Select K with the highest silhouette score
    best_k = int(ks[np.argmax(scores)])

    print(f"Optimal K: {best_k}")
    print(f"Silhouette score: {np.max(scores):.4f}")

    # Train K-Means with the selected K
    kmeans = KMeans(
        n_clusters=best_k,
        distance=distance
    ).fit(samples)

    # Elbow Method
    ks_elbow = np.arange(1, min(11, len(samples) + 1))

    potentials = potential_curve(
        samples,
        ks_elbow,
        distance=distance
    )

    plt.plot(ks_elbow, potentials, marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Potential")
    plt.title("Elbow Method")
    plt.show()

    # Silhouette Analysis (reuse calculated scores)
    plt.plot(ks, scores, marker="o")

    # Mark the optimal K
    plt.axvline(best_k, color="red", linestyle="--", label=f"Optimal K = {best_k}")

    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Analysis")
    plt.legend()
    plt.show()

    plotKmean()