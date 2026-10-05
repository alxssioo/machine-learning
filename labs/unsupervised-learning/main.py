import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from kmean import KMeans
from dbscan import DBSCAN
from distances import manhattan_distance, euclidean_distance

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
else:
    epsilon = float(input("Chose epsilon for DBSCAN: "))
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

    dbscan = DBSCAN(
        eps=epsilon,
        min_pts=5
    ).fit(samples)

    plotDBSCAN(dbscan, samples)
else:
    # Create a K-Means model with 3 clusters and the selected distance function
    # Fit the model to the samples:
    # initialize centroids -> assign points -> update centroids -> repeat until convergence
    kmeans = KMeans(
        n_clusters=3,
        distance=distance
    ).fit(samples)
    plotKmean()

