import numpy as np

def euclidean_distance(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Calculate the Euclidean distance between every row in U and every row in V.

    Args:
        U (np.ndarray):
            First set of points with shape (n_points_U, n_features).

        V (np.ndarray):
            Second set of points with shape (n_points_V, n_features).

    Returns:
        np.ndarray:
            Distance matrix with shape (n_points_U, n_points_V).

            Entry [i, j] contains the distance between
            U[i] and V[j].

    How it works:
        1. Check that both arrays are 2D and have the same number of features.

        2. U[:, np.newaxis] changes the shape of U from:
               (n_points_U, n_features)
           to:
               (n_points_U, 1, n_features)

        3. V[np.newaxis] changes the shape of V from:
               (n_points_V, n_features)
           to:
               (1, n_points_V, n_features)

        4. NumPy broadcasting subtracts every point in V
           from every point in U.

        5. np.linalg.norm(..., ord=2, axis=2) calculates the
           Euclidean distance for each pair of points.

           Euclidean distance:
               sqrt((x1-x2)^2 + (y1-y2)^2 + ...)
    """

    assert U.ndim == 2 and V.ndim == 2 and U.shape[1] == V.shape[1]
    distance = np.linalg.norm(U[:, np.newaxis] - V[np.newaxis], ord=2, axis=2)
    return distance


def manhattan_distance(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Calculate the Manhattan distance between every row in U and every row in V.

    Args:
        U (np.ndarray):
            First set of points with shape (n_points_U, n_features).

        V (np.ndarray):
            Second set of points with shape (n_points_V, n_features).

    Returns:
        np.ndarray:
            Distance matrix with shape (n_points_U, n_points_V).

            Entry [i, j] contains the distance between
            U[i] and V[j].

    How it works:
        The broadcasting works exactly like in euclidean_distance().

        The difference is ord=1.

        Manhattan distance:
            |x1-x2| + |y1-y2| + ...

        ord=1 therefore sums the absolute coordinate differences.
    """

    assert U.ndim == 2 and V.ndim == 2 and U.shape[1] == V.shape[1]
    distance = np.linalg.norm(U[:, np.newaxis] - V[np.newaxis], ord=1, axis=2)
    return distance