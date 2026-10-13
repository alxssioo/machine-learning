import numpy as np

from polynomial import PolynomialRegression
from evaluation import rmsd


def grid_search(X, y, degrees, lambdas, seed=42):
    """
    Find the best polynomial degree and regularization strength
    using an 80/20 training-validation split.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))

    split = int(0.8 * len(X))

    train_idx = indices[:split]
    val_idx = indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    results = []

    best_degree = None
    best_lambda = None
    best_error = np.inf

    for degree in degrees:
        for lam in lambdas:
            model = PolynomialRegression(
                degree=degree,
                regularizer=lam
            )

            model.fit(X_train, y_train)

            error = rmsd(
                y_val,
                model.predict(X_val)
            )

            results.append((degree, lam, error))

            if error < best_error:
                best_error = error
                best_degree = degree
                best_lambda = lam

    # Train the final model using all available data
    best_model = PolynomialRegression(
        degree=best_degree,
        regularizer=best_lambda
    )

    best_model.fit(X, y)

    return best_model, results