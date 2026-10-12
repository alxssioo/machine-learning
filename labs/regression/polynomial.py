import numpy as np
import json

from multivariate import MultivariateLinearRegression
from itertools import combinations_with_replacement


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    r"""Monomials with interactions up to `degree`, no bias column.

    *Input*: the feature matrix X with dimensions (M, N) and the maximum degree of the polynomial.
    *Output*: the transformed feature matrix Z with dimensions (M, P), where
    P = C(N + degree, degree) - 1 is the number of monomials of total degree 1, ..., degree
    (for a single feature, P = degree).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    columns: list[np.ndarray] = []
    for d in range(1, degree + 1):
        for comb in combinations_with_replacement(range(X.shape[1]), d):
            columns.append(np.prod(X[:, comb], axis=1))
    return np.column_stack(columns) if columns else np.empty((len(X), 0))

class PolynomialRegression:
    r"""Polynomial regression built on `MultivariateLinearRegression`.

    Constructed with a `degree`; after `fit`, the internal linear model lives in
    `self.model` and its parameters are accessible via `self.model.parameters`.
    The leading entry is the bias.
    """

    def __init__(self, degree, regularizer=0.0, method="normal_equation",
                 learning_rate=0.1, n_iterations=5000):
        self.degree = degree
        self.regularizer = regularizer
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegression":
        """Expand `X` to polynomial features and fit the internal linear model."""
        X_poly = polynomial_features(X, self.degree)

        self.model = MultivariateLinearRegression(
            method=self.method,
            learning_rate=self.learning_rate,
            n_iterations=self.n_iterations,
            regularizer=self.regularizer
        )
        self.model.fit(X_poly, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Expand `X` the same way as in fit() and return the internal model's prediction."""
        X_poly = polynomial_features(X, self.degree)

        return self.model.predict(X_poly)

    def save(self, path: str) -> None:
        """Write parameters, degree and hyperparameters to a JSON file."""
        with open(path, "w") as fh:
            json.dump({"parameters": np.asarray(self.model.parameters).tolist(),
                       "regularizer": self.regularizer,
                       "learning_rate": self.learning_rate,
                       "degree": self.degree}, fh)

    def load(self, path: str) -> "PolynomialRegression":
        """Load a model written by `save`; return `self` ready to `predict`."""
        with open(path) as fh:
            d = json.load(fh)
        self.degree = d["degree"]
        self.regularizer = d["regularizer"]
        self.learning_rate = d["learning_rate"]
        self.model = MultivariateLinearRegression(
            regularizer=self.regularizer, learning_rate=self.learning_rate)
        self.model.parameters = np.asarray(d["parameters"], dtype=float)
        return self

