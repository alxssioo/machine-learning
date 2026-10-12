import numpy as np
import json

class MultivariateLinearRegression:
    r"""Multivariate linear regression.

    Hyperparameters are set on construction. After `fit`, `self.parameters` holds the result.
    The leading entry is the bias.
    """

    def __init__(self, method="normal_equation", regularizer=0.0,
                 learning_rate=0.1, n_iterations=5000):
        self.method = method
        self.regularizer = regularizer
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.parameters = None
        self.cost_history = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultivariateLinearRegression":
        """Fit to features `X` (shape `(M, N)`) and targets `y`; return `self`."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        if self.method == "normal_equation":
            self._fit_normal_equation(X, y)
        elif self.method == "gradient_descent":
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError(f"unknown method {self.method!r}")
        return self

    # Task 2: normal equation  (+ Task 5a: L2 regularization)
    def _fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> None:
        r"""Set `self.parameters` from the (regularized) normal equation."""
        Xb = np.column_stack([np.ones(len(X)), X])
        n_params = Xb.shape[1]
        R = np.eye(n_params)
        R[0, 0] = 0.0

        self.parameters = np.linalg.solve(
            Xb.T @ Xb + self.regularizer * R,
            Xb.T @ y
        )

    # Task 6b: gradient descent
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        r"""Set `self.parameters` by regularized batch gradient descent."""
        Xb = np.column_stack([np.ones(len(X)), X])
        M, n_params = Xb.shape
        self.parameters = np.zeros(n_params)
        self.cost_history = []

        R = np.eye(n_params)
        R[0, 0] = 0.0

        for _ in range(self.n_iterations):
            error = Xb @ self.parameters - y
            cost = (
                np.sum(error ** 2)
                + self.regularizer * np.sum(self.parameters[1:] ** 2)
            ) / (2 * M)

            self.cost_history.append(cost)

            gradient = (
                Xb.T @ error
                + self.regularizer * (R @ self.parameters)
            ) / M

            self.parameters -= self.learning_rate * gradient

    def predict(self, X: np.ndarray) -> np.ndarray:
        r"""Predict using the fitted parameters."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xb = np.column_stack([np.ones(len(X)), X])
        return Xb @ self.parameters

    def save(self, path: str) -> None:
        """Write the fitted parameters and hyperparameters to a JSON file."""
        with open(path, "w") as fh:
            json.dump({"parameters": np.asarray(self.parameters).tolist(),
                       "regularizer": self.regularizer,
                       "learning_rate": self.learning_rate}, fh)

    def load(self, path: str) -> "MultivariateLinearRegression":
        """Load parameters and hyperparameters written by `save`; return `self`."""
        with open(path) as fh:
            d = json.load(fh)
        self.parameters = np.asarray(d["parameters"], dtype=float)
        self.regularizer = d["regularizer"]
        self.learning_rate = d["learning_rate"]
        return self