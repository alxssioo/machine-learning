import numpy as np

class UnivariateLinearRegression:
    r"""Univariate linear regression y = theta_0 + theta_1*x .

    The fitting `method` is chosen on construction; after `fit`, `self.theta_0`
    and `self.theta_1` hold the intercept and slope.
    """

    def __init__(self, method="closed_form", learning_rate=0.1, n_iterations=5000):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta_0 = None
        self.theta_1 = None
        self.cost_history = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "UnivariateLinearRegression":
        """Fit the line to 1-D inputs `x` and targets `y`; return `self`."""
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if self.method == "closed_form":
            self._fit_closed_form(x, y)
        elif self.method == "gradient_descent":
            self._fit_gradient_descent(x, y)
        else:
            raise ValueError(f"unknown method {self.method!r}")
        return self

    # Task 1a: closed form
    def _fit_closed_form(self, x: np.ndarray, y: np.ndarray) -> None:
        r"""Set `self.theta_1` and `self.theta_0` from the closed-form equations."""
        mu_x, mu_y = x.mean(), y.mean()
        self.theta_1 = np.sum((x - mu_x) * (y - mu_y)) / np.sum((x - mu_x) **2)
        self.theta_0 = mu_y - self.theta_1 * mu_x

    # Task 6a: gradient descent
    def _fit_gradient_descent(self, x: np.ndarray, y: np.ndarray) -> None:
        r"""Set `self.theta_0`, `self.theta_1` by gradient descent (Task 6a)."""
        M = len(x)
        self.theta_0, self.theta_1 = 0.0, 0.0
        self.cost_history = []

        for _ in range(self.n_iterations):
            y_pred = self.theta_0 + self.theta_1 * x
            error = y_pred - y

            grad_0 = np.sum(error) / M
            grad_1 = np.sum(error * x) / M

            cost = np.sum(error ** 2) / (2 * M)
            self.cost_history.append(cost)

            self.theta_0 -= self.learning_rate * grad_0
            self.theta_1 -= self.learning_rate * grad_1

    def predict(self, x: np.ndarray) -> np.ndarray:
        r"""$\hat y = \theta_0 + \theta_1 x$ for the fitted parameters."""
        x = np.asarray(x, dtype=float).ravel()
        return self.theta_0 + self.theta_1 * x