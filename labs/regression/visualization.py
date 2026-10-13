import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from polynomial import PolynomialRegression
from evaluation import r2_score, rmsd
from data import generate_assumption_datasets
from univariate import UnivariateLinearRegression


def plot_univariate(x, y, model):
    """Plot observations, fitted regression line, and residuals."""
    predictions = model.predict(x)
    indices = np.argsort(x)
    plt.figure()

    # Observed samples
    plt.scatter(x, y, alpha=0.6, label="Samples")
    # Fitted regression line
    plt.plot(
        x[indices],
        predictions[indices],
        color="red",
        label="Regression line"
    )

    # Residuals
    plt.vlines(
        x,
        np.minimum(y, predictions),
        np.maximum(y, predictions),
        colors="gray",
        linestyles="dashed",
        alpha=0.5
    )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Univariate Linear Regression")
    plt.legend()
    plt.show()


def plot_multivariate(X, y, model):
    """Plot the regression plane for a model with two features."""
    if X.shape[1] != 2:
        raise ValueError("3D visualization requires exactly two features")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Observed samples
    ax.scatter(
        X[:, 0],
        X[:, 1],
        y,
        alpha=0.6,
        label="Samples"
    )

    # Create a grid spanning both features
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)

    X1, X2 = np.meshgrid(x1, x2)

    grid = np.column_stack([
        X1.ravel(),
        X2.ravel()
    ])

    # Predict targets for every grid point
    predictions = model.predict(grid).reshape(X1.shape)

    # Fitted regression plane
    ax.plot_surface(
        X1,
        X2,
        predictions,
        alpha=0.4
    )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Y")
    ax.set_title("Multivariate Linear Regression")
    ax.legend()

    plt.show()


def plot_residuals(y, predictions):
    """Plot residuals against predicted values."""
    residuals = np.asarray(y) - np.asarray(predictions)

    plt.figure()

    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


def plot_residual_distribution(y, predictions):
    """Visualize the distribution of residuals."""
    residuals = np.asarray(y) - np.asarray(predictions)

    plt.figure()

    plt.hist(
        residuals,
        bins=20,
        density=True,
        alpha=0.7
    )

    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.title("Residual Distribution")
    plt.show()


def plot_cost_history(model):
    """Plot cost against iteration number for gradient descent."""
    if model.cost_history is None:
        print("No cost history available.")
        return

    plt.figure()

    plt.plot(model.cost_history)

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Gradient Descent Convergence")
    plt.show()


def plot_qq(y, predictions):
    """Q-Q plot to assess residual normality."""
    residuals = np.asarray(y) - np.asarray(predictions)
    stats.probplot(residuals, dist="norm", plot=plt)

    plt.title("Q-Q Plot of Residuals")
    plt.show()


def plot_standardized_residuals(y, predictions):
    """Plot standardized residuals against predicted values."""
    residuals = np.asarray(y) - np.asarray(predictions)

    std = np.std(residuals)

    if std == 0:
        raise ValueError("Cannot standardize residuals with zero variance")

    standardized = residuals / std

    plt.scatter(predictions, standardized, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")

    plt.xlabel("Predicted values")
    plt.ylabel("Standardized residuals")
    plt.title("Standardized Residual Plot")
    plt.show()


def plot_polynomial():
    rng = np.random.default_rng(9)

    x = np.sort(rng.uniform(-1, 1, size=40))
    true = np.sin(2.5 * x)
    y = true + rng.normal(0, 0.12, size=x.size)

    xs = np.linspace(-1, 1, 300)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for ax, degree in zip(axes, [1, 3, 9]):
        model = PolynomialRegression(degree=degree).fit(
            x.reshape(-1, 1), y
        )

        ax.scatter(x, y, s=14, alpha=0.6)

        ax.plot(
            xs,
            np.sin(2.5 * xs),
            "k--",
            lw=1,
            label="true"
        )

        ax.plot(
            xs,
            model.predict(xs.reshape(-1, 1)),
            "r-",
            lw=2,
            label="fit"
        )

        ax.set_title(
            f"degree {degree}, "
            f"$R^2$={r2_score(y, model.predict(x.reshape(-1, 1))):.3f}"
        )

        ax.set_ylim(-1.8, 1.8)
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_regression_assumptions():
    datasets = generate_assumption_datasets()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for ax, (name, (x, y)) in zip(axes.flat, datasets.items()):
        model = UnivariateLinearRegression().fit(x, y)

        predictions = model.predict(x)
        residuals = y - predictions

        if name == "Normality":
            ax.hist(residuals, bins=25)
            ax.set_xlabel("Residual")
            ax.set_ylabel("Count")

        else:
            ax.scatter(x, residuals, s=12)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("X")
            ax.set_ylabel("Residual")

        ax.set_title(f"{name} violated")

    fig.tight_layout()
    plt.show()


def plot_overfitting():
    rng = np.random.default_rng(9)

    x = np.sort(rng.uniform(-1, 1, size=40))
    y = np.sin(2.5 * x) + rng.normal(0, 0.12, size=x.size)

    # Split into training and validation data
    indices = rng.permutation(len(x))
    split = int(0.7 * len(x))

    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    xs = np.linspace(-1, 1, 300)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, degree in zip(axes, [1, 3, 12]):
        model = PolynomialRegression(degree=degree)
        model.fit(x_train.reshape(-1, 1), y_train)

        train_error = rmsd(
            y_train,
            model.predict(x_train.reshape(-1, 1))
        )

        val_error = rmsd(
            y_val,
            model.predict(x_val.reshape(-1, 1))
        )

        ax.scatter(x_train, y_train, label="Training")
        ax.scatter(x_val, y_val, label="Validation")

        ax.plot(
            xs,
            np.sin(2.5 * xs),
            "k--",
            label="True function"
        )

        ax.plot(
            xs,
            model.predict(xs.reshape(-1, 1)),
            "r-",
            label="Prediction"
        )

        ax.set_title(
            f"Degree {degree}\n"
            f"Train RMSD: {train_error:.3f}\n"
            f"Val RMSD: {val_error:.3f}"
        )

        ax.set_ylim(-1.8, 1.8)
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_regularization():
    rng = np.random.default_rng(11)

    x = np.sort(rng.uniform(-1, 1, size=18))
    y = np.sin(3.0 * x) + rng.normal(0, 0.15, size=x.size)

    xs = np.linspace(-1, 1, 300)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, lam, label in [
        (axes[0], 0.0, "No regularization"),
        (axes[1], 0.01, "Moderate regularization"),
        (axes[2], 30.0, "Strong regularization")
    ]:
        model = PolynomialRegression(
            degree=12,
            regularizer=lam
        )

        model.fit(x.reshape(-1, 1), y)

        ax.scatter(x, y, label="Samples")

        ax.plot(
            xs,
            np.sin(3.0 * xs),
            "k--",
            label="True function"
        )

        ax.plot(
            xs,
            model.predict(xs.reshape(-1, 1)),
            "r-",
            label="Prediction"
        )

        ax.set_title(f"λ = {lam}\n{label}")
        ax.set_ylim(-2, 2)
        ax.legend()

    fig.tight_layout()
    plt.show()


def plot_regularization_search():
    rng = np.random.default_rng(11)

    x = np.sort(rng.uniform(-1, 1, size=40))
    y = np.sin(3.0 * x) + rng.normal(0, 0.15, size=x.size)

    indices = rng.permutation(len(x))
    split = int(0.8 * len(x))

    tr = indices[:split]
    val = indices[split:]

    lambdas = np.logspace(-6, 2, 40)

    train_errors = []
    val_errors = []

    for lam in lambdas:
        model = PolynomialRegression(
            degree=9,
            regularizer=lam
        )

        model.fit(x[tr].reshape(-1, 1), y[tr])

        train_errors.append(
            rmsd(y[tr], model.predict(x[tr].reshape(-1, 1)))
        )

        val_errors.append(
            rmsd(y[val], model.predict(x[val].reshape(-1, 1)))
        )

    best_idx = np.argmin(val_errors)
    best_lambda = lambdas[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(lambdas, train_errors, label="Training RMSD")
    ax.plot(lambdas, val_errors, label="Validation RMSD")

    ax.axvline(
        best_lambda,
        linestyle="--",
        label=f"Best λ = {best_lambda:.5f}"
    )

    ax.set_xscale("log")
    ax.set_xlabel("λ")
    ax.set_ylabel("RMSD")
    ax.set_title("Choosing Regularization Strength")
    ax.legend()

    plt.show()

    print(f"Best lambda: {best_lambda:.6f}")
    print(f"Validation RMSD: {val_errors[best_idx]:.4f}")


def plot_hyperparameter_search(results):
    degrees = sorted(set(d for d, lam, error in results))
    lambdas = sorted(set(lam for d, lam, error in results))

    fig, ax = plt.subplots(figsize=(9, 6))

    for degree in degrees:
        errors = [
            error
            for d, lam, error in results
            if d == degree
        ]

        line, = ax.plot(
            lambdas[1:],
            errors[1:],
            label=f"Degree {degree}"
        )

        ax.scatter(
            [lambdas[1]],
            [errors[0]],
            marker="x",
            color=line.get_color()
        )

    ax.set_xscale("log")
    ax.set_xlabel("Regularization λ")
    ax.set_ylabel("Validation RMSD")
    ax.set_title("Polynomial Regression Hyperparameter Search")
    ax.legend()

    plt.show()
