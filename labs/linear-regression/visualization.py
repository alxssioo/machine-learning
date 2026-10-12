import numpy as np
import matplotlib.pyplot as plt


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