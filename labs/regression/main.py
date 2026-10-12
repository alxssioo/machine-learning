import numpy as np
import matplotlib.pyplot as plt

from data import generate_univariate_data, generate_multivariate_data
from univariate import UnivariateLinearRegression
from multivariate import MultivariateLinearRegression
from visualization import plot_qq, plot_standardized_residuals
from evaluation import evaluate
from visualization import plot_polynomial


def run_univariate():
    # Generate random data around a random line
    x, y, true_parameters = generate_univariate_data(
        n_samples=100,
        noise=2.0,
        seed=42
    )

    # Fit using the closed-form solution
    model = UnivariateLinearRegression(
        method="closed_form"
    ).fit(x, y)

    # Fit using gradient descent
    model_gd = UnivariateLinearRegression(
        method="gradient_descent",
        learning_rate=0.01,
        n_iterations=5000
    ).fit(x, y)

    print("\n--- Univariate Linear Regression ---")
    print("True parameters:", true_parameters)
    print("Closed form:", [model.theta_0, model.theta_1])
    print("Gradient descent:", [model_gd.theta_0, model_gd.theta_1])

    # Generate predictions
    predictions = model.predict(x)

    # Plot samples and regression lines
    indices = np.argsort(x)

    plt.scatter(x, y, alpha=0.6, label="Samples")

    plt.plot(
        x[indices],
        predictions[indices],
        label="Closed form"
    )

    plt.plot(
        x[indices],
        model_gd.predict(x)[indices],
        linestyle="--",
        label="Gradient descent"
    )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Univariate Linear Regression")
    plt.legend()
    plt.show()

    # Plot gradient descent convergence
    plt.figure()
    plt.plot(model_gd.cost_history)

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Univariate Gradient Descent")
    plt.show()

    plot_qq(y, predictions)
    plot_standardized_residuals(y, predictions)

    print("\nClosed-form evaluation:")
    evaluate(y, model.predict(x))

    print("\nGradient descent evaluation:")
    evaluate(y, model_gd.predict(x))


def run_multivariate():
    # Generate random data around a random plane
    X, y, true_parameters = generate_multivariate_data(
        n_samples=100,
        n_features=2,
        noise=2.0,
        seed=42
    )

    # Fit using the normal equation
    model = MultivariateLinearRegression(
        method="normal_equation",
        regularizer=0.0
    ).fit(X, y)

    # Fit using gradient descent
    model_gd = MultivariateLinearRegression(
        method="gradient_descent",
        learning_rate=0.01,
        n_iterations=5000
    ).fit(X, y)

    print("\n--- Multivariate Linear Regression ---")
    print("True parameters:", true_parameters)
    print("Normal equation:", model.parameters)
    print("Gradient descent:", model_gd.parameters)

    # Visualize the regression plane
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        X[:, 0],
        X[:, 1],
        y,
        alpha=0.6,
        label="Samples"
    )

    # Create grid for the regression plane
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 30)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 30)

    X1, X2 = np.meshgrid(x1, x2)

    grid = np.column_stack([
        X1.ravel(),
        X2.ravel()
    ])

    grid_predictions = model.predict(grid).reshape(X1.shape)

    ax.plot_surface(X1, X2, grid_predictions, alpha=0.4)

    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Y")
    ax.set_title("Multivariate Linear Regression")

    plt.show()

    # Plot gradient descent convergence
    plt.figure()
    plt.plot(model_gd.cost_history)

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Multivariate Gradient Descent")
    plt.show()

    predictions = model.predict(X)

    plot_qq(y, predictions)
    plot_standardized_residuals(y, predictions)

    print("\nNormal equation evaluation:")
    evaluate(y, model.predict(X))

    print("\nGradient descent evaluation:")
    evaluate(y, model_gd.predict(X))

def run_polynomial():
    print("\n--- Polynomial Regression ---")
    plot_polynomial()

if __name__ == "__main__":
    run_univariate()
    run_multivariate()
    run_polynomial()