import numpy as np

def generate_univariate_data(n_samples=100, noise=2.0, seed=None):
    rng = np.random.default_rng(seed)
    # Generate random line parameters
    theta_0 = rng.uniform(-10, 10)
    theta_1 = rng.uniform(-5, 5)
    # Generate points around the line
    x = rng.uniform(0, 10, n_samples)
    y = theta_0 + theta_1 * x + rng.normal(0, noise, n_samples)
    return x, y, np.array([theta_0, theta_1])


def generate_multivariate_data(n_samples=100, n_features=2, noise=2.0, seed=None):
    rng = np.random.default_rng(seed)
    # Generate random model parameters (including intercept)
    theta = rng.uniform(-5, 5, n_features + 1)
    # Generate random samples
    X = rng.uniform(0, 10, (n_samples, n_features))
    # Generate targets with Gaussian noise
    y = theta[0] + X @ theta[1:] + rng.normal(0, noise, n_samples)
    return X, y, theta
