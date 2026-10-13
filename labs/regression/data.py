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

def generate_assumption_datasets(seed=7):
    rng = np.random.default_rng(seed)

    datasets = {}

    # 1. Linearity violated: quadratic relationship
    x = np.linspace(-3, 3, 120)
    y = 0.5 * x**2 + rng.normal(0, 0.3, x.size)

    datasets["Linearity"] = (x, y)

    # 2. Independence violated: correlated errors
    x = np.linspace(0, 10, 200)
    noise = np.zeros(x.size)

    for i in range(1, x.size):
        noise[i] = 0.95 * noise[i - 1] + rng.normal(0, 0.3)

    y = 1.0 + 0.8 * x + noise

    datasets["Independence"] = (x, y)

    # 3. Normality violated: skewed errors
    x = np.linspace(0, 10, 200)
    y = 1.0 + 0.8 * x + rng.standard_exponential(x.size)**2 - 1.0

    datasets["Normality"] = (x, y)

    # 4. Homoscedasticity violated: increasing variance
    x = np.linspace(0, 10, 200)
    y = 1.0 + 0.8 * x + rng.normal(0, 0.05 + 0.25 * x)

    datasets["Homoscedasticity"] = (x, y)

    return datasets
