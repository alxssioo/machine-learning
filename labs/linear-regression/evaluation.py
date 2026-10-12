import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Mean squared error MSE."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Mean absolute error MAE."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return np.mean(np.abs(y_true- y_pred))


def rmsd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Root mean squared deviation RMSD."""
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""Coefficient of determination R^2."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


def evaluate(y: np.ndarray, predictions: np.ndarray) -> None:
    """Print all regression evaluation metrics."""
    print(f"MSE:  {mse(y, predictions):.4f}")
    print(f"MAE:  {mae(y, predictions):.4f}")
    print(f"RMSD: {rmsd(y, predictions):.4f}")
    print(f"R²:   {r2_score(y, predictions):.4f}")