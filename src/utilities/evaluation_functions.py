"""Evaluation functions.

This module provides functions used to evaluate our models such as RMSE.
"""

import math

from sklearn.metrics import mean_squared_error


def get_score(predictions, target_values):
    """Root-mean-squared error of predictions and target values.

    Args:
        predictions (np.array): Predictions of the model as a numpy array.
        target_values (np.array): Target values as a numpy array.

    Returns:
        int: RMSE.
    """
    mse = mean_squared_error(predictions, target_values)
    return math.sqrt(mse)
