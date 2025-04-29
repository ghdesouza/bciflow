"""
metric_functions.py

Description
-----------
This module contains the functions to calculate the metrics accuracy, Cohen's
kappa coefficient, logarithmic loss and root-mean-squared error. These
functions are used to evaluate the performance of the given data.

Dependencies
------------
pandas
sklearn.metrics

"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    log_loss,
    root_mean_squared_error,
)


def accuracy(correct: pd.Series, probs: pd.DataFrame) -> float:
    """Calculates the accuracy given the correct labels and the predicted
    probabilities.

    Description
    -----------
    Calculates the accuracy given the correct labels and the predicted
    probabilities.

    Parameters
    ----------
    correct : pandas.Series
            Correct labels.
    probs : pandas.DataFrame
            Predicted probabilities.

    Returns
    -------
    float
            Accuracy value.

    """

    return accuracy_score(correct, probs.idxmax(axis=1))


def kappa(correct: pd.Series, probs: pd.DataFrame) -> float:
    """Calculates the Cohen's kappa coefficient given the correct labels and
    the predicted probabilities.

    Description
    -----------
    Calculates the Cohen's kappa coefficient given the correct labels and the
    predicted probabilities.

    Parameters
    ----------
    correct : pandas.Series
            Correct labels.
    probs : pandas.DataFrame
            Predicted probabilities.

    Returns
    -------
    float
            Kappa value

    """

    return cohen_kappa_score(correct, probs.idxmax(axis=1))


def logloss(correct: pd.Series, probs: pd.DataFrame) -> float:
    """Calculates the logarithmic loss given the correct labels and the
    predicted probabilities.

    Description
    -----------
    Calculates the logarithmic loss given the correct labels and the predicted
    probabilities.

    Parameters
    ----------
    correct : pandas.Series
            Correct labels.
    probs : pandas.DataFrame
            Predicted probabilities.

    Returns
    -------
    float
            Logarithmic loss value.

    """

    return log_loss(correct, probs, labels=probs.columns)


def rmse(correct: pd.Series, probs: pd.DataFrame) -> float:
    """Calculates the root-mean-squared error given the correct labels and the
    predicted probabilities.

    Description
    -----------
    Calculates the root-mean-squared error given the correct labels and the
    predicted probabilities.

    Parameters
    ----------
    correct : pandas.Series
            Correct labels.
    probs : pandas.DataFrame
            Predicted probabilities.

    Returns
    -------
    float
            Root-mean-squared error value.

    """

    return root_mean_squared_error(pd.get_dummies(correct), probs)
