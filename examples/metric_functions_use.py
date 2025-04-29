import os, sys
project_directory = os.getcwd()
sys.path.append(project_directory)

import pandas as pd
from bciflow.modules.analysis.metric_functions import accuracy, kappa, logloss, rmse
import numpy as np


# Define the data as a dictionary
data = {
    "fold": [1] * 20,
    "tmin": [0.0] * 20,
    "true_label": [
        "right-hand", "left-hand", "left-hand", "right-hand", "right-hand",
        "left-hand", "left-hand", "right-hand", "right-hand", "left-hand",
        "right-hand", "right-hand", "right-hand", "left-hand", "left-hand",
        "left-hand", "right-hand", "left-hand", "right-hand", "left-hand"
    ],
    "left-hand": [
        0.5061, 0.5148, 0.5284, 0.53, 0.5124,
        0.5199, 0.514, 0.5005, 0.4933, 0.5236,
        0.5068, 0.4936, 0.5295, 0.5187, 0.5217,
        0.5101, 0.5117, 0.5272, 0.5042, 0.5256
    ],
    "right-hand": [
        0.4939, 0.4852, 0.4716, 0.47, 0.4876,
        0.4801, 0.486, 0.4995, 0.5067, 0.4764,
        0.4932, 0.5064, 0.4705, 0.4813, 0.4783,
        0.4899, 0.4883, 0.4728, 0.4958, 0.4744
    ]
}

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# Extract true labels
true_labels = df["true_label"]

# Extract predicted probabilities
prob_df = df[["left-hand", "right-hand"]]

# Calculate metrics
acc = accuracy(true_labels, prob_df)
kappa_score = kappa(true_labels, prob_df)
log_loss_value = logloss(true_labels, prob_df)
rmse_value = rmse(true_labels, prob_df)

# Display results
print(f"Accuracy: {acc:.4f}")
print(f"Cohen's Kappa: {kappa_score:.4f}")
print(f"Log Loss: {log_loss_value:.4f}")
print(f"RMSE: {rmse_value:.4f}")
