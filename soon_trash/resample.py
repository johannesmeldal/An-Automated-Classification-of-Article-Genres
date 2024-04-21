import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def multi_label_resample(X, y, strategy='over'):
    """
    Resample the dataset based on the strategy.
    :param X: Feature set.
    :param y: Binary encoded labels for multi-label classification.
    :param strategy: 'over' for oversampling, 'under' for undersampling.
    :return: Resampled X and y.
    """
    # Binarize y if it's not already in a binary form (list of lists or similar)
    if isinstance(y[0], list):
        mlb = MultiLabelBinarizer()
        y = mlb.fit_transform(y)

    # Count label frequency
    label_counts = np.sum(y, axis=0)

    # Determine the sampling strategy
    if strategy == 'over':
        sampler = RandomOverSampler(random_state=42)
    elif strategy == 'under':
        sampler = RandomUnderSampler(random_state=42)
    else:
        raise ValueError("Unsupported resampling strategy. Use 'over' or 'under'.")

    # Calculate the sampling strategy
    # Find the max number of occurrences of any label for oversampling
    # Find the min for undersampling
    if strategy == 'over':
        sampling_strategy = {i: max(label_counts) for i in range(y.shape[1])}
    else:
        sampling_strategy = {i: min(label_counts) for i in range(y.shape[1])}

    # Resample the dataset
    X_res, y_res = sampler.fit_resample(np.hstack((X, y)), y)
    
    # Separate the features and labels after resampling
    X_res = X_res[:, :-y.shape[1]]
    y_res = X_res[:, -y.shape[1]:]
    
    return X_res, y_res