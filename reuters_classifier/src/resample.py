import numpy as np

def custom_resample(X, y, strategy='over'):
    label_counts = np.sum(y, axis=0)
    max_count = np.max(label_counts)

    # Initialize lists to collect new samples
    resampled_X = []
    resampled_y = []

    if strategy == 'over':
        # Perform oversampling
        for label_index in range(y.shape[1]):
            indices = np.where(y[:, label_index] == 1)[0]
            num_to_add = max_count - label_counts[label_index]
            additional_indices = np.random.choice(indices, size=num_to_add, replace=True)
            resampled_X.extend(X[additional_indices])
            # For each index added, also add the corresponding label
            for idx in additional_indices:
                resampled_y.append(y[idx])

    # Combine the original data with the resampled data
    X_final = np.vstack((X, np.array(resampled_X).reshape(-1, 1)))
    y_final = np.vstack((y, np.array(resampled_y)))

    return X_final, y_final
