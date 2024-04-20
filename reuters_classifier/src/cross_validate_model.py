# src/cross_validate_model.py


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from skmultilearn.model_selection import IterativeStratification  # Correct import from scikit-multilearn
import joblib

def perform_cross_validation(X, y, n_splits=5, class_weight='balanced'):
    """
    Performs cross-validation using stratified K-Folds for multi-label data.
    """
    # Setting up the stratified K-Fold cross-validation
    stratifier = IterativeStratification(n_splits=n_splits, order=1, sample_distribution_per_fold=[1.0/n_splits] * n_splits)
    folds = list(stratifier.split(X, y))

    # Pipeline setup
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight=class_weight)))
    ])

    # Perform cross-validation using the defined folds
    f1_scores = []
    for train_indexes, test_indexes in folds:
        X_train, X_test = X[train_indexes], X[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        score = f1_score(y_test, predictions, average='micro')
        f1_scores.append(score)

    print("F1 Scores for each fold:", f1_scores)
    print("Average F1 Score:", np.mean(f1_scores))
