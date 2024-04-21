# train.py
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

def train_model(X_train, y_train, classifier_type='lr', model_path='models/text_classification_pipeline.joblib'):
    """ Trains a model based on the classifier type. """
    if classifier_type == 'lr':
        classifier = LogisticRegression(max_iter=1000, class_weight='balanced')
    elif classifier_type == 'svm':
        classifier = SVC(kernel='linear', probability=True)
    elif classifier_type == 'nb':
        classifier = MultinomialNB()
    else:
        raise ValueError("Unsupported classifier type")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', OneVsRestClassifier(classifier))
    ])
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, model_path)
    return pipeline

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def print_top10_vectorized_words_per_class(pipeline, classes):
    """
    Prints the top 10 words for each class based on the TfidfVectorizer within a trained pipeline.
    """
    vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['clf']
    feature_names = vectorizer.get_feature_names_out()
    for i, class_label in enumerate(classes):
        top10 = np.argsort(classifier.estimators_[i].coef_[0])[-10:]
        print(f"{class_label}: {', '.join(feature_names[top10])}")