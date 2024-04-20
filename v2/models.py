from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

def train_naive_bayes(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  
        ('clf', MultinomialNB()) 
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_logistic_regression(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  
        ('clf', LogisticRegression(max_iter=1000)) 
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def train_svm(X_train, y_train):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  
        ('clf', SVC()) 
    ])
    pipeline.fit(X_train, y_train)
    return pipeline
