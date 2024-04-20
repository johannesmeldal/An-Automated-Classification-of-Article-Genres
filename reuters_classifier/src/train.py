import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

def train_model(X_train, y_train, mlb, model_path='models/text_classification_pipeline.joblib'):
    """
    Trains the multi-label classification model using a pipeline and saves it to disk.
    Assumes that y_train is already transformed by MultiLabelBinarizer outside this function.
    
    Parameters:
    - X_train: Feature data for training.
    - y_train: Binarized labels for training.
    - mlb: Pre-fitted MultiLabelBinarizer to ensure consistent label handling.
    - model_path: Path where the trained model and binarizer should be saved.
    """
    try:
        # Save the pre-fitted MultiLabelBinarizer to disk for later use
        joblib.dump(mlb, f'{model_path}_mlb.joblib')

        # Define the pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
            ('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000)))  # Logistic Regression classifier
        ])

        # Train the classifier
        pipeline.fit(X_train, y_train)
        
        # Save the trained pipeline to disk
        joblib.dump(pipeline, model_path)

        return pipeline
    except Exception as e:
        print("An error occurred during model training:", str(e))
        return None
