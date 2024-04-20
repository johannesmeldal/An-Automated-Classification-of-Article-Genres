# main.py

from src.preprocess import (
    tokenize,
    remove_punctuation_and_lowercase,
    remove_stopwords,
    lemmatize_text,
    preprocess_targets
)
from src.train import train_model
from src.cross_validate_model import perform_cross_validation
from src.utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd

def preprocess_and_train(data_path, preprocess_options, model_path='models/text_classification_pipeline.joblib', perform_cv=False):
    try:
        df = load_data(data_path)
        df['text'] = df['TITLE'] + " " + df['BODY']  # Concatenating TITLE and BODY

        preprocessing_functions = {
            'tokenize': tokenize,
            'remove_punctuation_and_lowercase': remove_punctuation_and_lowercase,
            'remove_stopwords': remove_stopwords,
            'lemmatize_text': lemmatize_text
        }

        for option, enabled in preprocess_options.items():
            if enabled:
                func = preprocessing_functions.get(option)
                if func:
                    df['text'] = df['text'].apply(func)

        y, classes = preprocess_targets(df)
        df['text'] = df['text'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

        if perform_cv:
            # For cross-validation, no train-test split is needed
            mlb = MultiLabelBinarizer()
            y_binarized = mlb.fit_transform(df['TOPICS'])
            perform_cross_validation(df['text'], y_binarized, n_splits=5)
        else:
            # Usual train-test split approach
            X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.2, random_state=42)
            print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            pipeline = train_model(X_train, y_train, model_path)

            # Predict on the test set
            y_pred = pipeline.predict(X_test)
            
            # Calculate and print metrics
            print("Accuracy:", accuracy_score(y_test, y_pred))
            print("Precision:", precision_score(y_test, y_pred, average='micro'))
            print("Recall:", recall_score(y_test, y_pred, average='micro'))
            print("F1 Score:", f1_score(y_test, y_pred, average='micro'))

    except Exception as e:
        print("An error occurred during processing:", str(e))

def main():
    preprocess_options = {
        'tokenize': True,
        'remove_punctuation_and_lowercase': True,
        'remove_stopwords': True,
        'lemmatize_text': True,
    }
    data_path = 'data/article_data_fullbody.csv'
    preprocess_and_train(data_path, preprocess_options, perform_cv=True)  # Set perform_cv=True to perform cross-validation

if __name__ == '__main__':
    main()
