import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.preprocess import preprocess_text
from src.train import train_model, plot_confusion_matrix, print_top10_vectorized_words_per_class
from src.utils import load_data
from src.resample import custom_resample  # Assuming this is the correct path
from skmultilearn.model_selection import iterative_train_test_split

def preprocess_and_train(data_path, model_path='models/text_classification_pipeline.joblib', classifier_type='lr', train_prop=0.5, resample_strategy='over', use_resampling=False):
    df = load_data(data_path)
    df['text'] = df['TITLE'].fillna('') + " " + df['BODY'].fillna('')
    df['text'] = df['text'].apply(preprocess_text)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['TOPICS'])
    X = df['text'].values.reshape(-1, 1)

    # Conditional resampling based on the use_resampling flag
    if use_resampling:
        X, y = custom_resample(X, y, strategy=resample_strategy)

    # Stratified split and model training
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=1-train_prop)
    X_train = X_train.flatten()
    X_test = X_test.flatten()

    pipeline = train_model(X_train, y_train, classifier_type, model_path)

    if pipeline:
        y_pred = pipeline.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred, average='micro'))
        print("Recall:", recall_score(y_test, y_pred, average='micro'))
        print("F1 Score:", f1_score(y_test, y_pred, average='micro'))

        

def main():
    data_path = 'data/article_data_fullbody.csv'
    model_path = 'models/text_classification_pipeline.joblib'

    # differnt classifier types: 'lr', 'svm', 'nb'
    # different resampling strategies: 'over',

    preprocess_and_train(data_path, model_path, classifier_type='lr', train_prop=0.5, resample_strategy='over', use_resampling=False)

if __name__ == '__main__':
    main()
