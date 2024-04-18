# In main.py
import sgm_parser
import text_preprocessing
import feature_extraction
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

def main():

    # Parse SGM files
    articles_and_ = sgm_parser.parse_sgm_files('./data')

    # Extract articles and topics separately
    articles = [article[0] for article in articles_and_topics]
    topics = [article[1] for article in articles_and_topics]

    df = text_preprocessing.preprocess_data(articles)


    # Assuming you want to use titles as features or labels
    X = feature_extraction.extract_features(df['text'])  # Features from text
    y = df['title']  # Using title as labels for some tasks

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

    # # Parse SGM files
    # articles_and_topics = sgm_parser.parse_sgm_files('./data')  # Adjust path as necessary

    # # Extract articles and topics separately
    # articles = [article[0] for article in articles_and_topics]
    # topics = [article[1] for article in articles_and_topics]

    # # Preprocess data
    # df = text_preprocessing.preprocess_data(articles)  # Ensure this function processes a list of texts
    
    # # Assuming topics needs to be processed separately if they are multi-label
    # topics_encoded = feature_extraction.encode_topics(topics)  # Handle topics encoding if necessary

    # # Feature extraction
    # X = feature_extraction.extract_features(df['text'])  # Assuming text feature extraction only
    # y = topics_encoded  # Direct assignment if topics are already encoded

    # # Split data into training and testing sets
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # # Train the model
    # model = MultinomialNB()
    # model.fit(X_train, y_train)

    # # Evaluate the model
    # y_pred = model.predict(X_test)
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

