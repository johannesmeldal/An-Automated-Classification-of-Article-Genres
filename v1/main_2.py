from text_handler import clean_text
from models import create_embedding_layer, build_lstm_model, train_model

def main():
    # Load and preprocess data
    articles = get_data()
    cleaned_articles = [clean_text(article) for article in articles]

    # Vectorize text data
    x, y, tokenizer = sequence(cleaned_articles, labels, MAX_WORDS, MAX_LEN)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE)

    # Create embedding layer
    embedding_layer = create_embedding_layer(tokenizer, MAX_LEN)

    # Build and train LSTM model
    lstm_model = build_lstm_model(embedding_layer, num_classes)
    train_model(lstm_model, x_train, y_train, x_test, y_test)

    # Evaluate and save the model
    evaluate_model(lstm_model, x_test, y_test)
    save_model(lstm_model, 'genre_classification_model.h5')

if __name__ == "__main__":
    main()
