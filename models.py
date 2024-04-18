from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.metrics import Precision, Recall
from tensorflow.keras.layers import LSTM, Dense, Embedding, GRU
import numpy as np

def sequence(texts, labels, max_words, max_len):
    tokenizer = Tokenizer(num_words=max_words, split=' ')
    tokenizer.fit_on_texts(texts)
    x = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(x, max_len)
    y = np.array(labels)
    return x, y, tokenizer

def create_glove(tokenizer, max_len, glove_file_path):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f'Found {len(embeddings_index)} word vectors.')
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return Embedding(len(tokenizer.word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False)

def build_lstm_model(embedding, num_classes):
    model = Sequential()
    model.add(embedding)
    model.add(LSTM(128, dropout=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    return model

def build_gru_model(embedding, num_classes):
    model = Sequential()
    model.add(embedding)
    model.add(GRU(32, dropout=0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy', Precision(), Recall()])
    return model

def train_model(model, epochs, earlystopping, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1, validation_data=(x_test, y_test),
              callbacks=[earlystopping])
    return model

def save_model(model, model_path):
    model.save(model_path)

def evaluate_model(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)
    f1_score = 2 * (scores[2] * scores[3]) / (scores[2] + scores[3])
    print(f"Accuracy: {scores[1]:.2f} - Precision: {scores[2]:.2f} - Recall: {scores[3]:.2f} - F1 Score: {f1_score:.2f}")

def predict(text, model, tokenizer, max_len):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    return np.around(model.predict(padded), decimals=0).argmax(axis=1)[0]