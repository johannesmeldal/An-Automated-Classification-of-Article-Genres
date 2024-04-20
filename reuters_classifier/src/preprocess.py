import re
import string
from ast import literal_eval

import numpy as np  # Import to safely evaluate stringified lists
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import MultiLabelBinarizer

# Make sure you download the necessary NLTK data:
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

def tokenize(text):
    """
    Tokenizes the text into words using simple whitespace splitting.
    """
    return text.split()

def remove_punctuation_and_lowercase(tokens):
    """
    Removes punctuation from tokens and converts them to lowercase.
    """
    table = str.maketrans('', '', string.punctuation)
    return [token.translate(table).lower() for token in tokens]

def remove_stopwords(tokens):
    """
    Removes stopwords from a list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize_text(tokens):
    """
    Applies lemmatization to each token in the list.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    """
    Full preprocessing pipeline integrating tokenization, punctuation removal,
    stopwords removal, and lemmatization into a single function.
    """
    tokens = tokenize(text)
    tokens = remove_punctuation_and_lowercase(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    return tokens  # Return tokens for further processing or recomposition in the main script

def preprocess_targets(df):
    """
    Processes the 'TOPICS' column in the dataframe to prepare it for multi-label binarization.
    Converts string representations of lists into actual lists if necessary.
    """
    # Ensure that the 'TOPICS' entries are lists; convert stringified lists to actual lists
    df['TOPICS'] = df['TOPICS'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)

    # Before binarization
    print("Raw Topics Sample:", df['TOPICS'].head())

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['TOPICS'])

    print("Binarized labels sample:", y[:5])
    print("Label counts:", np.sum(y, axis=0))  # Summing binarized labels to see the distribution
    
    # # Debug: Print out some of the binarized labels to check them
    # print("Sample of binarized labels:", y[:5])
    # print("Label classes:", mlb.classes_)

    return y, mlb.classes_
