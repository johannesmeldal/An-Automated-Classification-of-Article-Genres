import re
import string
from ast import literal_eval

import numpy as np  # Import to safely evaluate stringified lists
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import MultiLabelBinarizer

# Make sure you download the necessary NLTK data:
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')

stemmer = PorterStemmer()

# Dictionary of common financial abbreviations
finance_terms = {
    'dlr': 'dollar',
    'dlrs': 'dollars',
    'mln': 'million',
    'mlns': 'millions',
    'bln': 'billion',
    'blns': 'billions',
    'pct': 'percent',
    'shr': 'shares',
    'revs': 'revenues',
    'cts': 'cents',
    'fx': 'foreign exchange',
    'bpd': 'barrels per day',
    'opec': 'organization of petroleum exporting countries',

    # TODO: add more if needed
}

def tokenize(text):
    """Tokenizes the text into words using simple whitespace splitting."""
    return text.split()

def remove_punctuation_and_lowercase(tokens):
    """Removes punctuation from tokens and converts them to lowercase."""
    table = str.maketrans('', '', string.punctuation)
    return [token.translate(table).lower() for token in tokens if isinstance(token, str)]

def remove_stopwords(tokens):
    """Removes stopwords from a list of tokens."""
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

def lemmatize_text(tokens):
    """Applies lemmatization to each token in the list."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]

def expand_abbreviations(text, abbr_dict):
    """Expands common abbreviations found in the text based on the provided dictionary."""
    pattern = re.compile(r'\b(' + '|'.join(abbr_dict.keys()) + r')\b')
    return pattern.sub(lambda x: abbr_dict[x.group()], text)

def preprocess_text(text):
    """Full preprocessing pipeline integrating all steps. Comment out manually to test the different parts."""
    text = expand_abbreviations(text, finance_terms)
    tokens = tokenize(text)
    tokens = remove_punctuation_and_lowercase(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_text(tokens)
    # tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)
