import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd

# Load English stopwords
stop_words = set(stopwords.words('english'))

# Read data from CSV file
data = pd.read_csv('article_data.csv')

# Filter rows where LEWISSPLIT is 'TRAIN'
train_data = data[data['LEWISSPLIT'] == 'TRAIN']

# Define a function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        # Remove punctuation and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Tokenize the text
        tokens = word_tokenize(text)
        # Convert tokens to lowercase and remove stopwords
        tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
        # Remove short tokens (length less than or equal to 3)
        tokens = [word for word in tokens if len(word) > 3]
        return tokens
    else:
        return []

# Apply the preprocessing function to the body column of the DataFrame
train_data['PREPROCESSED_BODY'] = train_data['BODY'].apply(preprocess_text)

# Filter out rows where the preprocessed body is empty
train_data = train_data[train_data['PREPROCESSED_BODY'].map(len) > 0]

print(train_data)
