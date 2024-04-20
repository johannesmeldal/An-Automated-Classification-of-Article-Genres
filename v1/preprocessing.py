import re
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

import sgm_parser



# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
def word_exist(word):
    if not word.isalpha():
        return False
    if not wordnet.synsets(word):  # Checks if a word exists (faster than checking if word in nltk.words)
        return False
    return True

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun if not found

def preprocess_text(text):
    # Remove numerical data
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters and symbols
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove single characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # POS tagging
    tagged_tokens = pos_tag(tokens)
    
    # Lemmatization and filtering out non-existent words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos_tag)) for token, pos_tag in tagged_tokens if word_exist(token)]
    
    # Lowercasing and removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    # Join the tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

if __name__ == "__main__":
    articles = sgm_parser.parse_sgm_files('./data')
    preprocessed_articles = []

    for article_text, topics in articles:
        preprocessed_text = preprocess_text(article_text)
        preprocessed_articles.append((preprocessed_text, topics))

    # print(preprocessed_articles[0][0])  # Print the first preprocessed article to check


# ----- Example Usage -----
    # # Tokenize the text
    # words = word_tokenize(preprocessed_articles[0][0])
    #
    # # Get the top 10 most common words
    # list = preprocessed_articles[0][0].split(" ")
    # # print(list)
    # top10 = nltk.FreqDist(list).most_common(10)
    # top10 = [word for word, _ in top10]
    # print(top10)  # Print the top 10 most common words in the first article
