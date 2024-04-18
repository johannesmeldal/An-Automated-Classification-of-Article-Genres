import pandas as pd
import sgm_parser
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import download

# Download necessary NLTK resources
download('punkt', quiet=True)
download('stopwords', quiet=True)
download('wordnet', quiet=True)

def clean_text(text):
    """
    Cleans the input text using various preprocessing steps.
    
    Args:
        text (str): The text to be cleaned.
    
    Returns:
        str: The cleaned text.
    """
    # Convert text to lowercase
    text = text.lower()
    
    # Remove unwanted characters and symbols
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    print("lol jeg kjører her")
    
    # Lemmatization
    # lemmatizer has the task 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Reconstruct the cleaned text
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def preprocess_data(articles_and_titles):
    """
    Preprocesses the data by applying text cleaning to each article.
    
    Args:
        articles_and_titles (list of tuples): Each tuple contains the text of the article and its title.
    
    Returns:
        pd.DataFrame: A DataFrame with cleaned text and titles.
    """
    # Create DataFrame
    df = pd.DataFrame(articles_and_titles, columns=['text', 'topics'])

    # Apply text cleaning to the 'text' column
    df['text'] = df['text'].apply(clean_text)
    
    return df

if __name__ == "__main__":
    articles = sgm_parser.parse_sgm_files('./data')

    df = preprocess_data(articles)
    print(df.head())  # Print the first few rows of the DataFrame
