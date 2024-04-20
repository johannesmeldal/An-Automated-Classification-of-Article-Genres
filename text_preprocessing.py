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

def chat_words_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            new_text.append(chat_words_map_dict[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

def clean_text(text):
    """
    Cleans the input text by focusing on financial data retention and formatting for NLP processing.
    """
    # lowercase text
    text = text.lower()
    

    # Custom handling for financial numbers (e.g., "2.55 dlrs" -> "2.55 dollars")
    text = re.sub(r'(\d+(\.\d+)?)(\s*)(dlrs|bu|cwt)', r'\1 \4', text) 

    # Remove special characters, preserving decimal points and hyphens within numbers
    text = re.sub(r'[^\w\s.-]', '', text)

    # Replace multiple spaces or newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords while retaining financial context
    finance_stop_words = set(stopwords.words('english')) - {'above', 'below', 'off', 'over', 'under'}
    tokens = [token for token in tokens if token not in finance_stop_words and len(token) > 1]

    # Optional: Lemmatization or stemming
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

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
    examplearticletext = articles[0][0]
    print("Example article text before cleaning:")
    print(examplearticletext)
    print("\nExample article text after cleaning:")
    print(clean_text(examplearticletext))
    # df = preprocess_data(articles)
    # print(df.head())  # Print the first few rows of the DataFrame
