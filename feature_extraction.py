from sklearn.feature_extraction.text import TfidfVectorizer

def extract_features(texts):
    tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_matrix