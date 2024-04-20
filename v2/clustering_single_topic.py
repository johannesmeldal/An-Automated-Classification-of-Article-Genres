import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine_similarity
from collections import defaultdict

# Load the list of all possible topics
with open('all-topics-strings.lc.txt', 'r') as f:
    all_topics = [topic.strip() for topic in f.readlines()]

# Load the dataset
data = pd.read_csv("article_data.csv")

# Extract text data and topics
X = data['BODY']
topics = data['TOPICS']

# Define the number of clusters
num_clusters = 100  # Set number of clusters equal to the number of topics

# Tokenize topics for each article
article_to_topics = defaultdict(list)
for i, row in data.iterrows():
    article_topics = eval(row['TOPICS'])  # Convert string representation to list
    for topic in article_topics:
        tokens = topic.split('-')  # Tokenize multi-word topics
        article_to_topics[i].extend([token.lower() for token in tokens])

# Drop rows with missing values
data = data.dropna(subset=['BODY']).copy()
X = data['BODY']

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_tfidf)

# Get cluster centroids
cluster_centroids = kmeans.cluster_centers_

# Assign each topic to the cluster with the highest similarity
topic_to_cluster = {}
for topic in all_topics:
    topic_vector = tfidf_vectorizer.transform([topic])
    similarity_scores = cosine_similarity(topic_vector, cluster_centroids)
    assigned_cluster = similarity_scores.argmax()
    topic_to_cluster[topic] = assigned_cluster

# Initialize a dictionary to store top words for each cluster
top_words_per_cluster = defaultdict(list)

# Get the feature names from the TF-IDF vectorizer
terms = tfidf_vectorizer.get_feature_names_out()

# Find the top words for each cluster
for cluster_idx, centroid in enumerate(cluster_centroids):
    top_indices = centroid.argsort()[-5:][::-1]  # Get indices of top 5 words
    top_words = [terms[i] for i in top_indices]  # Get the top words
    top_words_per_cluster[cluster_idx] = top_words

# Print the top words for each cluster
for cluster_idx, top_words in top_words_per_cluster.items():
    print(f"Cluster {cluster_idx}: {', '.join(top_words)}")

# Print topics and their associated clusters
for topic, cluster in topic_to_cluster.items():
    print(f"Topic: {topic}, Cluster: {cluster}")
