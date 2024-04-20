import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Load the dataset
data = pd.read_csv("article_data.csv")

# Extract text data and topic labels
X = data['BODY']
y = data['TOPICS']

# Define the number of clusters
num_clusters = 10  # You can adjust this number as desired

# Drop rows with missing values
data = data.dropna(subset=['BODY']).copy()
X = data['BODY']

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(X_tfidf)

# Get cluster assignments
cluster_labels = kmeans.labels_

# Assign topics to clusters
topic_to_cluster = defaultdict(list)
for topic, cluster in zip(y, cluster_labels):
    topic_to_cluster[topic].append(cluster)

# Analyze top words for each cluster
terms = tfidf_vectorizer.get_feature_names_out()
centroids = kmeans.cluster_centers_
top_words_per_cluster = []
for cluster_idx, centroid in enumerate(centroids):
    top_indices = centroid.argsort()[-10:][::-1]
    top_words = [terms[i] for i in top_indices]
    top_words_per_cluster.append(top_words)

# Print topics and top words per cluster
for topic, clusters in topic_to_cluster.items():
    print(f"Topic: {topic}")
    for cluster_idx in set(clusters):
        print(f"Cluster {cluster_idx}: {', '.join(top_words_per_cluster[cluster_idx])}")
    print("\n")
