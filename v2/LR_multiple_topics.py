import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load your training dataset
train_data = pd.read_csv("sample_train_data.csv")

# Load your testing dataset
test_data = pd.read_csv("sample_test_data.csv")

# Drop rows with missing values
train_data.dropna(subset=['BODY', 'TOPICS'], inplace=True)
test_data.dropna(subset=['BODY', 'TOPICS'], inplace=True)

# Split the data into training and testing sets
X_train, y_train = train_data['BODY'], train_data['TOPICS']
X_test, y_test = test_data['BODY'], test_data['TOPICS']

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
    ('clf', LogisticRegression(max_iter=1000))  # Logistic Regression classifier
])

# Train the classifier
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
