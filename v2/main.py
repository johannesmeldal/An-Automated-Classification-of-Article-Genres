import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from models import train_naive_bayes, train_logistic_regression, train_svm

# Load your dataset
data = pd.read_csv("article_data.csv")

# Drop rows with missing values
data.dropna(subset=['BODY', 'TOPICS'], inplace=True)

# Proportion of data for training
train_proportion = 0.5
max_count = 1000  # Maximum number of articles to use

# Determine number of articles for training and testing
num_total_articles = min(max_count, len(data))
num_train_articles = int(train_proportion * num_total_articles)
num_test_articles = num_total_articles - num_train_articles

# Separate training and testing data
train_data = data[data['LEWISSPLIT'] == 'TRAIN'].head(num_train_articles)
test_data = data[data['LEWISSPLIT'] == 'TEST'].head(num_test_articles)

# Split the data into training and testing sets
X_train, y_train = train_data['BODY'], train_data['TOPICS']
X_test, y_test = test_data['BODY'], test_data['TOPICS']

# Train the classifiers
nb_classifier = train_naive_bayes(X_train, y_train)
lr_classifier = train_logistic_regression(X_train, y_train)
svm_classifier = train_svm(X_train, y_train)

# Make predictions
y_pred_nb = nb_classifier.predict(X_test)
y_pred_lr = lr_classifier.predict(X_test)
y_pred_svm = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy_nb = accuracy_score(y_test, y_pred_nb)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("Naive Bayes Accuracy:", accuracy_nb)
print("Logistic Regression Accuracy:", accuracy_lr)
print("SVM Accuracy:", accuracy_svm)

# Compare models using confusion matrix
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")
plot_confusion_matrix(y_test, y_pred_svm, "SVM Confusion Matrix")
