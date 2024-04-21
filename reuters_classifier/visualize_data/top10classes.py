import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import ast  # Import ast module for safe evaluation

def load_data(file_path):
    """
    Loads the data from CSV and returns a DataFrame.
    """
    df = pd.read_csv(file_path)
    df.dropna(subset=['BODY', 'TOPICS'], inplace=True)
    # Safely convert stringified lists back into actual lists
    df['TOPICS'] = df['TOPICS'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return df

def visualize_top_classes(data_path):
    # Load the data
    df = load_data(data_path)

    # Binarize the labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['TOPICS'])

    # Calculate class frequencies
    class_counts = pd.Series(y.sum(axis=0), index=mlb.classes_).sort_values(ascending=False)

    # Select the top 10 classes
    top_classes = class_counts.head(10)
    print("Top 10 classes and their frequencies:")
    print(top_classes)

    # Plotting
    plt.figure(figsize=(12, 8))
    top_classes.plot(kind='bar')
    plt.title('Top 10 Classes Frequency')
    plt.xlabel('Class Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    data_path = '../data/article_data_fullbody.csv'  # Update this path if necessary
    visualize_top_classes(data_path)

if __name__ == '__main__':
    main()
