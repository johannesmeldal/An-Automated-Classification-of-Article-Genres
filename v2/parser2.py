from bs4 import BeautifulSoup # type: ignore
import os
import csv

# Directory containing .sgm files
directory = './data'
files = os.listdir(directory)

# Initialize dictionaries to store extracted data
article_data = {}

import re

# Function to tokenize text
def tokenize_text(text):
    # Remove non-alphanumeric characters and extra spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase and split into tokens
    tokens = text.lower().split()
    return ' '.join(tokens)

# Modify the data extraction part to tokenize the text
for file_name in files:
    if file_name.endswith('.sgm'):
        with open(os.path.join(directory, file_name), 'r', encoding='latin-1') as file:
            # Parse .sgm file using Beautiful Soup
            soup = BeautifulSoup(file, 'html.parser')

            # Find all <REUTERS> elements
            reuters_elements = soup.find_all('reuters')
            for reuters_element in reuters_elements:
                # Extract relevant information
                doc_id = reuters_element['newid']
                lewissplit = reuters_element['lewissplit']
                date = reuters_element.find('date').text.strip()
                topics = [topic.text.strip() for topic in reuters_element.find('topics').find_all('d')]
                
                # Check if <TEXT> tag exists
                text_element = reuters_element.find('text')
                if text_element:
                    # Extract title if it exists and tokenize
                    title_element = text_element.find('title')
                    title = tokenize_text(title_element.text.strip()) if title_element else ''
                    
                    # Extract body if it exists and tokenize
                    body_element = text_element.find('body')
                    body = tokenize_text(body_element.text.strip()) if body_element else ''
                else:
                    title = ''
                    body = ''

                # Store extracted data in article_data dictionary
                article_data[doc_id] = {
                    'LEWISSPLIT': lewissplit,
                    'DATE': date,
                    'TOPICS': topics,
                    'TITLE': title,
                    'BODY': body
                }

# Generate CSV file
def create_csv(title, data):
    with open(title, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['DOCID', 'LEWISSPLIT', 'DATE', 'TOPICS', 'TITLE', 'BODY'])
        writer.writeheader()
        for doc_id, article_info in data.items():
            writer.writerow({'DOCID': doc_id, **article_info})

# Create CSV file from extracted data
create_csv('article_data.csv', article_data)
