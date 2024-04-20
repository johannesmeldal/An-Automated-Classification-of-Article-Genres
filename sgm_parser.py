from bs4 import BeautifulSoup
import os

def parse_sgm_files(directory, filename='reut2-000.sgm'):
    file_path = os.path.join(directory, filename)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    soup = BeautifulSoup(content, 'html.parser')
    articles = []
    deleted_count = 0
    article_count = 0

    for reuter in soup.find_all('reuters'):
        
        topics_tag = reuter.find('topics')
        topics = [d.text for d in topics_tag.find_all('d')] if topics_tag else []

        title_tag = reuter.find('title')
        title = title_tag.text if title_tag else ""

        body_tag = reuter.find('body')
        body = body_tag.text if body_tag else ""

        text = title + " " + body

        if len(text) > 0 and len(topics) > 0:
            articles.append((text, topics))
            article_count += 1
        else:
            # print(f"TEXT{text} \n TOPICS{topics} \n")
            deleted_count += 1

    print(f"Deleted {deleted_count} articles with no body text")
    print(f"Kept {article_count} articles")
    return articles

# if __name__ == "__main__":
#     articles = parse_sgm_files('./data')
#     print(articles[1])  # Print the first article to check