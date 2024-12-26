import pandas as pd
import os
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter


# Function to read all .txt files in a directory
def load_articles(data_dir):
    articles = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as file:
                articles.append(file.read())
    return articles


def calculate_statistics(articles):
    stats = {
        "num_articles": len(articles),
        "word_counts": [],
        "sentence_counts": [],
        "char_counts": [],
        "vocabulary": Counter(),
        "average_sentence_length": 0
    }

    stop_words = set(stopwords.words('english'))
    total_words = 0  # Initialize total_words
    total_sentences = 0  # Initialize total_sentences

    for article in articles:
        # Tokenize
        words = word_tokenize(article)
        sentences = sent_tokenize(article)

        # Update stats
        word_count = len(words)
        sentence_count = len(sentences)

        stats["word_counts"].append(word_count)
        stats["sentence_counts"].append(sentence_count)
        stats["char_counts"].append(len(article))
        stats["vocabulary"].update([word.lower() for word in words if word.isalpha() and word.lower() not in stop_words])

        total_words += word_count
        total_sentences += sentence_count

    # Calculate average sentence length (words per sentence)
    if total_sentences > 0:  # Avoid division by zero
        stats["average_sentence_length"] = total_words / total_sentences

    return stats


def find_shortest_and_longest_articles(articles):
    shortest_article = min(articles, key=lambda x: len(word_tokenize(x)))
    longest_article = max(articles, key=lambda x: len(word_tokenize(x)))

    return shortest_article, longest_article


if __name__ == "__main__":
    folder_path = "../data/raw-documents"
    articles = load_articles(folder_path)

    stats = calculate_statistics(articles)

    # Display results
    print(f"Number of Articles: {stats['num_articles']}")
    print(f"Average Word Count: {sum(stats['word_counts']) / stats['num_articles']:.2f}")

    print(f"Median Word Count: {pd.Series(stats['word_counts']).median()}")
    print(f"Average Sentence Count: {sum(stats['sentence_counts']) / stats['num_articles']:.2f}")
    print(f"Median Sentence Count: {pd.Series(stats['sentence_counts']).median()}")
    print(f"Average Character Count: {sum(stats['char_counts']) / stats['num_articles']:.2f}")
    print(f"Total Word Count:{sum(stats['word_counts'])}")
    print("Average Sentence Length:", stats["average_sentence_length"])

    shortest_article, longest_article = find_shortest_and_longest_articles(articles)

    print("Shortest Article Word Count:", len(word_tokenize(shortest_article)))
    print("\nLongest Article Word Count:", len(word_tokenize(longest_article)))