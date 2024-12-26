import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import os
import re
import nltk
from nltk.corpus import stopwords
from glob import glob
from textblob import TextBlob
from wordcloud import WordCloud
from collections import Counter


def word_cloud_all(text):
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def most_common(text, num):
    # excludes the most common stop words

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    # Tokenize words and remove punctuation
    words = re.findall(r'\b\w+\b', text)  # Extract only alphanumeric words
    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stop_words]
    word_counts = Counter(filtered_words)
    most_common_words = word_counts.most_common(num)

    # Plot
    words, counts = zip(*most_common_words)
    sns.barplot(x=counts, y=words)
    plt.title('Top ' + str(num) + ' Most Frequent Words (Excluding Stopwords and Punctuation)')
    plt.xlabel('Frequency')
    plt.ylabel('Words')
    plt.show()

# PARTS OF SPEECH
def pos(text):
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1500000  # Adjust as needed, e.g., 1.5M characters
    doc = nlp(text)

    # Count POS tags
    pos_counts = Counter(token.pos_ for token in doc)

    # sort in descending order
    sorted_pos = pos_counts.most_common()
    pos_labels, pos_values = zip(*sorted_pos)

    # Visualize POS distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(y=pos_labels, x=pos_values, orient='h')
    # sns.barplot(x=list(pos_counts.keys()), y=list(pos_counts.values()))
    plt.title('Part-of-Speech Distribution')
    plt.xlabel('Frequency')
    plt.ylabel('POS Tag')
    plt.show()

# NAMED ENTITY RECOGNITION
def ner(text):
    """
    Label	        Explanation
    GPE	          Geopolitical entities: countries, cities, states (e.g., "France", "New York").
    ORG	          Organizations, companies, institutions (e.g., "Google", "UN").
    PERSON	      People, including fictional characters (e.g., "Barack Obama").
    DATE	        Absolute or relative dates (e.g., "June 5th", "last year").
    NORP	        Nationalities, religious groups, political groups (e.g., "American", "Buddhist").
    CARDINAL      Cardinal numbers (e.g., one, two).
    LOC	          Locations (other than GPE), e.g., mountain ranges, bodies of water.
    ORDINAL       Ordinal numbers (e.g., first, second).
    WORK_OF_ART   Titles of books, songs, movies, paintings, etc.
    TIME	        Specific times (e.g., "3 PM", "midnight").
    PERCENT       ??
    PRODUCT	      Products, vehicles, or objects (e.g., "iPhone", "Tesla").
    EVENT	        Named events (e.g., "World War II", "Olympics").
    FAC           Buildings, airports, highways, bridges.
    QUANTITY      Measurements (e.g., distance, weight).
    LAW           Named documents made into laws.
    LANGUAGE      Any named language.
    """

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 1500000  # Adjust as needed, e.g., 1.5M characters

    doc = nlp(text)

    entities = [ent.label_ for ent in doc.ents]
    entity_counts = Counter(entities)

    # sort in descending order
    sorted_entities = entity_counts.most_common()
    entity_labels, entity_values = zip(*sorted_entities)

    plt.figure(figsize=(12, 6))
    sns.barplot(y=entity_labels, x=entity_values, orient='h')  # Horizontal bars
    plt.title('Named Entity Distribution')
    plt.xlabel('Frequency')
    plt.ylabel('Entity Type')
    plt.show()

# SENTIMENT POLARITY
def sentiment_polatrity(folder_path):
    """
    Sentiment polarity measures the emotional tone of a piece of text on a scale
    that typically ranges from -1 to 1:
    -1: Strongly negative sentiment (e.g., sadness, anger, criticism).
    0: Neutral sentiment (e.g., factual or emotionless text).
    1: Strongly positive sentiment (e.g., happiness, praise, excitement).
    """

    sentiments = []

    for file_path in glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            sentiments.append(TextBlob(text).sentiment.polarity)

    print(f"Average Sentiment Polarity: {sum(sentiments) / len(sentiments):.2f}")

    plt.hist(sentiments, bins=20, edgecolor='black')
    plt.title('Distribution of Sentiment Polarity')
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Number of Articles')
    plt.show()

if __name__ == "__main__":
    folder_path = "../data/raw-documents"
    text = ""
    # Iterate through all .txt files in the folder
    for file_path in glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, 'r', encoding='utf-8') as file:
            text += file.read() + " "  # Add a space between files' content

    #word_cloud_all(text)

    #most_common(text, 20)

    #pos(text)

    #ner(text)

    sentiment_polatrity(folder_path)

