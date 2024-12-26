import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def clean_article(text):
    """
    Cleans a single article by removing non-ASCII characters,
    unwanted symbols, and normalizing whitespace.
    """
    # Remove non-ASCII characters (e.g., emojis, special symbols)
    text = text.encode("ascii", "ignore").decode("utf-8")

    # Remove unwanted symbols (e.g., @, #, $, etc.)
    text = re.sub(r"[^\w\s.,!?']", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def read_and_clean_articles(directory):
    """
    Reads all .txt articles from a directory, cleans them,
    and stores them in a dictionary.
    """
    articles = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            # Read the article
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                original_text = file.read()

            # Clean the article
            cleaned_text = clean_article(original_text)

            # Store in dictionary with the filename (without extension) as key
            articles[filename] = {
                "cleaned": cleaned_text
            }
    return articles

def remove_empty_articles(articles_dict):
    """
    Removes empty or near-empty articles from the dataset.
    An article is considered empty if its 'cleaned' text is
    empty or contains only whitespace.
    """
    non_empty_articles = {
        key: value
        for key, value in articles_dict.items()
        if value["cleaned"].strip()  # Check if 'cleaned' text is not empty
    }
    return non_empty_articles

def role_distribution(df):
    # show plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='main_role', data=df, order=df['main_role'].value_counts().index)
    plt.title('Distribution of Main Roles')
    plt.xlabel('Main Role')
    _ = plt.ylabel('Frequency')

    # print values for main roles
    df['main_role'].value_counts()

    # print values for fine-grained roles 1
    df['fine-grained_role_1'].value_counts()

    # print values for fine-grained roles 2
    df['fine-grained_role_2'].value_counts()


if __name__ == "__main__":
    # Path to the directory containing .txt files
    folder_path = "../data/raw-documents"
    annotation_path = "../data/subtask-1-annotations.txt"

    # Read and clean articles

    articles_dict = read_and_clean_articles(folder_path) #Dict with article id's and their cleaned versions
    cleaned_articles_dict= remove_empty_articles(articles_dict)
    # Access cleaned or original version of an article
    article_id = "EN_UA_DEV_26.txt"  # Replace with the desired filename (without .txt extension)
    print(f"Original number of articles: {len(articles_dict)}")
    print(f"Number of non-empty articles: {len(cleaned_articles_dict)}")
    print("\nCleaned Article:\n", articles_dict[article_id]["cleaned"], "\n")

    # Class imbalance

    # Read the file into a DataFrame
    df = pd.read_csv(annotation_path, sep="\t", header=None,
                     names=["article_id", "entity_mention", "start_offset", "end_offset", "main_role",
                            "fine-grained_role_1", "fine-grained_role_2"])
    print(df.head())

    role_distribution(df)

    # Check keys in articles_dict
    print("Keys in articles_dict:", len(articles_dict))
    print(df.shape)
    # Check unique article IDs in the DataFrame
    print("Unique article IDs in df:", len(df["article_id"].unique()))
    # Count duplicate entries
    num_duplicates = df["article_id"].duplicated().sum()
    print(f"Number of duplicate entries: {num_duplicates}")


