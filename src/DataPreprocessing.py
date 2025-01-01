import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def clean_article(text):
    '''
    Cleans a single article by removing non-ASCII characters, unwanted symbols, and normalizing whitespace.
    :param text: text from the article
    :return: cleaned text
    '''
    # Remove non-ASCII characters (e.g., emojis, special symbols)
    text = text.encode("ascii", "ignore").decode("utf-8")

    # Remove unwanted symbols (e.g., @, #, $, etc.)
    text = re.sub(r"[^\w\s.,!?']", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def read_and_clean_articles(folder_path):
    '''
    Reads all .txt articles from a directory, cleans them, and stores them in a dictionary.
    :param folder_path: folder containing news articles
    :return: dictionary containing cleaned articles
    '''
    articles = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            # Read the article
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as file:
                original_text = file.read()

            # Clean the article
            cleaned_text = clean_article(original_text)

            # Store in dictionary with the filename (without extension) as key
            articles[filename] = {
                "cleaned": cleaned_text
            }
    return articles


def remove_empty_articles(articles_dict):
    '''
    Removes empty or near-empty articles from the dataset.
    An article is considered empty if its 'cleaned' text is
    empty or contains only whitespace.
    :param articles_dict: dictionary containing cleaned articles as provided by read_and_clean_articles()
    :return: dictionary without empty articles
    '''
    non_empty_articles = {
        key: value
        for key, value in articles_dict.items()
        if value["cleaned"].strip()  # Check if 'cleaned' text is not empty
    }
    return non_empty_articles


def create_anno_df(annotation_path):
    '''
    Creates a DataFrame containing the articles annotations.
    :param annotation_path: file containing annotations
    :return: Output format:
                article_id	        entity_mention	start_offset	end_offset	main_role	fine-grained_role_1	fine-grained_role_2
             0	EN_CC_100013.txt	Bill Gates	    93	            102	        Antagonist	Deceiver	        Corrupt
             1	EN_CC_100013.txt	BBC	            1860	        1862	    Antagonist	Deceiver	        NaN
             ...
    '''
    anno_df = pd.read_csv(annotation_path, sep="\t", header=None,
                     names=["article_id", "entity_mention", "start_offset", "end_offset", "main_role",
                            "fine-grained_role_1", "fine-grained_role_2"])
    return anno_df


def remove_unlabelled_articles(cleaned_articles_dict, anno_df):
    '''
    Some articles do not appear in the annotation sheet.
    E.g article EN_CC_100000.txt has no labelled entities, which makes it uninteresting for training.
    :param cleaned_articles_dict: dictionary containing the cleaned articles as provided by read_and_clean_articles()
    :param anno_df: DataFrame containing the annotations as provided by create_anno_df()
    :return: dictionary containing cleaned articles which appear in the annotation sheet
    '''
    # Separate labeled and unlabeled articles
    labeled_df = anno_df[anno_df["entity_mention"].notnull()]  # Keep rows with labels
    # sub-dictionary for labeled articles
    labeled_article_ids = labeled_df["article_id"].unique()
    labeled_articles = {key: value['cleaned'] for key, value in cleaned_articles_dict.items() if
                        key in labeled_article_ids}
    return labeled_articles


def preprocess(folder_path, annotation_path):
    '''
    Provides full preprocessing steps for articles:
    - reads files into dictionaries
    - cleans articles
    - removes articles which do not have labelled entities from the annotation file
    :param folder_path: folder containing news articles
    :param annotation_path: file containing annotations
    :return:
    '''
    articles_dict = read_and_clean_articles(folder_path)
    cleaned_articles_dict = remove_empty_articles(articles_dict)
    anno_df = create_anno_df(annotation_path)
    labeled_articles = remove_unlabelled_articles(cleaned_articles_dict, anno_df)
    return labeled_articles

# ---------------------------------------------------- VISUALIZATION, NO FUNCTIONAL USE

def role_distribution(anno_df):
    # show plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x='main_role', data=anno_df, order=anno_df['main_role'].value_counts().index)
    plt.title('Distribution of Main Roles')
    plt.xlabel('Main Role')
    _ = plt.ylabel('Frequency')

    # print values for main roles
    anno_df['main_role'].value_counts()

    # print values for fine-grained roles 1
    anno_df['fine-grained_role_1'].value_counts()

    # print values for fine-grained roles 2
    anno_df['fine-grained_role_2'].value_counts()



# just for some testing and visualization
if __name__ == "__main__":
    # Path to the directory containing .txt files
    folder_path = "../data/raw-documents"
    annotation_path = "../data/subtask-1-annotations.txt"

    # Read and clean articles
    articles_dict = read_and_clean_articles(folder_path) #Dict with article id's and their cleaned versions
    cleaned_articles_dict = remove_empty_articles(articles_dict)
    # Access cleaned or original version of an article
    article_id = "EN_UA_DEV_26.txt"  # Replace with the desired filename (without .txt extension)
    print(f"Original number of articles: {len(articles_dict)}")
    print(f"Number of non-empty articles: {len(cleaned_articles_dict)}")
    print("\nCleaned Article:\n", articles_dict[article_id]["cleaned"], "\n")

    # Read the annotations into a DataFrame
    anno_df = create_anno_df(annotation_path)


    # ------------------------------------------------ CLASS IMBALANCE VISUALIZATION

    print(anno_df.head())
    role_distribution(anno_df)

    # Check keys in articles_dict
    print("Keys in articles_dict:", len(articles_dict))
    print(anno_df.shape)
    # Check unique article IDs in the DataFrame
    print("Unique article IDs in df:", len(anno_df["article_id"].unique()))
    # Count duplicate entries
    num_duplicates = anno_df["article_id"].duplicated().sum()
    print(f"Number of duplicate entries: {num_duplicates}")


