import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from DataPreprocessing import preprocess
#import torch
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


'''
Bert-large-uncased:
-Why I chose this: this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked) to make decisions, such as sequence classification, token classification or question answering.

-notes:

All text is lowercased during tokenization
Max token length is 550
Longer texts are truncated if exceeding this limit
Tokenized Articles:

All the labeled articles
Tokenized with Bertokenizer
All formated in Pytorch tensors
'''
def bert_tokenize(labeled_articles):
    # Tokenize each article and store the results
    tokenized_articles = {}

    for article_id, content in labeled_articles.items():
        # Tokenize the article content
        tokenized_output = tokenizer(
            content,
            truncation=True,  # Truncate if text exceeds max length
            return_tensors="pt"  # Return PyTorch tensors
        )

        # Store tokenized output
        tokenized_articles[article_id] = {
            "input_ids": tokenized_output["input_ids"][0].tolist(),
            "attention_mask": tokenized_output["attention_mask"][0].tolist()
        }
    return tokenized_articles

if __name__ == '__main__':
    # Path to the directory containing .txt files
    folder_path = "../data/raw-documents"
    annotation_path = "../data/subtask-1-annotations.txt"

    articles = preprocess(folder_path, annotation_path)
    tokenized_articles = bert_tokenize(articles)

    # example output for an entry
    print(tokenized_articles['EN_UA_DEV_26.txt'].keys())