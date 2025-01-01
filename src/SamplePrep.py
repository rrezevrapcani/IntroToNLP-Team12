from DataPreprocessing import preprocess, create_anno_df
import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")


'''
THE NEWS ARTICLES ARE COMBINED WITH THE ANNOTATION SHEET AND FIRST USEFUL SAMPLES FOR TRAINING
BERT ARE CREATED.
THIS IS STILL VERY UNFINISHED AS THE CREATION OF TRAINING SAMPLES PROBABLY HAS MUCH IMPACT ON 
THE EVENTUAL PERFORMANCE.
'''


def create_raw_samples(articles, annotations):
    '''
    Combines text, entity and label in a dictionary, no tokenization yet
    :param articles: preprocessed articles as provided by preprocess(folder_path, annotation_path)
    :param annotations: label annotations as provided by create_anno_df(annotation_path)
    :return: list of dictionaries, one dictionary contains 'text', 'entity' and 'labels'
             Output format example:
             {'text': 'Bill Gates Says He Is ...',
              'entity': 'Bill Gates',
              'labels': ['Antagonist', 'Deceiver']}
             Highlights the entity in 'text' like:
             '... Four Private Jets Bill  [ENTITY] Bill Gates [/ENTITY]  the right to fly ...'
    '''

    raw_samples = []

    for _, row in annotations.iterrows():
        article_id = row["article_id"]
        entity = row["entity_mention"]
        start = row["start_offset"]
        end = row["end_offset"]

        # Extract text and highlight entity
        if article_id in articles:
            article_text = articles[article_id]
            highlighted_text = (
                    article_text[:start]
                    + f" [ENTITY] {entity} [/ENTITY] "
                    + article_text[end:]
            )

            # Consolidate labels into a list
            labels = [row["main_role"]]
            if pd.notna(row["fine-grained_role_1"]):
                labels.append(row["fine-grained_role_1"])
            if pd.notna(row["fine-grained_role_2"]):
                labels.append(row["fine-grained_role_2"])

            # Append sample
            raw_samples.append({
                "text": highlighted_text,
                "entity": entity,
                "labels": labels
            })

    return raw_samples


def tokenize_samples(raw_samples, mlb):
    '''
    Tokenizes the extracted samples from create_raw_samples
    :param raw_samples: samples as provided by create_raw_samples
    :param mlb: multi label binarizer
    :return: tokenized samples
    '''
    tokenized_samples = []

    for sample in raw_samples:
        # Tokenize the text
        tokenized_output = tokenizer(
            sample["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )

        # Binarize labels
        label_vector = mlb.transform([sample["labels"]])[0]

        # Append tokenized sample
        tokenized_samples.append({
            "input_ids": tokenized_output["input_ids"][0],
            "attention_mask": tokenized_output["attention_mask"][0],
            "labels": label_vector
        })

    return tokenized_samples


def create_samples(folder_path, annotation_path):
    '''
    Provides semi-final training samples from the given files to feed into BERT
    :param folder_path: folder containing news articles
    :param annotation_path: file containing annotations
    :return: list of dictionaries, one dictionary has output format:
             {"input_ids": torch.tensor([...]),  # tokenized input IDs
              "attention_mask": torch.tensor([...]),  # attention mask
              "labels": [1, 0, 1, ...]}  # multi-label vector
    '''
    articles = preprocess(folder_path, annotation_path)
    annotations = create_anno_df(annotation_path)

    # Prepare labels for multi-label classification
    all_labels = annotations["main_role"].unique().tolist()
    all_labels += annotations["fine-grained_role_1"].dropna().unique().tolist()
    all_labels += annotations["fine-grained_role_2"].dropna().unique().tolist()
    all_labels = list(set(all_labels))  # Remove duplicates

    # mlb encodes multi-label classifications into a binary matrix representation
    mlb = MultiLabelBinarizer(classes=all_labels)
    mlb.fit([all_labels])  # Fit the label encoder

    print("number of all labels: ", len(all_labels))

    # create raw samples
    raw_samples = create_raw_samples(articles, annotations)
    # tokenize them
    tokenized_samples = tokenize_samples(raw_samples, mlb)
    return tokenized_samples


class MultiLabelDataset(Dataset):
    '''
    Custom PyTorch Dataset
    '''
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.samples[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.samples[idx]["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.samples[idx]["labels"], dtype=torch.float),
        }


# just for some testing
if __name__ == '__main__':
    # Path to the directory containing .txt files
    folder_path = "../data/raw-documents"
    annotation_path = "../data/subtask-1-annotations.txt"

    #samples = create_samples(folder_path, annotation_path)
    #print(len(samples[0]['labels']))
    #print("number of all labels: ", len(samples[2]))
    #print("number of all labels: ", len(samples["labels"][0]))

    #tokenized_samples = create_samples(folder_path, annotation_path)
    #print(tokenized_samples[:2])  # Check the first two tokenized samples


    #articles = preprocess(folder_path, annotation_path)
    #annotations = create_anno_df(annotation_path)

    #print(articles)
    #print(anno_df)

    #training_samples = create_raw_samples(articles, annotations)
    #print(training_samples[:2])  # Check the first two samples




