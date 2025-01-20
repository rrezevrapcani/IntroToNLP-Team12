import os
import pandas as pd
import re
import torch

'''
Corresponding respective multiple annotations to the same article
'''

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

def create_anno_df(annotation_path):
    '''
    Creates a DataFrame containing the articles annotations (ordened by article_id because the same article can have more than one line of information).
    :param annotation_path: file containing annotations
    :return: Output format:
                article_id	        entity_mention	start_offset	end_offset	main_role	fine-grained_role_1	fine-grained_role_2
             0	EN_CC_100013.txt	Bill Gates	    93	            102	        Antagonist	Deceiver	        Corrupt
             1	EN_CC_100013.txt	BBC	            1860	        1862	    Antagonist	Deceiver	        NaN
             ...
    '''
    anno_df = pd.read_csv(annotation_path, sep="\t", header=None,
                     names=["article_id", "entity_mention", "start_offset", "end_offset", "main_role",
                            "fine-grained_role_1", "fine-grained_role_2"]).sort_values(by="article_id")
    return anno_df

def read_data(articles_path, annotations_path):
    '''
    Correspond annotations to the articles.
    :param articles_path: folder containing news articles
    :param annotations_path: file containing annotations
    :return: two lists, one containing cleaned articles and one containg corresponded annotations
    '''
    articles = os.listdir(articles_path)
    anno_df = create_anno_df(annotations_path)

    articles_unique = anno_df["article_id"].unique()
    articles = [article for article in articles if article in articles_unique]

    articles_content = []
    annotations = []

    for article in articles:
        with open(os.path.join(articles_path, article), "r", encoding="utf-8") as file:
            original_text = file.read()
            cleaned_text = clean_article(original_text)

        article_annotations = anno_df[anno_df["article_id"] == article]
        current_annotation = []

        if len(article_annotations.index) == 0:
            current_annotation.append([
                None,
                None, 
                main2idx["O"],
                []
            ])
            
        for idx in article_annotations.index:
            entity = article_annotations.loc[idx, "entity_mention"]
            start = article_annotations.loc[idx, "start_offset"]
            end = article_annotations.loc[idx, "end_offset"]

            cleaned_text = (
                    cleaned_text[:start]
                    + f" [ENT] {entity} [/ENT] "
                    + cleaned_text[end:]
            )

            main_role = article_annotations.loc[idx, "main_role"]
            role_idx = main2idx[main_role]

            fine_grained_roles = []
            if pd.notna(article_annotations.loc[idx, "fine-grained_role_1"]):
                fine_grained_roles.append(article_annotations.loc[idx, "fine-grained_role_1"])
            if pd.notna(article_annotations.loc[idx, "fine-grained_role_2"]):
                fine_grained_roles.append(article_annotations.loc[idx, "fine-grained_role_2"])

            current_annotation.append([
                start,
                end,
                role_idx, 
                fine_grained_roles
            ])
        annotations.append(current_annotation)
        articles_content.append(cleaned_text)
    return articles_content, annotations


main2idx = {"Protagonist": 0, "Antagonist": 1, "Innocent": 2, "None": 3}
fine_grained2idx = [
    {"Guardian": 0, "Martyr": 1, "Peacemaker": 2, "Rebel": 3, "Underdog": 4, "Virtuous": 5},
    {"Instigator": 0, "Conspirator": 1, "Tyrant": 2, "Foreign Adversary": 3, "Traitor": 4, "Spy": 5, "Saboteur": 6, "Corrupt": 7, "Incompetent": 8, "Terrorist": 9, "Deceiver": 10, "Bigot": 11},
    {"Forgotten": 0, "Exploited": 1, "Victim": 2, "Scapegoat": 3},
    {}
]
    
class EntityFramingDataset(torch.utils.data.Dataset):
    def __init__(self, texts, annotations, tokenizer, max_length=512):
        self.texts = texts
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        annotations = self.annotations[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",  # Pad sequences to max_length
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        main_labels = torch.zeros(encoding.input_ids.size(1), dtype=torch.long)  # Initialize NER labels
        all_fine_grained_labels = []
        entity_spans = []  # Initialize entity spans list
        
        # Create zero-filled fine-grained labels for each role
        for i, role in enumerate(main2idx.keys()):
            if role != "None":
                all_fine_grained_labels.append(torch.zeros(len(fine_grained2idx[i])))

        # Fill in NER and fine-grained labels
        for start, end, role_idx, labels in annotations:
            if start is None or end is None:
                entity_spans.append((-1, -1))
                continue
            
            start_token = encoding.char_to_token(start) 
            end_token = encoding.char_to_token(end - 1)
            if start_token is not None and end_token is not None:
                main_labels[start_token:end_token + 1] = role_idx
                entity_spans.append((start_token, end_token))  # Store the entity span

            if role_idx < len(all_fine_grained_labels):
                for label in labels:
                    label_idx = fine_grained2idx[role_idx].get(label, None)
                    if label_idx is not None:
                        all_fine_grained_labels[role_idx][label_idx] = 1.0

        # Concatenate fine-grained labels (flatten them)
        fine_grained_labels_tensor = torch.cat(all_fine_grained_labels)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "main_labels": main_labels,
            "fine_grained_labels": fine_grained_labels_tensor,
        }

