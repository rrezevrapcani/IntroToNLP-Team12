import os
import pandas as pd
import re
import torch
from torch.utils.data import Dataset

main2idx = {"Protagonist": 0, "Antagonist": 1, "Innocent": 2}
fine_grained2idx = [
    {"Guardian": 0, "Martyr": 1, "Peacemaker": 2, "Rebel": 3, "Underdog": 4, "Virtuous": 5},
    {"Instigator": 0, "Conspirator": 1, "Tyrant": 2, "Foreign Adversary": 3, "Traitor": 4, "Spy": 5, "Saboteur": 6, "Corrupt": 7, "Incompetent": 8, "Terrorist": 9, "Deceiver": 10, "Bigot": 11},
    {"Forgotten": 0, "Exploited": 1, "Victim": 2, "Scapegoat": 3},
]

#deal with emojis in articles so the entities offset are not messed up
def replace_emojis_with_codes(text):
    def emoji_to_code(match):
        emoji = match.group()
        return ''.join(f'::' for char in emoji)

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF" 
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F700-\U0001F77F"  
        u"\U0001F780-\U0001F7FF" 
        u"\U0001F800-\U0001F8FF" 
        u"\U0001F900-\U0001F9FF"  
        u"\U0001FA00-\U0001FA6F" 
        u"\U0001FA70-\U0001FAFF"  
        u"\U00002702-\U000027B0"  
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(emoji_to_code, text)

def create_anno_df(annotation_path):
    '''
    Creates a DataFrame containing the articles annotations (ordened by article_id because the same article can have more than one line of information).
    :param annotation_path: file containing annotations
    :return: Output format:
                article_id	        entity_mention	start_offset	end_offset	main_role	fine-grained_role_1	fine-grained_role_2
             0	EN_CC_100013.txt	Bill Gates	    93	            102	        Antagonist	Deceiver	        Corrupt
             1	EN_CC_100013.txt	BBC	            1860	        1862	    Antagonist	Deceiver	        NaN
    '''
    anno_df = pd.read_csv(annotation_path, sep="\t", header=None,
                     names=["article_id", "entity_mention", "start_offset", "end_offset", "main_role",
                            "fine-grained_role_1", "fine-grained_role_2", "fine-grained_role_3"]).sort_values(by="article_id")
    return anno_df

def read_data(articles_path, annotations_path, with_annotations=True):
    '''
    Correspond annotations to the articles, and remove articles that do not have annotations.
    :param articles_path: folder containing news articles
    :param annotations_path: file containing annotations
    :param with_annotations: if True, return annotations with grount-truth labels, otherwise return only input format
    :return: two lists, one containing cleaned articles and one containg corresponded annotations
    '''
    articles = os.listdir(articles_path)
    anno_df = create_anno_df(annotations_path)
    # remove articles that do not have annotations
    articles = [article for article in articles if article in anno_df["article_id"].unique()]

    articles_content = []
    annotations = []

    for article in articles:
        with open(os.path.join(articles_path, article), "r", encoding="utf-8") as file:
            original_text = file.read()
        original_text = replace_emojis_with_codes(original_text)

        article_annotations = anno_df[anno_df["article_id"] == article]
        current_annotation = []
        

        for idx in article_annotations.index:
            entity = article_annotations.loc[idx, "entity_mention"]
            start = article_annotations.loc[idx, "start_offset"]
            end = article_annotations.loc[idx, "end_offset"]
            main_role = article_annotations.loc[idx, "main_role"]

            # debuging to see if offsets are correct
            # if entity != original_text[start:end + 1]:
            #     print(entity, original_text[start:end + 1])
            #     continue
            if not with_annotations:
                annotations.append((article, original_text, entity, start, end))
                continue
            role_idx = main2idx[main_role]
            fine_grained_roles = []
            if pd.notna(article_annotations.loc[idx, "fine-grained_role_1"]):
                fine_grained_roles.append(article_annotations.loc[idx, "fine-grained_role_1"])
            if pd.notna(article_annotations.loc[idx, "fine-grained_role_2"]):
                fine_grained_roles.append(article_annotations.loc[idx, "fine-grained_role_2"])
            if pd.notna(article_annotations.loc[idx, "fine-grained_role_3"]):
                fine_grained_roles.append(article_annotations.loc[idx, "fine-grained_role_3"])

            current_annotation.append([
                start,
                end,
                role_idx, 
                fine_grained_roles
            ])
        if with_annotations and len(current_annotation) == 0:
            continue
        if with_annotations:
            annotations.append(current_annotation)
        articles_content.append(original_text)
    return articles_content, annotations


class EntityFramingDataset(Dataset):
    def __init__(self, texts, annotations, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fine_role2idx = self._initialize_roles()
        
        # filter out data with entity within the max_length
        self.data = []
        for text, annotation in zip(texts, annotations):
            if self._has_valid_entities(text, annotation):
                self.data.append((text, annotation))
                
    
    def _initialize_roles(self):
        """
        Initialize fine-grained role to index mapping.
        ::return:: fine_role2idx: fine-grained role to index mapping(dict with the 22 fine-grained role as key and index as value)
        """
        protagonist_roles = ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"]
        antagonist_roles = ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", 
                            "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", 
                            "Terrorist", "Deceiver", "Bigot"]
        innocent_roles = ["Forgotten", "Exploited", "Victim", "Scapegoat"]
        fine_role2idx = {}
        for i, role in enumerate(protagonist_roles):
            fine_role2idx[role] = i
        for i, role in enumerate(antagonist_roles):
            fine_role2idx[role] = i + len(protagonist_roles)
        for i, role in enumerate(innocent_roles):
            fine_role2idx[role] = i + len(protagonist_roles) + len(antagonist_roles)
        return fine_role2idx

    def _has_valid_entities(self, text, annotation):
        """
        Check if the text and its annotation contain at least one valid entity.
        """
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        for entity_data in annotation:
            start, end, _, _ = entity_data
            token_start = encoding.char_to_token(start)
            token_end = encoding.char_to_token(end + 1)

            if token_start is not None and token_end is not None:
                if token_start < self.max_length and token_end + 1 < self.max_length:
                    return True  # at least one valid entity exists
        return False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, annotation = self.data[idx]
        
        # tokenize text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True  
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        token_type_ids = encoding["token_type_ids"].squeeze()
        
        entity_start_positions = []
        entity_end_positions = []
        main_role_labels = []
        fine_role_labels = []
        
        for entity_data in annotation:
            start, end, main_role_idx, fine_grained_roles = entity_data

            #find token indices for entity start and end character positions
            token_start = encoding.char_to_token(start)
            token_end = encoding.char_to_token(end + 1)

            if token_start is None or token_end is None:
                continue
            
            entity_start_positions.append(token_start)
            entity_end_positions.append(token_end - 1)
            main_role_labels.append(main_role_idx)
            
            # create fine-grained role label vector(size: [22])
            fine_label_vector = torch.zeros(22, dtype=torch.float)
            for role in fine_grained_roles:
                fine_label_vector[self.fine_role2idx[role]] = 1.0
            # print(fine_label_vector.count_nonzero() == len(fine_grained_roles))
            fine_role_labels.append(fine_label_vector)
        
        #convert to tensors
        entity_start_positions = torch.tensor(entity_start_positions, dtype=torch.long)
        entity_end_positions = torch.tensor(entity_end_positions, dtype=torch.long)
        main_role_labels = torch.tensor(main_role_labels, dtype=torch.long)
        fine_label_vector = torch.stack(fine_role_labels)

        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "entity_start_positions": entity_start_positions,
            "entity_end_positions": entity_end_positions,
            "main_role_labels": main_role_labels,
            "fine_role_labels": fine_label_vector
        }


        