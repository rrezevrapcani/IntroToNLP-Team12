import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import os
from model import EntityFramingModel
from dataset import EntityFramingDataset, role2idx, classes2idx, read_data 
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.nn.utils.rnn import pad_sequence

'''
FIRST DRAFT FINE-TUNING BERT FOR ENTITY FRAMING TASK

CONSIDERATIONS:
- The model is trained to predict the main role of an entity in a news article.
- The model is also trained to predict secondary roles for the entity.
- The secondary roles are specific to the main role.

'''

from torch.nn.utils.rnn import pad_sequence
import torch

# assure all sequences in a batch have the same length(there are data with more than one entity and data with no entity)
def collate_fn(batch):
    # find max length
    max_len = max(len(item['input_ids']) for item in batch)
    
    def pad_to_max_len(tensor, max_len, pad_value):
        pad_size = max_len - len(tensor)
        return torch.cat([tensor, torch.full((pad_size,), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_to_max_len(item['input_ids'], max_len, 0) for item in batch])
    attention_mask = torch.stack([pad_to_max_len(item['attention_mask'], max_len, 0) for item in batch])
    token_type_ids = torch.stack([pad_to_max_len(item['token_type_ids'], max_len, 0) for item in batch])
    ner_labels = torch.stack([pad_to_max_len(item['ner_labels'], max_len, -1) for item in batch])
    
    fine_grained_labels = torch.stack([item['fine_grained_labels'] for item in batch])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "ner_labels": ner_labels,
        "fine_grained_labels": fine_grained_labels,
    }


# Training Loop
def train_model(model, dataloader, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    # maybe something wrong here? also we have to save the best model..
    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            ner_labels = batch["ner_labels"].to(device)
            fine_grained_labels = batch["fine_grained_labels"].to(device)

            optimizer.zero_grad()
            
            loss, ner_logits, fine_grained_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                ner_labels=ner_labels,
                secondary_labels=fine_grained_labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    # save model checkpoint to test it later.
    model_save_path = os.path.join("checkpoints", f"model_epoch_{epoch + 1}.pt")
    torch.save(model.state_dict(), model_save_path)


# Evaluation Loop
def evaluate_model(model, dataloader, device):
    model.to(device)
    model.eval()

    # COMPLETE THIS FUNCTION. could not do it yet because I was trying to fix the training loop.

if __name__ == "__main__":
    # Load data
    texts, annotations = read_data("../data/EN/raw-documents", "../data/EN/subtask-1-annotations.txt")
    print(len(texts), len(annotations))

    # Split data
    train_texts, val_texts, train_annotations, val_annotations = train_test_split(
        texts, annotations, test_size=0.2, random_state=42
    )

    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Create datasets
    train_dataset = EntityFramingDataset(train_texts, train_annotations, tokenizer)
    val_dataset = EntityFramingDataset(val_texts, val_annotations, tokenizer)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print([len(classes2idx[i]) for i in range(3)])
    print(len(role2idx))
    #test it with multilingual bert  
    model = EntityFramingModel("bert-base-uncased", len(role2idx), [len(classes2idx[i]) for i in range(3)])

    # Move model to device
    model.to(device)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = 3 * len(train_loader)
    # to adjust the learning rate during training
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=5e-5, total_steps=num_training_steps
    )

    # Loss function
    loss_fn = nn.BCEWithLogitsLoss()

    # Train model
    train_model(model, train_loader, optimizer, 10, device)

    # Evaluate model
    evaluate_model(model, val_loader, device=device)
