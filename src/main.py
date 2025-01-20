import torch
import torch.nn as nn
import numpy as np
import os
from transformers import BertModel, BertTokenizerFast, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from model import EntityFramingModel
from dataset import EntityFramingDataset, main2idx, fine_grained2idx, read_data 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence

'''
FIRST DRAFT FINE-TUNING BERT FOR ENTITY FRAMING TASK

CONSIDERATIONS:
- The model is trained to predict the main role of an entity in a news article.
- The model is also trained to predict secondary roles for the entity.
- The secondary roles are specific to the main role.

'''

def collate_fn(batch):
    """
    Custom collate function to handle batches with different sequence lengths
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    main_labels = [item['main_labels'] for item in batch]
    fine_grained_labels = [item['fine_grained_labels'] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True)
    attention_masks = pad_sequence(attention_masks, batch_first=True)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    main_labels = pad_sequence(main_labels, batch_first=True)
    
    fine_grained_labels = torch.stack(fine_grained_labels)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_masks,
        'token_type_ids': token_type_ids,
        'main_labels': main_labels,
        'fine_grained_labels': fine_grained_labels
    }

# Training Loop
def train_model(model, dataloader, val_dataloader, optimizer, num_epochs, device):
    model.to(device)
    model.train()

    best_val_emr = float("inf")

    # maybe something wrong here? also we have to save the best model..
    for epoch in range(num_epochs):
        total_loss = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            main_labels = batch["main_labels"].to(device)
            fine_grained_labels = batch["fine_grained_labels"].to(device)

            optimizer.zero_grad()
            
            loss, ner_logits, fine_grained_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                main_labels=main_labels,
                fine_grained_labels=fine_grained_labels
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

        #call evaluate_model after each epoch
        metrics = evaluate_model(model, val_dataloader, device)
        print(f"{metrics}")

        val_emr = metrics['fine_grained_exact_match'] 
        if val_emr > best_val_emr:
            best_val_emr = val_emr
            model_save_path = os.path.join("checkpoints", f"best_model_epoch_{epoch + 1}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch + 1}")



# Evaluation Loop
def evaluate_model(model, dataloader, device, threshold=0.3):  # Lower threshold from 0.5 to 0.3
    model.eval()
    all_main_preds = []
    all_main_labels = []
    all_fine_grained_preds = []
    all_fine_grained_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            main_labels = batch["main_labels"].to(device)
            fine_grained_labels = batch["fine_grained_labels"].to(device)
            
            res = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            if len(res) == 2:
                main_logits, fine_grained_logits = res
            else:
                loss, main_logits, fine_grained_logits = res
            
            # main role predictions
            main_preds = torch.argmax(main_logits, dim=-1)
            
            # get predictions for fine-grained roles, as multi-label classification we use sigmoid and select responses above threshold
            fine_grained_concat = torch.cat(list(fine_grained_logits.values()), dim=1)
            fine_grained_preds = (torch.sigmoid(fine_grained_concat) > threshold).long()
            
            for i in range(len(main_labels)):
                mask = attention_mask[i].bool()
                all_main_preds.extend(main_preds[i][mask].cpu().numpy())
                all_main_labels.extend(main_labels[i][mask].cpu().numpy())
                
            all_fine_grained_preds.extend(fine_grained_preds.cpu().numpy())
            all_fine_grained_labels.extend(fine_grained_labels.cpu().numpy())
    
    all_main_preds = np.array(all_main_preds)
    all_main_labels = np.array(all_main_labels)
    all_fine_grained_preds = np.array(all_fine_grained_preds)
    all_fine_grained_labels = np.array(all_fine_grained_labels)
    
    # calculate metrics: accuracy for main role and exact match ratio for fine-grained roles
    main_accuracy = accuracy_score(all_main_labels, all_main_preds)

    fine_grained_exact_match = np.mean([
        np.array_equal(pred, label)
        for pred, label in zip(all_fine_grained_preds, all_fine_grained_labels)
        if label.sum() > 0 or pred.sum() > 0  
    ])
    
    return {
        'main_accuracy': main_accuracy,
        'fine_grained_exact_match': fine_grained_exact_match,
    }

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

    # # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EntityFramingModel("bert-base-uncased", len(main2idx), [len(fine_grained2idx[i]) for i in range(3)])

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
    train_model(model, train_loader, val_loader, optimizer, 10, device)
