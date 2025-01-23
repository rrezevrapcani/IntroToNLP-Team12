import torch
import torch.nn as nn
import numpy as np
import os
from transformers import BertTokenizerFast
from sklearn.model_selection import train_test_split
from model import EntityRoleClassifier
from dataset import EntityFramingDataset, read_data
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import argparse

def collate_fn(batch):
    """
    Custom collate function to handle variable-length entity annotations.
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    token_type_ids = torch.stack([item["token_type_ids"] for item in batch])
    entity_start_positions = [item["entity_start_positions"] for item in batch]
    entity_end_positions = [item["entity_end_positions"] for item in batch]
    main_role_labels = [item["main_role_labels"] for item in batch]
    fine_role_labels = [item["fine_role_labels"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "entity_start_positions": entity_start_positions,
        "entity_end_positions": entity_end_positions,
        "main_role_labels": main_role_labels,
        "fine_role_labels": fine_role_labels,
    }

def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device):
    """
    Trains the Entity Role Classifier Model.
    """
    model.to(device)
    best_val_accuracy = 0.0
    # using CrossEntropyLoss for main role classification and BCEWithLogitsLoss for fine-grained classification
    loss_fn_main = nn.CrossEntropyLoss()
    loss_fn_fine = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            entity_start_positions = batch["entity_start_positions"]
            entity_end_positions = batch["entity_end_positions"]

            main_role_labels = torch.cat(
                [labels[:len(positions)] for labels, positions in zip(batch["main_role_labels"], batch["entity_start_positions"])]
            ).to(device)

            fine_role_labels = torch.cat(
                [labels[:len(positions)] for labels, positions in zip(batch["fine_role_labels"], batch["entity_start_positions"])]
            ).to(device)
            
            optimizer.zero_grad()   

            outputs = model(input_ids, attention_mask, token_type_ids, entity_start_positions, entity_end_positions)

            # calculate main role loss
            main_loss = loss_fn_main(outputs["main_role_logits"], main_role_labels)
            
            # calculate fine-grained loss for all samples, but mask incorrect main roles
            batch_size = len(fine_role_labels)
            fine_loss = torch.tensor(0.0, device=device)
            
            for i in range(batch_size):
                main_pred = torch.argmax(outputs["main_role_logits"][i])
                main_true = main_role_labels[i]
                
                # get the relevant slice of fine-grained logits based on true main role
                if main_true == 0:  # protagonist
                    relevant_logits = outputs["fine_logits"][i, :6]
                    relevant_labels = fine_role_labels[i, :6]
                elif main_true == 1:  # antagonist
                    relevant_logits = outputs["fine_logits"][i, 6:18]
                    relevant_labels = fine_role_labels[i, 6:18]
                else:  # innocent
                    relevant_logits = outputs["fine_logits"][i, 18:]
                    relevant_labels = fine_role_labels[i, 18:]
                
                # calculate loss with a weight based on main role prediction
                weight = 1.0 if main_pred == main_true else 0.5
                sample_fine_loss = weight * loss_fn_fine(relevant_logits, relevant_labels)
                fine_loss += sample_fine_loss
                
            fine_loss = fine_loss / batch_size
            
            # combined loss with dynamic weighting
            loss = 1.2 * main_loss + fine_loss

            # backpropagation
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # evaluating epoch
        val_accuracy, val_fine_match_ratio = evaluate_model(model, val_loader, device)
        print(f"Validation Accuracy: {val_accuracy:.4f}, Fine-Grained Match Ratio: {val_fine_match_ratio:.4f}")

        #save model with best exact match ratio
        if val_fine_match_ratio > best_val_accuracy:
            best_val_accuracy = val_fine_match_ratio
            torch.save(model.state_dict(), "best_entity_role_classifier.pt")
            print("Saved best model.")


def evaluate_model(model, val_loader, device):
    """
    Evaluates the Entity Role Classifier Model.
    """
    model.to(device)
    model.eval()
    main_role_preds = []
    main_role_labels = []
    fine_role_preds = []
    fine_role_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tokens_type_ids = batch["token_type_ids"].to(device)
            entity_start_positions = batch["entity_start_positions"]
            entity_end_positions = batch["entity_end_positions"]
            batch_main_labels = torch.cat(batch["main_role_labels"]).to(device)
            batch_fine_labels = torch.cat(batch["fine_role_labels"]).to(device)

            outputs = model(input_ids, attention_mask, tokens_type_ids, entity_start_positions, entity_end_positions)

            # store main role predictions and labels
            main_role_preds.extend(torch.argmax(outputs["main_role_logits"], dim=-1).cpu().tolist())
            main_role_labels.extend(batch_main_labels.cpu().tolist())

            fine_role_preds.extend((outputs["fine_logits"] > 0.5).int().cpu().tolist())
            fine_role_labels.extend(batch_fine_labels.cpu().tolist())


    # calculate metrics
    main_accuracy = accuracy_score(main_role_labels, main_role_preds)
    fine_match_ratio = np.mean(
        [np.array_equal(p, l) for p, l in zip(fine_role_preds, fine_role_labels)]
    )

    return main_accuracy, fine_match_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Role Classifier")
    parser.add_argument("--data-path", type=str, default="../data/EN/raw-documents", help="Directory containing the data")
    parser.add_argument("--annotation-file", type=str, default="../data/EN/subtask-1-annotations.txt", help="Annotation file")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--num-epochs", type=int, default=10, help="Number of epochs to train")

    args = parser.parse_args()
    #load data
    texts, annotations = read_data(args.data_path, args.annotation_file)
    print(f"Loaded {len(texts)} articles with annotations.")

    #split data
    train_texts, val_texts, train_annotations, val_annotations = train_test_split(
        texts, annotations, test_size=0.2, random_state=42
    )

    #initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    #create datasets
    train_dataset = EntityFramingDataset(texts, annotations, tokenizer)
    texts, annotations = read_data("../dev_set/EN/subtask-1-documents", "../dev_set/EN/subtask-1-annotations.txt")
    val_dataset = EntityFramingDataset(texts, annotations, tokenizer)
    # train_dataset = EntityFramingDataset(train_texts, train_annotations, tokenizer)
    # val_dataset = EntityFramingDataset(val_texts, val_annotations, tokenizer)

    #create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn #custom collate function to handle variable-length entity annotations
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn #custom collate function to handle variable-length entity annotations
    )

    #initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntityRoleClassifier(args.model_name)

    #initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    #train model    
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device
    )