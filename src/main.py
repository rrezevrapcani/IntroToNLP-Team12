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
    entity_start_positions = [item["entity_start_positions"] for item in batch]
    entity_end_positions = [item["entity_end_positions"] for item in batch]
    main_role_labels = [item["main_role_labels"] for item in batch]
    fine_role_labels = [item["fine_role_labels"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
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
            entity_start_positions = batch["entity_start_positions"]
            entity_end_positions = batch["entity_end_positions"]

            #prepare main role labels to calculate loss
            main_role_labels = torch.cat(
                [labels[:len(positions)] for labels, positions in zip(batch["main_role_labels"], batch["entity_start_positions"])]
            ).to(device)

            #prepare fine-grained role labels to calculate loss, making sure they are in the same shape as the logits
            fine_role_labels = {"protagonist": [], "antagonist": [], "innocent": []}
            for labels, roles in zip(batch["fine_role_labels"],  batch["main_role_labels"]):
                for idx, label in enumerate(labels):
                    if(roles[idx] == 0): #protagonist
                        fine_role_labels["protagonist"].append(label[:6])   
                    if(roles[idx] == 1): #antagonist
                        fine_role_labels["antagonist"].append(label[6:18])
                    if(roles[idx] == 2):
                        fine_role_labels["innocent"].append(label[18:22])
            #convert to tensors
            for key in fine_role_labels:
                if len(fine_role_labels[key]) > 0:
                    fine_role_labels[key] = torch.stack(fine_role_labels[key]).to(device)
                    num_classes = model.fine_grained_classifiers[key][-2].out_features
                    fine_role_labels[key] = fine_role_labels[key][:, :num_classes]
                else:
                    #create empty tensor if no labels are available
                    num_classes = model.fine_grained_classifiers[key][-2].out_features
                    fine_role_labels[key] = torch.empty(0, num_classes).to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask, entity_start_positions, entity_end_positions)

            #calculate both losses
            main_loss = loss_fn_main(outputs["main_role_logits"], main_role_labels)
            fine_loss = 0.0

            for role_key in outputs["fine_logits"]:
                logits = outputs["fine_logits"][role_key]
                labels = fine_role_labels[role_key]
                
                if logits.size(0) > 0 and labels.size(0) > 0:  # ensure both tensors have entries
                    # align predictions and labels by truncating to the minimum size
                    min_size = min(logits.size(0), labels.size(0))
                    logits = logits[:min_size]
                    labels = labels[:min_size]

                    fine_loss += loss_fn_fine(logits, labels)
                    
                    # fine-grained loss is calculated only with the samples that the main role was predicted correctly

            loss = main_loss + fine_loss

            #backpropagation
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
    fine_role_preds = {"protagonist": [], "antagonist": [], "innocent": []}
    fine_role_labels_dict = {"protagonist": [], "antagonist": [], "innocent": []}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            entity_start_positions = batch["entity_start_positions"]
            entity_end_positions = batch["entity_end_positions"]
            batch_main_labels = torch.cat(batch["main_role_labels"]).to(device)

            #prepare fine-grained role labels
            fine_role_labels = {"protagonist": [], "antagonist": [], "innocent": []}
            for labels, roles in zip(
                batch["fine_role_labels"], batch["main_role_labels"]
            ):
                for idx, label in enumerate(labels):
                    if roles[idx] == 0:  # protagonist
                        fine_role_labels["protagonist"].append(label[:6])
                    if roles[idx] == 1:  # antagonist
                        fine_role_labels["antagonist"].append(label[6:18])
                    if roles[idx] == 2:  # innocent
                        fine_role_labels["innocent"].append(label[18:22])

            outputs = model(input_ids, attention_mask, entity_start_positions, entity_end_positions)

            # store main role predictions and labels
            main_role_preds.extend(torch.argmax(outputs["main_role_logits"], dim=-1).cpu().tolist())
            main_role_labels.extend(batch_main_labels.cpu().tolist())

            # store fine-grained role predictions and labels
            for role_key in outputs["fine_logits"]:
                logits = outputs["fine_logits"][role_key]
                labels = fine_role_labels[role_key]

                if len(labels) > 0 and not isinstance(labels, torch.Tensor):
                    labels = torch.stack(labels)

                if logits.size(0) > 0:  
                    fine_role_preds[role_key].extend((logits > 0.5).int().cpu().tolist())
                    fine_role_labels_dict[role_key].extend(labels.cpu().tolist())

    all_fine_preds = []
    all_fine_labels = []
    for role_key in fine_role_preds:
        all_fine_preds.extend(fine_role_preds[role_key])
        all_fine_labels.extend(fine_role_labels_dict[role_key])

    # calculate metrics
    main_accuracy = accuracy_score(main_role_labels, main_role_preds)
    fine_match_ratio = np.mean(
        [np.array_equal(p, l) for p, l in zip(all_fine_preds, all_fine_labels)]
    )

    return main_accuracy, fine_match_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Role Classifier")
    parser.add_argument("--data_path", type=str, default="../data/EN/raw-documents", help="Directory containing the data")
    parser.add_argument("--annotation_file", type=str, default="../data/EN/subtask-1-annotations.txt", help="Annotation file")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Pretrained model name")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and validation")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train")

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
    train_dataset = EntityFramingDataset(train_texts, train_annotations, tokenizer)
    val_dataset = EntityFramingDataset(val_texts, val_annotations, tokenizer)

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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-5,
        total_steps=len(train_loader) * 10
    )

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
