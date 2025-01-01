import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from torch.nn import BCEWithLogitsLoss
from transformers import get_scheduler
import numpy as np
from SamplePrep import MultiLabelDataset, create_samples

'''
THIS IS A FIRST DRAFT OF HOW TRAINING COULD LOOK.
THIS HAS NOT BEEN RUN SO NO GUARANTEE FOR RESULTS.
'''


# Training Loop
def train_model():
    model.train()
    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Evaluation Loop
def evaluate_model(loader):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Compute loss
            loss = loss_fn(logits, labels)
            total_loss += loss.item()

            # Collect predictions and labels for evaluation
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    print(f"Evaluation Loss: {avg_loss:.4f}")

    # Convert predictions to binary (threshold = 0.5)
    binary_preds = (np.array(all_preds) >= 0.5).astype(int)

    # Metrics (example: F1-score)
    from sklearn.metrics import f1_score
    f1 = f1_score(all_labels, binary_preds, average="weighted")
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    folder_path = "../data/raw-documents"
    annotation_path = "../data/subtask-1-annotations.txt"

    # samples is a list of tokenized examples in this format:
    # {"input_ids": ..., "attention_mask": ..., "labels": ...}
    samples = create_samples(folder_path, annotation_path)
    print("number of all labels: ", len(samples["labels"]))
    print("number of all labels: ", len(samples["labels"][0]))


    # Split data into train/val/test
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)
    val_samples, test_samples = train_test_split(test_samples, test_size=0.5, random_state=42)

    # Create PyTorch datasets
    train_dataset = MultiLabelDataset(train_samples)
    val_dataset = MultiLabelDataset(val_samples)
    test_dataset = MultiLabelDataset(test_samples)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # Load the BERT model for multi-label classification
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(samples[0]['labels']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Learning rate scheduler
    num_training_steps = len(train_loader) * 3  # Assuming 3 epochs
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Loss function
    loss_fn = BCEWithLogitsLoss()

    # Train and Evaluate
    train_model()
    evaluate_model(val_loader)  # Evaluate on validation set




