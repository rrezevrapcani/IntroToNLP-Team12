import torch
import pandas as pd
import argparse
from transformers import BertTokenizerFast
from model import EntityRoleClassifier
from dataset import read_data

def generate_predictions(test_data, model_checkpoint, output_file, device):
    """
    Generate predictions for the test set and save them to a .txt file.

    :param test_data: List of dictionaries with keys: article_id, entity_mention, start_offset, end_offset, text.
    :param model_checkpoint: Path to the saved model checkpoint.
    :param output_file: Path to the output .txt file.
    :param device: Torch device (e.g., "cuda" or "cpu").
    """
    # Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = EntityRoleClassifier("bert-base-uncased")
    model.load_state_dict(torch.load(model_checkpoint))
    model.to(device)
    model.eval()

    # Prepare output file
    with open(output_file, "w") as f:
        # Write header
        f.write("article_id\tentity_mention\tstart_offset\tend_offset\tmain_role(*)\tfine-grained_roles(*)\n")

        with torch.no_grad():
            print(test_data[0])
            for article_id, text, entity_mention, start_offset, end_offset in test_data:
                    # Tokenize input text
                    encoding = tokenizer(
                        text,
                        padding="max_length",
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                        return_offsets_mapping=True
                    )
                    input_ids = encoding["input_ids"].to(device)
                    attention_mask = encoding["attention_mask"].to(device)
                    offsets = encoding["offset_mapping"].squeeze()

                    # Map entity offsets to token indices
                    token_start = None
                    token_end = None
                    for idx, (start, end) in enumerate(offsets):
                        if start == start_offset:
                            token_start = idx
                        if end == end_offset:
                            token_end = idx
                            break

                    # Skip if the entity cannot be mapped
                    if token_start is None or token_end is None:
                        continue

                    # Forward pass
                    outputs = model(
                        input_ids,
                        attention_mask,
                        torch.tensor([[token_start]], dtype=torch.long).to(device),
                        torch.tensor([[token_end]], dtype=torch.long).to(device)
                    )

                    # Decode predictions
                    main_role_pred = torch.argmax(outputs["main_role_logits"], dim=-1).item()
                    main_role = ["Protagonist", "Antagonist", "Innocent"][main_role_pred]

                    fine_logits = outputs["fine_logits"]
                    fine_roles = []
                    if main_role == "Protagonist":
                        fine_roles = (fine_logits["protagonist"] > 0.7).nonzero(as_tuple=True)[1].cpu().tolist()
                    elif main_role == "Antagonist":
                        print(fine_logits["protagonist"])
                        fine_roles = (fine_logits["antagonist"] > 0.55).nonzero(as_tuple=True)[1].cpu().tolist()
                    elif main_role == "Innocent":
                        fine_roles = (fine_logits["innocent"] > 0.55).nonzero(as_tuple=True)[1].cpu().tolist()

                    # Map fine-grained role indices to their names
                    fine_role_names = []
                    if main_role == "Protagonist":
                        fine_role_names = ["Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous"]
                    elif main_role == "Antagonist":
                        fine_role_names = ["Instigator", "Conspirator", "Tyrant", "Foreign Adversary", 
                                        "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent", 
                                        "Terrorist", "Deceiver", "Bigot"]
                    elif main_role == "Innocent":
                        fine_role_names = ["Forgotten", "Exploited", "Victim", "Scapegoat"]

                    fine_roles_str = ", ".join([fine_role_names[idx] for idx in fine_roles])

                    # Write prediction to file
                    f.write(f"{article_id}\t{entity_mention}\t{start_offset}\t{end_offset}\t{main_role}\t{fine_roles_str}\n")

        print(f"Predictions saved to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Role Classifier")
    parser.add_argument("--data_path", type=str, default="../data/EN/raw-documents", help="Directory containing the data")
    parser.add_argument("--entities_mention", type=str, default="../data/EN/subtask-1-annotations.txt", help="Annotation file")
    parser.add_argument("--weights_path", type=str, default="best_entity_role_classifier.pt", help="Fine-tuning weights path")
    parser.add_argument("--output_file", type=str, default="predictions.txt", help="Output file whose predictions are to be saved")

    args = parser.parse_args()
    articles, annotations = read_data(args.data_path, args.entities_mention, with_annotations=False)


    generate_predictions(
        test_data=annotations,
        model_checkpoint=args.weights_path,
        output_file=args.output_file,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )


