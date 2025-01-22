import torch
import pandas as pd
import argparse
import os
from transformers import BertTokenizerFast
from dataset import read_data
from model import EntityRoleClassifier
from sklearn.metrics import accuracy_score

def generate_predictions(test_data, model_checkpoint, output_file, device, max_length=512, overlap=128):
    from transformers import BertTokenizerFast
    import torch
    import numpy as np

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    model = EntityRoleClassifier("bert-base-uncased")
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    model.to(device)
    model.eval()

    threshold = 0.51
    fine_role_names = [
        "Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous",  # Protagonist (0-5)
        "Instigator", "Conspirator", "Tyrant", "Foreign Adversary",
        "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent",
        "Terrorist", "Deceiver", "Bigot",  # Antagonist (6-17)
        "Forgotten", "Exploited", "Victim", "Scapegoat"  # Innocent (18-21)
    ]

    unique_predictions = {}

    with open(output_file, "w") as f:
        with torch.no_grad():
            for article_id, text, entity_mention, start_offset, end_offset in test_data:
                if not isinstance(text, str) or len(text.strip()) == 0:
                    print(f"Skipping empty or invalid text for article_id {article_id}")
                    continue

                words = text.split()
                total_chars = 0
                window_start_char = 0

                for i, word in enumerate(words):
                    if total_chars <= start_offset:
                        window_start_char = total_chars
                    total_chars += len(word) + 1

                context_before = text[max(0, window_start_char - max_length // 4):start_offset]
                entity_text = text[start_offset:end_offset + 1]
                context_after = text[end_offset + 1:min(len(text), end_offset + max_length // 4)]
                chunk_text = context_before + entity_text + context_after

                adjusted_start = len(context_before)
                adjusted_end = len(context_before) + len(entity_text) - 1

                try:
                    encoding = tokenizer(
                        chunk_text,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                        return_offsets_mapping=True
                    )

                    input_ids = encoding["input_ids"].to(device)
                    attention_mask = encoding["attention_mask"].to(device)

                    token_start = encoding.char_to_token(adjusted_start)
                    token_end = encoding.char_to_token(adjusted_end)

                    if token_start is not None and token_end is not None:
                        outputs = model(
                            input_ids,
                            attention_mask,
                            torch.tensor([[token_start]], dtype=torch.long).to(device),
                            torch.tensor([[token_end]], dtype=torch.long).to(device)
                        )

                        main_role_pred = torch.argmax(outputs["main_role_logits"], dim=-1).item()
                        main_role = ["Protagonist", "Antagonist", "Innocent"][main_role_pred]

                        fine_logits = torch.sigmoid(outputs["fine_logits"])
                        fine_logits = fine_logits.squeeze(0)

                        if main_role == "Protagonist":
                            relevant_indices = list(range(6))
                        elif main_role == "Antagonist":
                            relevant_indices = list(range(6, 18))
                        elif main_role == "Innocent":
                            relevant_indices = list(range(18, 22))

                        relevant_logits = fine_logits[relevant_indices]
                        print(relevant_logits)

                        topk_results = torch.topk(relevant_logits, min(2, len(relevant_indices)))
                        fine_roles = [
                            idx + relevant_indices[0]
                            for idx, score in zip(topk_results.indices.tolist(), topk_results.values.tolist())
                            if score > threshold
                        ]

                        fine_roles_str = ", ".join([fine_role_names[idx] for idx in fine_roles])

                        unique_predictions[(article_id, entity_mention, start_offset, end_offset)] = (main_role, fine_roles_str)

                except Exception as e:
                    print(f"Error processing entity {entity_mention} in article {article_id}: {str(e)}")
                    unique_predictions[(article_id, entity_mention, start_offset, end_offset)] = ("Innocent", "Victim")

        for (article_id, entity_mention, start_offset, end_offset), (main_role, fine_roles_str) in unique_predictions.items():
            f.write(f"{article_id}\t{entity_mention}\t{start_offset}\t{end_offset}\t{main_role}\t{fine_roles_str.replace(', ', '\t')}\n")



# not working
def calculate_metrics(predictions_file, gold_file):
    df_predictions = pd.read_csv(predictions_file, sep="\t", header=None)
    df_gold = pd.read_csv(gold_file, sep="\t", header=None)

    df_predictions.columns = ["article_id", "entity_mention", "start_offset", "end_offset", "main_role", "fine_roles"]
    df_gold.columns = ["article_id", "entity_mention", "start_offset", "end_offset", "main_role", "fine_roles"]

    df_merged = pd.merge(df_predictions, df_gold, on=["article_id", "entity_mention", "start_offset", "end_offset"], suffixes=("_pred", "_gold"))

    main_role_accuracy = accuracy_score(df_merged["main_role_gold"], df_merged["main_role_pred"])

    def exact_match_ratio(pred, gold):
        pred_set = set(pred.split(", "))
        gold_set = set(gold.split(", "))
        return pred_set == gold_set

    fine_role_exact_matches = df_merged.apply(lambda row: exact_match_ratio(row["fine_roles_pred"], row["fine_roles_gold"]), axis=1)
    fine_role_exact_match_ratio = fine_role_exact_matches.mean()

    return main_role_accuracy, fine_role_exact_match_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Role Classifier")
    parser.add_argument("--data-path", type=str, default="../dev_set/EN/subtask-1-documents", help="Directory containing the data")
    parser.add_argument("--entities-mention", type=str, default="../dev_set/EN/subtask-1-entity-mentions.txt", help="Annotation file")
    parser.add_argument("--model-checkpoint", type=str, default="best_entity_role_classifier.pt", help="Fine-tuning weights")
    parser.add_argument("--output-file", type=str, default="predictions.txt", help="Output file whose predictions are to be saved")

    args = parser.parse_args()
    articles, annotations = read_data(args.data_path, args.entities_mention, with_annotations=False)
    # print(annotations[0])
    print(len(annotations))


    generate_predictions(
        test_data=annotations,
        model_checkpoint=args.model_checkpoint,
        output_file=args.output_file,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # calculate_metrics("predictions.txt", "gold.txt")


