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
    model.load_state_dict(torch.load(model_checkpoint, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    threshold = 0.51
    fine_role_names = [
        "Guardian", "Martyr", "Peacemaker", "Rebel", "Underdog", "Virtuous",  # protagonist (0-5)
        "Instigator", "Conspirator", "Tyrant", "Foreign Adversary",
        "Traitor", "Spy", "Saboteur", "Corrupt", "Incompetent",
        "Terrorist", "Deceiver", "Bigot",  # antagonist (6-17)
        "Forgotten", "Exploited", "Victim", "Scapegoat"  # innocent (18-21)
    ]

    unique_predictions = {}

    with open(output_file, "w") as f:
        with torch.no_grad():
            for article_id, text, entity_mention, start_offset, end_offset in test_data:
                if not isinstance(text, str) or len(text.strip()) == 0:
                    print(f"Skipping empty or invalid text for article_id {article_id}")
                    continue

                text_length = len(text)
                window_start = 0
                window_end = max_length

                while window_start < text_length:
                    window_text = text[window_start:window_end]

                    encoding = tokenizer(
                        window_text,
                        padding="max_length",
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                        return_offsets_mapping=True
                    )

                    input_ids = encoding["input_ids"].to(device)
                    attention_mask = encoding["attention_mask"].to(device)
                    token_type_ids = encoding["token_type_ids"].to(device)
                    offsets_mapping = encoding["offset_mapping"][0].cpu().numpy()

                    # adjust start and end offsets
                    adjusted_start_offset = start_offset - window_start
                    adjusted_end_offset = end_offset - window_start

                    token_start = None
                    token_end = None

                    for idx, (start, end) in enumerate(offsets_mapping):
                        if start <= adjusted_start_offset < end:
                            token_start = idx
                        if start < adjusted_end_offset <= end:
                            token_end = idx

                    if token_start is not None and token_end is not None:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            entity_start_positions=torch.tensor([[token_start]], dtype=torch.long).to(device),
                            entity_end_positions=torch.tensor([[token_end]], dtype=torch.long).to(device)
                        )

                        main_role_pred = torch.argmax(outputs["main_role_logits"], dim=-1).item()
                        main_role = ["Protagonist", "Antagonist", "Innocent"][main_role_pred]

                        fine_logits = outputs["fine_logits"].squeeze(0).cpu()
                        print(fine_logits)

                        topk_results = torch.topk(fine_logits, min(2, len(fine_logits)))
                        fine_roles = [
                            idx 
                            for idx, score in zip(topk_results.indices.tolist(), topk_results.values.tolist())
                            if score > threshold
                        ]

                        fine_roles_str = "\t".join([fine_role_names[idx] for idx in fine_roles])

                        unique_predictions[(article_id, entity_mention, start_offset, end_offset)] = (main_role, fine_roles_str)

                    window_start += max_length - overlap
                    window_end = min(window_start + max_length, text_length)

        for (article_id, entity_mention, start_offset, end_offset), (main_role, fine_roles_str) in unique_predictions.items():
            f.write(f"{article_id}\t{entity_mention}\t{start_offset}\t{end_offset}\t{main_role}\t{fine_roles_str}\n")




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

