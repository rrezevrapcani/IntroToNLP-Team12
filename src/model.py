import torch
import torch.nn as nn
from transformers import BertModel

class EntityRoleClassifier(nn.Module):
    def __init__(self, bert_model_name):
        super(EntityRoleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name, ignore_mismatched_sizes=True)

        self.hidden_dim = self.bert.config.hidden_size
        self.hidden_layer_dim = 256

        # main role classification head
        self.main_role_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_layer_dim, 3)
        )

        # fine-grained classification heads
        self.fine_grained_classifiers = {
            "protagonist": nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_layer_dim, 6),
                nn.Sigmoid()
            ),
            "antagonist": nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_layer_dim, 12),
                nn.Sigmoid()
            ),
            "innocent": nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_layer_dim, 4),
                nn.Sigmoid()
            )
        }

    def forward(self, input_ids, attention_mask, token_type_ids, entity_start_positions, entity_end_positions):
        # numerical stability check for input (needs for BERTimbau)
        if torch.isnan(input_ids).any():
            raise ValueError("NaN found in input_ids")

        # ensure fine-grained heads are on the same device
        for key in self.fine_grained_classifiers:
            self.fine_grained_classifiers[key] = self.fine_grained_classifiers[key].to(input_ids.device)
        
        # all for BERTimbau to work
        # BERT forward pass with numerical checks
        try:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = outputs.last_hidden_state
        except RuntimeError as e:
            print(f"BERT forward pass error: {e}")
            raise

        # numerical stability for sequence output (bertimbau)
        if torch.isnan(sequence_output).any():
            sequence_output = torch.nan_to_num(sequence_output, nan=0.0)

        batch_size = input_ids.size(0)
        entity_embeddings = []

        # entity embedding extraction with numerical checks (bertimbau)
        for i in range(batch_size):
            for start_pos, end_pos in zip(entity_start_positions[i], entity_end_positions[i]):
                if start_pos == -1 or end_pos == -1:
                    continue
                start_pos = start_pos.item()
                end_pos = end_pos.item()

                # extract entity embeddings with context
                entity_span = sequence_output[i, start_pos:end_pos + 1]
                context_left = sequence_output[i, :start_pos].mean(dim=0) if start_pos > 0 else torch.zeros(self.hidden_dim, device=input_ids.device)
                context_right = sequence_output[i, end_pos + 1:].mean(dim=0) if end_pos + 1 < sequence_output.size(1) else torch.zeros(self.hidden_dim, device=input_ids.device)
                
                # numerical stability for embeddings
                entity_emb = torch.nan_to_num(
                    torch.cat([context_left, entity_span.mean(dim=0), context_right], dim=-1), 
                    nan=0.0
                )
                entity_embeddings.append(entity_emb)

        # numerical check for entity embeddings (bertimbau)
        entity_embeddings = torch.stack(entity_embeddings)
        if torch.isnan(entity_embeddings).any():
            entity_embeddings = torch.nan_to_num(entity_embeddings, nan=0.0)

        # main role classification (bertimbau)
        main_role_logits = self.main_role_classifier(entity_embeddings)
        main_role_probs = torch.softmax(main_role_logits, dim=-1)
        main_role_pred = torch.argmax(main_role_probs, dim=-1)

        # fine-grained classification
        fine_logits = []
        for idx, role in enumerate(main_role_pred):
            fine_logits.append(torch.zeros(22, device=input_ids.device))
            role_key = ["protagonist", "antagonist", "innocent"][role]
            logits = self.fine_grained_classifiers[role_key](entity_embeddings[idx])
            
            if role_key == "protagonist":
                fine_logits[idx][:6] = logits
            elif role_key == "antagonist":
                fine_logits[idx][6:18] = logits
            else:
                fine_logits[idx][18:] = logits

        fine_logits = torch.stack(fine_logits)

        return {
            "main_role_logits": main_role_logits,
            "main_role_probs": main_role_probs,
            "fine_logits": fine_logits,
        }