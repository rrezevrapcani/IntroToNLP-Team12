import torch
import torch.nn as nn
from transformers import BertModel
import numpy as np

class EntityRoleClassifier(nn.Module):
    """
    Entity Role Classifier model based on BERT.
    :input: input_ids: Input token IDs
    :input: attention_mask: Attention mask
    :input: entity_start_positions: List of entity start positions
    :output: main_role_logits: results of main role classification without softmax
    :output: main_role_probs: results of main role classification with softmax
    :output: fine_logits: results of fine-grained classification without sigmoid, depends on the classification of main role
    """
    def __init__(self, bert_model_name):
        super(EntityRoleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        self.hidden_dim = self.bert.config.hidden_size
        self.hidden_layer_dim = 512
        
        # main role classification head
        self.main_role_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_layer_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_layer_dim, 3) # number of main roles
        )
        
        # fine-grained classification heads
        self.fine_grained_classifiers = {
            "protagonist": nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_layer_dim, 6), # number of protagonist fine-grained roles
                nn.Sigmoid() # sigmoid activation for multi-label classification
            ),
            "antagonist": nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_layer_dim, 12),  # number of antagonist fine-grained roles
                nn.Sigmoid()
            ),
            "innocent": nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_layer_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_layer_dim, 4),  # number of innocent fine-grained roles
                nn.Sigmoid()
            )
        }

    def forward(self, input_ids, attention_mask, entity_start_positions, entity_end_positions):
        #making sure fine-grained heads are on the same device as the input
        for key in self.fine_grained_classifiers:
            self.fine_grained_classifiers[key] = self.fine_grained_classifiers[key].to(input_ids.device)

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state

        batch_size = input_ids.size(0)
        entity_embeddings = []
        #get all entities of the batch
        for i in range(batch_size):
            for start_pos, end_pos in zip(entity_start_positions[i], entity_end_positions[i]):
                if start_pos == -1 or end_pos == -1:
                    continue
                start_pos = start_pos.item()
                end_pos = end_pos.item()

                # extract entity embeddings
                entity_span = sequence_output[i, start_pos:end_pos + 1]
                entity_emb = torch.cat([entity_span[0], entity_span[-1]], dim=-1)
                entity_embeddings.append(entity_emb)

        entity_embeddings = torch.stack(entity_embeddings)

        # main role classification, apply softmax to get probabilities and argmax to get the predicted class because the fine-grained classification depends on the main role
        main_role_logits = self.main_role_classifier(entity_embeddings)
        main_role_probs = torch.softmax(main_role_logits, dim=-1)
        main_role_pred = torch.argmax(main_role_probs, dim=-1)

        #fine-grained classification, apply sigmoid to get probabilities
        fine_logits = []
        for idx, role in enumerate(main_role_pred):
            fine_logits.append(np.zeros(22))
            role_key = ["protagonist", "antagonist", "innocent"][role]
            logits = self.fine_grained_classifiers[role_key](entity_embeddings[idx])
            logits = logits.cpu().detach().numpy()  
            if role_key == "protagonist":
                fine_logits[idx][:6] = logits
            elif role_key == "antagonist":
                fine_logits[idx][6:18] = logits
            else:
                fine_logits[idx][18:] = logits

        fine_logits = np.array(fine_logits)

        #convert them to tensors
        fine_logits = torch.tensor(fine_logits, device=input_ids.device)
        
        return {
            "main_role_logits": main_role_logits,
            "main_role_probs": main_role_probs,
            "fine_logits": fine_logits,
        }