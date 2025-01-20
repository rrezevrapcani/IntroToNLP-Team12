import torch
import torch.nn as nn
from transformers import BertModel

'''
WHERE OUR FINE-TUNING MODEL IS DEFINED
'''

class EntityFramingModel(nn.Module):
    def __init__(self, model_name, num_main_classes, num_secondary_classes):
        super(EntityFramingModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

        # head for classifying main roles 
        self.main_head = nn.Linear(self.bert.config.hidden_size, num_main_classes)

        # heads for classifying fine-grained roles
        self.fine_grained_heads = nn.ModuleDict({
            "Protagonist": nn.Linear(self.bert.config.hidden_size, num_secondary_classes[0]),
            "Antagonist": nn.Linear(self.bert.config.hidden_size, num_secondary_classes[1]),
            "Innocent": nn.Linear(self.bert.config.hidden_size, num_secondary_classes[2]),
        })

    def forward(self, input_ids, attention_mask, token_type_ids, main_labels=None, fine_grained_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # for classify main role + identify entities
        pooled_output = outputs.pooler_output       # for classifying fine grained roles

        main_logits = self.main_head(sequence_output)
        fine_grained_logits = {
            role: head(pooled_output) for role, head in self.fine_grained_heads.items()
        }

        loss = None
        if main_labels is not None and fine_grained_labels is not None:
            # CROSS ENTROPY LOSS FOR NER? BEST CHOICE?
            loss_fct_ner = nn.CrossEntropyLoss()
            main_loss = loss_fct_ner(main_logits.view(-1, main_logits.size(-1)), main_labels.view(-1))
            # BINARY CROSS ENTROPY LOSS FOR FINE-GRAINED ROLES? BEST CHOICE?
            loss_fct_fine_grained = nn.BCEWithLogitsLoss()
            fine_grained_loss = loss_fct_fine_grained(
                torch.cat(list(fine_grained_logits.values()), dim=1),
                fine_grained_labels.float(),
            )
            loss = main_loss + fine_grained_loss

        return (loss, main_logits, fine_grained_logits) if loss is not None else (main_logits, fine_grained_logits)