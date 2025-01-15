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

        # head for classifying main roles and identifying entities
        self.ner_head = nn.Linear(self.bert.config.hidden_size, num_main_classes)

        # heads for classifying secondary roles
        self.secondary_heads = nn.ModuleDict({
            "Protagonist": nn.Linear(self.bert.config.hidden_size, num_secondary_classes[0]),
            "Antagonist": nn.Linear(self.bert.config.hidden_size, num_secondary_classes[1]),
            "Innocent": nn.Linear(self.bert.config.hidden_size, num_secondary_classes[2]),
        })

    def forward(self, input_ids, attention_mask, token_type_ids, ner_labels=None, secondary_labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # for classify main role + identify entities
        pooled_output = outputs.pooler_output       # for classifying fine grained roles

        ner_logits = self.ner_head(sequence_output)
        secondary_logits = {
            role: head(pooled_output) for role, head in self.secondary_heads.items()
        }

        loss = None
        if ner_labels is not None and secondary_labels is not None:
            # CROSS ENTROPY LOSS FOR NER? BEST CHOICE?
            loss_fct_ner = nn.CrossEntropyLoss()
            ner_loss = loss_fct_ner(ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1))
            # BINARY CROSS ENTROPY LOSS FOR SECONDARY ROLES? BEST CHOICE?
            loss_fct_secondary = nn.BCEWithLogitsLoss()
            secondary_loss = loss_fct_secondary(
                torch.cat(list(secondary_logits.values()), dim=1),
                secondary_labels.float(),
            )
            loss = ner_loss + secondary_loss

        return (loss, ner_logits, secondary_logits) if loss is not None else (ner_logits, secondary_logits)