import torch
import torch.nn as nn
from transformers import AutoModel

class SentenceModelWithLinearTransformation(nn.Module):
    def __init__(self, model_name, target_embedding_dim):
        super(SentenceModelWithLinearTransformation, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(model_name)
        text_hidden_size = self.text_encoder.config.hidden_size
        self.linear = nn.Linear(text_hidden_size, target_embedding_dim)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.mean_pooling(outputs, attention_mask)  # (batch_size, hidden_dim)
        transformed_output = self.linear(pooled_output)             # (batch_size, target_embedding_dim)
        return transformed_output
