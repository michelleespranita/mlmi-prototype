import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from einops import rearrange, repeat

class FusionModule(nn.Module):
    def __init__(self, dim=128, num_classes=1):
        super(FusionModule, self).__init__()
        configuration = BertConfig(num_hidden_layers=1, hidden_size=dim, num_attention_heads=8, output_attentions=True)
        self.fusion_transformer = BertModel(config=configuration)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # Multiplied by 2 for image and table modalities

    
    def forward(self, image_table_embeddings, position_ids, token_type_ids):
        '''
        image_table_embeddings: image + table token embeddings
        '''
        batch_size = image_table_embeddings.shape[0]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        image_table_embeddings = torch.cat((cls_tokens, image_table_embeddings), dim=1)

        # Add extra token_type_id for [CLS]
        new_token = torch.Tensor([torch.max(token_type_ids.unique()).item() + 1]).broadcast_to((batch_size, 1))
        token_type_ids = torch.cat([token_type_ids, new_token], dim=1).to(torch.int64)

        # Add extra position_id for [CLS]
        new_pos_token = torch.Tensor([0]).broadcast_to((batch_size, 1))
        position_ids = torch.cat([new_pos_token, position_ids], dim=1).to(torch.int64)

        trf_output = self.fusion_transformer(inputs_embeds=image_table_embeddings, position_ids=position_ids, token_type_ids=token_type_ids).last_hidden_state
        cls_embedding = trf_output[:, 0] # Shape: (b, dim)?

        mlp_output = self.mlp(cls_embedding)
        pred = torch.sigmoid(mlp_output)

        return pred