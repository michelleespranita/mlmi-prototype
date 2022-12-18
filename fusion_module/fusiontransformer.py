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

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    
    def forward(self, embeddings, position_ids, token_type_ids):
        '''
        embeddings: image + table token embeddings
        '''
        batch_size = embeddings.shape[0]

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = batch_size)
        embeddings = torch.cat((cls_tokens, embeddings), dim = 1)

        trf_output = self.fusion_transformer(inputs_embeds=embeddings, position_ids=position_ids, token_type_ids=token_type_ids)
        clf_embedding = trf_output[:, 0] # Shape: (b, dim)?
        print("clf embedding shape:", clf_embedding.shape)

        mlp_output = self.mlp(clf_embedding)
        pred = F.sigmoid(mlp_output)

        return pred