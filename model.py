import torch
import torch.nn as nn

from image_data_branch.get_image_embeddings import get_trained_CT_Morbidity_model, remove_linear_layers_from_trained_model
from tabular_data_branch.dnn import get_tabular_dnn
from tabular_data_branch.fttransformer import FTTransformer
from fusion_module.fusiontransformer import FusionModule

class MultimodalModel(nn.Module):
    def __init__(self, dim=128):
        super(MultimodalModel, self).__init__()

        # Image branch
        # pretrained_image_model = get_trained_CT_Morbidity_model()
        # self.cnn_image_branch = remove_linear_layers_from_trained_model(pretrained_image_model)
        self.mlp_image_branch = nn.Sequential(
            nn.Linear(10000, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Table branch
        self.dnn_table_branch = get_tabular_dnn()
        self.mlp_table_branch = nn.Sequential(
            nn.Linear(16, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        # self.table_branch = FTTransformer(
        #     categories = (2, 2),      # Gender and Udis
        #     num_continuous = 125,     # number of continuous values
        #     dim = dim,                # dimension of transformer input and output, paper set at 32
        #     dim_out = 1,              # dimension of MLP output (ignored here)
        #     depth = 1,                # depth, paper recommended 6
        #     heads = 8,                # heads, paper recommends 8
        #     attn_dropout = 0.1,       # post-attention dropout
        #     ff_dropout = 0.1          # feed forward dropout
        # )

        # Fusion Module
        self.fusion_module = FusionModule(dim=dim, num_classes=1)

        # Freeze pre-trained models
        # for param in self.cnn_image_branch.parameters():
        #     param.requires_grad = False
        for param in self.dnn_table_branch.parameters():
            param.requires_grad = False

    
    def forward(self, x_image, x_table):
        '''
        x_image: from image_embeddings.npy
        x_table: [x_cat, x_cont]
        '''
        batch_size = x_image.shape[0]
        num_img_tokens = x_image.shape[1]
        num_table_tokens = x_table.shape[1]

        out_image = self.mlp_image_branch(x_image)
        # out_image = self.mlp_image_branch(self.cnn_image_branch(x_image)) # Shape: (b, dim=128)
        out_table = self.mlp_table_branch(self.dnn_table_branch(torch.cat(x_table, dim=0).squeeze())) # Shape: (b, 16)
        # out_table = self.table_branch(x_table[0], x_table[1]) # Shape: (b, num_table_features, dim=128) # FTTransformer

        image_table_embeddings = torch.cat((out_image, out_table), dim=1)
        token_type_ids = torch.tensor(torch.cat([torch.zeros(batch_size, num_img_tokens), torch.ones(batch_size,num_table_tokens)], axis=-1), dtype=torch.int32)
        position_ids = torch.tensor([list(range(num_img_tokens)) + list(range(num_table_tokens)) for i in range(batch_size)], dtype=torch.int32)

        pred = self.fusion_module(image_table_embeddings, token_type_ids, position_ids)

        return pred
