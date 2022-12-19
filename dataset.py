import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from image_data_branch.get_top_pCTs import get_trained_CT_images_model, get_images_paths, create_csv_with_top_pCTs
from image_data_branch.get_image_embeddings import get_top_pCTs_paths, save_image_embeddings

class ImageTableDataset(Dataset):
    def __init__(self, split, image_filename="image_data_branch/image_embeddings_morbidity_test.npy", table_filename="tabular_data_branch/tabular_embeddings_morbidity_end.npy",
                label_filename="tabular_data_branch/matt_metadata_norm_morbidity.csv"):

        # ----- Image data -----
        full_image_embeds = torch.from_numpy(np.load(image_filename)) # (num_patients, 10000)

        # ----- Tabular data -----
        full_tabular_embeds = torch.from_numpy(np.load(table_filename))

        # ----- Label -----
        labels = list(pd.read_csv(label_filename, sep='\t')["Morbidity"])

        if split == "train_morbidity":
            with open("train_idx_morbidity.csv", "r") as f:
                train_idx = [int(idx.rstrip()) for idx in f]
                self.image_embeds = full_image_embeds[train_idx]
                self.tabular_embeds = full_tabular_embeds[train_idx]
                self.labels = [labels[idx] for idx in train_idx]
        elif split == "val_morbidity":
            with open("val_idx_morbidity.csv", "r") as f:
                val_idx = [int(idx.rstrip()) for idx in f]
                self.image_embeds = full_image_embeds[val_idx]
                self.tabular_embeds = full_tabular_embeds[val_idx]
                self.labels = [labels[idx] for idx in val_idx]
        elif split == "test_morbidity":
            with open("test_idx_morbidity.csv", "r") as f:
                test_idx = [int(idx.rstrip()) for idx in f]
                self.image_embeds = full_image_embeds[test_idx]
                self.tabular_embeds = full_tabular_embeds[test_idx]
                self.labels = [labels[idx] for idx in test_idx]

        # ----- Tabular data -----
        # full_tab_df = pd.read_csv(table_filename, sep='\t')
        # full_tab_df.drop(["Patient"], axis=1, inplace=True)
        # full_tab_df["addition"] = [0.5 for i in range(len(full_tab_df))] # Append extra column filled with 0.5s

        # cat_features = ["Gender", "Underlying diseases"]
        # cont_features = [col for col in list(full_tab_df.columns) if col not in cat_features+["Patient"]]

        # if split == "train_mortality":
        #     with open("train_idx_mortality.csv", "r") as f:
        #         train_idx = [int(idx.rstrip()) for idx in f]
        #         self.image_embeds = full_image_embeds[train_idx]
        #         self.tab_df = full_tab_df.iloc[train_idx]
        #         # self.x_cat = torch.Tensor(self.tab_df[cat_features].values).to(torch.int64)
        #         # self.x_cont = torch.Tensor(self.tab_df[cont_features].values)
        #         self.labels = list(self.tab_df["Mortality"])
        #         self.tab_df.drop(["Mortality"], axis=1, inplace=True)
        # elif split == "val_mortality":
        #     with open("val_idx_mortality.csv", "r") as f:
        #         test_idx = [int(idx.rstrip()) for idx in f]
        #         self.image_embeds = full_image_embeds[test_idx]
        #         self.tab_df = full_tab_df.iloc[test_idx]
        #         self.labels = list(self.tab_df["Mortality"])
        #         self.tab_df.drop(["Mortality"], axis=1, inplace=True)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, i):
        return {
            "image": self.image_embeds[i],
            "table": self.tabular_embeds[i],
            "label": self.labels[i]
        }
        # return {
        #     "image": self.image_embeds[i],
        #     "table": torch.Tensor(self.tab_df.iloc[i]),
        #     # "table": [self.x_cat[i], self.x_cont[i]],
        #     "label": self.labels[i]
        # }