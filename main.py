import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import numpy as np

from dataset import ImageTableDataset
from model import MultimodalModel
from train import train

# ----- Dataset -----
batch_size = 16

train_dataset = ImageTableDataset("train")
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageTableDataset("val")
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

dataset = ConcatDataset([train_dataset, val_dataset])

k = 10
splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# ----- Model -----
model = MultimodalModel()

# ----- Training -----
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
config = {
    "learning_rate": 0.001,
    "max_epochs": 50,
    "print_every_n": 10,
    "validate_every_n": 10,
    "experiment_name": "baseline"
}

results = {"val_acc": [], "val_auc": []}
for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    history = train(model, train_dataloader, val_dataloader, device, config)
    results["val_acc"].append(history["val_acc"])
    results["val_auc"].append(history["val_auc"])

print("Validation accuracy:", torch.mean(results["val_acc"]).item())
print("Validation AUC:", torch.mean(results["val_auc"]).item())