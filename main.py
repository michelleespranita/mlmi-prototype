import torch
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import numpy as np

from dataset import ImageTableDataset
from model import MultimodalModel
from train import train

# ----- Dataset -----
print("Loading datasets...")

batch_size = 2

train_dataset = ImageTableDataset("train_morbidity")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ImageTableDataset("val_morbidity")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ImageTableDataset("test_morbidity")
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# ----- Training -----
print("Start training...")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
config = {
    "learning_rate": 0.001,
    "max_epochs": 50,
    "print_every_n": 10,
    "validate_every_n": 10,
    "experiment_name": "baseline"
}

model = MultimodalModel()
history = train(model, train_dataloader, val_dataloader, device, config)
print(history)