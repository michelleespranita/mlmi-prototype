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

# dataset = ConcatDataset([train_dataset, val_dataset])
# labels = [dataset[i]['label'] for i in range(len(dataset))]

# k = 2
# splits = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


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

# results = {"val_acc": [], "val_auc": [], "val_sn": [], "val_sp": [], "val_ppv": [], "val_npv": [], "val_mcc": []}
model = MultimodalModel()
history = train(model, train_dataloader, val_dataloader, device, config)
print(history)

# for fold, (train_idx, val_idx) in enumerate(splits.split(dataset, labels)):
#     print('Fold {}'.format(fold + 1))

#     train_sampler = SubsetRandomSampler(train_idx)
#     val_sampler = SubsetRandomSampler(val_idx)
#     train_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
#     val_dataloader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

#     model = MultimodalModel()

#     history = train(model, train_dataloader, val_dataloader, device, config)
#     results["val_acc"].append(history["val_acc"])
#     results["val_auc"].append(history["val_auc"])
#     results["val_sn"].append(history["val_sn"])
#     results["val_sp"].append(history["val_sp"])
#     results["val_ppv"].append(history["val_ppv"])
#     results["val_npv"].append(history["val_npv"])
#     results["val_mcc"].append(history["val_mcc"])

# print("Validation accuracy:", torch.mean(torch.Tensor(results["val_acc"])).item())
# print("Validation AUC:", torch.mean(torch.Tensor(results["val_auc"])).item())
# print("Validation SN (Recall):", torch.mean(torch.Tensor(results["val_sn"])).item())
# print("Validation SP (Specificity):", torch.mean(torch.Tensor(results["val_sp"])).item())
# print("Validation PPV:", torch.mean(torch.Tensor(results["val_ppv"])).item())
# print("Validation NPV:", torch.mean(torch.Tensor(results["val_npv"])).item())
# print("Validation MCC:", torch.mean(torch.Tensor(results["val_mcc"])).item())