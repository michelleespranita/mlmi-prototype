import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

def train(model, train_dataloader, val_dataloader, device, config):
    history = {"val_acc": [], "val_auc": []}

    loss_criterion = nn.BinaryCrossEntropy()
    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    model.train()

    train_loss_running = 0.

    for epoch in range(config["max_epochs"]):
        for i, batch in enumerate(train_dataloader):
            pred = model(batch["image"], batch["table"])
            loss = loss_criterion(pred, batch["label"])
            
            loss.backward()
            optimizer.step()

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.
            
            # ----- VALIDATION -----
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                model.eval()

                loss_total_val = 0
                total, correct = 0, 0
                roc_auc_vals = []

                for batch_val in val_dataloader:
                    with torch.no_grad():
                        pred = model(batch_val["image"], batch_val["table"]) # Shape: (b, 1)

                    predicted_label = (pred>0.5).float()

                    total += predicted_label.shape[0]
                    correct += (predicted_label == batch_val['label']).sum().item()

                    loss_total_val += loss_criterion(pred, batch_val['label'])

                    roc_auc_val = roc_auc_score(batch_val['label'], pred)
                    roc_auc_vals.append(roc_auc_val)

                accuracy = 100 * correct / total
                roc_auc_val = torch.mean(roc_auc_vals).item()

                history["val_acc"].append(accuracy)
                history["val_auc"].append(roc_auc_val)

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(val_dataloader):.3f}, val_accuracy: {accuracy:.3f}%, val_roc_auc: {roc_auc_val}')

                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'checkpoints/{config["experiment_name"]}')
                    best_accuracy = accuracy

                model.train()

    history["val_acc"] = torch.mean(history["val_acc"]).item()
    history["val_auc"] = torch.mean(history["val_auc"]).item()

    return history
