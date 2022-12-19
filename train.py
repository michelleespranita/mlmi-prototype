import math
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, recall_score, precision_score, roc_curve, auc

def train(model, train_dataloader, val_dataloader, device, config):
    train_logger = SummaryWriter()
    num_classes = 3

    metrics = ["val_auc", "val_sn", "val_sp"]
    history = {}
    for label in range(num_classes):
        for metric in metrics:
            history[f"{metric}_{label}"] = []
    history["val_acc"] = []

    loss_criterion = nn.CrossEntropyLoss()
    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    model.train()

    train_loss_running = 0.
    best_val_loss = math.inf

    for epoch in range(config["max_epochs"]):
        for i, batch in enumerate(train_dataloader):
            pred = model(batch["image"], batch["table"])
            if len(pred.shape) == 1:
                pred = pred[None, :]
            loss = loss_criterion(pred, batch["label"].long())
            
            loss.backward()
            optimizer.step()

            train_loss_running += loss.item()
            iteration = epoch * len(train_dataloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_logger.add_scalar("train_loss", train_loss_running / config["print_every_n"], iteration)
                train_loss_running = 0.
            
            # ----- VALIDATION -----
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):
                model.eval()

                loss_total_val = 0
                total, correct = 0, 0
                val_gt_labels = []
                val_pred_labels = []

                for batch_val in val_dataloader:
                    with torch.no_grad():
                        pred = model(batch_val["image"], batch_val["table"]) # Shape: (b, 1)
                        if len(pred.shape) == 1:
                            pred = pred[None, :]

                    predicted_label = torch.argmax(pred, dim=1)

                    val_gt_labels.append(batch_val['label'].to(torch.float32))
                    val_pred_labels.append(predicted_label)

                    total += predicted_label.shape[0]
                    correct += (predicted_label == batch_val['label']).sum().item()

                    loss_total_val += loss_criterion(pred, batch_val['label'].long())

                # Calculate metrics
                # Accuracy
                accuracy = 100 * correct / total
                history["val_acc"].append(accuracy)
                train_logger.add_scalar("val_acc", accuracy, iteration)

                # AUC, Recall, Precision for each class
                sn = recall_score(torch.cat(val_gt_labels), torch.cat(val_pred_labels), average=None)
                sp = precision_score(torch.cat(val_gt_labels), torch.cat(val_pred_labels), average=None)
                for label in range(num_classes):
                    fpr, tpr, thresholds = roc_curve(torch.cat(val_gt_labels), torch.cat(val_pred_labels), pos_label = label) 
                    auroc = round(auc(fpr, tpr), 2)
                    history[f"val_auc_{label}"].append(auroc)
                    history[f"val_sn_{label}"].append(sn[label])
                    history[f"val_sp_{label}"].append(sp[label])
                    train_logger.add_scalar(f"val_auc_{label}", auroc, iteration)
                    train_logger.add_scalar(f"val_sn_{label}", sn[label], iteration)
                    train_logger.add_scalar(f"val_sp_{label}", sp[label], iteration)

                val_loss = loss_total_val / len(val_dataloader)
                train_logger.add_scalar("val_loss", val_loss, iteration)
                print(f'[{epoch:03d}/{i:05d}] val_loss: {val_loss:.3f}, val_accuracy: {accuracy:.3f}%')
                
                if val_loss < best_val_loss:
                    os.makedirs(f'checkpoints/{config["experiment_name"]}', exist_ok=True)
                    torch.save(model.state_dict(), f'checkpoints/{config["experiment_name"]}/best_model.ckpt')
                    best_val_loss = val_loss

                model.train()

    train_logger.close()

    return history
