import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef

def train(model, train_dataloader, val_dataloader, device, config):
    train_logger = SummaryWriter()

    history = {"val_acc": [], "val_auc": [], "val_sn": [], "val_sp": [], "val_ppv": [], "val_npv": [], "val_mcc": []}

    loss_criterion = nn.BCELoss()
    loss_criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = config["learning_rate"])

    model.train()

    train_loss_running = 0.
    best_roc_auc = 0.

    for epoch in range(config["max_epochs"]):
        for i, batch in enumerate(train_dataloader):
            pred = model(batch["image"], batch["table"])
            loss = loss_criterion(pred, batch["label"].to(torch.float32))
            
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

                    predicted_label = (pred>0.5).float()

                    val_gt_labels.append(batch_val['label'].to(torch.float32))
                    val_pred_labels.append(predicted_label)

                    total += predicted_label.shape[0]
                    correct += (predicted_label == batch_val['label']).sum().item()

                    loss_total_val += loss_criterion(pred, batch_val['label'].to(torch.float32))

                # Calculate metrics
                accuracy = 100 * correct / total
                roc_auc = roc_auc_score(torch.cat(val_gt_labels), torch.cat(val_pred_labels))
                tn, fp, fn, tp = confusion_matrix(torch.cat(val_gt_labels), torch.cat(val_pred_labels)).ravel()
                sn = tp / (tp + fn)
                sp = tn / (tn + fp)
                ppv = tp / (tp + fp)
                npv = tn / (tn + fn)
                mcc = matthews_corrcoef(torch.cat(val_gt_labels), torch.cat(val_pred_labels))

                history["val_acc"].append(accuracy)
                history["val_auc"].append(roc_auc)
                history["val_sn"].append(sn)
                history["val_sp"].append(sp)
                history["val_ppv"].append(ppv)
                history["val_npv"].append(npv)
                history["val_mcc"].append(mcc)

                train_logger.add_scalar("val_acc", accuracy, iteration)
                train_logger.add_scalar("val_auc", roc_auc, iteration)
                train_logger.add_scalar("val_sn", sn, iteration)
                train_logger.add_scalar("val_sp", sp, iteration)
                train_logger.add_scalar("val_ppv", ppv, iteration)
                train_logger.add_scalar("val_npv", npv, iteration)
                train_logger.add_scalar("val_mcc", mcc, iteration)

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val / len(val_dataloader):.3f}, val_accuracy: {accuracy:.3f}%, val_roc_auc: {roc_auc}')

                if roc_auc > best_roc_auc:
                    torch.save(model.state_dict(), f'checkpoints/{config["experiment_name"]}')
                    best_roc_auc = roc_auc

                model.train()

    history["val_acc"] = torch.mean(torch.Tensor(history["val_acc"])).item()
    history["val_auc"] = torch.mean(torch.Tensor(history["val_auc"])).item()
    history["val_sn"] = torch.mean(torch.Tensor(history["val_sn"])).item()
    history["val_sp"] = torch.mean(torch.Tensor(history["val_sp"])).item()
    history["val_ppv"] = torch.mean(torch.Tensor(history["val_ppv"])).item()
    history["val_npv"] = torch.mean(torch.Tensor(history["val_npv"])).item()
    history["val_mcc"] = torch.mean(torch.Tensor(history["val_mcc"])).item()

    train_logger.close()

    return history
