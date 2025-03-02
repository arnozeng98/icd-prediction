import os
import math
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

# Clean text: convert None or NaN to empty string.
def clean_text(value):
    if value is None:
        return ""
    if isinstance(value, float):
        try:
            if math.isnan(value):
                return ""
        except Exception:
            return ""
    return str(value)

# Compute evaluation metrics from model predictions.
def compute_metrics(model, dataloader, device):
    model.eval()
    all_main_preds = []
    all_main_labels = []
    all_other_preds = []
    all_other_labels = []
    composite_scores = []
    with torch.no_grad():
        for batch in dataloader:
            main_labels = batch["main_label"].to(device)
            other_labels = batch["other_label"].to(device)
            main_logits, other_logits = model(batch)
            main_preds = torch.argmax(main_logits, dim=1)
            other_probs = torch.sigmoid(other_logits)
            other_preds = (other_probs > 0.5).float()
            all_main_preds.extend(main_preds.cpu().numpy())
            all_main_labels.extend(main_labels.cpu().numpy())
            all_other_preds.extend(other_preds.cpu().numpy())
            all_other_labels.extend(other_labels.cpu().numpy())
            batch_size = main_labels.size(0)
            for i in range(batch_size):
                main_acc = 1.0 if main_preds[i] == main_labels[i] else 0.0
                true_other = other_labels[i].cpu().numpy()
                pred_other = other_preds[i].cpu().numpy()
                ratio = (1.0 if true_other.sum() == 0 
                         else (true_other * pred_other).sum() / true_other.sum())
                composite_scores.append(0.5 * main_acc + 0.5 * ratio)
    main_acc_metric = accuracy_score(all_main_labels, all_main_preds)
    other_f1 = f1_score(np.array(all_other_labels), np.array(all_other_preds),
                         average='macro', zero_division=0)
    composite = np.mean(composite_scores)
    return main_acc_metric, other_f1, composite

# Get output checkpoint directory path.
def get_output_checkpoint_dir(base_dir, epoch):
    return os.path.join(base_dir, "output_checkpoints", f"checkpoint-{epoch}")
