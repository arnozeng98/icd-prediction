import os
import math
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

# ------------------------------------------------------------------------
# Function: clean_text
# ------------------------------------------------------------------------
# Purpose:
#   This function cleans the input text by converting None values or
#   NaN (Not a Number) values to an empty string.
#
# Parameters:
#   value (any): The input value which may be None, NaN, or a valid text.
#
# Returns:
#   str: A valid string representation of the input.
#
# Implementation details:
#   - First, check if the input value is None.
#   - Check if the input is of type float and if it is NaN using math.isnan.
#   - In cases of exceptions during NaN-check, return an empty string.
#   - Otherwise, convert the input to a string.
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

# ------------------------------------------------------------------------
# Function: compute_metrics
# ------------------------------------------------------------------------
# Purpose:
#   Compute evaluation metrics based on model predictions vs. ground truth.
#
# Parameters:
#   model (torch.nn.Module): The neural network model used for prediction.
#   dataloader (torch.utils.data.DataLoader): DataLoader containing validation/test data.
#   device (torch.device): Device to perform computations on.
#
# Returns:
#   tuple: (main_acc_metric, other_f1, composite) where:
#     - main_acc_metric: Accuracy of primary diagnosis predictions.
#     - other_f1: Macro F1-score for secondary diagnosis prediction.
#     - composite: A combined metric averaging primary and secondary performance.
#
# Implementation details:
#   - Set the model to evaluation mode and disable gradient calculations.
#   - For each batch:
#       • Move labels to device.
#       • Get predictions from the model.
#       • Compute predictions for primary diagnosis using argmax.
#       • Compute predictions for secondary diagnosis by thresholding sigmoid outputs at 0.5.
#       • Save predictions and ground truth for overall metric calculation.
#       • For each sample in the batch, compute a composite score:
#         composite = 0.5 * (1 if prediction correct else 0) + 0.5 * (overlap ratio for secondary diagnosis)
#   - Calculate accuracy and macro F1-score.
def compute_metrics(model, dataloader, device):
    model.eval()
    all_main_preds = []
    all_main_labels = []
    all_other_preds = []
    all_other_labels = []
    composite_scores = []
    with torch.no_grad():
        for batch in dataloader:
            # Move ground truth labels to the specified device.
            main_labels = batch["main_label"].to(device)
            other_labels = batch["other_label"].to(device)
            # Forward pass: generate predictions for both tasks.
            main_logits, other_logits = model(batch)
            # Compute primary (main) predictions: index with maximum logit.
            main_preds = torch.argmax(main_logits, dim=1)
            # Compute secondary (other) predictions: apply sigmoid and threshold at 0.5.
            other_probs = torch.sigmoid(other_logits)
            other_preds = (other_probs > 0.5).float()
            
            # Collect all predictions and labels for metric calculation.
            all_main_preds.extend(main_preds.cpu().numpy())
            all_main_labels.extend(main_labels.cpu().numpy())
            all_other_preds.extend(other_preds.cpu().numpy())
            all_other_labels.extend(other_labels.cpu().numpy())
            
            # For each sample in the batch, compute an individual composite metric.
            batch_size = main_labels.size(0)
            for i in range(batch_size):
                # Calculate primary accuracy for this sample.
                main_acc = 1.0 if main_preds[i] == main_labels[i] else 0.0
                # Retrieve the true and predicted secondary labels as numpy arrays.
                true_other = other_labels[i].cpu().numpy()
                pred_other = other_preds[i].cpu().numpy()
                # Calculate ratio: if no true secondary labels, set ratio to 1.
                ratio = (1.0 if true_other.sum() == 0 
                         else (true_other * pred_other).sum() / true_other.sum())
                # Composite score: average of primary and secondary task scores.
                composite_scores.append(0.5 * main_acc + 0.5 * ratio)
    
    # Calculate overall accuracy for primary diagnosis.
    main_acc_metric = accuracy_score(all_main_labels, all_main_preds)
    # Calculate macro F1-score for secondary diagnosis.
    other_f1 = f1_score(np.array(all_other_labels), np.array(all_other_preds),
                         average='macro', zero_division=0)
    # Average composite score over all samples.
    composite = np.mean(composite_scores)
    return main_acc_metric, other_f1, composite

# ------------------------------------------------------------------------
# Function: get_output_checkpoint_dir
# ------------------------------------------------------------------------
# Purpose:
#   Generate the directory path where a model checkpoint should be saved.
#
# Parameters:
#   base_dir (str): The root directory of the project.
#   epoch (int): The current epoch number.
#
# Returns:
#   str: A string path combining base_dir, "output_checkpoints", and the checkpoint folder name.
#
# Implementation details:
#   - Constructs the directory path using os.path.join.
def get_output_checkpoint_dir(base_dir, epoch):
    return os.path.join(base_dir, "output_checkpoints", f"checkpoint-{epoch}")
