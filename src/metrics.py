import numpy as np

def calculate_acc(y_true_main, y_pred_main, y_true_other, y_pred_other):
    total_acc = 0
    for i in range(len(y_true_main)):
        primary_acc = 0.5 if y_pred_main[i] == y_true_main[i] else 0.0
        true_indices = set(np.where(y_true_other[i] == 1)[0])
        pred_indices = set(np.where(y_pred_other[i] == 1)[0])
        additional_acc = 0.5 * len(true_indices & pred_indices) / len(true_indices) if len(true_indices) > 0 else 0.0
        total_acc += primary_acc + additional_acc
    return total_acc / len(y_true_main)
