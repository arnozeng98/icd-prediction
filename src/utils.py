import json
import pandas as pd
from src.data_processing import combine_fields
from src.config import TEST_FILE_PATH, OUTPUT_FILE_PATH
import numpy as np

def read_data(file_path=TEST_FILE_PATH):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for record in data:
        record["combined_text"] = combine_fields(record)
    return pd.DataFrame(data)

def write_predictions(test_df, y_test_main, y_test_other_codes, output_file=OUTPUT_FILE_PATH):
    predictions = []
    for idx, record in test_df.iterrows():
        main_code = y_test_main[idx]
        other_codes = y_test_other_codes[idx]
        other_str = ";".join(other_codes) if other_codes else ""
        prediction_str = f"[{main_code}|{other_str}]"
        predictions.append({
            "病案标识": record["病案标识"],
            "预测结果": prediction_str
        })
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
    print(f"Test set prediction completed, results saved to {output_file}")

def calculate_acc(y_true_main, y_pred_main, y_true_other, y_pred_other):
    total_acc = 0
    for i in range(len(y_true_main)):
        primary_acc = 0.5 if y_pred_main[i] == y_true_main[i] else 0.0
        true_indices = set(np.where(y_true_other[i] == 1)[0])
        pred_indices = set(np.where(y_pred_other[i] == 1)[0])
        additional_acc = 0.5 * len(true_indices & pred_indices) / len(true_indices) if len(true_indices) > 0 else 0.0
        total_acc += primary_acc + additional_acc
    return total_acc / len(y_true_main)
