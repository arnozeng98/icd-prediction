import json
import pandas as pd
from src.data_utils import combine_fields
from src.config import TEST_FILE_PATH, OUTPUT_FILE_PATH

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
    print(f"测试集预测完成，结果保存至 {output_file}")
