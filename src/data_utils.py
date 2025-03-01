import json
import re
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from src.config import TRAIN_FILE_PATH

def combine_fields(record):
    fields = ["主诉", "现病史", "既往史", "个人史", "婚姻史", "家族史", 
              "入院情况", "入院诊断", "诊疗经过", "出院情况", "出院医嘱"]
    combined = []
    for field in fields:
        text = record.get(field, "")
        if not isinstance(text, str):
            text = ""
        if text:
            text = re.sub(r'\s+', ' ', text.strip())
            combined.append(text)
    return " ".join(combined)

def load_and_preprocess_data(train_file_path=TRAIN_FILE_PATH):
    with open(train_file_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    for record in train_data:
        record["combined_text"] = combine_fields(record)
    train_df = pd.DataFrame(train_data)
    y_true_main = [record["主要诊断编码"] for record in train_data]
    y_true_other = [record["其他诊断编码"].split(";") if record["其他诊断编码"] else [] for record in train_data]
    mlb = MultiLabelBinarizer()
    y_additional = mlb.fit_transform(y_true_other)
    X_text = train_df["combined_text"]
    X_train_text_main, X_val_text_main, y_train_main, y_val_main = train_test_split(
        X_text, y_true_main, test_size=0.2, random_state=42
    )
    X_train_text_other, X_val_text_other, y_train_other, y_val_other = train_test_split(
        X_text, y_additional, test_size=0.2, random_state=42
    )
    return (X_train_text_main, X_val_text_main, y_train_main, y_val_main,
            X_train_text_other, X_val_text_other, y_train_other, y_val_other, mlb)
