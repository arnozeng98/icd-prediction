import json
import re
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from src.config import TRAIN_FILE_PATH
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.bert_utils import compute_bert_embeddings, bert_tokenizer, bert_model_embed, device

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
            combined.append(f"{field}: {text}")
    return " ".join(combined)

def load_and_preprocess_data(train_file_path=TRAIN_FILE_PATH):
    with open(train_file_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    for record in train_data:
        record["combined_text"] = combine_fields(record)
    train_df = pd.DataFrame(train_data)
    y_true_main = [record["主要诊断编码"] for record in train_data]
    y_true_other = [record["其他诊断编码"].split(";") if record.get("其他诊断编码") else [] for record in train_data]
    mlb = MultiLabelBinarizer()
    y_additional = mlb.fit_transform(y_true_other)
    X_text = train_df["combined_text"]
    
    # Convert main diagnosis labels to integer labels
    label_encoder = LabelEncoder()
    y_true_main_encoded = label_encoder.fit_transform(y_true_main)
    
    X_train_text_main, X_val_text_main, y_train_main, y_val_main = train_test_split(
        X_text, y_true_main_encoded, test_size=0.2, random_state=42
    )
    X_train_text_other, X_val_text_other, y_train_other, y_val_other = train_test_split(
        X_text, y_additional, test_size=0.2, random_state=42
    )
    
    # Check for NaN values and handle them
    X_train_text_main = X_train_text_main.fillna("")
    X_val_text_main = X_val_text_main.fillna("")
    X_train_text_other = X_train_text_other.fillna("")
    X_val_text_other = X_val_text_other.fillna("")
    
    return (X_train_text_main, X_val_text_main, y_train_main, y_val_main,
            X_train_text_other, X_val_text_other, y_train_other, y_val_other, mlb, label_encoder)

def extract_features(X_train_text, X_val_text):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_train_bert = compute_bert_embeddings(X_train_text.tolist(), bert_tokenizer, bert_model_embed, max_len=128, device=device)
    X_val_bert = compute_bert_embeddings(X_val_text.tolist(), bert_tokenizer, bert_model_embed, max_len=128, device=device)
    scaler_tfidf = StandardScaler(with_mean=False)
    X_train_tfidf_scaled = scaler_tfidf.fit_transform(X_train_tfidf.toarray())
    X_val_tfidf_scaled = scaler_tfidf.transform(X_val_tfidf.toarray())
    scaler_bert = StandardScaler()
    X_train_bert_scaled = scaler_bert.fit_transform(X_train_bert)
    X_val_bert_scaled = scaler_bert.transform(X_val_bert)
    X_train_fused = np.hstack([X_train_tfidf_scaled, X_train_bert_scaled])
    X_val_fused = np.hstack([X_val_tfidf_scaled, X_val_bert_scaled])
    return X_train_fused, X_val_fused, vectorizer, scaler_tfidf, scaler_bert
