import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from src.bert_utils import compute_bert_embeddings, bert_tokenizer, bert_model_embed, device

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
