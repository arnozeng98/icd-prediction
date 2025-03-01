import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.data_utils import load_and_preprocess_data
from src.feature_extraction import extract_features
from src.model_training import train_bert_model
from src.svm_utils import train_svm
from src.bert_utils import compute_bert_embeddings, bert_tokenizer, bert_model_embed, device
from src.io_utils import read_data, write_predictions
from src.metrics import calculate_acc
from src.config import TRAIN_FILE_PATH, TEST_FILE_PATH, OUTPUT_FILE_PATH

# 读取训练集数据
(X_train_text_main, X_val_text_main, y_train_main, y_val_main,
 X_train_text_other, X_val_text_other, y_train_other, y_val_other, mlb) = load_and_preprocess_data(TRAIN_FILE_PATH)

X_train_fused, X_val_fused, vectorizer, scaler_tfidf, scaler_bert = extract_features(X_train_text_main, X_val_text_main)

best_svm = train_svm(X_train_fused, y_train_main)
y_pred_main = best_svm.predict(X_val_fused)
print("Best SVM parameters:", best_svm.get_params())

num_classes_other = y_train_other.shape[1]
model, best_thresholds = train_bert_model(X_train_text_other, y_train_other, X_val_text_other, y_val_other, bert_tokenizer, num_classes_other, device)

# 计算验证集上的Acc评分
y_pred_other_binary = np.zeros_like(y_val_other)  # 假设y_pred_other_binary已经计算出来
acc_score = calculate_acc(y_val_main, y_pred_main, y_val_other, y_pred_other_binary)
print(f"Final Acc Score on Validation: {acc_score:.4f}")

# 读取测试集数据
test_df = read_data(TEST_FILE_PATH)
print(f"test_df type: {type(test_df)}")  # 添加调试信息
X_test_text = test_df["combined_text"]

# 主诊断预测：使用之前训练好的 TF-IDF 与 BERT 特征融合
X_test_tfidf = vectorizer.transform(X_test_text)
X_test_bert = compute_bert_embeddings(X_test_text.tolist(), bert_tokenizer, bert_model_embed, max_len=128, device=device)
X_test_tfidf_dense = X_test_tfidf.toarray()
X_test_fused = np.hstack([scaler_tfidf.transform(X_test_tfidf_dense), scaler_bert.transform(X_test_bert)])
y_test_main = best_svm.predict(X_test_fused)

# 其他诊断预测：构造测试集 Dataset 与 DataLoader
class ICDDatasetTest(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }

test_dataset = ICDDatasetTest(X_test_text.tolist(), bert_tokenizer, max_len=128)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model.eval()
y_test_pred = []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        y_test_pred.append(logits.cpu().numpy())
y_test_pred = np.concatenate(y_test_pred, axis=0)

# 定义 apply_thresholds 函数
def apply_thresholds(logits, thresholds):
    preds = np.zeros_like(logits, dtype=int)
    for i in range(logits.shape[1]):
        preds[:, i] = (logits[:, i] > thresholds[i]).astype(int)
    for i in range(preds.shape[0]):
        if preds[i].sum() == 0:
            preds[i][np.argmax(logits[i])] = 1
    return preds

y_test_pred_binary = apply_thresholds(y_test_pred, best_thresholds)
# 利用 mlb 将二值结果转换为 ICD 编码列表
y_test_other_codes = mlb.inverse_transform(y_test_pred_binary)

# 输出预测结果
write_predictions(test_df, y_test_main, y_test_other_codes, OUTPUT_FILE_PATH)