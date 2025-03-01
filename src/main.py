import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.data_processing import load_and_preprocess_data, extract_features
from src.dl_models import train_bert_model
from src.bert_utils import compute_bert_embeddings, bert_tokenizer, bert_model_embed, device
from src.utils import read_data, write_predictions, calculate_acc
from src.config import TRAIN_FILE_PATH, TEST_FILE_PATH, OUTPUT_FILE_PATH
from src.ml_models import train_stacking_model

# Load training data
(X_train_text_main, X_val_text_main, y_train_main, y_val_main,
 X_train_text_other, X_val_text_other, y_train_other, y_val_other, mlb, label_encoder) = load_and_preprocess_data(TRAIN_FILE_PATH)

X_train_fused, X_val_fused, vectorizer, scaler_tfidf, scaler_bert = extract_features(X_train_text_main, X_val_text_main)

# Train Stacking model
stacking_clf = train_stacking_model(X_train_fused, y_train_main)
y_pred_main = stacking_clf.predict(X_val_fused)

# Format and print the best stacking model parameters
best_params = stacking_clf.get_params()
formatted_params = "\n".join([f"{key}: {value}" for key, value in best_params.items()])
print(f"Best Stacking model parameters:\n{formatted_params}")

# Calculate single-label prediction accuracy
main_accuracy = np.mean(y_pred_main == y_val_main)
print(f"Main diagnosis accuracy: {main_accuracy:.4f}")

num_classes_other = y_train_other.shape[1]
model, best_thresholds = train_bert_model(X_train_text_other, y_train_other, X_val_text_other, y_val_other, bert_tokenizer, num_classes_other, device)

# Calculate Acc score on validation set
y_pred_other_binary = np.zeros_like(y_val_other)  # Assuming y_pred_other_binary is already calculated
acc_score = calculate_acc(y_val_main, y_pred_main, y_val_other, y_pred_other_binary)
print(f"Final Acc Score on Validation: {acc_score:.4f}")

# Load test data
test_df = read_data(TEST_FILE_PATH)
print(f"test_df type: {type(test_df)}")  # Add debug information
X_test_text = test_df["combined_text"]

# Main diagnosis prediction: use previously trained TF-IDF and BERT feature fusion
X_test_tfidf = vectorizer.transform(X_test_text)
X_test_bert = compute_bert_embeddings(X_test_text.tolist(), bert_tokenizer, bert_model_embed, max_len=128, device=device)
X_test_tfidf_dense = X_test_tfidf.toarray()
X_test_fused = np.hstack([scaler_tfidf.transform(X_test_tfidf_dense), scaler_bert.transform(X_test_bert)])
y_test_main = stacking_clf.predict(X_test_fused)

# Convert integer labels back to original ICD codes
y_test_main = label_encoder.inverse_transform(y_test_main)

# Other diagnosis prediction: construct test set Dataset and DataLoader
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

# Define apply_thresholds function
def apply_thresholds(logits, thresholds):
    preds = np.zeros_like(logits, dtype=int)
    for i in range(logits.shape[1]):
        preds[:, i] = (logits[:, i] > thresholds[i]).astype(int)
    for i in range(preds.shape[0]):
        if preds[i].sum() == 0:
            preds[i][np.argmax(logits[i])] = 1
    return preds

y_test_pred_binary = apply_thresholds(y_test_pred, best_thresholds)
# Use mlb to convert binary results to ICD code list
y_test_other_codes = mlb.inverse_transform(y_test_pred_binary)

# Output prediction results
write_predictions(test_df, y_test_main, y_test_other_codes, OUTPUT_FILE_PATH)