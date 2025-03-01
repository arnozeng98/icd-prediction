import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import numpy as np
from src.bert_utils import bert_tokenizer

class RoBertaICDClassifier(nn.Module):
    def __init__(self, num_classes_other, dropout_rate=0.3):
        super(RoBertaICDClassifier, self).__init__()
        self.roberta = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(self.roberta.config.hidden_size, num_classes_other)
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

class ICDDataset(Dataset):
    def __init__(self, texts, labels_other, tokenizer, max_len=128):
        self.texts = texts
        self.labels_other = labels_other
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_other = self.labels_other[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label_other": torch.tensor(label_other, dtype=torch.float),
        }

def train_bert_model(X_train_text_other, y_train_other, X_val_text_other, y_val_other, bert_tokenizer, num_classes_other, device):
    pos_weight = (len(y_train_other) - np.sum(y_train_other, axis=0)) / (np.sum(y_train_other, axis=0) + 1e-9)
    pos_weight = np.clip(pos_weight, a_min=None, a_max=10.0)
    pos_weight_tensor = torch.FloatTensor(pos_weight).to(device)
    criterion_other = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    model = RoBertaICDClassifier(num_classes_other, dropout_rate=0.3).to(device)
    train_dataset = ICDDataset(X_train_text_other.tolist(), y_train_other, bert_tokenizer, max_len=128)
    val_dataset = ICDDataset(X_val_text_other.tolist(), y_val_other, bert_tokenizer, max_len=128)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    total_training_steps = len(train_loader) * 15
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_training_steps), num_training_steps=total_training_steps)
    num_epochs = 15
    freeze_epochs = 1
    for epoch in range(num_epochs):
        if epoch < freeze_epochs:
            for param in model.roberta.parameters():
                param.requires_grad = False
        else:
            for param in model.roberta.parameters():
                param.requires_grad = True
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label_other"].to(device)
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion_other(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label_other"].to(device)
                logits = model(input_ids, attention_mask)
                val_preds.append(logits.cpu().numpy())
                val_labels.append(labels.cpu().numpy())
        val_preds = np.concatenate(val_preds, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        best_thresholds = []
        for i in range(num_classes_other):
            best_f1 = 0
            best_thr = 0.5
            for thr in np.arange(0.1, 0.91, 0.05):
                preds = (val_preds[:, i] > thr).astype(int)
                f1 = f1_score(val_labels[:, i], preds, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr
            best_thresholds.append(best_thr)
        best_thresholds = np.array(best_thresholds)
        val_preds_binary = np.zeros_like(val_preds, dtype=int)
        for i in range(num_classes_other):
            val_preds_binary[:, i] = (val_preds[:, i] > best_thresholds[i]).astype(int)
        for i in range(val_preds_binary.shape[0]):
            if val_preds_binary[i].sum() == 0:
                val_preds_binary[i][np.argmax(val_preds[i])] = 1
        macro_f1 = f1_score(val_labels, val_preds_binary, average="macro", zero_division=0)
        print(f"Epoch {epoch+1} Validation Macro F1: {macro_f1:.4f}")
    return model, best_thresholds
