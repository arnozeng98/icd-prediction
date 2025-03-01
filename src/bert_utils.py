import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_embed = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext").to(device)
bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        return encoding["input_ids"].squeeze(), encoding["attention_mask"].squeeze()

def compute_bert_embeddings(texts, tokenizer, model, max_len=128, batch_size=16, device=device):
    model.eval()
    dataset = TextDataset(texts, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    embeddings = []
    with torch.no_grad():
        for input_ids, attention_mask in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_emb = outputs.pooler_output.cpu().numpy()
            embeddings.append(batch_emb)
    return np.concatenate(embeddings, axis=0)
