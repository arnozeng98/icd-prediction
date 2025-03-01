import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model_embed = AutoModel.from_pretrained("hfl/chinese-roberta-wwm-ext").to(device)
bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

def compute_bert_embeddings(texts, tokenizer, model, max_len=128, batch_size=16, device=device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            encoding = tokenizer(batch_texts, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            batch_emb = outputs.pooler_output.cpu().numpy()
            embeddings.append(batch_emb)
    return np.concatenate(embeddings, axis=0)
