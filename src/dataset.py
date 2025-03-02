import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import clean_text
from config import MAX_LENGTH, OTHER_CODES, main2id, other2id

# ICDDataset loads JSON data and tokenizes text fields.
class ICDDataset(Dataset):
    def __init__(self, data_path, tokenizer: AutoTokenizer, max_length=MAX_LENGTH, is_test=False):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        text_fields = [
            "主诉", "现病史", "既往史", "个人史", "婚姻史", "家族史",
            "入院情况", "入院诊断", "诊疗经过", "出院情况", "出院医嘱"
        ]
        for item in raw:
            sample = {}
            for field in text_fields:
                txt = item.get(field, "")
                if txt is None or (isinstance(txt, float) and math.isnan(txt)):
                    txt = ""
                else:
                    txt = str(txt)
                tokenized = tokenizer(
                    txt,
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                sample[f"{field}_input_ids"] = tokenized.input_ids.squeeze(0)
                sample[f"{field}_attention_mask"] = tokenized.attention_mask.squeeze(0)
            sample["text"] = " ".join([clean_text(item.get(field, "")) for field in text_fields]).strip()
            if not is_test:
                main_label = item.get("主要诊断编码", "").strip()
                sample["main_label"] = main2id.get(main_label, -1)
                other_label = item.get("其他诊断编码", "")
                other_list = [code.strip() for code in other_label.split(';') if code.strip()]
                other_vec = np.zeros(len(OTHER_CODES), dtype=np.float32)
                for code in other_list:
                    if code in other2id:
                        other_vec[other2id[code]] = 1.0
                sample["other_label"] = torch.tensor(other_vec)
            if is_test:
                sample["病案标识"] = item.get("病案标识", "")
            self.samples.append(sample)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
