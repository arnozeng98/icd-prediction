import json
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils import clean_text
from config import MAX_LENGTH, OTHER_CODES, main2id, other2id, FIELD_NAMES

# ICDDataset loads and processes raw JSON data for ICD coding.
# It tokenizes the text fields and converts diagnosis codes into numerical labels.
class ICDDataset(Dataset):
    def __init__(self, data_path, tokenizer: AutoTokenizer, max_length=MAX_LENGTH, is_test=False):
        self.samples = []
        # Open and load the JSON data file.
        with open(data_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        
        # Instead of defining text_fields locally,
        # use FIELD_NAMES imported from config (each field corresponds to a section in the record).
        text_fields = FIELD_NAMES  # e.g. ["主诉", "现病史", "既往史", ...]
        
        # Process each data entry in the raw JSON.
        for item in raw:
            sample = {}
            # Process each text field: clean and tokenize.
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
            # Combine all text fields into one string (may be used for debugging or output).
            sample["text"] = " ".join([clean_text(item.get(field, "")) for field in text_fields]).strip()
            
            # For training samples, process labels.
            if not is_test:
                # Process primary diagnosis code (main label)
                main_label = item.get("主要诊断编码", "").strip()
                sample["main_label"] = main2id.get(main_label, -1)
                # Process secondary diagnosis codes: split by ';', trim, and create one-hot vector.
                other_label = item.get("其他诊断编码", "")
                other_list = [code.strip() for code in other_label.split(';') if code.strip()]
                other_vec = np.zeros(len(OTHER_CODES), dtype=np.float32)
                for code in other_list:
                    if code in other2id:
                        other_vec[other2id[code]] = 1.0
                sample["other_label"] = torch.tensor(other_vec)
            # For test samples, also retain the case ID.
            if is_test:
                sample["病案标识"] = item.get("病案标识", "")
            
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
