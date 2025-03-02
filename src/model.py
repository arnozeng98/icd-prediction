import torch
import torch.nn as nn
from transformers import AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_main, num_other):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.field_names = [
            "主诉", "现病史", "既往史", "个人史", "婚姻史", "家族史",
            "入院情况", "入院诊断", "诊疗经过", "出院情况", "出院医嘱"
        ]
        hidden_size = self.encoder.config.hidden_size
        self.fusion = nn.Sequential(
            nn.Linear(len(self.field_names) * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.main_classifier = nn.Linear(hidden_size, num_main)
        self.other_classifier = nn.Linear(hidden_size, num_other)
    def forward(self, batch):
        field_reps = []
        for field in self.field_names:
            outputs = self.encoder(
                input_ids=batch[f"{field}_input_ids"].to(device),
                attention_mask=batch[f"{field}_attention_mask"].to(device)
            )
            cls_rep = outputs.last_hidden_state[:, 0, :]
            field_reps.append(cls_rep)
        concatenated = torch.cat(field_reps, dim=1)
        fused = self.fusion(concatenated)
        main_logits = self.main_classifier(fused)
        other_logits = self.other_classifier(fused)
        return main_logits, other_logits
