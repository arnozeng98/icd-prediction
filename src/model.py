import torch
import torch.nn as nn
from transformers import AutoModel
from config import DEVICE, FIELD_NAMES  # Import device and field names from config

# Detailed Model Explanation:
# The MultiTaskModel is designed for ICD coding prediction.
# It uses a pre-trained Transformer as the base encoder for multiple text fields.
# For each field, the model obtains the [CLS] output and then concatenates all field representations.
# The concatenated vector is passed through a fusion layer (fully connected layer with GELU activation and dropout)
# to obtain fused features.
# Finally, two separate classifier heads process these fused features:
# one for primary diagnosis classification (single-label) and one for secondary diagnosis classification (multi-label).

class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_main, num_other):
        super().__init__()
        # Load the pre-trained transformer model as the shared encoder.
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Use field names imported from config; each represents a section in the medical record.
        self.field_names = FIELD_NAMES
        
        # Get the hidden size of the encoder.
        hidden_size = self.encoder.config.hidden_size
        
        # Fusion layer: Reduces the concatenated features from all fields.
        # It consists of:
        # - A Linear layer that maps the concatenated vector back to the encoder's hidden size.
        # - A GELU activation function to introduce non-linearity.
        # - A Dropout layer to regularize and prevent overfitting.
        self.fusion = nn.Sequential(
            nn.Linear(len(self.field_names) * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Primary Diagnosis Classifier:
        # This head outputs logits for a single-label classification task.
        self.main_classifier = nn.Linear(hidden_size, num_main)
        
        # Secondary Diagnosis Classifier:
        # This head outputs logits for a multi-label classification task.
        self.other_classifier = nn.Linear(hidden_size, num_other)
    
    def forward(self, batch):
        # List to accumulate representations (CLS tokens) for each field.
        field_reps = []
        # Iterate over all defined fields.
        for field in self.field_names:
            # Pass the tokenized input for each field through the encoder.
            outputs = self.encoder(
                input_ids=batch[f"{field}_input_ids"].to(DEVICE),
                attention_mask=batch[f"{field}_attention_mask"].to(DEVICE)
            )
            # Extract the [CLS] token representation from the output.
            cls_rep = outputs.last_hidden_state[:, 0, :]
            field_reps.append(cls_rep)
        
        # Concatenate the representations from all fields along the feature dimension.
        concatenated = torch.cat(field_reps, dim=1)
        # Fuse the concatenated vector by passing it through the fusion layer.
        fused = self.fusion(concatenated)
        # Compute the logits for both primary and secondary diagnosis tasks.
        main_logits = self.main_classifier(fused)
        other_logits = self.other_classifier(fused)
        # Return the logits.
        return main_logits, other_logits
