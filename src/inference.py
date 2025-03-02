import os
import json

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import (MODEL_NAME, MAX_LENGTH, BATCH_SIZE, MAIN_CODES, 
                    OTHER_CODES, num_main, num_other, CHECKPOINT_EPOCH, DEVICE)
from dataset import ICDDataset
from model import MultiTaskModel

# Disable oneDNN optimizations for consistent results.
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def inference():
    # Set base directory and data paths
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    test_path = os.path.join(data_dir, "ICD-Coding-test-A.json")
    
    # Initialize the tokenizer from the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Create the test dataset and DataLoader
    test_dataset = ICDDataset(test_path, tokenizer, max_length=MAX_LENGTH, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Build the checkpoint path using configuration
    checkpoint_path = os.path.join(base_dir, "output_checkpoints",
                                   CHECKPOINT_EPOCH, "pytorch_model.bin")
    
    # Initialize the multi-task model and load the pretrained weights.
    # The "weights_only=True" flag is set to avoid unwanted pickle operations.
    model = MultiTaskModel(MODEL_NAME, num_main, num_other).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE,
                                     weights_only=True))
    # Set the model to evaluation mode
    model.eval()
    
    predictions = []
    # Inference loop: perform prediction on each batch without gradient computation.
    with torch.no_grad():
        for batch in test_loader:
            # Compute model outputs for both primary and secondary diagnoses.
            main_logits, other_logits = model(batch)
            main_preds = torch.argmax(main_logits, dim=1).cpu().numpy()
            other_probs = torch.sigmoid(other_logits).cpu().numpy()
            
            # Iterate over each sample in the current batch
            for i in range(len(main_preds)):
                # Retrieve the case ID (handle list or single case scenario)
                case_id = (batch["病案标识"][i] if isinstance(batch["病案标识"], list)
                           else batch["病案标识"])
                main_code = MAIN_CODES[main_preds[i]]
                # For secondary diagnoses, select codes with probability > 0.5.
                other_codes = [OTHER_CODES[idx] for idx, prob in 
                               enumerate(other_probs[i]) if prob > 0.5]
                pred_str = f"[{main_code}|{';'.join(other_codes)}]"
                # Append results with proper keys ("病案标识" and "预测结果")
                predictions.append({
                    "病案标识": case_id,
                    "预测结果": pred_str
                })
    
    # Define the output prediction file path.
    pred_path = os.path.join(data_dir, "ICD-Coding-test-A-predictions.json")
    # Write predictions to JSON file with UTF-8 encoding (supporting Chinese characters)
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to: {pred_path}")

if __name__ == "__main__":
    inference()
