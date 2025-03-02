import os
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE, MAIN_CODES, OTHER_CODES, num_main, num_other
from dataset import ICDDataset
from model import MultiTaskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def inference():
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    test_path = os.path.join(data_dir, "ICD-Coding-test-A.json")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_dataset = ICDDataset(test_path, tokenizer, max_length=MAX_LENGTH, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    checkpoint_path = os.path.join(base_dir, "output_checkpoints", "checkpoint-<epoch>", "pytorch_model.bin")
    model = MultiTaskModel(MODEL_NAME, num_main, num_other).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for batch in test_loader:
            main_logits, other_logits = model(batch)
            main_preds = torch.argmax(main_logits, dim=1).cpu().numpy()
            other_probs = torch.sigmoid(other_logits).cpu().numpy()
            for i in range(len(main_preds)):
                case_id = (batch["病案标识"][i] if isinstance(batch["病案标识"], list)
                           else batch["病案标识"])
                main_code = MAIN_CODES[main_preds[i]]
                other_codes = [
                    OTHER_CODES[idx] for idx, prob in enumerate(other_probs[i])
                    if prob > 0.5
                ]
                pred_str = f"[{main_code}|{';'.join(other_codes)}]"
                predictions.append({
                    "Case_ID": case_id,
                    "Prediction": pred_str
                })
    
    pred_path = os.path.join(data_dir, "ICD-Coding-test-A-predictions.json")
    with open(pred_path, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Predictions saved to: {pred_path}")

if __name__ == "__main__":
    inference()
