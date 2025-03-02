import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from copy import deepcopy

from config import MODEL_NAME, MAX_LENGTH, BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, PATIENCE, LEARNING_RATE, MAIN_CODES, OTHER_CODES, num_main, num_other
from dataset import ICDDataset
from model import MultiTaskModel
from utils import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def train():
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, "data")
    train_path = os.path.join(data_dir, "ICD-Coding-train.json")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    full_dataset = ICDDataset(train_path, tokenizer, max_length=MAX_LENGTH)
    train_size = len(full_dataset) - int(0.2 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    model = MultiTaskModel(MODEL_NAME, num_main, num_other).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')
    
    best_composite = -1
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS+1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0
        total_steps = len(train_loader)
        
        for step, batch in enumerate(train_loader):
            with autocast(device_type='cuda', enabled=True):
                main_labels = batch["main_label"].to(device)
                other_labels = batch["other_label"].to(device)
                main_logits, other_logits = model(batch)
                loss_main = torch.nn.CrossEntropyLoss()(main_logits, main_labels)
                loss_other = torch.nn.BCEWithLogitsLoss()(other_logits, other_labels)
                loss = (loss_main + loss_other) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == total_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        
        main_acc, other_f1, composite = compute_metrics(model, val_loader, device)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Main Acc: {main_acc:.4f} | Val Other F1: {other_f1:.4f} | Composite: {composite:.4f}")
        
        if composite > best_composite:
            best_composite = composite
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            print("Best model found, saving state...")
            output_checkpoint_dir = os.path.join(base_dir, "output_checkpoints", f"checkpoint-{epoch}")
            os.makedirs(output_checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_checkpoint_dir, "pytorch_model.bin"))
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
        
        torch.cuda.empty_cache()
        
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Training complete, best composite metric: {best_composite:.4f}")

if __name__ == "__main__":
    train()
