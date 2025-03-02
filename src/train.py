import os
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from torch.optim import AdamW
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from copy import deepcopy

# Import configuration variables including paths and device.
from config import (MODEL_NAME, MAX_LENGTH, BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS,
                    PATIENCE, LEARNING_RATE, num_main, num_other, DATA_DIR, TRAIN_FILE, 
                    OUTPUT_CHECKPOINT_BASE, DEVICE)
from dataset import ICDDataset
from model import MultiTaskModel
from utils import compute_metrics

# Disable oneDNN optimizations (for consistency)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def train():
    # Use BASE_DIR and DATA_DIR from config for file paths.
    # TRAIN_FILE specifies the training dataset filename.
    train_path = os.path.join(DATA_DIR, TRAIN_FILE)
    
    # Initialize the tokenizer using the specified pretrained model.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load the full dataset using ICDDataset.
    full_dataset = ICDDataset(train_path, tokenizer, max_length=MAX_LENGTH)
    
    # Split the dataset into training and validation sets (80% training, 20% validation).
    train_size = len(full_dataset) - int(0.2 * len(full_dataset))
    train_dataset, val_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    # Create DataLoader instances for training and validation.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize the multi-task model and transfer it to the configured DEVICE.
    model = MultiTaskModel(MODEL_NAME, num_main, num_other).to(DEVICE)
    
    # Set up the optimizer with the specified learning rate.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # Initialize the gradient scaler for mixed precision training.
    scaler = GradScaler('cuda')
    
    # Variables for early stopping and best model tracking.
    best_composite = -1
    best_model_state = None
    patience_counter = 0
    
    # Training loop over epochs.
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()  # Set model to training mode.
        optimizer.zero_grad()  # Reset gradients.
        epoch_loss = 0
        total_steps = len(train_loader)
        
        # Iterate over each batch in the training DataLoader.
        for step, batch in enumerate(train_loader):
            # Enable mixed precision training with autocast.
            with autocast(device_type='cuda', enabled=True):
                # Move labels to DEVICE.
                main_labels = batch["main_label"].to(DEVICE)
                other_labels = batch["other_label"].to(DEVICE)
                # Forward pass: obtain predictions for both tasks.
                main_logits, other_logits = model(batch)
                # Calculate loss for primary diagnosis using cross entropy.
                loss_main = torch.nn.CrossEntropyLoss()(main_logits, main_labels)
                # Calculate loss for secondary diagnoses using BCE with logits.
                loss_other = torch.nn.BCEWithLogitsLoss()(other_logits, other_labels)
                # Scale loss for gradient accumulation.
                loss = (loss_main + loss_other) / GRAD_ACCUM_STEPS
            
            # Backward pass with gradient scaling.
            scaler.scale(loss).backward()
            
            # Perform optimizer step after GRAD_ACCUM_STEPS batches.
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == total_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
        
        # After each epoch, compute validation metrics.
        main_acc, other_f1, composite = compute_metrics(model, val_loader, DEVICE)
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | "
              f"Val Main Acc: {main_acc:.4f} | Val Other F1: {other_f1:.4f} | Composite: {composite:.4f}")
        
        # Check if current model is the best so far (based on composite metric).
        if composite > best_composite:
            best_composite = composite
            best_model_state = deepcopy(model.state_dict())
            patience_counter = 0
            print("Best model found, saving state...")
            # Build the checkpoint directory path using the OUTPUT_CHECKPOINT_BASE from config.
            output_checkpoint_dir = os.path.join(OUTPUT_CHECKPOINT_BASE, f"checkpoint-{epoch}")
            os.makedirs(output_checkpoint_dir, exist_ok=True)
            # Save the model state.
            torch.save(model.state_dict(), os.path.join(output_checkpoint_dir, "pytorch_model.bin"))
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{PATIENCE}")
        
        # Early stopping condition.
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
        
        # Free CUDA cache after each epoch.
        torch.cuda.empty_cache()
        
    # Load the best model state if available.
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Training complete, best composite metric: {best_composite:.4f}")

if __name__ == "__main__":
    train()
