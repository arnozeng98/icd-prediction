import torch
import numpy as np
import sys
from train import train
from inference import inference

# Enable optimized settings for CUDA matmul and cuDNN
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|inference]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    if mode == "train":
        train()
    elif mode == "inference":
        inference()
    else:
        print("Invalid mode. Choose either 'train' or 'inference'.")
        sys.exit(1)

if __name__ == "__main__":
    main()