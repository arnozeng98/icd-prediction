"""
Global configuration for the ICD Coding Prediction project.

This file contains settings for:
1. Model and Tokenizer configuration.
2. Training hyperparameters.
3. Diagnosis codes and label mappings.
4. File paths for data and checkpoints.
5. Device and other runtime configurations.

Modify the values below to suit your system and experiment requirements.
"""

import os
import torch

# ============================
# 1. Model and Tokenizer Settings
# ============================
# Pretrained model name (used for both encoding and tokenizer initialization).
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
# Maximum sequence length for tokenization.
MAX_LENGTH = 256
# Batch size used in both training and inference.
BATCH_SIZE = 2

# ============================
# 2. Training Hyperparameters
# ============================
# Number of gradient accumulation steps.
GRAD_ACCUM_STEPS = 4
# Total number of epochs for training.
NUM_EPOCHS = 200
# Early stopping patience: number of epochs with no improvement after which training stops.
PATIENCE = 10
# Learning rate for the optimizer.
LEARNING_RATE = 2e-5

# ============================
# 3. Diagnosis Codes and Label Mappings
# ============================
# List of primary diagnosis codes.
MAIN_CODES = [
    'I10.x00x032', 'I20.000', 'I20.800x007', 'I21.401', 'I50.900x018'
]
# List of secondary/other diagnosis codes.
OTHER_CODES = [
    'E04.101', 'E04.102', 'E11.900', 'E14.900x001', 'E72.101', 'E78.500',
    'E87.600', 'I10.x00x023', 'I10.x00x024', 'I10.x00x027', 'I10.x00x028',
    'I10.x00x031', 'I10.x00x032', 'I20.000', 'I25.102', 'I25.103', 'I25.200',
    'I31.800x004', 'I38.x01', 'I48.x01', 'I48.x02', 'I49.100x001', 'I49.100x002',
    'I49.300x001', 'I49.300x002', 'I49.400x002', 'I49.400x003', 'I49.900',
    'I50.900x007', 'I50.900x008', 'I50.900x010', 'I50.900x014', 'I50.900x015',
    'I50.900x016', 'I50.900x018', 'I50.907', 'I63.900', 'I67.200x011',
    'I69.300x002', 'I70.203', 'I70.806', 'J18.900', 'J98.414', 'K76.000',
    'K76.807', 'N19.x00x002', 'N28.101', 'Q24.501', 'R42.x00x004',
    'R91.x00x003', 'Z54.000x033', 'Z95.501', 'Z98.800x612'
]

# Create label-to-index mappings for both task types.
main2id = {code: idx for idx, code in enumerate(MAIN_CODES)}
other2id = {code: idx for idx, code in enumerate(OTHER_CODES)}
# Total number of classes for primary and secondary diagnoses.
num_main = len(MAIN_CODES)
num_other = len(OTHER_CODES)

# ============================
# 4. File Paths and Checkpoint Settings
# ============================
# Base directory (current working directory).
BASE_DIR = os.getcwd()
# Data directory where training and testing files are stored.
DATA_DIR = os.path.join(BASE_DIR, "data")
# Filename for the training file.
TRAIN_FILE = "ICD-Coding-train.json"
# Base directory for saving checkpoint files.
OUTPUT_CHECKPOINT_BASE = os.path.join(BASE_DIR, "output_checkpoints")
# The name of the checkpoint folder to be used during inference (should be set to the best epoch's folder).
CHECKPOINT_EPOCH = "checkpoint-51"

# ============================
# 5. Device and Field Configuration
# ============================
# Device configuration: Use GPU if available, otherwise Fall back to CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Field names to be used in the model. Each represents a specific section in the medical record.
FIELD_NAMES = [
    "主诉", "现病史", "既往史", "个人史", "婚姻史", "家族史",
    "入院情况", "入院诊断", "诊疗经过", "出院情况", "出院医嘱"
]
