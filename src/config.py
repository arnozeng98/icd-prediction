import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

TRAIN_FILE_PATH = os.path.join(DATA_DIR, "ICD-Coding-train.json")
TEST_FILE_PATH = os.path.join(DATA_DIR, "ICD-Coding-test-A.json")
OUTPUT_FILE_PATH = os.path.join(DATA_DIR, "ICD-Coding-test-A-predictions.json")
