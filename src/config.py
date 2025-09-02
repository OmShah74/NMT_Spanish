# src/config.py

import os
import torch

# --- Project Paths ---
# Points to the root 'NMT' directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

# --- Data Files ---
TRAIN_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_df.pkl')
VAL_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'val_df.pkl')
TEST_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_df.pkl') # Added for completeness
VOCAB_ES_PATH = os.path.join(PROCESSED_DATA_DIR, 'vocab_es.pkl')
VOCAB_EN_PATH = os.path.join(PROCESSED_DATA_DIR, 'vocab_en.pkl')

# --- Model Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
CLIP = 1 # Gradient clipping value

# --- Special Token Indices ---
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# --- For Inference ---
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'best-nmt-model.pt')
MAX_TRANSLATION_LEN = 50 # Maximum length for generated sentences