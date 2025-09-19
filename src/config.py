# src/config.py

import os
import torch

# --- Project Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, 'saved_models')

# --- Data Files ---
TRAIN_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_df.pkl')
VAL_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'val_df.pkl')
TEST_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'test_df.pkl')

# --- NEW: SentencePiece Model Paths ---
SP_MODEL_PATH_ES = os.path.join(PROCESSED_DATA_DIR, 'sp_es.model')
SP_MODEL_PATH_EN = os.path.join(PROCESSED_DATA_DIR, 'sp_en.model')
SP_TRAIN_TEXT_ES = os.path.join(PROCESSED_DATA_DIR, 'train_es.txt')
SP_TRAIN_TEXT_EN = os.path.join(PROCESSED_DATA_DIR, 'train_en.txt')


# --- Model Hyperparameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 25
CLIP = 1

# --- Transformer Model Hyperparameters ---
D_MODEL = 512
N_HEAD = 8
D_HID = 2048
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DROPOUT = 0.1
# --- NEW: Vocabulary Size is now a hyperparameter ---
VOCAB_SIZE = 16000 # The size of our subword vocabulary

# --- Special Token Indices (Used by SentencePiece) ---
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

# --- For Inference ---
BEST_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, 'best-transformer-model.pt')
MAX_TRANSLATION_LEN = 100 # Increased max length