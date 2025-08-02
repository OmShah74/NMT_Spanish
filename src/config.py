import os
import torch

# --- Project Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SAVED_MODELS_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), 'saved_models')

# --- Data Files ---
TRAIN_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'train_df.pkl')
VAL_DF_PATH = os.path.join(PROCESSED_DATA_DIR, 'val_df.pkl')
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

# --- Special Token Indices ---
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3