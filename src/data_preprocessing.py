# src/data_preprocessing.py

import pandas as pd
import re
import os
import pickle
from sklearn.model_selection import train_test_split

# Import the single source of truth for Vocabulary from dataset.py
from .dataset import Vocabulary 
# Import paths from the centralized config file
from . import config

# --- Data Loading and Normalization ---
def load_data(raw_data_dir):
    spanish_file_path = os.path.join(raw_data_dir, 'europarl-v7.es-en.es')
    english_file_path = os.path.join(raw_data_dir, 'europarl-v7.es-en.en')

    print("Loading Spanish data from:", spanish_file_path)
    with open(spanish_file_path, 'r', encoding='utf-8') as f:
        spanish_sentences = f.readlines()

    print("Loading English data from:", english_file_path)
    with open(english_file_path, 'r', encoding='utf-8') as f:
        english_sentences = f.readlines()

    if len(spanish_sentences) != len(english_sentences):
        raise ValueError("The Spanish and English files do not have the same number of lines.")

    df = pd.DataFrame({'spanish': spanish_sentences, 'english': english_sentences})
    print(f"Successfully loaded {len(df)} sentence pairs.")
    return df

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text

def main():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    try:
        df = load_data(config.RAW_DATA_DIR)
    except FileNotFoundError:
        print(f"Error: Raw data files not found in '{config.RAW_DATA_DIR}'.")
        return

    print("\nNormalizing text data...")
    df['spanish_normalized'] = df['spanish'].apply(normalize_text)
    df['english_normalized'] = df['english'].apply(normalize_text)
    print("Normalization complete.")
    
    df = df[(df['spanish_normalized'] != '') & (df['english_normalized'] != '')].dropna().reset_index(drop=True)
    print(f"Dataframe shape after cleaning: {df.shape}")

    # Use a subset for faster processing. Increase this number for better results.
    df = df.head(100000)
    print(f"\nUsing a subset of {len(df)} sentences for processing.")
    
    train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    print("\nBuilding vocabularies from the training data...")
    spanish_vocab = Vocabulary("spanish")
    english_vocab = Vocabulary("english")

    for sentence in train_df['spanish_normalized']:
        spanish_vocab.add_sentence(sentence)
    for sentence in train_df['english_normalized']:
        english_vocab.add_sentence(sentence)

    print(f"Spanish vocabulary size: {spanish_vocab.n_words}")
    print(f"English vocabulary size: {english_vocab.n_words}")
    
    print("\nSaving data splits and vocabularies...")
    
    train_df.to_pickle(config.TRAIN_DF_PATH)
    val_df.to_pickle(config.VAL_DF_PATH)
    test_df.to_pickle(config.TEST_DF_PATH)
    print(f"Saved DataFrame splits to '{config.PROCESSED_DATA_DIR}'.")

    with open(config.VOCAB_ES_PATH, 'wb') as f:
        pickle.dump(spanish_vocab, f)
    with open(config.VOCAB_EN_PATH, 'wb') as f:
        pickle.dump(english_vocab, f)
    print(f"Saved vocabularies to '{config.PROCESSED_DATA_DIR}'.")

    print("\n--- Preprocessing Finished Successfully! ---")
    print("Next step: Run src/train.py")

if __name__ == '__main__':
    main()