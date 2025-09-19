# src/data_preprocessing.py

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import spacy
import sentencepiece as spm
from tqdm import tqdm

from . import config

# --- 1. Linguistic Preprocessing with spaCy ---
def preprocess_with_spacy(sentences, spacy_model):
    """
    Performs lemmatization and basic cleaning on a list of sentences.
    """
    processed_sentences = []
    for doc in tqdm(spacy_model.pipe(sentences), total=len(sentences), desc="Lemmatizing"):
        lemmas = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space]
        processed_sentences.append(" ".join(lemmas))
    return processed_sentences

# --- 2. Train SentencePiece Model (NEW ROBUST VERSION) ---
def train_sentencepiece_model(text_file_path, model_path_prefix, vocab_size):
    """
    Trains a SentencePiece model using a dictionary of arguments to avoid path issues.
    """
    print(f"Training SentencePiece model from {text_file_path}...")
    
    # --- THIS IS THE NEW, CORRECTED METHOD ---
    # We pass arguments as a dictionary, which is safer than a command string.
    # This completely avoids any issues with spaces in file paths.
    spm.SentencePieceTrainer.train(
        input=text_file_path,
        model_prefix=model_path_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=config.PAD_IDX,
        unk_id=config.UNK_IDX,
        bos_id=config.SOS_IDX,
        eos_id=config.EOS_IDX
    )
    print(f"SentencePiece model saved to {model_path_prefix}.model")

def main():
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    # --- Load Raw Data ---
    spanish_file = os.path.join(config.RAW_DATA_DIR, 'europarl-v7.es-en.es')
    english_file = os.path.join(config.RAW_DATA_DIR, 'europarl-v7.es-en.en')
    with open(spanish_file, 'r', encoding='utf-8') as f: es_sents = f.readlines()
    with open(english_file, 'r', encoding='utf-8') as f: en_sents = f.readlines()
    
    df = pd.DataFrame({'spanish': es_sents, 'english': en_sents})
    df = df.head(500000)
    print(f"Loaded {len(df)} raw sentence pairs.")

    # --- Linguistic Preprocessing ---
    print("\nLoading spaCy models...")
    nlp_es = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
    nlp_en = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    df['spanish_processed'] = preprocess_with_spacy(df['spanish'].tolist(), nlp_es)
    df['english_processed'] = preprocess_with_spacy(df['english'].tolist(), nlp_en)
    
    # --- Data Filtering ---
    df = df[(df['spanish_processed'] != '') & (df['english_processed'] != '')].dropna().reset_index(drop=True)
    df['es_len'] = df['spanish_processed'].str.split().str.len()
    df['en_len'] = df['english_processed'].str.split().str.len()
    
    df = df[(df['es_len'] < 100) & (df['en_len'] < 100)]
    df = df[(df['es_len'] / df['en_len'] < 1.5)]
    df = df[(df['en_len'] / df['es_len'] < 1.5)]
    print(f"Dataframe shape after cleaning and filtering: {df.shape}")

    # --- Data Splitting ---
    train_df, test_val_df = train_test_split(df, test_size=0.1, random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
    print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}, Test set: {len(test_df)}")

    # --- Train SentencePiece Tokenizers (on training data only) ---
    train_df['spanish_processed'].to_csv(config.SP_TRAIN_TEXT_ES, header=False, index=False)
    train_df['english_processed'].to_csv(config.SP_TRAIN_TEXT_EN, header=False, index=False)

    train_sentencepiece_model(config.SP_TRAIN_TEXT_ES, config.SP_MODEL_PATH_ES.replace('.model', ''), config.VOCAB_SIZE)
    train_sentencepiece_model(config.SP_TRAIN_TEXT_EN, config.SP_MODEL_PATH_EN.replace('.model', ''), config.VOCAB_SIZE)
    
    # --- Save Final DataFrames ---
    train_df.to_pickle(config.TRAIN_DF_PATH)
    val_df.to_pickle(config.VAL_DF_PATH)
    test_df.to_pickle(config.TEST_DF_PATH)
    print(f"\nProcessed DataFrame splits saved to '{config.PROCESSED_DATA_DIR}'.")
    
    # Clean up temporary text files
    os.remove(config.SP_TRAIN_TEXT_ES)
    os.remove(config.SP_TRAIN_TEXT_EN)

    print("\n--- Advanced Preprocessing Finished Successfully! ---")

if __name__ == '__main__':
    main()