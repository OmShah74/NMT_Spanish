import pandas as pd
import re
import os
import pickle
from sklearn.model_selection import train_test_split

# === CHANGED SECTION START ===
# REMOVE the old Vocabulary class definition from this file.
# IMPORT it from dataset.py, which is now the single source of truth.
from dataset import Vocabulary
# === CHANGED SECTION END ===


# --- Data Loading and Normalization ---
def load_data(raw_data_dir):
    """Loads the Spanish and English sentence pairs from the raw data directory."""
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
    """Performs text normalization steps on a single sentence."""
    text = text.lower()
    text = re.sub(r'([?.!,])', r' \1 ', text)
    text = re.sub(r'[^a-z?.!,]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    """Main function to execute the full preprocessing pipeline."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = os.path.join(project_root, 'data', 'raw')
    processed_data_path = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_data_path, exist_ok=True)

    try:
        df = load_data(raw_data_path)
    except FileNotFoundError:
        print("Error: Raw data files not found in 'data/raw/'.")
        return

    print("\nNormalizing text data...")
    df['spanish_normalized'] = df['spanish'].apply(normalize_text)
    df['english_normalized'] = df['english'].apply(normalize_text)
    print("Normalization complete.")
    
    df = df[(df['spanish_normalized'] != '') & (df['english_normalized'] != '')]
    df = df.dropna().reset_index(drop=True)
    print(f"Dataframe shape after cleaning: {df.shape}")

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
    
    train_df.to_pickle(os.path.join(processed_data_path, 'train_df.pkl'))
    val_df.to_pickle(os.path.join(processed_data_path, 'val_df.pkl'))
    test_df.to_pickle(os.path.join(processed_data_path, 'test_df.pkl'))
    print("Saved DataFrame splits to 'data/processed/'.")

    with open(os.path.join(processed_data_path, 'vocab_es.pkl'), 'wb') as f:
        pickle.dump(spanish_vocab, f)
    with open(os.path.join(processed_data_path, 'vocab_en.pkl'), 'wb') as f:
        pickle.dump(english_vocab, f)
    print("Saved vocabularies to 'data/processed/'.")

    print("\n--- Preprocessing Finished Successfully! ---")
    print("Next step: Run train.py")

if __name__ == '__main__':
    # Since we are running this from the /content/NMT/src directory,
    # we need to add 'src' to the path so it can find the 'dataset' module.
    import sys
    sys.path.insert(0, os.path.abspath('.'))
    main()