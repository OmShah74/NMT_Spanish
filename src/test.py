# src/test.py

import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import random

# Import custom modules
from . import config
from .model import create_model
from .dataset import get_loader, Vocabulary

# Import the BLEU score metric from torchtext


def test_model(model, iterator, criterion, en_vocab):
    """
    Performs a full evaluation on the test set, including BLEU score.
    """
    model.eval()
    epoch_loss = 0

    # Lists to store the actual and predicted sentences for BLEU score calculation
    targets_corpus = []
    generated_corpus = []

    with torch.no_grad():
        progress_bar = tqdm(iterator, desc="Testing", leave=True)
        for i, batch in enumerate(progress_bar):
            src, src_len, trg = batch
            src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

            # --- Forward Pass ---
            # Turn off teacher forcing
            output = model(src, src_len, trg, teacher_forcing_ratio=0)
            
            # --- Loss Calculation ---
            output_dim = output.shape[-1]
            output_for_loss = output[1:].view(-1, output_dim)
            trg_for_loss = trg[1:].view(-1)
            loss = criterion(output_for_loss, trg_for_loss)
            epoch_loss += loss.item()

            # --- Sentence Generation for BLEU ---
            # Get the predicted word indices by taking the argmax
            # Shape: [trg_len, batch_size]
            predicted_indices = output.argmax(2)

            # Transpose to iterate through sentences in the batch
            # Shape: [batch_size, trg_len]
            predicted_indices = predicted_indices.permute(1, 0).cpu().numpy()
            trg_indices = trg.permute(1, 0).cpu().numpy()

            # Convert indices to words for each sentence in the batch
            for i in range(len(predicted_indices)):
                pred_sentence = []
                for idx in predicted_indices[i]:
                    if idx == config.EOS_IDX:
                        break # Stop at End of Sentence token
                    pred_sentence.append(en_vocab.index2word[idx])
                
                trg_sentence = []
                for idx in trg_indices[i]:
                    # Ignore the <SOS> token in the reference
                    if idx == config.SOS_IDX: continue
                    if idx == config.EOS_IDX: break
                    trg_sentence.append(en_vocab.index2word[idx])

                generated_corpus.append(pred_sentence)
                # For bleu_score, each reference must be in a list of lists
                targets_corpus.append([trg_sentence])
                
    # --- Calculate Metrics ---
    final_loss = epoch_loss / len(iterator)
    perplexity = torch.exp(torch.tensor(final_loss))
    
    # Calculate BLEU score (multiplied by 100 for readability)
    bleu = corpus_bleu(targets_corpus, generated_corpus) * 100

    # Print some example translations
    print("\n--- Example Translations ---")
    for _ in range(5):
        idx = random.randint(0, len(generated_corpus) - 1)
        print(f"Reference : {' '.join(targets_corpus[idx][0])}")
        print(f"Generated : {' '.join(generated_corpus[idx])}\n")
    print("--------------------------\n")

    return final_loss, perplexity.item(), bleu

def main():
    """Main function to load the model and run the test evaluation."""
    print("Loading test data and vocabularies...")
    # Load the English vocabulary to convert indices back to words
    _, _, en_vocab = get_loader(
        df_path=config.TEST_DF_PATH,
        source_vocab_path=config.VOCAB_ES_PATH,
        target_vocab_path=config.VOCAB_EN_PATH,
        batch_size=config.BATCH_SIZE,
        pad_idx=config.PAD_IDX,
        shuffle=False
    )
    
    # We only need the test loader itself for evaluation
    test_loader, es_vocab, _ = get_loader(
        df_path=config.TEST_DF_PATH,
        source_vocab_path=config.VOCAB_ES_PATH,
        target_vocab_path=config.VOCAB_EN_PATH,
        batch_size=config.BATCH_SIZE,
        pad_idx=config.PAD_IDX,
        shuffle=False
    )
    
    print("Loading trained model...")
    INPUT_DIM = es_vocab.n_words
    OUTPUT_DIM = en_vocab.n_words
    
    # Initialize the model architecture
    model = create_model(INPUT_DIM, OUTPUT_DIM, config, config.DEVICE)
    
    # Load the saved best model weights
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
    # Define the loss function, making sure to ignore padding
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    # Run the evaluation
    test_loss, test_ppl, test_bleu = test_model(model, test_loader, criterion, en_vocab)

    print("\n--- Test Set Performance ---")
    print(f"Loss      : {test_loss:.3f}")
    print(f"Perplexity: {test_ppl:7.3f}")
    print(f"BLEU Score: {test_bleu:.2f}")
    print("----------------------------")

if __name__ == '__main__':
    main()