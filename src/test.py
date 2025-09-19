import torch
import torch.nn as nn
from tqdm import tqdm
import pickle
import random
from nltk.translate.bleu_score import corpus_bleu

from . import config
from .model import create_model, generate_square_subsequent_mask
from .dataset import get_loader

# Auto-regressive decoding function for evaluation
def greedy_decode(model, src, max_len, start_symbol_idx, device):
    src = src.to(device)
    src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool).to(device)
    memory = model.encode(src, src_mask)
    
    ys = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)
    
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == config.EOS_IDX: break
            
    return ys

def test_model(model, iterator, criterion, sp_model_en):
    model.eval()
    targets_corpus = []
    generated_corpus = []

    progress_bar = tqdm(iterator, desc="Testing Model", leave=True)
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            src, _, trg = batch
            src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)
            
            for j in range(src.size(1)): # Iterate through batch dimension
                src_sentence = src[:, j].view(-1, 1)
                trg_sentence_ids = trg[:, j]

                translated_indices = greedy_decode(model, src_sentence, config.MAX_TRANSLATION_LEN, config.SOS_IDX, config.DEVICE)
                
                # Decode generated IDs to subwords
                pred_tokens = sp_model_en.decode_ids(translated_indices.flatten().tolist())
                generated_corpus.append(pred_tokens.split())
                
                # Decode target IDs to subwords for reference
                trg_tokens = sp_model_en.decode_ids(trg_sentence_ids.tolist())
                # Filter out special tokens for BLEU calculation
                trg_tokens_filtered = [token for token in trg_tokens.split() if token not in ['<s>', '</s>', '<pad>']]
                targets_corpus.append([trg_tokens_filtered])

    bleu = corpus_bleu(targets_corpus, generated_corpus) * 100
    
    print("\n--- Example Translations ---")
    for _ in range(3):
        idx = random.randint(0, len(generated_corpus) - 1)
        print(f"Reference : {' '.join(targets_corpus[idx][0])}")
        print(f"Generated : {' '.join(generated_corpus[idx])}\n")

    return bleu

def main():
    print("--- Starting Final Evaluation on Test Set ---")
    test_loader, _, sp_model_en = get_loader(config.TEST_DF_PATH, config.SP_MODEL_PATH_ES, config.SP_MODEL_PATH_EN, config.BATCH_SIZE, config.PAD_IDX, shuffle=False)
    
    model = create_model(config.VOCAB_SIZE, config.VOCAB_SIZE, config, config.DEVICE)
    print(f"Loading trained model weights from {config.BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

    test_bleu = test_model(model, test_loader, criterion, sp_model_en)

    print("\n--- Test Set Performance ---")
    print(f"BLEU Score: {test_bleu:.2f}")
    print("----------------------------")

if __name__ == '__main__':
    main()







# # src/test.py

# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import pickle
# from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
# import random

# # Import custom modules
# from . import config
# from .model import create_model
# from .dataset import get_loader, Vocabulary

# # Import the BLEU score metric from torchtext


# def test_model(model, iterator, criterion, en_vocab):
#     """
#     Performs a full evaluation on the test set, including BLEU score.
#     """
#     model.eval()
#     epoch_loss = 0

#     # Lists to store the actual and predicted sentences for BLEU score calculation
#     targets_corpus = []
#     generated_corpus = []

#     with torch.no_grad():
#         progress_bar = tqdm(iterator, desc="Testing", leave=True)
#         for i, batch in enumerate(progress_bar):
#             src, src_len, trg = batch
#             src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

#             # --- Forward Pass ---
#             # Turn off teacher forcing
#             output = model(src, src_len, trg, teacher_forcing_ratio=0)
            
#             # --- Loss Calculation ---
#             output_dim = output.shape[-1]
#             output_for_loss = output[1:].view(-1, output_dim)
#             trg_for_loss = trg[1:].view(-1)
#             loss = criterion(output_for_loss, trg_for_loss)
#             epoch_loss += loss.item()

#             # --- Sentence Generation for BLEU ---
#             # Get the predicted word indices by taking the argmax
#             # Shape: [trg_len, batch_size]
#             predicted_indices = output.argmax(2)

#             # Transpose to iterate through sentences in the batch
#             # Shape: [batch_size, trg_len]
#             predicted_indices = predicted_indices.permute(1, 0).cpu().numpy()
#             trg_indices = trg.permute(1, 0).cpu().numpy()

#             # Convert indices to words for each sentence in the batch
#             for i in range(len(predicted_indices)):
#                 pred_sentence = []
#                 for idx in predicted_indices[i]:
#                     if idx == config.EOS_IDX:
#                         break # Stop at End of Sentence token
#                     pred_sentence.append(en_vocab.index2word[idx])
                
#                 trg_sentence = []
#                 for idx in trg_indices[i]:
#                     # Ignore the <SOS> token in the reference
#                     if idx == config.SOS_IDX: continue
#                     if idx == config.EOS_IDX: break
#                     trg_sentence.append(en_vocab.index2word[idx])

#                 generated_corpus.append(pred_sentence)
#                 # For bleu_score, each reference must be in a list of lists
#                 targets_corpus.append([trg_sentence])
                
#     # --- Calculate Metrics ---
#     final_loss = epoch_loss / len(iterator)
#     perplexity = torch.exp(torch.tensor(final_loss))
    
#     # Calculate BLEU score (multiplied by 100 for readability)
#     bleu = corpus_bleu(targets_corpus, generated_corpus) * 100

#     # Print some example translations
#     print("\n--- Example Translations ---")
#     for _ in range(5):
#         idx = random.randint(0, len(generated_corpus) - 1)
#         print(f"Reference : {' '.join(targets_corpus[idx][0])}")
#         print(f"Generated : {' '.join(generated_corpus[idx])}\n")
#     print("--------------------------\n")

#     return final_loss, perplexity.item(), bleu

# def main():
#     """Main function to load the model and run the test evaluation."""
#     print("Loading test data and vocabularies...")
#     # Load the English vocabulary to convert indices back to words
#     _, _, en_vocab = get_loader(
#         df_path=config.TEST_DF_PATH,
#         source_vocab_path=config.VOCAB_ES_PATH,
#         target_vocab_path=config.VOCAB_EN_PATH,
#         batch_size=config.BATCH_SIZE,
#         pad_idx=config.PAD_IDX,
#         shuffle=False
#     )
    
#     # We only need the test loader itself for evaluation
#     test_loader, es_vocab, _ = get_loader(
#         df_path=config.TEST_DF_PATH,
#         source_vocab_path=config.VOCAB_ES_PATH,
#         target_vocab_path=config.VOCAB_EN_PATH,
#         batch_size=config.BATCH_SIZE,
#         pad_idx=config.PAD_IDX,
#         shuffle=False
#     )
    
#     print("Loading trained model...")
#     INPUT_DIM = es_vocab.n_words
#     OUTPUT_DIM = en_vocab.n_words
    
#     # Initialize the model architecture
#     model = create_model(INPUT_DIM, OUTPUT_DIM, config, config.DEVICE)
    
#     # Load the saved best model weights
#     model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
#     # Define the loss function, making sure to ignore padding
#     criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)

#     # Run the evaluation
#     test_loss, test_ppl, test_bleu = test_model(model, test_loader, criterion, en_vocab)

#     print("\n--- Test Set Performance ---")
#     print(f"Loss      : {test_loss:.3f}")
#     print(f"Perplexity: {test_ppl:7.3f}")
#     print(f"BLEU Score: {test_bleu:.2f}")
#     print("----------------------------")

# if __name__ == '__main__':
#     main()