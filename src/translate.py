# src/translate.py

import torch
import torch.nn.functional as F
import argparse
import sentencepiece as spm

from . import config
from .model import create_model, generate_square_subsequent_mask

def beam_search_decode(model, src, max_len, start_symbol_idx, device, beam_size):
    """
    Performs beam search decoding.
    """
    model.eval()
    src = src.to(device)
    
    # --- Step 1: Encode the source sentence ---
    src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool).to(device)
    memory = model.encode(src, src_mask)
    memory = memory.to(device)

    # --- Step 2: Initialize the beam ---
    # The beam starts with a single hypothesis: the <SOS> token.
    # Each item in the beam is a tuple (sequence, score).
    initial_seq = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)
    beam = [(initial_seq, 0.0)]
    completed_hypotheses = []

    # --- Step 3: The Decoding Loop ---
    for _ in range(max_len):
        all_candidates = []
        
        # --- Step 4: Expand each hypothesis in the beam ---
        for seq, score in beam:
            # If the last token is <EOS>, this hypothesis is complete.
            if seq[-1].item() == config.EOS_IDX:
                completed_hypotheses.append((seq, score))
                continue

            # Get the model's prediction for the next token
            tgt_mask = (generate_square_subsequent_mask(seq.size(0), device)).to(device)
            out = model.decode(seq, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            
            # Use log probabilities for numerical stability
            log_probs = F.log_softmax(prob, dim=-1)
            
            # Get the top `beam_size` candidates for the next token
            top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)

            # Create new hypotheses from the expanded candidates
            for i in range(beam_size):
                next_word_idx = top_indices[0][i].reshape(1, 1)
                next_log_prob = top_log_probs[0][i].item()
                
                new_seq = torch.cat([seq, next_word_idx], dim=0)
                new_score = score + next_log_prob
                all_candidates.append((new_seq, new_score))

        # --- Step 5: Prune the beam ---
        # Sort all candidates by score and keep only the top `beam_size`
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        beam = ordered[:beam_size]
        
        # --- Step 6: Check for termination ---
        # If the best hypothesis is complete, and we have enough completed ones, we can stop early.
        if len(completed_hypotheses) >= beam_size:
            break

    # If some hypotheses are still active, add them to the completed list
    completed_hypotheses.extend(beam)
    
    # --- Step 7: Select the best hypothesis ---
    # Normalize scores by length to avoid favoring shorter sentences
    best_hypothesis = sorted(completed_hypotheses, key=lambda x: x[1] / len(x[0]), reverse=True)
    
    return best_hypothesis[0][0]


def translate(model, sentence, sp_model_es, sp_model_en, device):
    model.eval()
    
    tokens = [config.SOS_IDX] + sp_model_es.encode_as_ids(sentence) + [config.EOS_IDX]
    src_tensor = torch.LongTensor(tokens).view(-1, 1)
    
    with torch.no_grad():
        # --- CHANGED LINE ---
        # Call the new beam search function
        translated_indices = beam_search_decode(
            model, src_tensor, config.MAX_TRANSLATION_LEN, 
            config.SOS_IDX, device, config.BEAM_SIZE
        )
        
    translated_text = sp_model_en.decode_ids(translated_indices.flatten().tolist())
    return translated_text.replace('<s>', '').replace('</s>', '').strip()

def main(sentence):
    sp_es = spm.SentencePieceProcessor()
    sp_es.load(config.SP_MODEL_PATH_ES)
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(config.SP_MODEL_PATH_EN)
        
    model = create_model(config.VOCAB_SIZE, config.VOCAB_SIZE, config, config.DEVICE)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
    print("\n--- Translating with Beam Search ---")
    translation = translate(model, sentence, sp_es, sp_en, config.DEVICE)
    print(f"Original (es): {sentence}")
    print(f"Translated (en): {translation}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate a Spanish sentence using Beam Search.')
    parser.add_argument('sentence', type=str, help='The Spanish sentence to translate.')
    args = parser.parse_args()
    main(args.sentence)

# src/translate.py

# import torch
# import re
# import argparse
# import sentencepiece as spm

# from . import config
# from .model import create_model, generate_square_subsequent_mask

# def greedy_decode(model, src, max_len, start_symbol_idx, device):
#     src = src.to(device)
#     src_mask = (torch.zeros(src.shape[0], src.shape[0])).type(torch.bool).to(device)
#     memory = model.encode(src, src_mask)
    
#     ys = torch.ones(1, 1).fill_(start_symbol_idx).type(torch.long).to(device)
    
#     for i in range(max_len - 1):
#         memory = memory.to(device)
#         tgt_mask = (generate_square_subsequent_mask(ys.size(0), device)).to(device)
        
#         out = model.decode(ys, memory, tgt_mask)
#         out = out.transpose(0, 1)
        
#         prob = model.generator(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         next_word = next_word.item()

#         ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        
#         if next_word == config.EOS_IDX:
#             break
            
#     return ys

# def translate(model, sentence, sp_model_es, sp_model_en, device):
#     model.eval()
    
#     # Use SentencePiece to tokenize the input sentence
#     tokens = [config.SOS_IDX] + sp_model_es.encode_as_ids(sentence) + [config.EOS_IDX]
#     src_tensor = torch.LongTensor(tokens).view(-1, 1)
    
#     with torch.no_grad():
#         translated_indices = greedy_decode(model, src_tensor, config.MAX_TRANSLATION_LEN, config.SOS_IDX, device)
        
#     # Decode the resulting indices back to a string
#     translated_text = sp_model_en.decode_ids(translated_indices.flatten().tolist())
    
#     # Clean up special tokens
#     return translated_text.replace('<s>', '').replace('</s>', '').strip()

# def main(sentence):
#     print("Loading SentencePiece models...")
#     sp_es = spm.SentencePieceProcessor()
#     sp_es.load(config.SP_MODEL_PATH_ES)
    
#     sp_en = spm.SentencePieceProcessor()
#     sp_en.load(config.SP_MODEL_PATH_EN)
        
#     print("Loading Transformer model...")
#     model = create_model(config.VOCAB_SIZE, config.VOCAB_SIZE, config, config.DEVICE)
#     model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
#     print("\n--- Translating ---")
#     translation = translate(model, sentence, sp_es, sp_en, config.DEVICE)
#     print(f"Original (es): {sentence}")
#     print(f"Translated (en): {translation}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Translate a Spanish sentence using the Transformer model.')
#     parser.add_argument('sentence', type=str, help='The Spanish sentence to translate.')
#     args = parser.parse_args()
#     main(args.sentence)









# # src/translate.py

# import torch
# import pickle
# import re
# import argparse

# # Import custom modules
# from . import config
# from .model import create_model
# from .dataset import Vocabulary # We need the Vocabulary class definition

# def normalize_text(text):
#     """Performs the same text normalization as in preprocessing."""
#     text = text.lower().strip()
#     text = re.sub(r"([?.!,¿])", r" \1 ", text)
#     text = re.sub(r'[" "]+', " ", text)
#     text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
#     text = text.strip()
#     return text

# def translate_sentence(sentence, es_vocab, en_vocab, model, device, max_len=50):
#     model.eval()

#     # Normalize and tokenize the source sentence
#     normalized_sentence = normalize_text(sentence)
#     tokens = [es_vocab.word2index.get(word, es_vocab.word2index['<UNK>']) for word in normalized_sentence.split(' ')]
    
#     # Add <SOS> and <EOS> tokens
#     tokens = [es_vocab.word2index['<SOS>']] + tokens + [es_vocab.word2index['<EOS>']]
    
#     src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
#     src_len = torch.LongTensor([len(tokens)]).to('cpu') # Lengths must be on CPU

#     with torch.no_grad():
#         encoder_outputs, hidden = model.encoder(src_tensor, src_len)

#     # Start decoding with the <SOS> token
#     trg_indexes = [en_vocab.word2index['<SOS>']]
    
#     for i in range(max_len):
#         trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
#         with torch.no_grad():
#             output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        
#         pred_token = output.argmax(1).item()
#         trg_indexes.append(pred_token)

#         # Stop if the model predicts the <EOS> token
#         if pred_token == en_vocab.word2index['<EOS>']:
#             break
            
#     # Convert token indices back to words
#     trg_tokens = [en_vocab.index2word[i] for i in trg_indexes]
    
#     # Return the translated sentence, removing <SOS> and <EOS>
#     return ' '.join(trg_tokens[1:-1])


# def main(sentence_to_translate):
#     print("Loading vocabularies...")
#     with open(config.VOCAB_ES_PATH, 'rb') as f:
#         es_vocab = pickle.load(f)
#     with open(config.VOCAB_EN_PATH, 'rb') as f:
#         en_vocab = pickle.load(f)
        
#     print("Loading model...")
#     INPUT_DIM = es_vocab.n_words
#     OUTPUT_DIM = en_vocab.n_words
    
#     model = create_model(INPUT_DIM, OUTPUT_DIM, config, config.DEVICE)
#     model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
#     print("Model loaded successfully.\n")
    
#     translation = translate_sentence(sentence_to_translate, es_vocab, en_vocab, model, config.DEVICE, config.MAX_TRANSLATION_LEN)
    
#     print(f"Original (es): {sentence_to_translate}")
#     print(f"Translated (en): {translation}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Translate a Spanish sentence to English.')
#     parser.add_argument('sentence', type=str, help='The Spanish sentence to translate.')
#     args = parser.parse_args()
    
#     main(args.sentence)