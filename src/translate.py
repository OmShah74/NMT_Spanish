# src/translate.py

import torch
import pickle
import re
import argparse

# Import custom modules
from . import config
from .model import create_model
from .dataset import Vocabulary # We need the Vocabulary class definition

def normalize_text(text):
    """Performs the same text normalization as in preprocessing."""
    text = text.lower().strip()
    text = re.sub(r"([?.!,¿])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)
    text = text.strip()
    return text

def translate_sentence(sentence, es_vocab, en_vocab, model, device, max_len=50):
    model.eval()

    # Normalize and tokenize the source sentence
    normalized_sentence = normalize_text(sentence)
    tokens = [es_vocab.word2index.get(word, es_vocab.word2index['<UNK>']) for word in normalized_sentence.split(' ')]
    
    # Add <SOS> and <EOS> tokens
    tokens = [es_vocab.word2index['<SOS>']] + tokens + [es_vocab.word2index['<EOS>']]
    
    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)
    src_len = torch.LongTensor([len(tokens)]).to('cpu') # Lengths must be on CPU

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    # Start decoding with the <SOS> token
    trg_indexes = [en_vocab.word2index['<SOS>']]
    
    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        # Stop if the model predicts the <EOS> token
        if pred_token == en_vocab.word2index['<EOS>']:
            break
            
    # Convert token indices back to words
    trg_tokens = [en_vocab.index2word[i] for i in trg_indexes]
    
    # Return the translated sentence, removing <SOS> and <EOS>
    return ' '.join(trg_tokens[1:-1])


def main(sentence_to_translate):
    print("Loading vocabularies...")
    with open(config.VOCAB_ES_PATH, 'rb') as f:
        es_vocab = pickle.load(f)
    with open(config.VOCAB_EN_PATH, 'rb') as f:
        en_vocab = pickle.load(f)
        
    print("Loading model...")
    INPUT_DIM = es_vocab.n_words
    OUTPUT_DIM = en_vocab.n_words
    
    model = create_model(INPUT_DIM, OUTPUT_DIM, config, config.DEVICE)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    
    print("Model loaded successfully.\n")
    
    translation = translate_sentence(sentence_to_translate, es_vocab, en_vocab, model, config.DEVICE, config.MAX_TRANSLATION_LEN)
    
    print(f"Original (es): {sentence_to_translate}")
    print(f"Translated (en): {translation}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate a Spanish sentence to English.')
    parser.add_argument('sentence', type=str, help='The Spanish sentence to translate.')
    args = parser.parse_args()
    
    main(args.sentence)