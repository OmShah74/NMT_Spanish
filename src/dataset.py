# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd
# No need to import config here, pad_idx is passed directly

# This is the single, canonical definition of the Vocabulary class.
class Vocabulary:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.word2count = {}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

class NMTDataset(Dataset):
    """Custom Dataset for loading Spanish-English sentence pairs."""
    def __init__(self, df, source_vocab, target_vocab):
        self.df = df
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_sentence = self.df.iloc[index]['spanish_normalized']
        target_sentence = self.df.iloc[index]['english_normalized']

        source_tensor, source_len = self.numericalize(source_sentence, self.source_vocab)
        target_tensor, _ = self.numericalize(target_sentence, self.target_vocab)

        return source_tensor, source_len, target_tensor
    
    def numericalize(self, sentence, vocab):
        """Converts a sentence string into a tensor and returns its length."""
        tokens = [vocab.word2index['<SOS>']]
        for word in sentence.split(' '):
            tokens.append(vocab.word2index.get(word, vocab.word2index['<UNK>']))
        tokens.append(vocab.word2index['<EOS>'])
        
        return torch.tensor(tokens), len(tokens)

class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        source_tensors = [item[0] for item in batch]
        source_lengths = torch.tensor([item[1] for item in batch], dtype=torch.int64)
        target_tensors = [item[2] for item in batch]

        # batch_first=False is the expected format for our model
        padded_sources = pad_sequence(source_tensors, batch_first=False, padding_value=self.pad_idx)
        padded_targets = pad_sequence(target_tensors, batch_first=False, padding_value=self.pad_idx)

        return padded_sources, source_lengths, padded_targets

def get_loader(df_path, source_vocab_path, target_vocab_path, batch_size, pad_idx, shuffle=True):
    df = pd.read_pickle(df_path)
    with open(source_vocab_path, 'rb') as f:
        source_vocab = pickle.load(f)
    with open(target_vocab_path, 'rb') as f:
        target_vocab = pickle.load(f)

    dataset = NMTDataset(df, source_vocab, target_vocab)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collate(pad_idx=pad_idx)
    )
    
    return loader, source_vocab, target_vocab