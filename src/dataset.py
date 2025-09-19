# src/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import sentencepiece as spm

from . import config

# The old Vocabulary class is no longer needed.

class NMTDataset(Dataset):
    """
    Custom Dataset for loading sentence pairs and tokenizing them
    on-the-fly using SentencePiece.
    """
    def __init__(self, df, sp_model_es, sp_model_en):
        self.df = df
        self.sp_model_es = sp_model_es
        self.sp_model_en = sp_model_en

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        source_sentence = self.df.iloc[index]['spanish_processed']
        target_sentence = self.df.iloc[index]['english_processed']

        # Numericalize using SentencePiece
        source_tensor = self.numericalize(source_sentence, self.sp_model_es)
        target_tensor = self.numericalize(target_sentence, self.sp_model_en)

        # We no longer need to return the length, as the Transformer model
        # will use a padding mask instead.
        return source_tensor, target_tensor
    
    def numericalize(self, sentence, sp_model):
        """Converts a sentence string into a tensor of subword IDs."""
        # SentencePiece already adds SOS/EOS tokens if specified during training.
        # We'll add them manually for explicit control.
        ids = [config.SOS_IDX] + sp_model.encode_as_ids(sentence) + [config.EOS_IDX]
        return torch.tensor(ids)

class Collate:
    """
    A collate function to process a batch of samples.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        source_tensors = [item[0] for item in batch]
        target_tensors = [item[1] for item in batch]

        # Pad sequences to the same length in the batch
        padded_sources = pad_sequence(source_tensors, batch_first=False, padding_value=self.pad_idx)
        padded_targets = pad_sequence(target_tensors, batch_first=False, padding_value=self.pad_idx)

        # The dataloader no longer returns lengths
        return padded_sources, None, padded_targets # Return None for src_len

def get_loader(df_path, sp_model_path_es, sp_model_path_en, batch_size, pad_idx, shuffle=True):
    """
    Creates and returns a DataLoader.
    """
    df = pd.read_pickle(df_path)
    
    # Load the trained SentencePiece models
    sp_es = spm.SentencePieceProcessor()
    sp_es.load(sp_model_path_es)
    
    sp_en = spm.SentencePieceProcessor()
    sp_en.load(sp_model_path_en)

    dataset = NMTDataset(df, sp_es, sp_en)
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=Collate(pad_idx=pad_idx)
    )
    
    # Return the loader and the loaded SentencePiece models
    return loader, sp_es, sp_en