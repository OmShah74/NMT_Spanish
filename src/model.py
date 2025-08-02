import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F

# --- 1. The Enhanced Bidirectional Encoder ---
class Encoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: Tensor, src_len: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Processes the source sentence.
        
        Args:
            src (Tensor): The source sentence tensor. Shape: [src_len, batch_size]
            src_len (Tensor): The lengths of the source sentences. Shape: [batch_size]
            
        Returns:
            outputs (Tensor): The top-layer hidden state from both directions.
                              Shape: [src_len, batch_size, hidden_dim * 2]
            hidden (Tensor): The final hidden state, transformed for the decoder.
                             Shape: [num_layers, batch_size, hidden_dim]
        """
        # src shape: [src_len, batch_size]
        embedded = self.dropout(self.embedding(src))
        # embedded shape: [src_len, batch_size, embed_dim]
        
        # Pack sequence for more efficient processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'), enforce_sorted=False)
        
        packed_outputs, hidden = self.rnn(packed_embedded)
        
        # Unpack sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)
        # outputs shape: [src_len, batch_size, hidden_dim * 2]
        # hidden shape: [num_layers * 2, batch_size, hidden_dim]

        # The hidden state needs to be transformed to be used by the unidirectional decoder.
        # We concatenate the final forward and backward hidden states of each layer.
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # hidden shape: [batch_size, hidden_dim]
        # We need to expand it to match the number of layers in the decoder
        hidden = hidden.unsqueeze(0).repeat(self.num_layers, 1, 1)
        # hidden shape: [num_layers, batch_size, hidden_dim]

        return outputs, hidden

# --- 2. The Enhanced Attention Mechanism ---
class Attention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear((hidden_dim * 2) + hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden: Tensor, encoder_outputs: Tensor) -> Tensor:
        """
        Calculates attention weights.
        
        Args:
            hidden (Tensor): The previous hidden state of the decoder. Shape: [1, batch_size, hidden_dim]
            encoder_outputs (Tensor): Outputs from the bidirectional encoder. Shape: [src_len, batch_size, hidden_dim * 2]
        
        Returns:
            attention_weights (Tensor): Shape: [batch_size, src_len]
        """
        src_len = encoder_outputs.shape[0]
        
        # Repeat the decoder hidden state src_len times
        hidden = hidden[[-1]].repeat(src_len, 1, 1)
        # hidden shape: [src_len, batch_size, hidden_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        hidden = hidden.permute(1, 0, 2)
        # encoder_outputs shape: [batch_size, src_len, hidden_dim * 2]
        # hidden shape: [batch_size, src_len, hidden_dim]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy shape: [batch_size, src_len, hidden_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention shape: [batch_size, src_len]
        
        return F.softmax(attention, dim=1)

# --- 3. The Enhanced Decoder ---
class Decoder(nn.Module):
    def __init__(self, output_dim: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float, attention: Attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.rnn = nn.GRU((hidden_dim * 2) + embed_dim, hidden_dim, num_layers, dropout=dropout)
        self.fc_out = nn.Linear((hidden_dim * 2) + hidden_dim + embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input: Tensor, hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Performs one decoding step.
        
        Args:
            input (Tensor): Input token. Shape: [batch_size]
            hidden (Tensor): Previous hidden state. Shape: [num_layers, batch_size, hidden_dim]
            encoder_outputs (Tensor): Outputs from the encoder. Shape: [src_len, batch_size, hidden_dim * 2]
            
        Returns:
            prediction (Tensor): Raw output scores. Shape: [batch_size, output_dim]
            hidden (Tensor): Updated hidden state. Shape: [num_layers, batch_size, hidden_dim]
        """
        input = input.unsqueeze(0)
        # input shape: [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded shape: [1, batch_size, embed_dim]
        
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        # a shape: [batch_size, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs shape: [batch_size, src_len, hidden_dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        # weighted shape: [batch_size, 1, hidden_dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        # weighted shape: [1, batch_size, hidden_dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input shape: [1, batch_size, (hidden_dim * 2) + embed_dim]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output shape: [1, batch_size, hidden_dim]
        # hidden shape: [num_layers, batch_size, hidden_dim]
        
        # Prediction now uses the embedded word, the RNN output, and the weighted context
        prediction = self.fc_out(torch.cat((output.squeeze(0), weighted.squeeze(0), embedded.squeeze(0)), dim=1))
        # prediction shape: [batch_size, output_dim]
        
        return prediction, hidden

# --- 4. The Seq2Seq Wrapper (Now handles source lengths) ---
class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src: Tensor, src_len: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        encoder_outputs, hidden = self.encoder(src, src_len)
        
        input = trg[0, :] # <SOS> token
        
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            
        return outputs

# --- Helper function for model creation ---
def create_model(input_dim, output_dim, config, device):
    enc = Encoder(input_dim, config.EMBED_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT)
    attn = Attention(config.HIDDEN_SIZE)
    dec = Decoder(output_dim, config.EMBED_SIZE, config.HIDDEN_SIZE, config.NUM_LAYERS, config.DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01)
            else:
                nn.init.constant_(param.data, 0)
                
    model.apply(init_weights)
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    return model

# --- Example of how to use it (for testing) ---
if __name__ == '__main__':
    import config

    INPUT_DIM = 1000
    OUTPUT_DIM = 1200
    
    model = create_model(INPUT_DIM, OUTPUT_DIM, config, config.DEVICE)
    
    # Create dummy data
    src_len_val, trg_len_val, batch_size = 15, 17, config.BATCH_SIZE
    source_tensor = torch.randint(1, INPUT_DIM, (src_len_val, batch_size)).to(config.DEVICE) # Avoid padding token
    # Create dummy source lengths (must be <= src_len_val)
    source_lengths = torch.randint(5, src_len_val + 1, (batch_size,)).sort(descending=True).values
    
    target_tensor = torch.randint(1, OUTPUT_DIM, (trg_len_val, batch_size)).to(config.DEVICE)
    
    # Run a forward pass
    output = model(source_tensor, source_lengths, target_tensor)
    
    print("\n--- Model Test ---")
    print(f"Source tensor shape: {source_tensor.shape}")
    print(f"Source lengths shape: {source_lengths.shape}")
    print(f"Target tensor shape: {target_tensor.shape}")
    print(f"Model output shape: {output.shape}")
    print("------------------")
    
    assert output.shape == (trg_len_val, batch_size, OUTPUT_DIM), "Output shape is incorrect!"
    print("\nModel test passed! The output shape is correct.")