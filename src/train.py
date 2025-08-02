import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os

# Import our custom modules
import config
# === CHANGED LINE START ===
# Explicitly import Vocabulary alongside get_loader
from dataset import get_loader, Vocabulary
# === CHANGED LINE END ===
from model import create_model

# Import for Automatic Mixed Precision (AMP)
from torch.cuda.amp import GradScaler, autocast

def train(model, iterator, optimizer, criterion, clip, scaler):
    """Performs one epoch of training using mixed precision."""
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        src, src_len, trg = batch
        src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(src, src_len, trg)
            output_dim = output.shape[-1]
            output_reshaped = output[1:].view(-1, output_dim)
            trg_reshaped = trg[1:].view(-1)
            loss = criterion(output_reshaped, trg_reshaped)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """Evaluates the model on the validation set."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, src_len, trg = batch
            src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

            with autocast():
                output = model(src, src_len, trg, teacher_forcing_ratio=0) 
                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                trg = trg[1:].view(-1)
                loss = criterion(output, trg)

            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    """Calculates the time taken for an epoch."""
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    """Main function to start the training process."""
    print("Loading data...")
    train_loader, es_vocab, en_vocab = get_loader(
        df_path=config.TRAIN_DF_PATH,
        source_vocab_path=config.VOCAB_ES_PATH,
        target_vocab_path=config.VOCAB_EN_PATH,
        batch_size=config.BATCH_SIZE,
        pad_idx=config.PAD_IDX,
        shuffle=True
    )
    
    val_loader, _, _ = get_loader(
        df_path=config.VAL_DF_PATH,
        source_vocab_path=config.VOCAB_ES_PATH,
        target_vocab_path=config.VOCAB_EN_PATH,
        batch_size=config.BATCH_SIZE,
        pad_idx=config.PAD_IDX,
        shuffle=False
    )
    print("Data loaded successfully.")

    print(f"Device: {config.DEVICE}")
    INPUT_DIM = es_vocab.n_words
    OUTPUT_DIM = en_vocab.n_words
    
    model = create_model(INPUT_DIM, OUTPUT_DIM, config, config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    
    scaler = GradScaler()
    best_valid_loss = float('inf')
    CLIP = 1

    print("\nStarting training...")
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_loader, optimizer, criterion, CLIP, scaler)
        valid_loss = evaluate(model, val_loader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
            model_path = os.path.join(config.SAVED_MODELS_DIR, 'best-nmt-model.pt')
            torch.save(model.state_dict(), model_path)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
    print("\nTraining finished.")

if __name__ == '__main__':
    main()