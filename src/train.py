# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
from tqdm import tqdm

from . import config
from .dataset import get_loader
from .model import create_model, create_mask

def train_epoch(model, iterator, optimizer, criterion, clip, scaler):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(iterator, desc="Training Epoch", leave=True)

    for i, batch in enumerate(progress_bar):
        src, _, trg = batch # src_len is not used
        src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)
        
        trg_input = trg[:-1, :]
        
        _, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, trg_input, config.PAD_IDX, config.DEVICE)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16):
            output = model(src, trg_input, src_padding_mask, tgt_padding_mask, tgt_mask)
            
            output_dim = output.shape[-1]
            output_reshaped = output.reshape(-1, output_dim)
            trg_reshaped = trg[1:, :].reshape(-1)
            loss = criterion(output_reshaped, trg_reshaped)
        
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(iterator)

def evaluate_epoch(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    progress_bar = tqdm(iterator, desc="Validating Epoch", leave=True)
    
    with torch.no_grad():
        for i, batch in enumerate(progress_bar):
            src, _, trg = batch
            src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)
            trg_input = trg[:-1, :]
            _, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, trg_input, config.PAD_IDX, config.DEVICE)

            with torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16):
                output = model(src, trg_input, src_padding_mask, tgt_padding_mask, tgt_mask)
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                trg = trg[1:, :].reshape(-1)
                loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
    return epoch_loss / len(iterator)

def main():
    print("--- Starting Training Process ---")
    print(f"Loading data using batch size: {config.BATCH_SIZE}")
    train_loader, _, _ = get_loader(config.TRAIN_DF_PATH, config.SP_MODEL_PATH_ES, config.SP_MODEL_PATH_EN, config.BATCH_SIZE, config.PAD_IDX)
    val_loader, _, _ = get_loader(config.VAL_DF_PATH, config.SP_MODEL_PATH_ES, config.SP_MODEL_PATH_EN, config.BATCH_SIZE, config.PAD_IDX, shuffle=False)
    
    print(f"Device: {config.DEVICE}")
    model = create_model(config.VOCAB_SIZE, config.VOCAB_SIZE, config, config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    scaler = torch.amp.GradScaler('cuda') if config.DEVICE == 'cuda' else None
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True, factor=0.5)

    best_valid_loss = float('inf')
    print(f"\nStarting training for {config.NUM_EPOCHS} epochs...")
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config.CLIP, scaler)
        valid_loss = evaluate_epoch(model, val_loader, criterion)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) - (epoch_mins * 60))
        
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"Epoch {epoch+1:02}: Validation loss improved. Saving model to {config.BEST_MODEL_PATH}")

        print("-" * 50)
        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        print("-" * 50)

    print("\n--- Training Finished ---")

if __name__ == '__main__':
    main()









# # src/train.py

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import time
# import math
# import os
# import contextlib
# from tqdm import tqdm # Import tqdm

# # Import our custom modules
# from . import config
# from .dataset import get_loader
# from .model import create_model

# def train(model, iterator, optimizer, criterion, clip, scaler, epoch):
#     """
#     Performs one epoch of training with a progress bar.
#     """
#     model.train()
#     epoch_loss = 0
    
#     ctx = torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16) if scaler else contextlib.nullcontext()

#     # Wrap the iterator with tqdm for a progress bar
#     progress_bar = tqdm(iterator, desc=f"Training Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=True)

#     for i, batch in enumerate(progress_bar):
#         src, src_len, trg = batch
#         src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)
        
#         optimizer.zero_grad()
        
#         with ctx:
#             output = model(src, src_len, trg)
#             output_dim = output.shape[-1]
#             output_reshaped = output[1:].view(-1, output_dim)
#             trg_reshaped = trg[1:].view(-1)
#             loss = criterion(output_reshaped, trg_reshaped)
        
#         if scaler:
#             scaler.scale(loss).backward()
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#             optimizer.step()
        
#         epoch_loss += loss.item()
        
#         # Update the progress bar with the current loss
#         progress_bar.set_postfix(loss=loss.item())
        
#     return epoch_loss / len(iterator)

# def evaluate(model, iterator, criterion, epoch):
#     """Evaluates the model on the validation set with a progress bar."""
#     model.eval()
#     epoch_loss = 0
    
#     ctx = torch.amp.autocast(device_type=config.DEVICE, dtype=torch.float16) if config.DEVICE == 'cuda' else contextlib.nullcontext()

#     # Wrap the iterator with tqdm
#     progress_bar = tqdm(iterator, desc=f"Validating Epoch {epoch+1}/{config.NUM_EPOCHS}", leave=True)
    
#     with torch.no_grad():
#         for i, batch in enumerate(progress_bar):
#             src, src_len, trg = batch
#             src, trg = src.to(config.DEVICE), trg.to(config.DEVICE)

#             with ctx:
#                 output = model(src, src_len, trg, teacher_forcing_ratio=0) 
#                 output_dim = output.shape[-1]
#                 output = output[1:].view(-1, output_dim)
#                 trg = trg[1:].view(-1)
#                 loss = criterion(output, trg)

#             epoch_loss += loss.item()
#             progress_bar.set_postfix(loss=loss.item())
            
#     return epoch_loss / len(iterator)

# def epoch_time(start_time, end_time):
#     elapsed_time = end_time - start_time
#     elapsed_mins = int(elapsed_time / 60)
#     elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
#     return elapsed_mins, elapsed_secs

# def main():
#     print("Loading data...")
#     train_loader, es_vocab, en_vocab = get_loader(
#         df_path=config.TRAIN_DF_PATH,
#         source_vocab_path=config.VOCAB_ES_PATH,
#         target_vocab_path=config.VOCAB_EN_PATH,
#         batch_size=config.BATCH_SIZE,
#         pad_idx=config.PAD_IDX,
#         shuffle=True
#     )
    
#     val_loader, _, _ = get_loader(
#         df_path=config.VAL_DF_PATH,
#         source_vocab_path=config.VOCAB_ES_PATH,
#         target_vocab_path=config.VOCAB_EN_PATH,
#         batch_size=config.BATCH_SIZE,
#         pad_idx=config.PAD_IDX,
#         shuffle=False
#     )
#     print("Data loaded successfully.")

#     print(f"Device: {config.DEVICE}")
#     INPUT_DIM = es_vocab.n_words
#     OUTPUT_DIM = en_vocab.n_words
    
#     model = create_model(INPUT_DIM, OUTPUT_DIM, config, config.DEVICE)

#     optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
#     criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    
#     scaler = torch.amp.GradScaler('cuda') if config.DEVICE == 'cuda' else None
    
#     if scaler:
#         print("Automatic Mixed Precision (AMP) enabled.")
#     else:
#         print("Running on CPU. Automatic Mixed Precision (AMP) is disabled.")

#     best_valid_loss = float('inf')

#     print("\nStarting training...")
#     for epoch in range(config.NUM_EPOCHS):
#         start_time = time.time()
        
#         train_loss = train(model, train_loader, optimizer, criterion, config.CLIP, scaler, epoch)
#         valid_loss = evaluate(model, val_loader, criterion, epoch)
        
#         end_time = time.time()
        
#         epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
#         if valid_loss < best_valid_loss:
#             best_valid_loss = valid_loss
#             os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
#             torch.save(model.state_dict(), config.BEST_MODEL_PATH)
        
#         print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
#         print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
#         print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        
#     print(f"\nTraining finished. Best model saved to {config.BEST_MODEL_PATH}")

# if __name__ == '__main__':
#     main()