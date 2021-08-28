"""
Adapted from this blog post:
https://charon.me/posts/pytorch/pytorch_seq2seq_5
"""
import os
import time
import math
import torch
from tqdm import tqdm

from utils import epoch_time, save_model


def train_epoch(model, data_loader, optimizer, criterion, clip, epoch):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(data_loader, total=len(data_loader), desc=f"Training {epoch+1}"):
        optimizer.zero_grad()
        output, _ = model(src, tgt[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt = tgt[:,1:].contiguous().view(-1)
        loss = criterion(output, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate(model, data_loader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(data_loader, total=len(data_loader), desc="Evaluating"):
            output, _ = model(src, tgt[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[:,1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def train(model, criterion, train_dataloader, valid_dataloader, optimizer, num_epochs, clip):
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion, clip, epoch)
        valid_loss = evaluate(model, valid_dataloader, criterion)
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(model)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
