import logging
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tqdm import tqdm

from src.configs import PROJECT_PATH
from src.eval_lstm import evaluate_rouge


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: AutoTokenizer,
    device: torch.device,
    epochs: int = 3,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    pad_token_id: int = 0,
    save_dir: Union[str, Path] = PROJECT_PATH / "models"
) -> Tuple[nn.Module, str]:       
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    model = model.to(device)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_rouge = 0.0
    best_model_path = save_dir / "best_model.pt"

    train_losses = []
    val_losses = []
    val_rouges = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)

            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                logits, _ = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        torch.cuda.empty_cache()

        val_rouge_score = evaluate_rouge(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device
        )
        v = val_rouge_score["rougeL"]
        val_rouge_l = v.mid.fmeasure if hasattr(v, "mid") else float(v)
        val_rouges.append(val_rouge_l)

        torch.cuda.empty_cache()

        if val_rouge_l > best_rouge:
            best_rouge = val_rouge_l
            torch.save(model.state_dict(), best_model_path)

        epoch_path = save_dir / f"model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), epoch_path)

    plt.figure(figsize=(15, 5))
    epochs_range = range(1, epochs + 1)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs_range, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_rouges, 'r-', label='Val ROUGE-L')
    plt.xlabel('Epoch')
    plt.ylabel('ROUGE-L')
    plt.title('Validation ROUGE-L')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    return model, best_model_path