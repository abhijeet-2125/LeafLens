import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ml.training.config import (
    BEST_MODEL_PATH,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    WEIGHT_DECAY)


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def run_epoch(model, dataloader, criterion, optimizer=None):
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples
    return average_loss, accuracy

def train_model(model, train_loader, val_loader, label_to_idx):
    if DEVICE == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=2)
    best_val_accuracy = 0.0
    history = []
    for epoch in range(EPOCHS):
        train_loss, train_accuracy = run_epoch(model,train_loader,criterion,optimizer)
        val_loss, val_accuracy = run_epoch(model,val_loader,criterion)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": current_lr})

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.6f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "label_to_idx": label_to_idx,
                    "idx_to_label": {index: label for label, index in label_to_idx.items()}},BEST_MODEL_PATH)
            print("  Best model checkpoint saved.")
    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    return model, history