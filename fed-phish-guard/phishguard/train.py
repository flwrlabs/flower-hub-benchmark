"""
Phishing URL Classification - Training Script
"""

from __future__ import annotations

import time

import math
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device, max_grad_norm: float = 1.0, show_progress: bool = False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = tqdm(dataloader, desc="Training", leave=False, disable=not show_progress)
    for batch in progress:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device).float()

        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, pos_weight, device, show_progress: bool = False):
    """Evaluate model on a dataset."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_probs = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False, disable=not show_progress):
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device).float()

        logits = model(input_ids).squeeze(-1)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        total_loss += float(loss.item())
        num_batches += 1
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    auc = 0.0
    if len(set(all_labels)) >= 2:
        auc = float(roc_auc_score(all_labels, all_probs))

    metrics = {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "auc": auc,
    }

    return metrics, all_labels, all_preds


def summarize_history(history):
    """Calculate average train metrics over all epochs."""
    if not history:
        raise ValueError("History is empty.")

    num_epochs = len(history)

    avg_train_loss = sum(epoch["train_loss"] for epoch in history) / num_epochs
    avg_train_acc = sum(epoch["train_accuracy"] for epoch in history) / num_epochs
    avg_train_f1 = sum(epoch["train_f1"] for epoch in history) / num_epochs

    return {
        "avg_train_loss": avg_train_loss,
        "avg_train_accuracy": avg_train_acc,
        "avg_train_f1": avg_train_f1,
    }


def train(
    model,
    train_loader,
    pos_weight,
    lr: float,
    device: torch.device,
    num_epochs: int = 20,
    weight_decay: float = 1e-4,
):
    """Training loop without local validation."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)

        start_time = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time

        train_metrics, _, _ = evaluate(model, train_loader, pos_weight, device)
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train Loss: {train_loss:.4f} | Time: {train_time:.1f}s")
        print(
            f"Train Eval Loss: {train_metrics['loss']:.4f} | "
            f"Acc: {train_metrics['accuracy']:.4f} | "
            f"Precision: {train_metrics['precision']:.4f} | "
            f"Recall: {train_metrics['recall']:.4f} | "
            f"F1: {train_metrics['f1']:.4f} | "
            f"AUC: {train_metrics['auc']:.4f}"
        )
        print(f"LR: {current_lr:.2e}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_eval_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "train_auc": train_metrics["auc"],
                "lr": current_lr,
            }
        )

    return history, num_epochs


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))
