from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from preprocess import preprocess_data
from dataset import NSLKDDDataset
from model import MLPIntrusionDetector


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(train_loader.dataset)


def evaluate_loss(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(data_loader.dataset)


def plot_losses(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(RESULTS_DIR / "learning_curve.png", dpi=300)
    plt.close()


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    train_dataset = NSLKDDDataset(X_train, y_train)
    val_dataset = NSLKDDDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False
    )

    input_dim = X_train.shape[1]

    model = MLPIntrusionDetector(input_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

    torch.save(model.state_dict(), RESULTS_DIR / "mlp_model.pt")
    plot_losses(train_losses, val_losses)

    print("Training complete.")
    print("Saved model to results/mlp_model.pt")
    print("Saved learning curve to results/learning_curve.png")


if __name__ == "__main__":
    main()