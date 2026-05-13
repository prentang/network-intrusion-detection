from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoencoderIntrusionDetector
from preprocess import preprocess_data


RESULTS_DIR = Path("results")


def compute_reconstruction_errors(model, data_loader, device):
    model.eval()

    errors = []

    with torch.no_grad():
        for X_batch, in data_loader:
            X_batch = X_batch.to(device)

            reconstructed = model(X_batch)
            batch_errors = torch.mean((X_batch - reconstructed) ** 2, dim=1)

            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )

    display.plot()
    plt.title("Autoencoder Confusion Matrix")
    plt.savefig(RESULTS_DIR / "autoencoder_confusion_matrix.png", dpi=300)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    val_tensor = torch.tensor(X_val, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

    val_loader = DataLoader(
        TensorDataset(val_tensor),
        batch_size=256,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(test_tensor),
        batch_size=256,
        shuffle=False
    )

    input_dim = X_train.shape[1]

    model = AutoencoderIntrusionDetector(input_dim).to(device)
    model.load_state_dict(
        torch.load(RESULTS_DIR / "autoencoder_model.pt", map_location=device)
    )

    val_errors = compute_reconstruction_errors(
        model,
        val_loader,
        device
    )

    test_errors = compute_reconstruction_errors(
        model,
        test_loader,
        device
    )

    normal_val_errors = val_errors[y_val == 0]
    threshold = np.percentile(normal_val_errors, 95)

    y_pred = (test_errors > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Attack"]
    )

    cm = confusion_matrix(y_test, y_pred)

    print("Threshold:", threshold)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print()
    print(report)
    print("Confusion Matrix:")
    print(cm)

    save_confusion_matrix(y_test, y_pred)

    with open(RESULTS_DIR / "autoencoder_metrics.txt", "w") as file:
        file.write(f"Threshold: {threshold}\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1-score: {f1}\n\n")
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    print("Saved metrics to results/autoencoder_metrics.txt")
    print("Saved confusion matrix to results/autoencoder_confusion_matrix.png")


if __name__ == "__main__":
    main()