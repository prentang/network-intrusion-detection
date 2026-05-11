from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from preprocess import preprocess_data
from dataset import NSLKDDDataset
from model import MLPIntrusionDetector


RESULTS_DIR = Path("results")


def get_predictions(model, data_loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)

            logits = model(X_batch)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).int()

            all_labels.extend(y_batch.numpy().flatten())
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_probabilities.extend(probabilities.cpu().numpy().flatten())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Attack"]
    )

    display.plot()
    plt.title("Confusion Matrix")
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    test_dataset = NSLKDDDataset(X_test, y_test)

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False
    )

    input_dim = X_train.shape[1]

    model = MLPIntrusionDetector(input_dim).to(device)
    model.load_state_dict(
        torch.load(RESULTS_DIR / "mlp_model.pt", map_location=device)
    )

    y_true, y_pred, y_prob = get_predictions(
        model,
        test_loader,
        device
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["Normal", "Attack"]
    )

    cm = confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print()
    print(report)
    print("Confusion Matrix:")
    print(cm)

    save_confusion_matrix(y_true, y_pred)

    with open(RESULTS_DIR / "metrics.txt", "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1-score: {f1}\n\n")
        file.write("Classification Report:\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    print("Saved metrics to results/metrics.txt")
    print("Saved confusion matrix to results/confusion_matrix.png")


if __name__ == "__main__":
    main()