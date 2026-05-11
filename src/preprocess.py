from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from columns import NSL_KDD_COLUMNS


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_nsl_kdd():
    train_path = DATA_DIR / "KDDTrain+.txt"
    test_path = DATA_DIR / "KDDTest+.txt"

    train_df = pd.read_csv(train_path, names=NSL_KDD_COLUMNS)
    test_df = pd.read_csv(test_path, names=NSL_KDD_COLUMNS)

    return train_df, test_df


def convert_to_binary_label(df):
    df = df.copy()

    # normal = 0
    # anything else = attack = 1
    df["binary_label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

    return df


def preprocess_data():
    train_df, test_df = load_nsl_kdd()

    train_df = convert_to_binary_label(train_df)
    test_df = convert_to_binary_label(test_df)

    feature_cols = [
        col for col in NSL_KDD_COLUMNS
        if col not in ["label", "difficulty"]
    ]

    categorical_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [
        col for col in feature_cols
        if col not in categorical_cols
    ]

    X_train_full = train_df[feature_cols]
    y_train_full = train_df["binary_label"]

    X_test = test_df[feature_cols]
    y_test = test_df["binary_label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )

    try:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_cols),
            ("categorical", one_hot_encoder, categorical_cols),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    joblib.dump(preprocessor, RESULTS_DIR / "preprocessor.joblib")

    return (
        X_train_processed,
        X_val_processed,
        X_test_processed,
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
    )


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    print("Preprocessing complete.")
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)