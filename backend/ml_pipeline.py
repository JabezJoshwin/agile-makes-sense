import os

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor


DATA_PATH = "backend/data/processed/structured_dataset.csv"

MODEL_SAVE_PATH = "backend/models/"


def main():
    df = pd.read_csv(DATA_PATH)

    print("Dataset size:", len(df))

    # -----------------------------
    # Embedding Model (CPU-only for portability)
    # -----------------------------
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    print("Generating embeddings...")
    X_embeddings = embedder.encode(df["text"].tolist(), show_progress_bar=True)

    # -----------------------------
    # Category Classification
    # -----------------------------
    label_encoder = LabelEncoder()
    y_class = label_encoder.fit_transform(df["category"])

    X_train, X_test, y_train, y_test = train_test_split(
        X_embeddings, y_class, test_size=0.2, random_state=42
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss"
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # -----------------------------
    # Priority Regression
    # -----------------------------
    y_priority = df["priority_score"].values

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_embeddings, y_priority, test_size=0.2, random_state=42
    )

    reg = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="reg:squarederror"
    )

    reg.fit(X_train_r, y_train_r)

    y_pred_r = reg.predict(X_test_r)

    print("\nPriority MAE:", mean_absolute_error(y_test_r, y_pred_r))

    # -----------------------------
    # Save Models
    # -----------------------------
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    joblib.dump(embedder, MODEL_SAVE_PATH + "embedder.pkl")
    joblib.dump(clf, MODEL_SAVE_PATH + "classifier.pkl")
    joblib.dump(reg, MODEL_SAVE_PATH + "regressor.pkl")
    joblib.dump(label_encoder, MODEL_SAVE_PATH + "label_encoder.pkl")

    print("\nModels saved successfully.")


if __name__ == "__main__":
    main()