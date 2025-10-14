from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from filters.tfidf import TfidfFilter
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def main():
    LOGGER.info("▶️  Обучение TF-IDF модели")
    
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "messages.csv"
    model_path = Path(__file__).resolve().parents[1] / "models" / "tfidf_model.pkl"
    
    tfidf_filter = TfidfFilter(
        model_path=model_path,
        dataset_path=dataset_path
    )
    
    tfidf_filter.train()
    
    if not dataset_path.exists():
        LOGGER.error("Dataset not found")
        return
    
    df = pd.read_csv(dataset_path)
    if len(df) < 10:
        LOGGER.warning("Dataset too small for evaluation")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"],
        test_size=0.3,
        random_state=42,
        stratify=df["label"] if len(df) > 20 else None
    )
    
    y_pred = tfidf_filter.model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    LOGGER.info(f"✅ Accuracy: {accuracy:.4f}")
    
    print("\n" + "="*50)
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path} ({len(df)} samples)")
    print(f"Test size: {len(X_test)} samples")
    print(f"Accuracy: {accuracy:.4f}")
    print("="*50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))


if __name__ == "__main__":
    main()
