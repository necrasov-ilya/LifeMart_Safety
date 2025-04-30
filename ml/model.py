"""
ml/model.py
────────────────────────────────────────────────────────
Machine-learning модуль антиспам-бота.

Содержит класс `SpamClassifier`, который:
• Загружает / обучает модель (TF-IDF + MultinomialNB).
• Делает прогнозы спам/не-спам.
• Умеет пополнять датасет и переобучаться по порогу.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# ───────────────────────────────
# Пути по умолчанию (переопределяются при инициализации).
# ───────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = ROOT_DIR / "data" / "messages.csv"
DEFAULT_MODEL_PATH = ROOT_DIR / "ml" / "model.pkl"

LOGGER = logging.getLogger("SpamClassifier")


class SpamClassifier:
    """
    Класс-обёртка над sklearn-пайплайном TF-IDF → MultinomialNB.

    Parameters
    ----------
    dataset_path : str | Path
        CSV-файл с колонками ``message`` и ``label`` (0 – не спам, 1 – спам).
    model_path : str | Path
        Файл, куда сериализуется обученная модель (joblib).
    retrain_threshold : int
        Сколько новых размеченных примеров нужно накопить,
        прежде чем автоматически переобучить модель.
    """

    def __init__(
        self,
        dataset_path: str | Path = DEFAULT_DATASET_PATH,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        retrain_threshold: int | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.retrain_threshold: int = int(
            retrain_threshold or os.getenv("RETRAIN_THRESHOLD", 100)
        )
        self._new_samples: int = 0  # счётчик добавленных примеров
        self.model: Pipeline | None = None

        self._ensure_dataset_exists()
        self._load_or_train()

    # ───────────────────────────────
    # Публичный API
    # ───────────────────────────────
    def predict(self, text: str) -> int:
        """Вернуть 1, если спам, иначе 0."""
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        return int(self.model.predict([text])[0])

    def update_dataset(self, message: str, label: int, *, autosave: bool = True) -> None:
        """
        Добавить новый пример и, при необходимости, переобучить модель.

        • Не добавляет дубликаты (message+label уникальны).
        • После ``retrain_threshold`` новых записей модель переобучается.
        """
        label = int(label)
        if label not in (0, 1):
            raise ValueError("Label must be 0 or 1.")

        df = pd.read_csv(self.dataset_path)
        duplicate_mask = (df["message"] == message) & (df["label"] == label)
        if duplicate_mask.any():
            LOGGER.debug("Sample already exists in dataset, skipping append.")
            return

        df.loc[len(df)] = [message, label]  # type: ignore[index]
        if autosave:
            df.to_csv(self.dataset_path, index=False)
            LOGGER.info("Added new sample to dataset: %.60s… | label=%d", message, label)

        self._new_samples += 1
        if self._new_samples >= self.retrain_threshold:
            LOGGER.info(
                "Retrain threshold (%d) reached – starting retraining.",
                self.retrain_threshold,
            )
            self.train()
            self._new_samples = 0

    def train(self) -> None:
        """Полное переобучение модели на актуальном датасете."""
        df = pd.read_csv(self.dataset_path)
        X, y = df["message"], df["label"]

        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(stop_words=None)),
                ("clf", MultinomialNB()),
            ]
        )

        LOGGER.info("Training model on %d samples…", len(df))
        pipeline.fit(X, y)
        self.model = pipeline
        dump(pipeline, self.model_path)
        LOGGER.info("Model saved to %s", self.model_path.as_posix())

    # ───────────────────────────────
    # Служебные методы
    # ───────────────────────────────
    def _load_or_train(self) -> None:
        if self.model_path.exists():
            LOGGER.info("Loading model from %s", self.model_path.as_posix())
            self.model = load(self.model_path)
        else:
            LOGGER.warning("Model file not found – training from scratch.")
            self.train()

    def _ensure_dataset_exists(self) -> None:
        """Создаёт пустой датасет, если файла ещё нет."""
        if not self.dataset_path.exists():
            LOGGER.warning(
                "Dataset file %s not found – creating empty template.",
                self.dataset_path.as_posix(),
            )
            self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=["message", "label"]).to_csv(self.dataset_path, index=False)

    # ───────────────────────────────
    # Утилиты
    # ───────────────────────────────
    def batch_predict(self, texts: Iterable[str]) -> list[int]:
        """Неблокирующий прогноз списка сообщений (для возможного батчинга)."""
        if not self.model:
            raise RuntimeError("Model is not loaded.")
        return list(map(int, self.model.predict(list(texts))))


# ───────────────────────────────
# Локальный тест (python -m ml.model)
# ───────────────────────────────
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    clf = SpamClassifier()
    sample = "Зарабатывай от 500 $ в день! Пиши «+»."
    print(f'"{sample}" →', clf.predict(sample))
