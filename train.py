"""
train.py
────────────────────────────────────────────────────────
Скрипт ручного / первичного обучения анти-спам модели.

Что делает:
1. Поднимает централизованный логгер (utils.logger).
2. Создаёт экземпляр `SpamClassifier`
   – если `ml/model.pkl` существует, он будет загружен;
   – если нет, модель обучится с нуля на data/messages.csv.
3. Явно вызывает `.train()` для гарантированного переобучения,
   чтобы учесть все свежие данные.
4. Выводит путь к сохранённому .pkl и базовую метрику accuracy
   (валидация train-test splitом 80/20).

Запуск:
    python train.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config.config import settings
from ml.model import SpamClassifier
from utils.logger import get_logger

LOGGER = get_logger(__name__)


def evaluate(clf: SpamClassifier, test_csv: Path) -> Tuple[int, float]:
    """
    Быстрая оценка accuracy на hold-out (20 % датасета).
    Возвращает (test_size, accuracy).
    """
    df = pd.read_csv(test_csv)
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    y_pred = clf.model.predict(X_test)  # type: ignore[attr-defined]
    acc = accuracy_score(y_test, y_pred)
    return len(X_test), acc


def main() -> None:
    LOGGER.info("▶️  Ручное обучение модели — старт")
    clf = SpamClassifier(retrain_threshold=settings.RETRAIN_THRESHOLD)

    # Всегда переобучаем, чтобы быть уверенными в актуальности
    clf.train()

    # Мини-оценка качества
    test_size, acc = evaluate(clf, clf.dataset_path)
    LOGGER.info("✅ Обучение завершено. Test size: %d, accuracy: %.4f", test_size, acc)
    LOGGER.info("📦 Модель сохранена в: %s", clf.model_path.resolve())

    print("\n=== Итог ===")
    print(f"Модель сохранена в: {clf.model_path.resolve()}")
    print(f"Test-accuracy на {test_size} примерах: {acc:.4f}")


if __name__ == "__main__":
    # Если скрипт запущен в stdout-только среде (без лог-файла),
    # убедимся, что root-логгер пишет хотя бы в консоль.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=settings.LOG_LEVEL,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    main()
