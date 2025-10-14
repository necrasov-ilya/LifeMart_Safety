"""
scripts/train_meta.py
────────────────────────────────────────────────────────
Обучение мета-классификатора на базе логистической регрессии.

Пайплайн:
1. Чистка датасета (дубликаты, конфликты)
2. Получение эмбеддингов через Ollama
3. Вычисление центроидов spam/ham
4. Генерация фичей
5. Обучение LogisticRegression + калибровка
6. Сохранение артефактов в models/
7. Отчет с метриками
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    classification_report,
    confusion_matrix,
    brier_score_loss
)
from joblib import dump

from config.config import settings
from utils.logger import get_logger
from utils.textprep import prepare_for_embedding, extract_patterns_from_raw

LOGGER = get_logger(__name__)


async def get_embeddings_batch(texts: list[str], model: str, base_url: str) -> list[np.ndarray]:
    """Получает эмбеддинги для батча текстов через Ollama."""
    try:
        import httpx
    except ImportError:
        LOGGER.error("httpx not installed. Install with: pip install httpx")
        return []
    
    embeddings = []
    total = len(texts)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, text in enumerate(texts, 1):
            try:
                # Нормализация текста
                normalized = prepare_for_embedding(text)
                
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": normalized}
                )
                
                if response.status_code == 200:
                    embedding = np.array(response.json()["embedding"])
                    embeddings.append(embedding)
                    
                    if i % 10 == 0:
                        LOGGER.info(f"Получено эмбеддингов: {i}/{total}")
                else:
                    LOGGER.error(f"Ollama API error for text {i}: {response.status_code}")
                    embeddings.append(None)
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            except Exception as e:
                LOGGER.error(f"Failed to get embedding for text {i}: {e}")
                embeddings.append(None)
    
    return embeddings


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Чистка датасета: дубликаты, конфликты, валидация."""
    LOGGER.info(f"Dataset shape before cleaning: {df.shape}")
    
    # Удаляем пустые сообщения
    df = df[df['message'].notna() & (df['message'].str.strip() != '')]
    LOGGER.info(f"After removing empty messages: {df.shape}")
    
    # Проверяем конфликтные лейблы (одно сообщение с разными метками)
    duplicates = df.groupby('message')['label'].nunique()
    conflicts = duplicates[duplicates > 1]
    
    if len(conflicts) > 0:
        LOGGER.warning(f"Found {len(conflicts)} conflicting labels:")
        for msg, _ in conflicts.head(5).items():
            LOGGER.warning(f"  '{msg[:50]}...'")
        
        # Удаляем конфликтные сообщения
        df = df[~df['message'].isin(conflicts.index)]
        LOGGER.info(f"After removing conflicts: {df.shape}")
    
    # Удаляем точные дубликаты
    df = df.drop_duplicates(subset=['message', 'label'])
    LOGGER.info(f"After removing duplicates: {df.shape}")
    
    # Проверяем баланс классов
    spam_count = (df['label'] == 1).sum()
    ham_count = (df['label'] == 0).sum()
    LOGGER.info(f"Class balance: spam={spam_count}, ham={ham_count} ({spam_count/(spam_count+ham_count):.1%} spam)")
    
    if spam_count < 10 or ham_count < 10:
        raise ValueError(f"Insufficient data: need at least 10 examples per class. Got spam={spam_count}, ham={ham_count}")
    
    return df


def build_feature_matrix(
    df: pd.DataFrame,
    embeddings: list[np.ndarray],
    spam_centroid: np.ndarray,
    ham_centroid: np.ndarray,
    kw_scores: np.ndarray,
    tfidf_scores: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    """Строит матрицу фичей для обучения."""
    features_list = []
    feature_names = []
    
    # Добавляем названия фичей
    feature_names.extend(['sim_spam', 'sim_ham', 'sim_diff', 'kw_score', 'tfidf_score'])
    
    for i, (_, row) in enumerate(df.iterrows()):
        text = row['message']
        emb = embeddings[i]
        
        row_features = []
        
        # 1. Косинусные дистанции
        if emb is not None:
            sim_spam = float(np.dot(emb, spam_centroid) / (np.linalg.norm(emb) * np.linalg.norm(spam_centroid)))
            sim_ham = float(np.dot(emb, ham_centroid) / (np.linalg.norm(emb) * np.linalg.norm(ham_centroid)))
            sim_diff = sim_spam - sim_ham
        else:
            sim_spam = sim_ham = sim_diff = 0.0
        
        row_features.extend([sim_spam, sim_ham, sim_diff])
        
        # 2. Scores от фильтров
        row_features.append(kw_scores[i])
        row_features.append(tfidf_scores[i])
        
        # 3. Паттерны
        patterns = extract_patterns_from_raw(text)
        row_features.extend([
            float(patterns['has_money']),
            float(patterns['money_count']),
            float(patterns['has_age']),
            float(patterns['has_cta_plus']),
            float(patterns['has_dm']),
            float(patterns['has_contact']),
            float(patterns['has_remote']),
            float(patterns['has_legal']),
            float(patterns['has_casino']),
            patterns['obfuscation_ratio'],
        ])
        
        features_list.append(row_features)
    
    # Добавляем названия паттерн-фичей (только один раз)
    if len(feature_names) == 5:
        feature_names.extend([
            'has_money', 'money_count', 'has_age', 'has_cta_plus', 'has_dm',
            'has_contact', 'has_remote', 'has_legal', 'has_casino', 'obfuscation_ratio'
        ])
    
    return np.array(features_list), feature_names


async def main():
    LOGGER.info("="*60)
    LOGGER.info("TRAINING META CLASSIFIER")
    LOGGER.info("="*60)
    
    # Пути
    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "data" / "messages.csv"
    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Проверка датасета
    if not dataset_path.exists():
        LOGGER.error(f"Dataset not found: {dataset_path}")
        return
    
    # 1. Загрузка и чистка датасета
    LOGGER.info("\n[1/7] Loading and cleaning dataset...")
    df = pd.read_csv(dataset_path)
    df = clean_dataset(df)
    
    # 2. Стратифицированный сплит
    LOGGER.info("\n[2/7] Splitting dataset...")
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    LOGGER.info(f"Train: {len(train_df)}, Val: {len(val_df)}")
    
    # 3. Получение эмбеддингов
    LOGGER.info("\n[3/7] Getting embeddings from Ollama...")
    model = settings.EMBEDDING_MODEL_ID or settings.OLLAMA_MODEL or "nomic-embed-text"
    base_url = settings.OLLAMA_BASE_URL or "http://localhost:11434"
    
    LOGGER.info(f"Model: {model}, URL: {base_url}")
    
    train_texts = train_df['message'].tolist()
    val_texts = val_df['message'].tolist()
    
    train_embeddings = await get_embeddings_batch(train_texts, model, base_url)
    val_embeddings = await get_embeddings_batch(val_texts, model, base_url)
    
    # Фильтруем строки с failed embeddings
    train_valid_mask = [e is not None for e in train_embeddings]
    val_valid_mask = [e is not None for e in val_embeddings]
    
    if not all(train_valid_mask):
        LOGGER.warning(f"Removing {sum(not m for m in train_valid_mask)} train samples with failed embeddings")
        train_df = train_df[train_valid_mask].reset_index(drop=True)
        train_embeddings = [e for e in train_embeddings if e is not None]
    
    if not all(val_valid_mask):
        LOGGER.warning(f"Removing {sum(not m for m in val_valid_mask)} val samples with failed embeddings")
        val_df = val_df[val_valid_mask].reset_index(drop=True)
        val_embeddings = [e for e in val_embeddings if e is not None]
    
    # 4. Вычисление центроидов
    LOGGER.info("\n[4/7] Computing centroids...")
    train_spam_embeddings = [train_embeddings[i] for i in range(len(train_df)) if train_df.iloc[i]['label'] == 1]
    train_ham_embeddings = [train_embeddings[i] for i in range(len(train_df)) if train_df.iloc[i]['label'] == 0]
    
    spam_centroid = np.mean(train_spam_embeddings, axis=0)
    ham_centroid = np.mean(train_ham_embeddings, axis=0)
    
    LOGGER.info(f"Spam centroid shape: {spam_centroid.shape}")
    LOGGER.info(f"Ham centroid shape: {ham_centroid.shape}")
    
    # Сохранение центроидов
    centroids_path = models_dir / "centroids.npz"
    np.savez(centroids_path, spam_centroid=spam_centroid, ham_centroid=ham_centroid)
    LOGGER.info(f"Saved centroids to {centroids_path}")
    
    # 5. Генерация фичей (используем dummy scores для Keyword/TF-IDF)
    LOGGER.info("\n[5/7] Building feature matrices...")
    
    # Dummy scores (в реальности нужно было бы прогнать через фильтры)
    train_kw_scores = np.zeros(len(train_df))
    train_tfidf_scores = np.zeros(len(train_df))
    val_kw_scores = np.zeros(len(val_df))
    val_tfidf_scores = np.zeros(len(val_df))
    
    X_train, feature_names = build_feature_matrix(
        train_df, train_embeddings, spam_centroid, ham_centroid,
        train_kw_scores, train_tfidf_scores
    )
    
    X_val, _ = build_feature_matrix(
        val_df, val_embeddings, spam_centroid, ham_centroid,
        val_kw_scores, val_tfidf_scores
    )
    
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    
    LOGGER.info(f"X_train shape: {X_train.shape}")
    LOGGER.info(f"X_val shape: {X_val.shape}")
    LOGGER.info(f"Features: {len(feature_names)}")
    
    # 6. Обучение LogisticRegression
    LOGGER.info("\n[6/7] Training Logistic Regression...")
    
    logreg = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    
    logreg.fit(X_train, y_train)
    LOGGER.info("LogisticRegression trained")
    
    # Калибровка вероятностей
    LOGGER.info("Calibrating probabilities...")
    calibrator = CalibratedClassifierCV(logreg, method='isotonic', cv='prefit')
    calibrator.fit(X_val, y_val)
    LOGGER.info("Calibrator trained")
    
    # 7. Метрики
    LOGGER.info("\n[7/7] Evaluating metrics...")
    
    # Предсказания
    y_pred_proba = calibrator.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    LOGGER.info(f"ROC-AUC: {roc_auc:.4f}")
    
    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall, precision)
    LOGGER.info(f"PR-AUC: {pr_auc:.4f}")
    
    # Brier Score
    brier = brier_score_loss(y_val, y_pred_proba)
    LOGGER.info(f"Brier Score: {brier:.4f}")
    
    # Classification Report
    LOGGER.info("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Ham', 'Spam']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    LOGGER.info(f"\nConfusion Matrix:\n{cm}")
    
    # Precision @ high threshold
    high_threshold = 0.85
    y_pred_high = (y_pred_proba >= high_threshold).astype(int)
    if y_pred_high.sum() > 0:
        precision_high = (y_pred_high & y_val).sum() / y_pred_high.sum()
        LOGGER.info(f"Precision @ {high_threshold}: {precision_high:.2%}")
    
    # 8. Сохранение артефактов
    LOGGER.info("\nSaving artifacts...")
    
    # Модель
    logreg_path = models_dir / "meta_logreg.joblib"
    dump(logreg, logreg_path)
    LOGGER.info(f"Saved LogisticRegression to {logreg_path}")
    
    # Калибратор
    calibrator_path = models_dir / "meta_calibrator.joblib"
    dump(calibrator, calibrator_path)
    LOGGER.info(f"Saved calibrator to {calibrator_path}")
    
    # Feature spec
    feature_spec_path = models_dir / "feature_spec.json"
    with open(feature_spec_path, 'w') as f:
        json.dump({'features': feature_names}, f, indent=2)
    LOGGER.info(f"Saved feature spec to {feature_spec_path}")
    
    # Отчет
    report_path = models_dir / "train_meta_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("META CLASSIFIER TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Train samples: {len(train_df)}\n")
        f.write(f"Val samples: {len(val_df)}\n")
        f.write(f"Features: {len(feature_names)}\n\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n")
        f.write(f"PR-AUC: {pr_auc:.4f}\n")
        f.write(f"Brier Score: {brier:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_val, y_pred, target_names=['Ham', 'Spam']))
        f.write(f"\nConfusion Matrix:\n{cm}\n\n")
        f.write(f"Suggested thresholds:\n")
        f.write(f"  META_THRESHOLD_HIGH: 0.85 (precision-focused)\n")
        f.write(f"  META_THRESHOLD_MEDIUM: 0.65 (balanced)\n")
    
    LOGGER.info(f"Saved report to {report_path}")
    
    LOGGER.info("\n" + "="*60)
    LOGGER.info("TRAINING COMPLETE!")
    LOGGER.info("="*60)
    LOGGER.info(f"\nArtifacts saved to: {models_dir}")
    LOGGER.info("\nTo use the meta classifier:")
    LOGGER.info("  1. Set USE_META_CLASSIFIER=True in .env")
    LOGGER.info("  2. Restart the bot")
    LOGGER.info("  3. Use /meta_info to verify")


if __name__ == "__main__":
    asyncio.run(main())
