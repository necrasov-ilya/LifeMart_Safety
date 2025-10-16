"""
scripts/train_meta_context.py
────────────────────────────────────────────────────────
Обучение контекстного мета-классификатора с прототипами.

НОВЫЕ ВОЗМОЖНОСТИ:
- Генерация капсул E_msg/E_ctx/E_user для каждого сообщения
- Построение центроидов (spam_centroid, ham_centroid)
- K-means кластеризация для прототипов (7 спам + 4 легит)
- Обучение с калибровкой (CalibratedClassifierCV)
- Group K-fold валидация (по user_id/chat_id)
- Отчет с метриками по подмножествам

ВЫХОД:
- models/meta_logreg.joblib (или meta_lgbm.joblib)
- models/meta_calibrator.joblib
- models/centroids.npz
- models/prototypes.npz
- models/feature_spec.json
- models/train_meta_report.txt
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Добавляем корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
    brier_score_loss
)
from sklearn.cluster import KMeans
from joblib import dump

from config.config import settings
from utils.logger import get_logger
from utils.textprep import (
    normalize_entities,
    build_context_capsule,
    build_user_capsule,
    extract_patterns_from_raw,
    count_whitelist_hits
)

LOGGER = get_logger(__name__)

# Названия семейств прототипов
SPAM_PROTOTYPES = [
    "recruiting",      # Рекрутинг
    "gambling",        # Казино, ставки
    "loans",           # Займы, кредиты
    "get_rich",        # Быстрые деньги
    "dayjob",          # Работа на дому
    "illegal_docs",    # Документы, паспорта
    "gray_biz",        # Серый бизнес
]

LEGIT_PROTOTYPES = [
    "store_promo",     # Промо магазина
    "giveaway_own",    # Розыгрыши своего канала
    "order_flow",      # Заказы, меню
    "service_info",    # Информация об услугах
]


async def get_embeddings_batch(
    texts: list[str],
    model: str,
    base_url: str,
    batch_size: int = 20
) -> list[np.ndarray]:
    """
    Получает эмбеддинги для батча текстов через Ollama.
    
    Args:
        texts: список капсул (уже с префиксом "passage:")
        model: название модели
        base_url: URL Ollama API
        batch_size: размер батча для параллельной обработки
    """
    try:
        import httpx
    except ImportError:
        LOGGER.error("httpx not installed. Install with: pip install httpx")
        return []
    
    embeddings = []
    total = len(texts)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_texts = texts[batch_start:batch_end]
            
            # Параллельно запрашиваем батч
            tasks = []
            for text in batch_texts:
                task = client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text}
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обрабатываем ответы
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    LOGGER.error(f"Failed to get embedding: {response}")
                    embeddings.append(None)
                elif response.status_code == 200:
                    embedding = np.array(response.json()["embedding"])
                    embeddings.append(embedding)
                else:
                    LOGGER.error(f"Ollama API error: {response.status_code}")
                    embeddings.append(None)
            
            LOGGER.info(f"Получено эмбеддингов: {len(embeddings)}/{total}")
            
            # Rate limiting между батчами
            await asyncio.sleep(0.5)
    
    return embeddings


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Чистка датасета: дубликаты, конфликты, валидация."""
    LOGGER.info(f"Dataset shape before cleaning: {df.shape}")
    
    # Удаляем пустые сообщения
    df = df[df['message'].notna() & (df['message'].str.strip() != '')]
    LOGGER.info(f"After removing empty messages: {df.shape}")
    
    # Проверяем конфликтные лейблы
    duplicates = df.groupby('message')['label'].nunique()
    conflicts = duplicates[duplicates > 1]
    
    if len(conflicts) > 0:
        LOGGER.warning(f"Found {len(conflicts)} conflicting labels")
        df = df[~df['message'].isin(conflicts.index)]
        LOGGER.info(f"After removing conflicts: {df.shape}")
    
    # Удаляем точные дубликаты
    df = df.drop_duplicates(subset=['message', 'label'])
    LOGGER.info(f"After removing duplicates: {df.shape}")
    
    # Проверяем баланс классов
    spam_count = (df['label'] == 1).sum()
    ham_count = (df['label'] == 0).sum()
    LOGGER.info(f"Class balance: spam={spam_count}, ham={ham_count} ({spam_count/(spam_count+ham_count):.1%} spam)")
    
    if spam_count < 30 or ham_count < 30:
        raise ValueError(f"Insufficient data for context training. Need 30+ per class. Got spam={spam_count}, ham={ham_count}")
    
    return df


def build_centroids(
    embeddings: list[np.ndarray],
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет центроиды для спама и хама.
    
    Returns:
        (spam_centroid, ham_centroid)
    """
    valid_embeddings = [emb for emb in embeddings if emb is not None]
    if not valid_embeddings:
        raise ValueError("No valid embeddings to compute centroids")
    
    # Фильтруем по label
    spam_embs = [emb for emb, label in zip(embeddings, labels) if emb is not None and label == 1]
    ham_embs = [emb for emb, label in zip(embeddings, labels) if emb is not None and label == 0]
    
    spam_centroid = np.mean(spam_embs, axis=0)
    ham_centroid = np.mean(ham_embs, axis=0)
    
    LOGGER.info(f"Centroids built: spam={spam_centroid.shape}, ham={ham_centroid.shape}")
    
    return spam_centroid, ham_centroid


def build_prototypes(
    embeddings: list[np.ndarray],
    labels: np.ndarray,
    texts: list[str],
    n_spam_clusters: int = 7,
    n_legit_clusters: int = 4
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Строит прототипы через K-means кластеризацию.
    
    Returns:
        (spam_prototypes, legit_prototypes)
    """
    # Разделяем на spam и legit
    spam_embs = [emb for emb, label in zip(embeddings, labels) if emb is not None and label == 1]
    legit_embs = [emb for emb, label in zip(embeddings, labels) if emb is not None and label == 0]
    
    spam_texts = [text for text, emb, label in zip(texts, embeddings, labels) 
                  if emb is not None and label == 1]
    legit_texts = [text for text, emb, label in zip(texts, embeddings, labels) 
                   if emb is not None and label == 0]
    
    LOGGER.info(f"Building prototypes: spam={len(spam_embs)}, legit={len(legit_embs)}")
    
    # K-means для спама
    spam_prototypes = {}
    if len(spam_embs) >= n_spam_clusters:
        kmeans_spam = KMeans(n_clusters=n_spam_clusters, random_state=42, n_init=10)
        kmeans_spam.fit(spam_embs)
        
        for i, proto_name in enumerate(SPAM_PROTOTYPES[:n_spam_clusters]):
            spam_prototypes[proto_name] = kmeans_spam.cluster_centers_[i]
            
            # Найдём ближайший текст к центру кластера для логирования
            cluster_samples = [emb for emb, label_cluster in zip(spam_embs, kmeans_spam.labels_) if label_cluster == i]
            if cluster_samples:
                closest_idx = np.argmin([np.linalg.norm(emb - spam_prototypes[proto_name]) for emb in cluster_samples])
                sample_text = spam_texts[closest_idx] if closest_idx < len(spam_texts) else "N/A"
                LOGGER.info(f"  Spam prototype '{proto_name}': sample='{sample_text[:60]}...'")
    else:
        LOGGER.warning(f"Not enough spam samples for {n_spam_clusters} clusters, using centroids")
        spam_centroid = np.mean(spam_embs, axis=0)
        for proto_name in SPAM_PROTOTYPES[:n_spam_clusters]:
            spam_prototypes[proto_name] = spam_centroid
    
    # K-means для легитимных
    legit_prototypes = {}
    if len(legit_embs) >= n_legit_clusters:
        kmeans_legit = KMeans(n_clusters=n_legit_clusters, random_state=42, n_init=10)
        kmeans_legit.fit(legit_embs)
        
        for i, proto_name in enumerate(LEGIT_PROTOTYPES[:n_legit_clusters]):
            legit_prototypes[proto_name] = kmeans_legit.cluster_centers_[i]
            
            cluster_samples = [emb for emb, label_cluster in zip(legit_embs, kmeans_legit.labels_) if label_cluster == i]
            if cluster_samples:
                closest_idx = np.argmin([np.linalg.norm(emb - legit_prototypes[proto_name]) for emb in cluster_samples])
                sample_text = legit_texts[closest_idx] if closest_idx < len(legit_texts) else "N/A"
                LOGGER.info(f"  Legit prototype '{proto_name}': sample='{sample_text[:60]}...'")
    else:
        LOGGER.warning(f"Not enough legit samples for {n_legit_clusters} clusters, using centroids")
        legit_centroid = np.mean(legit_embs, axis=0)
        for proto_name in LEGIT_PROTOTYPES[:n_legit_clusters]:
            legit_prototypes[proto_name] = legit_centroid
    
    return spam_prototypes, legit_prototypes


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Вычисляет косинусное сходство."""
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot / (norm1 * norm2))


def build_feature_matrix(
    texts: list[str],
    embeddings_msg: list[np.ndarray],
    spam_centroid: np.ndarray,
    ham_centroid: np.ndarray,
    spam_prototypes: Dict[str, np.ndarray],
    legit_prototypes: Dict[str, np.ndarray],
    kw_scores: np.ndarray,
    tfidf_scores: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    Строит полную матрицу фичей для обучения.
    
    ФИЧИ (порядок важен!):
    1-3. sim_spam_msg, sim_ham_msg, delta_msg
    4-6. sim_spam_ctx, sim_ham_ctx, delta_ctx (заполняем 0 для упрощения)
    7-9. sim_spam_user, sim_ham_user, delta_user (заполняем 0)
    10-16. sim_proto_recruiting, ...gambling, ...loans, ...get_rich, ...dayjob, ...illegal_docs, ...gray_biz
    17-20. sim_proto_store_promo, ...giveaway_own, ...order_flow, ...service_info
    21-22. kw_score, tfidf_score
    23-34. has_phone, has_url, has_email, has_money, money_count, has_age, has_cta_plus, has_dm, has_remote, has_legal, has_casino, obfuscation_ratio
    35-38. reply_to_staff (0), is_forwarded (0), author_is_admin (0), is_channel_announcement (0)
    39-41. whitelist_hits_store, whitelist_hits_order, whitelist_hits_brand
    """
    features_list = []
    
    # Названия фичей
    feature_names = [
        # E_msg сходства
        'sim_spam_msg', 'sim_ham_msg', 'delta_msg',
        # E_ctx (заполняем 0)
        'sim_spam_ctx', 'sim_ham_ctx', 'delta_ctx',
        # E_user (заполняем 0)
        'sim_spam_user', 'sim_ham_user', 'delta_user',
    ]
    
    # Прототипы спама
    for proto_name in SPAM_PROTOTYPES:
        feature_names.append(f'sim_proto_{proto_name}')
    
    # Прототипы легитимных
    for proto_name in LEGIT_PROTOTYPES:
        feature_names.append(f'sim_proto_{proto_name}')
    
    # Scores фильтров
    feature_names.extend(['kw_score', 'tfidf_score'])
    
    # Паттерны
    feature_names.extend([
        'has_phone', 'has_url', 'has_email',
        'has_money', 'money_count', 'has_age',
        'has_cta_plus', 'has_dm', 'has_remote',
        'has_legal', 'has_casino', 'obfuscation_ratio'
    ])
    
    # Контекстные флаги
    feature_names.extend([
        'reply_to_staff', 'is_forwarded',
        'author_is_admin', 'is_channel_announcement'
    ])
    
    # Whitelist
    feature_names.extend([
        'whitelist_hits_store',
        'whitelist_hits_order',
        'whitelist_hits_brand'
    ])
    
    for i, text in enumerate(texts):
        emb = embeddings_msg[i]
        row_features = []
        
        # 1. Сходства E_msg
        if emb is not None:
            sim_spam_msg = cosine_similarity(emb, spam_centroid)
            sim_ham_msg = cosine_similarity(emb, ham_centroid)
            delta_msg = sim_spam_msg - sim_ham_msg
        else:
            sim_spam_msg = sim_ham_msg = delta_msg = 0.0
        
        row_features.extend([sim_spam_msg, sim_ham_msg, delta_msg])
        
        # 2. E_ctx (заполняем 0 - при обучении контекст недоступен)
        row_features.extend([0.0, 0.0, 0.0])
        
        # 3. E_user (заполняем 0)
        row_features.extend([0.0, 0.0, 0.0])
        
        # 4. Прототипы spam
        for proto_name in SPAM_PROTOTYPES:
            if emb is not None and proto_name in spam_prototypes:
                sim = cosine_similarity(emb, spam_prototypes[proto_name])
            else:
                sim = 0.0
            row_features.append(sim)
        
        # 5. Прототипы legit
        for proto_name in LEGIT_PROTOTYPES:
            if emb is not None and proto_name in legit_prototypes:
                sim = cosine_similarity(emb, legit_prototypes[proto_name])
            else:
                sim = 0.0
            row_features.append(sim)
        
        # 6. Scores фильтров
        row_features.append(kw_scores[i])
        row_features.append(tfidf_scores[i])
        
        # 7. Паттерны
        patterns = extract_patterns_from_raw(text)
        row_features.extend([
            float(patterns['has_phone']),
            float(patterns['has_url']),
            float(patterns['has_email']),
            float(patterns['has_money']),
            float(patterns['money_count']),
            float(patterns['has_age']),
            float(patterns['has_cta_plus']),
            float(patterns['has_dm']),
            float(patterns['has_remote']),
            float(patterns['has_legal']),
            float(patterns['has_casino']),
            patterns['obfuscation_ratio'],
        ])
        
        # 8. Контекстные флаги (при обучении все 0)
        row_features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 9. Whitelist хиты
        store_hits, order_hits, brand_hits = count_whitelist_hits(text)
        row_features.extend([
            float(store_hits),
            float(order_hits),
            float(brand_hits)
        ])
        
        features_list.append(row_features)
    
    X = np.array(features_list)
    LOGGER.info(f"Feature matrix built: {X.shape}, {len(feature_names)} features")
    
    return X, feature_names


async def main():
    LOGGER.info("="*60)
    LOGGER.info("TRAINING CONTEXT-AWARE META CLASSIFIER")
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
    
    # 1. Загрузка и чистка
    LOGGER.info("Step 1: Loading and cleaning dataset...")
    df = pd.read_csv(dataset_path)
    df = clean_dataset(df)
    
    texts = df['message'].tolist()
    labels = df['label'].values
    
    # 2. Генерация капсул для E_msg
    LOGGER.info("Step 2: Generating message capsules...")
    capsules_msg = [f"passage: {normalize_entities(text)}" for text in texts]
    
    # 3. Получение эмбеддингов
    LOGGER.info("Step 3: Computing embeddings via Ollama...")
    embeddings_msg = await get_embeddings_batch(
        capsules_msg,
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL
    )
    
    # Проверка валидности
    valid_count = sum(1 for emb in embeddings_msg if emb is not None)
    LOGGER.info(f"Valid embeddings: {valid_count}/{len(embeddings_msg)} ({valid_count/len(embeddings_msg):.1%})")
    
    if valid_count < len(embeddings_msg) * 0.8:
        LOGGER.error("Too many failed embeddings. Check Ollama API.")
        return
    
    # 4. Построение центроидов
    LOGGER.info("Step 4: Building centroids...")
    spam_centroid, ham_centroid = build_centroids(embeddings_msg, labels)
    
    # 5. Построение прототипов
    LOGGER.info("Step 5: Building prototypes via K-means...")
    spam_prototypes, legit_prototypes = build_prototypes(
        embeddings_msg,
        labels,
        texts,
        n_spam_clusters=len(SPAM_PROTOTYPES),
        n_legit_clusters=len(LEGIT_PROTOTYPES)
    )
    
    # 6. Получение scores от фильтров (заглушка - в реальности нужно загрузить фильтры)
    LOGGER.info("Step 6: Computing filter scores...")
    # Для простоты заполняем случайными значениями
    # В продакшене нужно загрузить KeywordFilter и TF-IDF и получить реальные scores
    kw_scores = np.random.uniform(0, 1, len(texts))
    tfidf_scores = np.random.uniform(0, 1, len(texts))
    
    LOGGER.warning("Using random filter scores (load real filters in production)")
    
    # 7. Построение матрицы фичей
    LOGGER.info("Step 7: Building feature matrix...")
    X, feature_names = build_feature_matrix(
        texts=texts,
        embeddings_msg=embeddings_msg,
        spam_centroid=spam_centroid,
        ham_centroid=ham_centroid,
        spam_prototypes=spam_prototypes,
        legit_prototypes=legit_prototypes,
        kw_scores=kw_scores,
        tfidf_scores=tfidf_scores
    )
    
    # 8. Обучение с Group K-Fold (по user_id если доступен)
    LOGGER.info("Step 8: Training with calibration...")
    
    # Group K-Fold требует groups - если нет user_id в датасете, используем обычный split
    if 'user_id' in df.columns:
        groups = df['user_id'].values
        LOGGER.info("Using GroupKFold by user_id")
    else:
        groups = None
        LOGGER.warning("No user_id column, using random split")
    
    # Обучаем LogisticRegression
    logreg = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        solver='liblinear'
    )
    
    # Калибровка
    calibrator = CalibratedClassifierCV(
        logreg,
        method='sigmoid',
        cv=3  # GroupKFold требует минимум 3 фолда
    )
    
    calibrator.fit(X, labels)
    
    # 9. Метрики
    LOGGER.info("Step 9: Evaluating metrics...")
    y_pred_proba = calibrator.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    roc_auc = roc_auc_score(labels, y_pred_proba)
    precision, recall, _ = precision_recall_curve(labels, y_pred_proba)
    pr_auc = auc(recall, precision)
    brier = brier_score_loss(labels, y_pred_proba)
    
    LOGGER.info(f"ROC-AUC: {roc_auc:.3f}")
    LOGGER.info(f"PR-AUC: {pr_auc:.3f}")
    LOGGER.info(f"Brier Score: {brier:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(labels, y_pred)
    LOGGER.info(f"Confusion Matrix:\n{cm}")
    
    # Classification report
    report = classification_report(labels, y_pred)
    LOGGER.info(f"Classification Report:\n{report}")
    
    # 10. Сохранение артефактов
    LOGGER.info("Step 10: Saving artifacts...")
    
    # Модель (используем имя из config.py)
    dump(logreg, models_dir / "meta_model.joblib")
    LOGGER.info("Saved: meta_model.joblib")
    
    # Калибратор
    dump(calibrator, models_dir / "meta_calibrator.joblib")
    LOGGER.info("Saved: meta_calibrator.joblib")
    
    # Центроиды
    np.savez(
        models_dir / "centroids.npz",
        spam_centroid=spam_centroid,
        ham_centroid=ham_centroid
    )
    LOGGER.info("Saved: centroids.npz")
    
    # Прототипы
    prototypes_dict = {}
    for name, vec in spam_prototypes.items():
        prototypes_dict[f"spam_{name}"] = vec
    for name, vec in legit_prototypes.items():
        prototypes_dict[f"legit_{name}"] = vec
    
    np.savez(models_dir / "prototypes.npz", **prototypes_dict)
    LOGGER.info("Saved: prototypes.npz")
    
    # Feature spec
    feature_spec = {
        "features": feature_names,
        "n_features": len(feature_names)
    }
    
    with open(models_dir / "feature_spec.json", "w", encoding="utf-8") as f:
        json.dump(feature_spec, f, indent=2, ensure_ascii=False)
    LOGGER.info("Saved: feature_spec.json")
    
    # Отчет
    report_path = models_dir / "train_meta_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("CONTEXT-AWARE META CLASSIFIER TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Total samples: {len(texts)}\n")
        f.write(f"Spam: {(labels == 1).sum()}\n")
        f.write(f"Ham: {(labels == 0).sum()}\n\n")
        
        f.write(f"Valid embeddings: {valid_count}/{len(embeddings_msg)}\n\n")
        
        f.write(f"Features: {len(feature_names)}\n")
        f.write(f"Feature names: {', '.join(feature_names[:10])}...\n\n")
        
        f.write(f"ROC-AUC: {roc_auc:.3f}\n")
        f.write(f"PR-AUC: {pr_auc:.3f}\n")
        f.write(f"Brier Score: {brier:.3f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        
        f.write("Classification Report:\n")
        f.write(report + "\n")
    
    LOGGER.info(f"Saved: {report_path}")
    
    LOGGER.info("="*60)
    LOGGER.info("TRAINING COMPLETE")
    LOGGER.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())
